#!/usr/bin/env python3
"""
ambientmapper genotyping — probabilistic merge + summarize (+ optional decontam)

This module consumes per-read multi-genome evidence exported by `assign` and
produces per-cell calls (single / doublet / indistinguishable / ambiguous),
with ambient-aware priors, optional design priors, and a summary report.

CLI (Typer):
  ambientmapper genotyping \
      --assign 'work/assign_chunks/*.parquet' \
      --outdir work/merge \
      [--design design.csv] [--sample SAMPLE] \
      [--min-reads 300] [--beta 0.5] [--tau-drop 8]

Inputs expected from `assign` (flexible; auto-detected when possible):
  Required columns:
    - barcode (str)
    - read_id (str or int)
    - genome (str)
  Recommended score columns (at least one of AS/MAPQ/NM should exist):
    - AS (alignment score, higher better)
    - MAPQ (mapping quality, higher better)
    - NM (edits / mismatches, lower better)
  Optional per-genome p-values for the winning-vs-runner-up ambiguity model:
    - p_as, p_mq  (0..1, lower means stronger winner)

Design priors (optional): CSV with any of:
  - barcode (exact) → {genome: weight,...}
  - plate, well, genome_prior_* columns, OR a wide one-hot schema per genome.
  Use --design-key to choose key column among {barcode, plate, well}.

Outputs:
  - <outdir>/<sample>_cells_calls.tsv.gz         (per-cell calls + QC)
  - <outdir>/<sample>_Reads_to_discard.csv.gz    (optional per-read drops)
  - <outdir>/<sample>_BCs_PASS_by_mapping.csv    (legacy compatible summary)
  - <outdir>/<sample>_qc_report.pdf              (summary plots)

Notes:
  * This file is intentionally self-contained. If you later want to split into
    submodules (io.py, model.py, summarize.py), you can copy the corresponding
    sections out cleanly.
  * The likelihood used here is a pragmatic composite likelihood based on
    per-read soft posteriors from `assign`. It is calibrated for model selection
    (BIC) and decontamination heuristics as discussed with Fabio.
"""
from __future__ import annotations

import gzip
import io
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import typer
from pydantic import BaseModel
from tqdm import tqdm
import glob


# Optional but nice to have for summary plots
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except Exception:
    HAS_PLOTTING = False

app = typer.Typer(add_completion=False, no_args_is_help=True)

# ------------------------------
# Config models
# ------------------------------

class MergeConfig(BaseModel):
    beta: float = 0.5                 # softmax temperature for score fusion
    w_as: float = 1.0                 # weight for AS (higher better)
    w_mapq: float = 0.5               # weight for MAPQ (higher better)
    w_nm: float = 0.25                # weight for NM (lower better)
    ambient_const: float = 1e-3       # per-read ambient mass before renorm
    p_eps: float = 1e-3               # floor for p-value penalty gamma = max(eps,1-p)
    min_reads: int = 300              # minimum reads to attempt a confident call
    single_mass_min: float = 0.85     # mass threshold for single call
    doublet_minor_min: float = 0.20   # minor fraction threshold for doublet
    bic_margin: float = 6.0           # ΔBIC to accept more complex model
    near_tie_margin: float = 2.0      # ΔBIC below which genomes are indistinguishable
    tau_drop: float = 8.0             # posterior-odds threshold to drop contrary reads
    alpha_grid: float = 0.02          # step for ambient fraction grid in [0, 0.5]
    rho_grid: float = 0.05            # step for doublet mixture grid in [0.1, 0.9]
    max_alpha: float = 0.5            # cap ambient fraction
    topk_genomes: int = 3             # candidate genomes per barcode
    sample: str = "sample"

# ------------------------------
# I/O helpers
# ------------------------------

def _read_table(path: Path) -> pd.DataFrame:
    suf = "".join(path.suffixes).lower()
    # Parquet (plain or gz)
    if suf.endswith(".parquet") or suf.endswith(".parquet.gz"):
        return pd.read_parquet(path)
    # Delimited (plain or gz)
    if suf.endswith(".csv") or suf.endswith(".csv.gz"):
        return pd.read_csv(path)
    if suf.endswith(".tsv") or suf.endswith(".tsv.gz") or suf.endswith(".txt") or suf.endswith(".txt.gz"):
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported file format: {path}")

def _coerce_assign_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns from assign outputs to the schema expected by genotyping."""
    rename = {}
    if "BC" in df.columns: rename["BC"] = "barcode"
    if "Read" in df.columns: rename["Read"] = "read_id"
    if "Genome" in df.columns: rename["Genome"] = "genome"
    if "p_as_decile" in df.columns: rename["p_as_decile"] = "p_as"
    if "p_mq_decile" in df.columns: rename["p_mq_decile"] = "p_mq"
    out = df.rename(columns=rename)

    for col in ("barcode", "read_id", "genome"):
        if col in out.columns:
            out[col] = out[col].astype(str)

    # If p_* are deciles in 1..10, scale to 0..1
    for pcol in ("p_as", "p_mq"):
        if pcol in out.columns:
            out[pcol] = pd.to_numeric(out[pcol], errors="coerce")
            if out[pcol].max(skipna=True) and out[pcol].max(skipna=True) > 1.0:
                out[pcol] = out[pcol] / 10.0

    return out


def _reduce_alignments_to_per_genome(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse alignment-level evidence to one row per (barcode, read_id, genome).
    Strategy: AS=max, MAPQ=max, NM=min, and keep p-values if present.
    """
    have = set(df.columns)
    keys = ["barcode", "read_id", "genome"]
    for k in keys:
        if k not in have:
            raise ValueError(f"Expected column '{k}' after schema coercion.")

    agg = {}
    if "AS" in have: agg["AS"] = "max"
    if "MAPQ" in have: agg["MAPQ"] = "max"
    if "NM" in have: agg["NM"] = "min"
    # p-values are per (read,BC) not per-genome; keep min if multiple show up
    if "p_as" in have: agg["p_as"] = "min"
    if "p_mq" in have: agg["p_mq"] = "min"

    if not agg:
        # we must at least have the keys; return distinct keys
        return df.drop_duplicates(keys)[keys]

    return (
        df.groupby(keys, observed=True)
          .agg(agg)
          .reset_index()
    )

def _process_one(args):
    bc, L_block, cand, eta_dict, cfg_dict = args
    # Rehydrate config and ambient profile
    cfg = MergeConfig(**cfg_dict)
    eta = pd.Series(eta_dict)

    read_count = L_block["read_id"].nunique()
    if read_count < cfg.min_reads:
        res = {
            "barcode": bc, "call": "ambiguous", "genome_1": None, "genome_2": None,
            "alpha": np.nan, "rho": np.nan, "purity": 0.0, "minor": 0.0, "bic_best": np.nan,
            "bic_single": np.nan, "bic_doublet": np.nan, "delta_bic": np.nan,
            "n_reads": read_count, "n_effective": float(L_block["L"].sum()),
            "concordance": np.nan, "indistinguishable_set": "", "notes": "low_reads"
        }
        return res, pd.DataFrame()

    best = _select_model_for_barcode(L_block, eta, cfg, cand)
    best["barcode"] = bc
    best["notes"] = ""

    drops = _reads_to_discard(L_block, best, cfg)
    return best, drops


REQUIRED_COLS = {"barcode", "read_id", "genome"}
SCORE_COLS = ["AS", "MAPQ", "NM"]
PVAL_COLS = ["p_as", "p_mq"]

# ------------------------------
# Core math utilities
# ------------------------------

def _zscore_series(x: np.ndarray) -> np.ndarray:
    # robust z: center by median, scale by MAD (fallback to std if MAD small)
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad < 1e-6:
        s = x.std(ddof=0)
        s = 1.0 if s < 1e-6 else s
        return (x - med) / s
    return (x - med) / (1.4826 * mad)

def _fuse_support(df_grp: pd.DataFrame, cfg: MergeConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Given rows for one read (same barcode+read_id) across genomes,
    return arrays aligned to df_grp rows: fused score S, gamma penalty, softmax weights.
    """
    S_parts = []
    if "AS" in df_grp.columns:
        S_parts.append(cfg.w_as * _zscore_series(df_grp["AS"].to_numpy(dtype=float)))
    if "MAPQ" in df_grp.columns:
        S_parts.append(cfg.w_mapq * _zscore_series(df_grp["MAPQ"].to_numpy(dtype=float)))
    if "NM" in df_grp.columns:
        S_parts.append(-cfg.w_nm * _zscore_series(df_grp["NM"].to_numpy(dtype=float)))  # lower is better
    S = np.sum(S_parts, axis=0) if S_parts else np.zeros(len(df_grp), dtype=float)

    if "p_as" in df_grp.columns or "p_mq" in df_grp.columns:
        pa = df_grp["p_as"].to_numpy(dtype=float) if "p_as" in df_grp.columns else np.ones(len(df_grp))
        pm = df_grp["p_mq"].to_numpy(dtype=float) if "p_mq" in df_grp.columns else np.ones(len(df_grp))
        pmin = np.minimum(pa, pm)
        gamma = np.maximum(cfg.p_eps, 1.0 - pmin)
    else:
        gamma = np.ones(len(df_grp), dtype=float)

    w = np.exp(cfg.beta * S) * gamma
    return S, gamma, w


def _compute_read_posteriors(df: pd.DataFrame, cfg: MergeConfig) -> pd.DataFrame:
    """Compute per-read soft posteriors L_{r,g} from assign rows.

    Input df columns must include REQUIRED_COLS and ideally some SCORE_COLS and/or PVAL_COLS.
    Output DataFrame has columns: barcode, read_id, genome, L (posterior), L_amb (same for all rows of a read),
    and any passthrough columns present in input.
    """
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"assign table missing required columns: {missing}")

    # sort so groupby is stable & memory friendly
    df = df.sort_values(["barcode", "read_id"]).reset_index(drop=True)
    df["barcode"] = df["barcode"].astype("category")
    df["genome"] = df["genome"].astype("category")

    records = []
    # Efficient group iteration
    for (_, _), grp in df.groupby(["barcode", "read_id"], sort=False):
        S, gamma, w = _fuse_support(grp, cfg)
        # ambient const mass
        amb = cfg.ambient_const
        total = w.sum() + amb
        L = w / total
        L_amb = amb / total
        out = grp[["barcode", "read_id", "genome"]].copy()
        out["L"] = L
        out["L_amb"] = L_amb
        records.append(out)
    return pd.concat(records, ignore_index=True)


def _aggregate_expected_counts(Ldf: pd.DataFrame) -> pd.DataFrame:
    """Return per-barcode expected counts per genome: C_{b,g} = sum_r L_{r,g}.
    Output columns: barcode, genome, C, n_reads.
    """
    grp = Ldf.groupby(["barcode", "genome"], observed=True)["L"].sum().rename("C").reset_index()
    n_reads = Ldf.groupby("barcode")["read_id"].nunique().rename("n_reads").reset_index()
    return grp.merge(n_reads, on="barcode", how="left")


def _estimate_ambient_profile(Ldf: pd.DataFrame, low_read_threshold: int = 200) -> pd.Series:
    """Crude ambient estimate: use low-read barcodes to infer genome mix.
    Return pd.Series indexed by genome with probabilities summing to 1.
    """
    n_by_bc = Ldf.groupby("barcode")["read_id"].nunique()
    empties = set(n_by_bc[n_by_bc < low_read_threshold].index)
    if not empties:
        # fallback: use all reads (rare)
        df_use = Ldf
    else:
        df_use = Ldf[Ldf["barcode"].isin(empties)]
    mix = df_use.groupby("genome")["L"].sum()
    s = mix.sum()
    if s <= 0:
        # uniform fallback
        mix = pd.Series(1.0, index=sorted(Ldf["genome"].unique()))
        s = mix.sum()
    eta = mix / s
    return eta

# ------------------------------
# Model selection per barcode
# ------------------------------

def _loglik_for_params(L_block: pd.DataFrame, eta: pd.Series, 
                       model: str, g1: str, g2: Optional[str],
                       alpha: float, rho: float = 0.5) -> float:
    """Composite log-likelihood for a barcode block of rows (one per (read,genome)).

    L_block columns: genome, read_id, L, L_amb

    Model:
      - single: theta_g = 1 for g1; effective mixture over genomes = (1-alpha)*delta_{g1} + alpha*eta
      - doublet: (1-alpha)*(rho*delta_{g1} + (1-rho)*delta_{g2}) + alpha*eta

    The per-read likelihood is sum_g w_g * L_{r,g}, where w_g is the effective mixture weight.
    We do not include L_amb separately here because eta already absorbs ambient genome mix.
    """
    # Precompute effective weights per genome
    genomes = L_block["genome"].unique()
    eta_g = eta.reindex(genomes).fillna(0.0).to_numpy()

    if model == "single":
        w = np.where(L_block["genome"].to_numpy() == g1,
                     (1.0 - alpha) + alpha * eta.reindex([g1]).fillna(0.0).to_numpy()[0],
                     alpha * eta_g)
    elif model == "doublet":
        is_g1 = (L_block["genome"].to_numpy() == g1)
        is_g2 = (L_block["genome"].to_numpy() == g2)
        mix = np.where(is_g1, rho, np.where(is_g2, 1.0 - rho, 0.0))
        w = (1.0 - alpha) * mix + alpha * eta_g
    else:
        raise ValueError("model must be 'single' or 'doublet'")

    # Per-read sums: sum over genomes present for that read
    # Build by grouping by read_id
    arr_wL = w * L_block["L"].to_numpy()
    df_tmp = L_block[["read_id"]].copy()
    df_tmp["wL"] = arr_wL
    s = df_tmp.groupby("read_id")["wL"].sum().to_numpy()

    # Numerical safety
    s = np.clip(s, 1e-12, None)
    return float(np.log(s).sum())


def _bic(loglik: float, n_params: int, n_reads: int) -> float:
    return -2.0 * loglik + n_params * math.log(max(n_reads, 1))


def _select_model_for_barcode(L_block: pd.DataFrame, eta: pd.Series, cfg: MergeConfig,
                              candidate_genomes: Sequence[str]) -> Dict:
    """Evaluate single and (optionally) doublet models for a barcode and pick best by BIC.
    Returns a dict with call, genomes, alpha, rho, bic values, purity, etc.
    """
    read_count = L_block["read_id"].nunique()

    # Single candidates: each genome in candidate_genomes
    # Doublet candidates: all unordered pairs among candidate_genomes
    alphas = np.arange(0.0, cfg.max_alpha + 1e-9, cfg.alpha_grid)
    rhos = np.arange(0.1, 0.9 + 1e-9, cfg.rho_grid)

    best = {"bic": float("inf"), "model": None}
    evals = []

    # SINGLE MODELS
    for g1 in candidate_genomes:
        best_local = {"bic": float("inf")}
        for a in alphas:
            ll = _loglik_for_params(L_block, eta, model="single", g1=g1, g2=None, alpha=a)
            bic = _bic(ll, n_params=1, n_reads=read_count)  # only alpha
            if bic < best_local["bic"]:
                best_local = {"model": "single", "g1": g1, "alpha": a, "rho": 1.0, "ll": ll, "bic": bic}
        evals.append(best_local)
        if best_local["bic"] < best["bic"]:
            best = best_local

    # DOUBLET MODELS
    if len(candidate_genomes) >= 2:
        for i in range(len(candidate_genomes)):
            for j in range(i + 1, len(candidate_genomes)):
                g1, g2 = candidate_genomes[i], candidate_genomes[j]
                best_local = {"bic": float("inf")}
                for a in alphas:
                    for r in rhos:
                        ll = _loglik_for_params(L_block, eta, model="doublet", g1=g1, g2=g2, alpha=a, rho=r)
                        bic = _bic(ll, n_params=2, n_reads=read_count)  # alpha + rho
                        if bic < best_local["bic"]:
                            best_local = {"model": "doublet", "g1": g1, "g2": g2, "alpha": a, "rho": r, "ll": ll, "bic": bic}
                evals.append(best_local)
                if best_local["bic"] < best["bic"]:
                    best = best_local

    # Derive additional QC metrics
    # Compute mass per genome (expected from L) under the best model, ignoring ambient part
    mass = L_block.groupby("genome")["L"].sum().sort_values(ascending=False)
    

    # default outputs
    out = {
        "model": best.get("model", None),
        "genome_1": best.get("g1", None),
        "genome_2": best.get("g2", None),
        "alpha": best.get("alpha", 0.0),
        "rho": best.get("rho", 1.0),
        "bic_best": best.get("bic", float("inf")),
        "bic_single": min([e["bic"] for e in evals if e.get("model") == "single"], default=float("inf")),
        "bic_doublet": min([e["bic"] for e in evals if e.get("model") == "doublet"], default=float("inf")),
        "n_reads": read_count,
        "n_effective": float(L_block["L"].sum()),
    }
    out["delta_bic"] = min(out["bic_doublet"], out["bic_single"]) - out["bic_best"]

    # Purity definition: (1-alpha) * major component weight inside model
    if out["model"] == "single":
        purity = (1.0 - out["alpha"])  # all non-ambient goes to g1
        minor = 0.0
    elif out["model"] == "doublet":
        purity = (1.0 - out["alpha"]) * max(out["rho"], 1.0 - out["rho"])
        minor = min(out["rho"], 1.0 - out["rho"]) * (1.0 - out["alpha"]) 
    else:
        purity, minor = 0.0, 0.0

    out["purity"] = purity
    out["minor"] = minor

    # Determine call per decision rules
    call = "ambiguous"
    indist = None

    # Near-tie check among singletons (indistinguishable genomes)
    singles_sorted = sorted([e for e in evals if e.get("model") == "single"], key=lambda x: x["bic"])[:2]
    if len(singles_sorted) == 2 and (singles_sorted[1]["bic"] - singles_sorted[0]["bic"]) < cfg.near_tie_margin:
        indist = (singles_sorted[0]["g1"], singles_sorted[1]["g1"])  # report pair

    if out["model"] == "single" and out["purity"] >= cfg.single_mass_min and (out["bic_doublet"] - out["bic_best"]) >= cfg.bic_margin:
        call = "single"
    elif out["model"] == "doublet" and out["minor"] >= cfg.doublet_minor_min and (out["bic_single"] - out["bic_best"]) >= cfg.bic_margin:
        call = "doublet"
    elif indist is not None:
        call = "indistinguishable"

    out["call"] = call
    out["indistinguishable_set"] = ",".join(indist) if indist is not None else ""

    # Concordance: fraction of reads whose argmax genome equals chosen major genome
    # Compute per-read argmax
    maj = out["genome_1"]
    if maj is None:
        concord = 0.0
    else:
        # For each read, sum L by genome, argmax, compare to maj
        x = L_block.groupby(["read_id", "genome"])["L"].sum().reset_index()
        argmax = x.sort_values(["read_id", "L"], ascending=[True, False]).drop_duplicates("read_id")["genome"]
        concord = float((argmax == maj).mean())
    out["concordance"] = concord

    return out

# ------------------------------
# Decontamination (per-read drop list)
# ------------------------------

def _reads_to_discard(L_block: pd.DataFrame, call_row: Dict, cfg: MergeConfig) -> pd.DataFrame:
    """Given per-read posteriors for a barcode and its call, decide which reads to drop.
    Returns DataFrame with: barcode, read_id, top_genome, assigned_genome, L_top, L_assigned, posterior_odds, reason
    """
    call = call_row.get("call")
    g1 = call_row.get("genome_1")
    g2 = call_row.get("genome_2")

    # aggregate per-read per-genome L to ensure unique entries
    X = L_block.groupby(["read_id", "genome"], observed=True)["L"].sum().reset_index()
    # per read argmax
    idx = X.sort_values(["read_id", "L"], ascending=[True, False]).drop_duplicates("read_id").index
    X_top = X.loc[idx, ["read_id", "genome", "L"]].rename(columns={"genome": "top_genome", "L": "L_top"})
    # merge assigned genome L (could be NaN if not present for a read)
    if call in ("single", "indistinguishable"):
        assigned = g1
    else:
        assigned = None  # for doublets we do not drop

    if assigned is None:
        return pd.DataFrame(columns=["barcode", "read_id", "top_genome", "assigned_genome", "L_top", "L_assigned", "posterior_odds", "reason"]).astype({"barcode": str})

    L_assigned = X[X["genome"] == assigned][["read_id", "L"]].rename(columns={"L": "L_assigned"})
    M = X_top.merge(L_assigned, on="read_id", how="left")
    M["L_assigned"] = M["L_assigned"].fillna(0.0)
    # posterior odds: (max other) / (assigned)
    M["posterior_odds"] = np.where(M["L_assigned"] > 0, M["L_top"] / M["L_assigned"], np.inf)

    # reason for drop if top != assigned and odds >= tau
    M["reason"] = np.where((M["top_genome"] != assigned) & (M["posterior_odds"] >= cfg.tau_drop), "contrary_genome", "")
    M = M[M["reason"] != ""].copy()
    M["assigned_genome"] = assigned
    # attach barcode
    bc = L_block["barcode"].iloc[0]
    M["barcode"] = bc
    return M[["barcode", "read_id", "top_genome", "assigned_genome", "L_top", "L_assigned", "posterior_odds", "reason"]]

# ------------------------------
# Summaries / plots
# ------------------------------

def _write_gzip_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        df.to_csv(f, index=False)


def _render_qc_report(calls: pd.DataFrame, out_pdf: Path, sample: str) -> None:
    if not HAS_PLOTTING:
        typer.echo("[warn] matplotlib/seaborn not available; skipping QC report")
        return
    sns.set_context("talk")
    fig = plt.figure(figsize=(12, 10))
    gs = fig.add_gridspec(2, 2)

    ax0 = fig.add_subplot(gs[0, 0])
    sns.histplot(calls["n_reads"], bins=50, ax=ax0)
    ax0.set_title(f"Reads per cell — {sample}")

    ax1 = fig.add_subplot(gs[0, 1])
    sns.histplot(calls["purity"], bins=50, ax=ax1)
    ax1.set_title("Purity distribution")

    ax2 = fig.add_subplot(gs[1, 0])
    sns.scatterplot(data=calls, x="n_reads", y="purity", hue="call", ax=ax2, s=12)
    ax2.set_title("Purity vs Reads")
    ax2.legend(markerscale=2)

    ax3 = fig.add_subplot(gs[1, 1])
    sns.countplot(data=calls, x="call", ax=ax3)
    ax3.set_title("Calls count")

    fig.tight_layout()
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)

# ------------------------------
# Main driver
# ------------------------------

def _load_assign_tables(assign_glob: str) -> pd.DataFrame:
  files = [Path(p) for p in glob.glob(assign_glob, recursive=True)]
  if not files:
    raise FileNotFoundError(f"No assign files found for pattern: {assign_glob}")
  frames = []
  for fp in files:
    df = _read_table(fp)
    df = _coerce_assign_schema(df)
    
    # Ensure required keys exist; if not, warn & skip this file
    if not {"barcode", "read_id", "genome"} <= set(df.columns):
      typer.echo(f"[warn] Missing required columns in {fp.name}; skipping.")
      continue

    df = _reduce_alignments_to_per_genome(df)
    frames.append(df)
        
  if not frames:
    raise ValueError("No valid assign tables after schema coercion; aborting.")
  
  out = pd.concat(frames, ignore_index=True)
  out = out.dropna(subset=["barcode","read_id","genome"])
  return out

def _topk_genomes_per_barcode(C: pd.DataFrame, k: int) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for bc, sub in C.groupby("barcode", sort=False):
        top = sub.sort_values("C", ascending=False)["genome"].head(k).tolist()
        out[bc] = top
    return out


@app.command("genotyping")
def genotyping(
    assign: str = typer.Option(..., help="Path or glob to assign outputs (csv/tsv/parquet, possibly gz)."),
    outdir: Path = typer.Option(Path("merge_out"), help="Output directory."),
    sample: str = typer.Option("sample", help="Sample name used for output filenames."),
    min_reads: int = typer.Option(300, help="Minimum reads to attempt confident calls."),
    beta: float = typer.Option(0.5, help="Softmax temperature for score fusion."),
    w_as: float = typer.Option(1.0, help="Weight for AS."),
    w_mapq: float = typer.Option(0.5, help="Weight for MAPQ."),
    w_nm: float = typer.Option(0.25, help="Weight for NM (penalty)."),
    ambient_const: float = typer.Option(1e-3, help="Ambient constant mass per read before normalization."),
    tau_drop: float = typer.Option(8.0, help="Posterior-odds threshold to drop contrary reads in singles."),
    topk_genomes: int = typer.Option(3, help="Number of candidate genomes per barcode to consider."),
    make_report: bool = typer.Option(True, help="Render QC PDF report."),
):
    """Run the full merge pipeline: read-level posteriors → ambient estimate → per-cell calls → optional decontam → summaries."""
    cfg = MergeConfig(
        beta=beta, w_as=w_as, w_mapq=w_mapq, w_nm=w_nm,
        ambient_const=ambient_const, min_reads=min_reads,
        tau_drop=tau_drop, topk_genomes=topk_genomes, sample=sample,
    )

    typer.echo("[1/5] Loading assign outputs…")
    df = _load_assign_tables(assign)
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise typer.BadParameter(f"Assign inputs missing required columns: {missing}")

    typer.echo("[2/5] Computing per-read posteriors…")
    Ldf = _compute_read_posteriors(df, cfg)

    typer.echo("[3/5] Aggregating expected counts & estimating ambient profile…")
    C = _aggregate_expected_counts(Ldf)
    eta = _estimate_ambient_profile(Ldf)

    typer.echo("[4/5] Per-cell model selection & calls…")
    # Build candidate genomes per barcode
    topk = _topk_genomes_per_barcode(C, cfg.topk_genomes)

    # Evaluate models per barcode
    rows = []
    drops = []
    for bc, L_block in tqdm(Ldf.groupby("barcode"), total=Ldf["barcode"].nunique()):
        cand = topk.get(bc, [])
        if len(cand) == 0:
            continue
        # Low-depth guard
        read_count = L_block["read_id"].nunique()
        if read_count < cfg.min_reads:
            res = {
                "barcode": bc, "call": "ambiguous", "genome_1": None, "genome_2": None,
                "alpha": np.nan, "rho": np.nan, "purity": 0.0, "minor": 0.0, "bic_best": np.nan,
                "bic_single": np.nan, "bic_doublet": np.nan, "delta_bic": np.nan,
                "n_reads": read_count, "n_effective": float(L_block["L"].sum()),
                "concordance": np.nan, "indistinguishable_set": "", "notes": "low_reads"
            }
            rows.append(res)
            continue

        best = _select_model_for_barcode(L_block, eta, cfg, cand)
        best["barcode"] = bc
        best["notes"] = ""
        rows.append(best)

        # per-read drops for singles/indistinguishable
        D = _reads_to_discard(L_block, best, cfg)
        if len(D):
            drops.append(D)

    calls = pd.DataFrame(rows)
    if drops:
        drops_df = pd.concat(drops, ignore_index=True)
    else:
        drops_df = pd.DataFrame(columns=["barcode", "read_id", "top_genome", "assigned_genome", "L_top", "L_assigned", "posterior_odds", "reason"]).astype({"barcode": str})

    # Legacy-compatible summary (PASS_by_mapping), keep but now enriched
    legacy = calls.copy()
    legacy["AssignedGenome"] = legacy["genome_1"].fillna("")
    pass_mask = legacy["call"].isin(["single", "doublet", "indistinguishable"])  # you can tighten if desired
    legacy_out = legacy.loc[pass_mask, ["barcode", "AssignedGenome", "call", "purity", "n_reads"]]

    outdir.mkdir(parents=True, exist_ok=True)
    _write_gzip_df(calls, outdir / f"{sample}_cells_calls.tsv.gz")
    _write_gzip_df(drops_df, outdir / f"{sample}_Reads_to_discard.csv.gz")
  #
    legacy_out.to_csv(outdir / f"{sample}_BCs_PASS_by_mapping.csv", index=False)

    # Optional QC PDF
    if make_report and not calls.empty:
        _render_qc_report(calls, outdir / f"{sample}_qc_report.pdf", sample=sample)

    # Optional debugging matrix
    if not C.empty:
        mat = (
            C.pivot(index="barcode", columns="genome", values="C")
             .sort_index(axis=1)
             .fillna(0.0)
        )
        mat.to_csv(outdir / f"{sample}_expected_counts_by_genome.csv")
        # or: mat.to_csv(outdir / f"{sample}_expected_counts_by_genome.csv.gz")

    typer.echo("[5/5] Done.")

if __name__ == "__main__":
    app()
