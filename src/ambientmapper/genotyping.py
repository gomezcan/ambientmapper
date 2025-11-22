#!/usr/bin/env python3
"""
ambientmapper genotyping — posterior-aware merge → per-cell genotype calls (+ optional decontam)
Two-pass, streaming implementation.

This module consumes per-read, multi-genome evidence exported by `assign` and
emits per-cell genotype calls: {single, doublet, indistinguishable, ambiguous}.
All decisions are data-driven from `assign` outputs (no plate/design priors here).
An optional QC report and per-read drop list (for single/indistinguishable calls)
are produced to aid downstream cleaning.

Two-pass execution
  Pass 1: Stream input TSV.GZ chunks → compute per-read posteriors L(r,g),
          accumulate expected counts C(b,g) + n_reads(b), and spill per-read L
          into barcode-hashed shard files on disk.
  Pass 2: Walk shard files → group rows per barcode → model selection (single vs
          doublet with ambient) → write calls and optional read-drop table.

CLI (Typer):
  ambientmapper genotyping \
      --assign 'work/cell_map_ref_chunks/*_filtered.tsv.gz' \
      --outdir work/merge --sample SAMPLE \
      [--min-reads 300] [--beta 0.5] [--tau-drop 8] \
      [--topk-genomes 3] [--ambient-const 1e-3] \
      [--report/--no-report] [--threads 1]

Inputs expected from `assign` (flexible; auto-detected when possible)
  Required columns (after coercion):
    - barcode (str)
    - read_id (str or int)
    - genome (str)
  Recommended score columns (any subset; improves separation):
    - AS   (alignment score; higher is better)
    - MAPQ (mapping quality; higher is better)
    - NM   (edits/mismatches; lower is better)
  Optional per-read ambiguity p-values (if your assign pipeline emits them):
    - p_as, p_mq  — in [0,1]; smaller = stronger winner-vs-runner-up evidence

Behavior (high level)
  1) Schema-coerce and collapse to one row per (barcode, read_id, genome):
       AS=max; MAPQ=max; NM=min; p_* = min
  2) For each read, fuse available scores with robust z-scaling and a p-value
     penalty; softmax (temperature beta) + tiny ambient mass → soft posteriors L(r,g).
  3) Aggregate expected counts per (barcode, genome): C(b,g) = Σ_r L(r,g).
     Estimate an ambient profile eta(g) from low-read barcodes (<200 reads).
  4) For each barcode, pick top-k candidate genomes by C(b,g). Grid-search
     single vs doublet models with ambient (alpha) and, for doublets, mixture (rho);
     select by BIC with conservative margins.
  5) Emit call + QC metrics.

Outputs (current simplified design)
  - <outdir>/<sample>_cells_calls.tsv.gz
      One row per barcode with:
        call, genome_1, genome_2, alpha, rho, purity, minor,
        bic_best, bic_single, bic_doublet, delta_bic,
        n_reads, n_effective, concordance, indistinguishable_set, notes,
        p_top1, p_top2, p_top3, top3_sum, entropy,
        top_genome, best_genome, status_flag, ...
  - <outdir>/<sample>_BCs_PASS_by_mapping.csv
      Legacy-compatible summary: barcode, AssignedGenome (=best_genome), call, purity, n_reads.
  - <outdir>/<sample>_expected_counts_by_genome.csv
      Barcode × genome expected-count matrix (debug/inspection).

Notes
  * This module is intentionally self-contained. If you later split into
    io.py / model.py / summarize.py, sections can be moved cleanly.
  * The likelihood is a pragmatic composite built on per-read soft posteriors
    from `assign`, tuned for model selection (BIC) and downstream heuristics.
"""

from __future__ import annotations

import gzip
import json
import math
import os
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob
import hashlib
import tempfile
import shutil
import warnings
from itertools import islice
from typing import Set

import numpy as np
import pandas as pd
import typer
from pydantic import BaseModel
from tqdm import tqdm

# Optional but nice to have for summary plots (currently unused)
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from typer.models import OptionInfo
    HAS_PLOTTING = True
except Exception:
    HAS_PLOTTING = False

# Ensure OptionInfo exists even if plotting import failed
try:
    from typer.models import OptionInfo  # type: ignore[assignment]
except Exception:
    class OptionInfo:  # fallback so isinstance(x, OptionInfo) is valid
        pass

app = typer.Typer(add_completion=False, no_args_is_help=True)

# ------------------------------
# Config models
# ------------------------------

class MergeConfig(BaseModel):
    beta: float = 0.5                 # softmax temperature for score fusion
    w_as: float = 0.5                 # weight for AS (higher better)
    w_mapq: float = 1                 # weight for MAPQ (higher better)
    w_nm: float = 1                   # weight for NM (lower better)
    ambient_const: float = 1e-3       # per-read ambient mass before renorm
    p_eps: float = 1e-3               # floor for p-value penalty gamma = max(eps,1-p)
    min_reads: int = 100              # minimum reads to attempt a confident call
    single_mass_min: float = 0.85     # mass threshold for single call
    doublet_minor_min: float = 0.20   # minor fraction threshold for doublet
    bic_margin: float = 6.0           # ΔBIC to accept more complex model
    near_tie_margin: float = 2.0      # ΔBIC below which genomes are indistinguishable
    tau_drop: float = 8.0             # posterior-odds threshold (now unused)
    alpha_grid: float = 0.02          # step for ambient fraction grid in [0, 0.5]
    rho_grid: float = 0.05            # step for doublet mixture grid in [0.1, 0.9]
    max_alpha: float = 0.5            # cap ambient fraction
    topk_genomes: int = 3             # candidate genomes per barcode
    sample: str = "sample"
    shards: int = 32                 # number of barcode-hash shards
    chunk_rows: int = 5_000_000       # streaming chunk size for input TSVs

# ------------------------------
# I/O helpers
# ------------------------------

NEEDED_COLS: Sequence[str] = ("barcode", "read_id", "genome", "AS", "MAPQ", "NM")
DTYPES: Dict[str, str] = {
    "barcode": "string",
    "read_id": "string",
    "genome": "string",   # cast to category later
    "AS": "int32",
    "MAPQ": "int16",
    "NM": "int16",
}

REQUIRED_COLS = {"barcode", "read_id", "genome"}
SCORE_COLS = ["AS", "MAPQ", "NM"]
PVAL_COLS = ["p_as", "p_mq"]


def _coerce_assign_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns from assign outputs to the schema expected by genotyping."""
    rename: Dict[str, str] = {}
    if "BC" in df.columns:
        rename["BC"] = "barcode"
    if "Read" in df.columns:
        rename["Read"] = "read_id"
    if "Genome" in df.columns:
        rename["Genome"] = "genome"
    if "p_as_decile" in df.columns:
        rename["p_as_decile"] = "p_as"
    if "p_mq_decile" in df.columns:
        rename["p_mq_decile"] = "p_mq"
    out = df.rename(columns=rename)

    for col in ("barcode", "read_id", "genome"):
        if col in out.columns:
            out[col] = out[col].astype(str)

    # If p_* are deciles in 1..10, scale to 0..1
    for pcol in ("p_as", "p_mq"):
        if pcol in out.columns:
            out[pcol] = pd.to_numeric(out[pcol], errors="coerce")
            max_val = out[pcol].max(skipna=True)
            if max_val and max_val > 1.0:
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

    agg: Dict[str, str] = {}
    if "AS" in have:
        agg["AS"] = "max"
    if "MAPQ" in have:
        agg["MAPQ"] = "max"
    if "NM" in have:
        agg["NM"] = "min"
    # p-values are per (read,BC) not per-genome; keep min if multiple show up
    if "p_as" in have:
        agg["p_as"] = "min"
    if "p_mq" in have:
        agg["p_mq"] = "min"

    if not agg:
        # At least return the keys
        return df.drop_duplicates(keys)[keys]

    return (
        df.groupby(keys, observed=True)
          .agg(agg)
          .reset_index()
    )


def _optint(x, default: int) -> int:
    if isinstance(x, OptionInfo):
        d = getattr(x, "default", None)
        return int(d) if d is not None else int(default)
    return int(x)

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
    S_parts: List[np.ndarray] = []
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

    records: List[pd.DataFrame] = []
    for (_, _), grp in df.groupby(["barcode", "read_id"], sort=False, observed=True):
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


def _aggregate_expected_counts_from_chunk(Ldf: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return two frames:
       C_chunk: per-barcode expected counts per genome: columns [barcode, genome, C]
       N_chunk: per-barcode n_reads: columns [barcode, n_reads]
    """
    C_chunk = (
        Ldf.groupby(["barcode", "genome"], observed=True)["L"]
            .sum().rename("C").reset_index()
    )
    N_chunk = (
        Ldf.groupby("barcode", observed=True)["read_id"]
            .nunique().rename("n_reads").reset_index()
    )
    return C_chunk, N_chunk


def _loglik_for_params(L_block: pd.DataFrame,
                       eta: pd.Series,
                       model: str,
                       g1: str,
                       g2: Optional[str],
                       alpha: float,
                       rho: float = 0.5) -> float:
    """Composite log-likelihood for a barcode block of rows (one per (read,genome))."""
    genomes = L_block["genome"].unique()
    eta_g = eta.reindex(genomes).fillna(0.0).to_numpy()

    if model == "single":
        eta_g1 = eta.reindex([g1]).fillna(0.0).to_numpy()[0]
        w = np.where(
            L_block["genome"].to_numpy() == g1,
            (1.0 - alpha) + alpha * eta_g1,
            alpha * eta_g,
        )
    elif model == "doublet":
        is_g1 = (L_block["genome"].to_numpy() == g1)
        is_g2 = (L_block["genome"].to_numpy() == g2)
        mix = np.where(is_g1, rho, np.where(is_g2, 1.0 - rho, 0.0))
        w = (1.0 - alpha) * mix + alpha * eta_g
    else:
        raise ValueError("model must be 'single' or 'doublet'")

    arr_wL = w * L_block["L"].to_numpy()
    df_tmp = L_block[["read_id"]].copy()
    df_tmp["wL"] = arr_wL
    s = df_tmp.groupby("read_id", observed=True)["wL"].sum().to_numpy()
    s = np.clip(s, 1e-12, None)
    return float(np.log(s).sum())


def _bic(loglik: float, n_params: int, n_reads: int) -> float:
    return -2.0 * loglik + n_params * math.log(max(n_reads, 1))


def _select_model_for_barcode(L_block: pd.DataFrame,
                              eta: pd.Series,
                              cfg: MergeConfig,
                              candidate_genomes: Sequence[str]) -> Dict:
    read_count = int(L_block["read_id"].nunique())

    if not candidate_genomes:
        return {
            "model": None,
            "genome_1": None,
            "genome_2": None,
            "alpha": np.nan,
            "rho": np.nan,
            "bic_best": np.inf,
            "bic_single": np.inf,
            "bic_doublet": np.inf,
            "delta_bic": np.nan,
            "n_reads": read_count,
            "n_effective": float(L_block["L"].sum()),
            "purity": 0.0,
            "minor": 0.0,
            "call": "ambiguous",
            "indistinguishable_set": "",
            "concordance": 0.0,
            "p_top1": 0.0,
            "p_top2": 0.0,
            "p_top3": 0.0,
            "top3_sum": 0.0,
            "entropy": 0.0,
            "top_genome": None,
            "suspect_multiplet": False,
            "multiplet_reason": "",
        }

    alphas = np.arange(0.0, cfg.max_alpha + 1e-9, cfg.alpha_grid)
    rhos = np.arange(0.1, 0.9 + 1e-9, cfg.rho_grid)

    best: Dict = {"bic": float("inf"), "model": None}
    evals: List[Dict] = []

    # Singles
    for g1 in candidate_genomes:
        best_local: Dict = {"bic": float("inf")}
        for a in alphas:
            ll = _loglik_for_params(L_block, eta, model="single", g1=g1, g2=None, alpha=a)
            bic = _bic(ll, n_params=1, n_reads=read_count)
            if bic < best_local["bic"]:
                best_local = {"model": "single", "g1": g1, "alpha": a, "rho": 1.0, "ll": ll, "bic": bic}
        evals.append(best_local)
        if best_local["bic"] < best["bic"]:
            best = best_local

    # Doublets
    if len(candidate_genomes) >= 2:
        for i in range(len(candidate_genomes)):
            for j in range(i + 1, len(candidate_genomes)):
                g1, g2 = candidate_genomes[i], candidate_genomes[j]
                best_local = {"bic": float("inf")}
                for a in alphas:
                    for r in rhos:
                        ll = _loglik_for_params(L_block, eta, model="doublet", g1=g1, g2=g2, alpha=a, rho=r)
                        bic = _bic(ll, n_params=2, n_reads=read_count)
                        if bic < best_local["bic"]:
                            best_local = {
                                "model": "doublet",
                                "g1": g1,
                                "g2": g2,
                                "alpha": a,
                                "rho": r,
                                "ll": ll,
                                "bic": bic,
                            }
                evals.append(best_local)
                if best_local["bic"] < best["bic"]:
                    best = best_local

    # QC props and top genome
    mass = L_block.groupby("genome", observed=True)["L"].sum().sort_values(ascending=False)
    tot = float(mass.sum())
    props = (mass / max(tot, 1e-12)).to_numpy()
    genomes_sorted = mass.index.to_list()
    top_genome = genomes_sorted[0] if len(genomes_sorted) > 0 else None

    p1 = float(props[0]) if len(props) > 0 else 0.0
    p2 = float(props[1]) if len(props) > 1 else 0.0
    p3 = float(props[2]) if len(props) > 2 else 0.0
    top3_sum = p1 + p2 + p3
    entropy = float(-np.sum(props * np.log(np.clip(props, 1e-12, 1.0)))) if tot > 0 else 0.0

    out: Dict = {
        "model": best.get("model", None),
        "genome_1": best.get("g1", None),
        "genome_2": best.get("g2", None),
        "alpha": best.get("alpha", 0.0),
        "rho": best.get("rho", 1.0),
        "bic_best": best.get("bic", float("inf")),
        "bic_single": min(
            [e["bic"] for e in evals if e.get("model") == "single"],
            default=float("inf"),
        ),
        "bic_doublet": min(
            [e["bic"] for e in evals if e.get("model") == "doublet"],
            default=float("inf"),
        ),
        "n_reads": read_count,
        "n_effective": float(L_block["L"].sum()),
    }
    out["delta_bic"] = min(out["bic_doublet"], out["bic_single"]) - out["bic_best"]

    # Purity & minor
    if out["model"] == "single":
        purity = (1.0 - out["alpha"])
        minor = 0.0
    elif out["model"] == "doublet":
        purity = (1.0 - out["alpha"]) * max(out["rho"], 1.0 - out["rho"])
        minor = (1.0 - out["alpha"]) * min(out["rho"], 1.0 - out["rho"])
    else:
        purity, minor = 0.0, 0.0
    out["purity"] = float(purity)
    out["minor"] = float(minor)

    # Call decision
    call = "ambiguous"
    indist: Optional[Tuple[str, str]] = None

    singles_sorted = sorted(
        [e for e in evals if e.get("model") == "single"],
        key=lambda x: x["bic"],
    )[:2]
    if len(singles_sorted) == 2 and (singles_sorted[1]["bic"] - singles_sorted[0]["bic"]) < cfg.near_tie_margin:
        indist = (singles_sorted[0]["g1"], singles_sorted[1]["g1"])

    if out["model"] == "single" and out["purity"] >= cfg.single_mass_min and \
            (out["bic_doublet"] - out["bic_best"]) >= cfg.bic_margin:
        call = "single"
    elif out["model"] == "doublet" and out["minor"] >= cfg.doublet_minor_min and \
            (out["bic_single"] - out["bic_best"]) >= cfg.bic_margin:
        call = "doublet"
    elif indist is not None:
        call = "indistinguishable"

    out["call"] = call
    out["indistinguishable_set"] = ",".join(indist) if indist is not None else ""

    # Concordance of per-read argmax with major genome
    maj = out["genome_1"]
    if maj is None:
        concord = 0.0
    else:
        x = (
            L_block.groupby(["read_id", "genome"], observed=True)["L"]
                  .sum().reset_index()
        )
        argmax = (
            x.sort_values(["read_id", "L"], ascending=[True, False])
             .drop_duplicates("read_id")["genome"]
        )
        concord = float((argmax == maj).mean())
    out["concordance"] = concord

    # Multiplet suspicion heuristics
    suspect = False
    reasons: List[str] = []
    weak_doublet_edge = (
        (out["model"] == "doublet" and (out["bic_single"] - out["bic_best"]) < 4.0)
        or (out["model"] == "single" and (out["bic_doublet"] - out["bic_best"]) < 4.0)
    )
    if p3 >= 0.15 and top3_sum >= 0.85:
        suspect = True
        reasons.append(f"top3={top3_sum:.2f} with p3={p3:.2f}")
    if weak_doublet_edge and out["purity"] < 0.75:
        suspect = True
        reasons.append(f"weak_deltaBIC + purity={out['purity']:.2f}")

    out.update(
        {
            "p_top1": p1,
            "p_top2": p2,
            "p_top3": p3,
            "top3_sum": top3_sum,
            "entropy": entropy,
            "top_genome": top_genome,
            "suspect_multiplet": bool(suspect),
            "multiplet_reason": ";".join(reasons),
        }
    )

    return out

# ------------------------------
# Summaries / plots
# ------------------------------

def _write_gzip_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as f:
        df.to_csv(f, index=False)

# ------------------------------
# Two-pass streaming helpers
# ------------------------------

def _barcode_to_shard_idx(barcode: str, shards: int) -> int:
    h = hashlib.sha1(barcode.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little") % max(shards, 1)


def _open_shard_handles(shard_dir: Path, shards: int) -> List[gzip.GzipFile]:
    shard_dir.mkdir(parents=True, exist_ok=True)
    handles: List[gzip.GzipFile] = []
    for i in range(shards):
        fp = shard_dir / f"shard_{i:02d}.tsv.gz"
        # open in append-text mode
        handles.append(gzip.open(fp, "at"))
    return handles


def _close_shard_handles(handles: List[gzip.GzipFile]) -> None:
    for h in handles:
        try:
            h.close()
        except Exception:
            pass


def _write_L_chunk_to_shards(Ldf: pd.DataFrame,
                             shard_handles: List[gzip.GzipFile],
                             shards: int,
                             write_header_flags: Dict[int, bool]) -> None:
    # write rows to shard by barcode-hash. Keep a small header tracker per shard.
    # Ensure consistent column order
    cols = ["barcode", "read_id", "genome", "L", "L_amb"]
    Ldf = Ldf[cols].copy()
    for bc, sub in Ldf.groupby("barcode", sort=False, observed=True):
        idx = _barcode_to_shard_idx(str(bc), shards)
        h = shard_handles[idx]
        if not write_header_flags[idx]:
            h.write("\t".join(cols) + "\n")
            write_header_flags[idx] = True
        # write sub as TSV without header
        sub.to_csv(h, sep="\t", header=False, index=False)

def _iter_shard_rows(shard_fp: Path, chunksize: int = 2_000_000):
    """
    Stream rows from a per-read posterior shard, coercing L/L_amb to floats
    and dropping header repeats + rows without posteriors.
    """
    try:
        it = pd.read_csv(
            shard_fp,
            sep="\t",
            compression="gzip",
            chunksize=chunksize,
            dtype={
                "barcode": "string",   # or "category" if you prefer
                "read_id": "string",
                "genome": "string",    # or "category"
                "L": "string",
                "L_amb": "string",
            },
        )
    except FileNotFoundError:
        return

    for chunk in it:
        if chunk.empty:
            continue

        # 1) Drop repeated header rows inside the file
        #    (these have literal 'barcode', 'read_id', 'genome', 'L', 'L_amb' as values)
        is_header_like = (
            (chunk["barcode"] == "barcode")
            & (chunk["read_id"] == "read_id")
        )
        if is_header_like.any():
            chunk = chunk[~is_header_like]

        if chunk.empty:
            continue

        # 2) Coerce L / L_amb → numeric; invalid tokens -> NaN
        chunk["L"] = pd.to_numeric(chunk["L"], errors="coerce")
        chunk["L_amb"] = pd.to_numeric(chunk["L_amb"], errors="coerce")

        # 3) Keep only rows with valid posteriors for downstream use
        mask_valid = chunk["L"].notna() & chunk["L_amb"].notna()
        if not mask_valid.any():
            continue

        chunk = chunk.loc[mask_valid].copy()

        # Optional: downcast to float32 to save memory
        chunk["L"] = chunk["L"].astype("float32")
        chunk["L_amb"] = chunk["L_amb"].astype("float32")

        yield chunk


def _process_barcode_group(bc: str,
                           L_block: pd.DataFrame,
                           cand: List[str],
                           eta: pd.Series,
                           cfg: MergeConfig) -> Dict:
    best = _select_model_for_barcode(L_block, eta, cfg, cand)
    best["barcode"] = bc
    best["notes"] = ""
    return best

# --- narrow columns early: place near DTYPES/NEEDED_COLS ---
READ_COLS = [
    "BC",
    "Read",
    "Genome",
    "AS",
    "MAPQ",
    "NM",
    "p_as_decile",
    "p_mq_decile",
    "barcode",
    "read_id",
    "genome",
    "p_as",
    "p_mq",
]  # tolerate pre-coerced

# --- helper: file listing once ---
def _list_input_files(assign_glob: str) -> List[Path]:
    return [
        Path(p)
        for p in glob.glob(assign_glob, recursive=True)
        if p.endswith("_filtered.tsv.gz")
        or p.endswith(".tsv.gz")
        or p.endswith("_filtered.tsv")
        or p.endswith(".tsv")
    ]

# --- faster chunk iterator: only needed columns, bigger chunks, C engine ---
def _iter_input_chunks(files: Sequence[Path],
                       chunk_rows: int) -> Iterator[pd.DataFrame]:
    for fp in files:
        try:
            it = pd.read_csv(
                fp,
                sep="\t",
                usecols=lambda c: c in READ_COLS,
                dtype=None,
                chunksize=chunk_rows,
                engine="c",
                low_memory=False,
                memory_map=True,
                on_bad_lines="skip",
            )
        except Exception as e:
            typer.echo(f"[warn] Failed to open {fp}: {e}; skipping.")
            continue
        for chunk in it:
            yield chunk

# --- Pass 1 worker: computes C/N and writes its own shards ---
def _pass1_worker(args) -> Tuple[pd.DataFrame, pd.DataFrame, Path]:
    files, cfg_dict, worker_idx, tmp_root = args
    cfg = MergeConfig(**cfg_dict)
    shard_dir = tmp_root / f"L_shards_w{worker_idx:02d}"
    shard_handles = _open_shard_handles(shard_dir, cfg.shards)
    header_written = {i: False for i in range(cfg.shards)}

    C_all_list: List[pd.DataFrame] = []
    N_all_list: List[pd.DataFrame] = []

    for raw in _iter_input_chunks(files, cfg.chunk_rows):
        try:
            df = _coerce_assign_schema(raw)
            df = _reduce_alignments_to_per_genome(df)
            df = df.dropna(subset=["barcode", "read_id", "genome"])
            if df.empty:
                continue
            Ldf = _compute_read_posteriors(df, cfg)
            C_chunk = (
                Ldf.groupby(["barcode", "genome"], observed=True)["L"]
                   .sum().rename("C").reset_index()
            )
            N_chunk = (
                Ldf.groupby("barcode", observed=True)["read_id"]
                   .nunique().rename("n_reads").reset_index()
            )
            C_all_list.append(C_chunk)
            N_all_list.append(N_chunk)
            _write_L_chunk_to_shards(Ldf, shard_handles, cfg.shards, header_written)
        except Exception as e:
            typer.echo(f"[warn] skipping chunk due to error: {e}")
            continue

    _close_shard_handles(shard_handles)
    if not C_all_list:
        # empty sentinel
        return (
            pd.DataFrame(columns=["barcode", "genome", "C"]),
            pd.DataFrame(columns=["barcode", "n_reads"]),
            shard_dir,
        )

    C_all = (
        pd.concat(C_all_list, ignore_index=True)
          .groupby(["barcode", "genome"], observed=True)["C"]
          .sum().reset_index()
    )
    N_all = (
        pd.concat(N_all_list, ignore_index=True)
          .groupby("barcode", observed=True)["n_reads"]
          .sum().reset_index()
    )
    return C_all, N_all, shard_dir


# --- Merge shards from workers into final shard set without recompression ---
def _concat_worker_shards(worker_dirs: Sequence[Path],
                          final_dir: Path,
                          shards: int) -> None:
    final_dir.mkdir(parents=True, exist_ok=True)
    for i in range(shards):
        out = final_dir / f"shard_{i:02d}.tsv.gz"
        with gzip.open(out, "wb") as fout:
            for wd in worker_dirs:
                part = wd / f"shard_{i:02d}.tsv.gz"
                if not part.exists():
                    continue
                with gzip.open(part, "rb") as fin:
                    shutil.copyfileobj(fin, fout)

# --- Pass 1 dispatcher: can go parallel over files ---
def _pass1_stream_build(assign_glob: str,
                        cfg: MergeConfig,
                        tmp_dir: Path,
                        pass1_workers: int = 1) -> Tuple[pd.DataFrame, pd.DataFrame, Path]:
    files = _list_input_files(assign_glob)
    if not files:
        raise FileNotFoundError(f"No assign files found for pattern: {assign_glob}")

    if pass1_workers <= 1:
        # single-worker path reuses worker logic
        C_all, N_all, shard_dir = _pass1_worker((files, cfg.model_dump(), 0, tmp_dir))
        return C_all, N_all, shard_dir

    # split file list across workers
    chunks: List[List[Path]] = [[] for _ in range(pass1_workers)]
    for i, fp in enumerate(files):
        chunks[i % pass1_workers].append(fp)

    args = [(chunks[i], cfg.model_dump(), i, tmp_dir) for i in range(pass1_workers)]
    worker_dirs: List[Path] = []
    C_parts: List[pd.DataFrame] = []
    N_parts: List[pd.DataFrame] = []

    with ProcessPoolExecutor(max_workers=pass1_workers) as ex:
        for C_i, N_i, shard_dir_i in tqdm(
            ex.map(_pass1_worker, args),
            total=len(args),
            desc="[pass1] workers",
        ):
            C_parts.append(C_i)
            N_parts.append(N_i)
            worker_dirs.append(shard_dir_i)

    # reduce aggregations
    C_all = (
        pd.concat(C_parts, ignore_index=True)
          .groupby(["barcode", "genome"], observed=True)["C"]
          .sum().reset_index()
    )
    N_all = (
        pd.concat(N_parts, ignore_index=True)
          .groupby("barcode", observed=True)["n_reads"]
          .sum().reset_index()
    )

    # merge per-worker shard trees into one final shard tree
    final_shard_dir = tmp_dir / "L_shards"
    _concat_worker_shards(worker_dirs, final_shard_dir, cfg.shards)
    return C_all, N_all, final_shard_dir

# ------------------------------
# Main driver
# ------------------------------

@app.command("genotyping")
def genotyping(
    assign: str = typer.Option(..., help="Path or glob to assign outputs (*.tsv[.gz] from assign)."),
    outdir: Path = typer.Option(Path("merge_out"), help="Output directory."),
    sample: str = typer.Option("sample", help="Sample name used for output filenames."),
    min_reads: int = typer.Option(100, help="Minimum reads to attempt confident calls."),
    beta: float = typer.Option(0.5, help="Softmax temperature for score fusion."),
    w_as: float = typer.Option(1.0, help="Weight for AS."),
    w_mapq: float = typer.Option(0.5, help="Weight for MAPQ."),
    w_nm: float = typer.Option(0.25, help="Weight for NM (penalty)."),
    ambient_const: float = typer.Option(1e-3, help="Ambient constant mass per read before normalization."),
    tau_drop: float = typer.Option(8.0, help="(unused) Posterior-odds threshold for drops."),
    topk_genomes: int = typer.Option(3, help="Number of candidate genomes per barcode to consider."),
    make_report: bool = typer.Option(True, help="(unused) Render QC PDF report."),
    threads: int = typer.Option(1, help="Parallel workers for per-cell model selection (Pass 2)."),
    shards: int = typer.Option(32, help="Number of on-disk shards for pass-1 spill."),
    chunk_rows: int = typer.Option(1_000_000, help="Input chunk size for streaming read."),
    pass1_workers: Optional[int] = typer.Option(None, help="Parallel workers for Pass 1 (file-level). If None, use --threads."),
):
    """Run the two-pass pipeline: streaming posterior calc → ambient estimate → per-cell calls."""
    shards = _optint(shards, 32)
    chunk_rows = _optint(chunk_rows, 2_000_000)
    threads = max(1, int(threads))

    if pass1_workers is None:
        pass1_workers = threads
    else:
        pass1_workers = max(1, int(pass1_workers))

    cfg = MergeConfig(
        beta=beta,
        w_as=w_as,
        w_mapq=w_mapq,
        w_nm=w_nm,
        ambient_const=ambient_const,
        min_reads=min_reads,
        tau_drop=tau_drop,
        topk_genomes=topk_genomes,
        sample=sample,
        shards=shards,
        chunk_rows=chunk_rows,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    tmp_dir = outdir / f"tmp_{sample}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Pass 1 ----------------
    typer.echo("[1/5] Pass 1: streaming inputs → per-read posteriors + accumulators")
    C_all, N_all, shard_dir = _pass1_stream_build(
        assign,
        cfg,
        tmp_dir,
        pass1_workers=pass1_workers,
    )

    # Estimate ambient from shards using low-read barcodes
    typer.echo("[2/5] Estimating ambient profile from low-read barcodes")
    low_reads = set(N_all[N_all["n_reads"] < 200]["barcode"])
    if not low_reads:
        warnings.warn("No low-read barcodes found; ambient will be estimated from all L rows.")

    mix: Dict[str, float] = {}
    for shard_fp in sorted(shard_dir.glob("shard_*.tsv.gz")):
        for chunk in _iter_shard_rows(shard_fp):
            if low_reads:
                chunk = chunk[chunk["barcode"].isin(low_reads)]
            if chunk.empty:
                continue
            add = chunk.groupby("genome", observed=True)["L"].sum()
            for g, v in add.items():
                mix[g] = mix.get(g, 0.0) + float(v)
    if not mix:
        # Fallback: uniform
        genomes = sorted(C_all["genome"].unique())
        eta = pd.Series({g: 1.0 / len(genomes) for g in genomes})
    else:
        s = sum(mix.values())
        eta = pd.Series({g: v / s for g, v in mix.items()})

    # Candidate genomes per barcode
    typer.echo("[3/5] Selecting top-k candidate genomes per barcode")

    def _topk_genomes_per_barcode(C: pd.DataFrame, k: int) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for bc, sub in C.groupby("barcode", sort=False, observed=True):
            top = (
                sub.sort_values("C", ascending=False)["genome"]
                   .head(k).astype(str).tolist()
            )
            out[str(bc)] = top
        return out

    topk = _topk_genomes_per_barcode(C_all, cfg.topk_genomes)

    # ---------------- Pass 2 ----------------
    typer.echo("[4/5] Pass 2: per-cell model selection over shards")
    rows: List[Dict] = []

    def _process_shard(shard_fp: Path) -> List[Dict]:
        local_rows: List[Dict] = []
        # Build in-memory blocks per barcode for this shard
        blocks: Dict[str, List[pd.DataFrame]] = {}
        for chunk in _iter_shard_rows(shard_fp):
            for bc, sub in chunk.groupby("barcode", sort=False, observed=True):
                blocks.setdefault(str(bc), []).append(
                    sub[["barcode", "read_id", "genome", "L", "L_amb"]]
                )
        for bc, parts in blocks.items():
            L_block = pd.concat(parts, ignore_index=True)
            cand = topk.get(bc, [])
            if not cand or L_block.empty:
                continue
            best = _process_barcode_group(bc, L_block, cand, eta, cfg)
            local_rows.append(best)
        return local_rows

    shard_files = sorted(shard_dir.glob("shard_*.tsv.gz"))
    if threads <= 1:
        for fp in tqdm(shard_files, desc="[pass2] shards"):
            r = _process_shard(fp)
            rows.extend(r)
    else:
        typer.echo(f"   → using {threads} workers at shard level")
        with ProcessPoolExecutor(max_workers=threads) as ex:
            futures = [ex.submit(_process_shard, fp) for fp in shard_files]
            for fut in tqdm(as_completed(futures), total=len(futures)):
                r = fut.result()
                rows.extend(r)

    calls = pd.DataFrame(rows)

    # Best genotype per cell: prefer top_genome, fall back to genome_1
    if "top_genome" in calls.columns:
        calls["best_genome"] = calls["top_genome"].fillna(calls["genome_1"])
    else:
        calls["best_genome"] = calls["genome_1"]

    # Simplified status flag
    def _status_flag(call: str) -> str:
        if call == "single":
            return "single"
        if call == "doublet":
            return "double"
        if call == "indistinguishable":
            return "low_confidence"
        return "ambiguous"

    calls["status_flag"] = calls["call"].astype(str).map(_status_flag)

    # Legacy-compatible summary (PASS_by_mapping)
    legacy = calls.copy()
    legacy["AssignedGenome"] = legacy["best_genome"].fillna("")
    pass_mask = legacy["call"].isin(["single", "doublet", "indistinguishable"])
    legacy_out = legacy.loc[
        pass_mask, ["barcode", "AssignedGenome", "call", "purity", "n_reads"]
    ]

    outdir.mkdir(parents=True, exist_ok=True)
    _write_gzip_df(calls, outdir / f"{sample}_cells_calls.tsv.gz")
    legacy_out.to_csv(outdir / f"{sample}_BCs_PASS_by_mapping.csv", index=False)

    # Expected-count matrix
    if not C_all.empty:
        mat = (
            C_all.pivot(index="barcode", columns="genome", values="C")
                .sort_index(axis=1)
                .fillna(0.0)
        )
        mat.to_csv(outdir / f"{sample}_expected_counts_by_genome.csv")

    # Cleanup tmp
    try:
        shutil.rmtree(tmp_dir)
    except Exception:
        pass

    typer.echo("[5/5] Done.")


if __name__ == "__main__":
    app()
