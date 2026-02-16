#!/usr/bin/env python3
"""
src/ambientmapper/genotyping.py

ambientmapper genotyping — posterior-aware merge → per-cell genotype calls (+ empty-aware)

Key properties (v2):
- Promiscuous-ambiguous filtering (MAPQ window) to prevent ambiguous-read inflation.
- Posterior mass per read (winner-only or softmax).
- Ambient learning (eta) from an "empty-like" seed refined by JSD.
- Deterministic spill policy ("new shard policy"):
    Prefer chunk-defined partitions from `cell_map_ref_chunk_*.txt` barcode lists.
    Each chunk produces ONE spill file: L_chunk_<chunk_id>.tsv.gz
    This guarantees a barcode is called exactly once (bulletproof; no contiguity assumptions).
  Fallback:
    If no chunk lists are provided, use a carry-over streaming caller per hash-shard file.

Notes:
- Input is expected to be assign "filtered" outputs with columns like:
  Read, BC, Genome, AS, MAPQ, NM, assigned_class, ...
  (gz or plain tsv accepted)
"""

from __future__ import annotations

import glob
import gzip
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import typer
from pydantic import BaseModel

app = typer.Typer(add_completion=False, no_args_is_help=True)

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------


class MergeConfig(BaseModel):
    # Fusion / scoring -> posteriors
    beta: float = 1.0
    w_as: float = 1.0
    w_mapq: float = 1.0
    w_nm: float = 1.0
    ambient_const: float = 1e-3  # ambient pseudocount in posterior denominator

    # Calling thresholds / gates (used)
    min_reads: int = 5
    single_mass_min: float = 0.6
    doublet_minor_min: float = 0.20
    bic_margin: float = 6.0              # empty override: empty if bic_empty + bic_margin < bic_best
    near_tie_margin: float = 2.0         # if BIC gap single vs doublet <= near_tie_margin -> downgrade label
    ratio_top1_top2_min: float = 2.0

    # Ambient learning
    eta_iters: int = 2
    eta_seed_quantile: float = 0.02      # initial seed fraction by n_effective
    empty_tau_quantile: float = 0.95     # refine seed by JSD threshold quantile of most eta-like

    # Empty gate (structure) to restrict eta candidates
    empty_top1_max: float = 0.6
    empty_ratio12_max: float = 1.5
    empty_reads_max: Optional[int] = None

    # Search grid
    alpha_grid: float = 0.02
    rho_grid: float = 0.05
    max_alpha: float = 0.5
    topk_genomes: int = 3

    # System / IO
    sample: str = "sample"
    chunk_rows: int = 5_000_000
    pass2_chunksize: int = 200_000

    # Posterior mode
    winner_only: bool = True

    # Pass 1 parallelism (files)
    pass1_workers: int = 1

    # Promiscuous ambiguous filter (applies only to assigned_class=="ambiguous")
    max_hits: Optional[int] = None          # e.g. 3
    hits_delta_mapq: Optional[float] = None # e.g. 2.0
    max_rows_per_read_guard: Optional[int] = 500


# --------------------------------------------------------------------------------------
# Helpers: schema and coercion
# --------------------------------------------------------------------------------------


def _coerce_assign_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize input schema from assign outputs.

    Expected upstream columns might include:
      Read, BC, Genome, AS, MAPQ, NM, XAcount, assigned_class, p_as, p_mq
    """
    m = {
        "BC": "barcode",
        "Read": "read_id",
        "Genome": "genome",
        "p_as_decile": "p_as",
        "p_mq_decile": "p_mq",
    }
    out = df.rename(columns=m)

    required = ["barcode", "read_id", "genome", "AS", "MAPQ", "NM"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    # strings
    out["barcode"] = out["barcode"].astype("string")
    out["read_id"] = out["read_id"].astype("string")
    out["genome"] = out["genome"].astype("string")

    # numerics
    for c in ["AS", "MAPQ", "NM"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    if "assigned_class" in out.columns:
        out["assigned_class"] = out["assigned_class"].astype("string")
    else:
        out["assigned_class"] = pd.Series(["ambiguous"] * len(out), dtype="string")

    return out


def _reduce_alignments_to_per_genome(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse potentially multiple rows per (barcode, read_id, genome) by keeping:
      AS=max, MAPQ=max, NM=min, assigned_class=first
    """
    keys = ["barcode", "read_id", "genome"]
    agg = {"AS": "max", "MAPQ": "max", "NM": "min", "assigned_class": "first"}
    return df.groupby(keys, observed=True, sort=False).agg(agg).reset_index()


# --------------------------------------------------------------------------------------
# Promiscuous ambiguous filter (NumPy)
# --------------------------------------------------------------------------------------


def _filter_promiscuous_ambiguous_reads(
    df: pd.DataFrame, cfg: MergeConfig
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Drop ambiguous rows for (barcode, read_id) groups where the number of "near-top" hits
    exceeds cfg.max_hits, using MAPQ >= (best_MAPQ - hits_delta_mapq).

    Only applies to assigned_class == "ambiguous".
    """
    if cfg.max_hits is None or cfg.hits_delta_mapq is None:
        return df, {"enabled": False}

    assigned = df["assigned_class"].to_numpy()
    is_amb = assigned == "ambiguous"
    if not is_amb.any():
        return df, {"enabled": True, "dropped_groups": 0, "dropped_rows": 0}

    amb_idx = np.flatnonzero(is_amb)
    amb = df.loc[is_amb, ["barcode", "read_id", "MAPQ"]]

    h = pd.util.hash_pandas_object(amb[["barcode", "read_id"]], index=False).to_numpy(np.uint64)
    codes, _ = pd.factorize(h, sort=False)
    MQ = amb["MAPQ"].to_numpy(dtype=np.float32, copy=False)

    valid = (codes >= 0) & np.isfinite(MQ)
    if not valid.all():
        codes = codes[valid]
        MQ = MQ[valid]
        amb_idx = amb_idx[valid]
        if codes.size == 0:
            keep = np.ones(len(df), dtype=bool)
            keep[np.flatnonzero(is_amb)] = False
            return df.loc[keep].copy(), {
                "enabled": True,
                "dropped_groups": 0,
                "dropped_rows": int(is_amb.sum()),
                "note": "all ambiguous rows invalid",
            }

    n_codes = int(codes.max()) + 1

    guard_dropped = 0
    if cfg.max_rows_per_read_guard is not None and cfg.max_rows_per_read_guard > 0:
        counts = np.bincount(codes, minlength=n_codes)
        bad_guard = np.flatnonzero(counts > int(cfg.max_rows_per_read_guard))
        if bad_guard.size:
            guard_dropped = int(bad_guard.size)
            keep_g = ~np.isin(codes, bad_guard)
            codes = codes[keep_g]
            MQ = MQ[keep_g]
            amb_idx = amb_idx[keep_g]
            if codes.size == 0:
                keep = np.ones(len(df), dtype=bool)
                keep[np.flatnonzero(is_amb)] = False
                return df.loc[keep].copy(), {
                    "enabled": True,
                    "dropped_groups": guard_dropped,
                    "dropped_rows": int(is_amb.sum()),
                    "note": "guard removed all ambiguous groups",
                }
            n_codes = int(codes.max()) + 1

    best_mq = np.full(n_codes, -np.inf, dtype=np.float32)
    np.maximum.at(best_mq, codes, MQ)

    delta = np.float32(cfg.hits_delta_mapq)
    is_near = MQ >= (best_mq[codes] - delta)
    near_counts = np.bincount(codes[is_near], minlength=n_codes)
    bad_codes = np.flatnonzero(near_counts > int(cfg.max_hits))

    if bad_codes.size == 0:
        return df, {"enabled": True, "dropped_groups": guard_dropped, "dropped_rows": 0}

    drop_mask = np.isin(codes, bad_codes)

    keep = np.ones(len(df), dtype=bool)
    keep[amb_idx[drop_mask]] = False

    return df.loc[keep].copy(), {
        "enabled": True,
        "dropped_groups": int(bad_codes.size) + guard_dropped,
        "dropped_rows": int(drop_mask.sum()),
    }


# --------------------------------------------------------------------------------------
# Core: posterior computation (vectorized)
# --------------------------------------------------------------------------------------


def _compute_read_posteriors(df: pd.DataFrame, cfg: MergeConfig) -> pd.DataFrame:
    """
    Build per-row posterior mass L (and per-read ambient mass L_amb).

    If winner_only:
      - choose best genome per (barcode, read_id) by score
      - set L=1 for that row, L_amb=ambient_const (stored on row)

    Else:
      - compute softmax over genomes per read-group with stable exp
      - L_amb is ambient_const / (sum_w + ambient_const)
    """
    df = df[["barcode", "read_id", "genome", "AS", "MAPQ", "NM"]].copy()
    df = df.dropna(subset=["barcode", "read_id", "genome"])
    if df.empty:
        return df.assign(L=np.array([], dtype=float), L_amb=np.array([], dtype=float))[["barcode", "read_id", "genome", "L", "L_amb"]]

    AS = df["AS"].to_numpy(np.float32, copy=False)
    MQ = df["MAPQ"].to_numpy(np.float32, copy=False)
    NM = df["NM"].to_numpy(np.float32, copy=False)

    rid = (df["barcode"].astype(str) + "::" + df["read_id"].astype(str))
    codes = rid.astype("category").cat.codes.to_numpy(np.int32)

    score = (AS * float(cfg.w_as)) + (MQ * float(cfg.w_mapq)) - (NM * float(cfg.w_nm))

    if cfg.winner_only:
        df["_code"] = codes
        df["_score"] = score
        df = df.sort_values(["_code", "_score"], ascending=[True, False]).drop_duplicates("_code", keep="first")
        df["L"] = 1.0
        df["L_amb"] = float(cfg.ambient_const)
        return df[["barcode", "read_id", "genome", "L", "L_amb"]]

    n_codes = int(codes.max()) + 1 if codes.size else 0
    max_per = np.full(n_codes, -np.inf, dtype=np.float32)
    np.maximum.at(max_per, codes, score)
    s = score - max_per[codes]

    w = np.exp(np.clip(float(cfg.beta) * s, -50, 50)).astype(np.float32)
    denom = np.bincount(codes, weights=w).astype(np.float32) + float(cfg.ambient_const)

    df["L"] = w / denom[codes]
    df["L_amb"] = float(cfg.ambient_const) / denom[codes]
    return df[["barcode", "read_id", "genome", "L", "L_amb"]]


# --------------------------------------------------------------------------------------
# Likelihood / BIC / JSD
# --------------------------------------------------------------------------------------


def _loglik_empty(L_block: pd.DataFrame, eta: pd.Series) -> float:
    eta_row = eta.reindex(L_block["genome"]).fillna(0.0).to_numpy(dtype=float)
    wL = eta_row * L_block["L"].to_numpy(dtype=float)
    s = (
        pd.DataFrame({"read_id": L_block["read_id"].values, "wL": wL})
        .groupby("read_id", sort=False)["wL"]
        .sum()
        .to_numpy()
    )
    return float(np.log(np.clip(s, 1e-12, None)).sum())


def _loglik_for_params(
    L_block: pd.DataFrame,
    eta: pd.Series,
    model: str,
    g1: str,
    g2: Optional[str],
    alpha: float,
    rho: float = 0.5,
) -> float:
    gn = L_block["genome"].to_numpy()
    eta_row = eta.reindex(L_block["genome"]).fillna(0.0).to_numpy(dtype=float)

    if model == "single":
        w = np.where(gn == g1, (1.0 - alpha) + alpha * float(eta.get(g1, 0.0)), alpha * eta_row)
    else:
        assert g2 is not None
        mix = np.where(gn == g1, rho, np.where(gn == g2, 1.0 - rho, 0.0))
        w = (1.0 - alpha) * mix + alpha * eta_row

    wL = w * L_block["L"].to_numpy(dtype=float)
    s = (
        pd.DataFrame({"read_id": L_block["read_id"].values, "wL": wL})
        .groupby("read_id", sort=False)["wL"]
        .sum()
        .to_numpy()
    )
    return float(np.log(np.clip(s, 1e-12, None)).sum())


def _bic(loglik: float, n_params: int, n_reads: int) -> float:
    return -2.0 * loglik + n_params * math.log(max(n_reads, 1))


def _barcode_jsd_from_mass(mass: pd.Series, eta: pd.Series, normalize: bool = True) -> float:
    idx = sorted(set(mass.index) | set(eta.index))
    p = mass.reindex(idx).fillna(0.0).to_numpy(dtype=float)
    q = eta.reindex(idx).fillna(0.0).to_numpy(dtype=float)
    p = p / (p.sum() + 1e-12)
    q = q / (q.sum() + 1e-12)
    m = 0.5 * (p + q)

    p2 = np.clip(p, 1e-12, None)
    q2 = np.clip(q, 1e-12, None)
    m2 = np.clip(m, 1e-12, None)

    js = 0.5 * (np.sum(p2 * np.log(p2 / m2)) + np.sum(q2 * np.log(q2 / m2)))
    return js / math.log(2.0) if normalize else js


# --------------------------------------------------------------------------------------
# Model selection + gates (BIC + confidence labels)
# --------------------------------------------------------------------------------------


def _select_model_for_barcode(
    L_block: pd.DataFrame,
    eta: pd.Series,
    cfg: MergeConfig,
    candidate_genomes: Sequence[str],
) -> Dict[str, Any]:
    read_count = int(L_block["read_id"].nunique())

    mass = L_block.groupby("genome", sort=False)["L"].sum().sort_values(ascending=False)
    tot = float(mass.sum())
    props = (mass / max(tot, 1e-12)).to_numpy()

    p_top1 = float(props[0]) if len(props) > 0 else 0.0
    p_top2 = float(props[1]) if len(props) > 1 else 0.0
    ratio12 = (p_top1 / max(p_top2, 1e-12)) if p_top2 > 0 else float("inf")

    out: Dict[str, Any] = {
        "n_reads": read_count,
        "n_effective": tot,
        "p_top1": p_top1,
        "p_top2": p_top2,
        "ratio12": ratio12,
        "top_genome": (mass.index[0] if len(mass) else None),
        "jsd_to_eta": _barcode_jsd_from_mass(mass, eta, normalize=True),
    }

    # empty model (0 params)
    out["bic_empty"] = _bic(_loglik_empty(L_block, eta), 0, read_count)

    best_single: Optional[Dict[str, Any]] = None
    best_doublet: Optional[Dict[str, Any]] = None

    if read_count >= cfg.min_reads and candidate_genomes:
        alphas = np.arange(0.0, float(cfg.max_alpha) + 1e-9, float(cfg.alpha_grid))

        # single: 1 param (alpha)
        for g1 in candidate_genomes:
            for a in alphas:
                bic = _bic(_loglik_for_params(L_block, eta, "single", g1, None, float(a)), 1, read_count)
                if best_single is None or bic < best_single["bic"]:
                    best_single = {"model": "single", "g1": g1, "alpha": float(a), "bic": float(bic)}

        # doublet: 2 params (alpha, rho)
        if len(candidate_genomes) >= 2:
            rhos = np.arange(0.1, 0.9 + 1e-9, float(cfg.rho_grid))
            for i in range(len(candidate_genomes)):
                for j in range(i + 1, len(candidate_genomes)):
                    g1, g2 = candidate_genomes[i], candidate_genomes[j]
                    for a in alphas:
                        for r in rhos:
                            bic = _bic(
                                _loglik_for_params(L_block, eta, "doublet", g1, g2, float(a), float(r)),
                                2,
                                read_count,
                            )
                            if best_doublet is None or bic < best_doublet["bic"]:
                                best_doublet = {
                                    "model": "doublet",
                                    "g1": g1,
                                    "g2": g2,
                                    "alpha": float(a),
                                    "rho": float(r),
                                    "bic": float(bic),
                                }

    bic_s = best_single["bic"] if best_single else float("inf")
    bic_d = best_doublet["bic"] if best_doublet else float("inf")
    best_non_empty = best_single if bic_s <= bic_d else best_doublet

    bic_best = min(bic_s, bic_d)
    bic_second = max(bic_s, bic_d)
    bic_gap = float(bic_second - bic_best) if np.isfinite(bic_second) else float("inf")

    out["bic_gap_sd"] = float(bic_gap)

    out["call"] = "low_reads" if read_count < cfg.min_reads else "uncalled"
    out["model"] = None
    out["genome_1"] = None
    out["genome_2"] = None
    out["bic_best"] = None
    out["alpha"] = None
    out["rho"] = None

    if best_non_empty is None:
        return out

    out["bic_best"] = float(best_non_empty["bic"])
    out["model"] = best_non_empty["model"]
    out["genome_1"] = best_non_empty.get("g1")
    out["genome_2"] = best_non_empty.get("g2")
    out["alpha"] = best_non_empty.get("alpha")
    out["rho"] = best_non_empty.get("rho")

    # BIC-based empty override
    if (out["bic_empty"] + float(cfg.bic_margin)) < float(best_non_empty["bic"]):
        out["call"] = "empty"
        return out

    minor_frac = float(min(p_top1, p_top2)) if (p_top1 > 0 and p_top2 > 0) else 0.0
    out["doublet_minor_frac"] = float(minor_frac)

    near_tie = (bic_gap <= float(cfg.near_tie_margin))
    out["near_tie_sd"] = bool(near_tie)

    if best_non_empty["model"] == "single":
        passes_mass = (p_top1 >= float(cfg.single_mass_min))
        passes_ratio = (ratio12 >= float(cfg.ratio_top1_top2_min))
        if passes_mass and passes_ratio and (not near_tie):
            out["call"] = "single_clean"
        else:
            out["call"] = "dirty_singlet"
    else:
        passes_minor = (minor_frac >= float(cfg.doublet_minor_min))
        if passes_minor and (not near_tie):
            out["call"] = "doublet"
        else:
            out["call"] = "weak_doublet"

    return out


# --------------------------------------------------------------------------------------
# Chunk policy helpers (new shard policy)
# --------------------------------------------------------------------------------------


_CHUNK_RE_DEFAULT = re.compile(r"chunk_(\d+)", re.IGNORECASE)


def _infer_chunk_id_from_path(p: Path, chunk_id_regex: str) -> Optional[str]:
    rx = re.compile(chunk_id_regex)
    m = rx.search(p.name)
    return m.group(1) if m else None


def _load_chunk_barcode_lists(cell_chunks_glob: str, chunk_id_regex: str) -> Dict[str, set]:
    """
    Load cell_map_ref_chunk_*.txt files.

    Returns:
      dict[chunk_id] = set(barcodes)
    """
    out: Dict[str, set] = {}
    rx = re.compile(chunk_id_regex)
    files = sorted(glob.glob(cell_chunks_glob))
    if not files:
        raise ValueError(f"No files matched --cell-chunks-glob: {cell_chunks_glob}")

    for fp in files:
        m = rx.search(Path(fp).name)
        if not m:
            continue
        cid = m.group(1)
        with open(fp, "rt") as fh:
            bcs = set()
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                bcs.add(line.split()[0])
        out[cid] = bcs

    if not out:
        raise ValueError(
            f"Matched chunk files but could not parse any chunk_id using regex={chunk_id_regex}"
        )
    return out


# --------------------------------------------------------------------------------------
# Pass 1: stream assign files -> (C_all, N_all) + spill
# --------------------------------------------------------------------------------------


@dataclass
class Pass1Outputs:
    C: pd.DataFrame
    N: pd.DataFrame
    spill_path: Path
    chunk_id: str


def _pass1_process_one_file_chunkspill(
    fp: Path,
    cfg: MergeConfig,
    spill_path: Path,
    chunk_barcodes: Optional[set] = None,
) -> Pass1Outputs:
    """
    Stream assign input, compute per-read posteriors, emit:
      - C (barcode x genome) mass
      - N (barcode) reads
      - spill file with columns barcode,read_id,genome,L,L_amb

    If chunk_barcodes is provided, restrict processing to those barcodes.
    """
    spill_path.parent.mkdir(parents=True, exist_ok=True)

    wrote_header = False
    C_parts: List[pd.DataFrame] = []
    N_parts: List[pd.DataFrame] = []

    with gzip.open(spill_path, "wt") as oh:
        for raw in pd.read_csv(fp, sep="\t", chunksize=int(cfg.chunk_rows), low_memory=False):
            df = _coerce_assign_schema(raw)
            df = _reduce_alignments_to_per_genome(df)
            df = df.dropna(subset=["barcode", "read_id", "genome"])
            if df.empty:
                continue

            if chunk_barcodes is not None:
                df = df[df["barcode"].isin(chunk_barcodes)]
                if df.empty:
                    continue

            df, _meta = _filter_promiscuous_ambiguous_reads(df, cfg)
            if df.empty:
                continue

            Ldf = _compute_read_posteriors(df, cfg)
            if Ldf.empty:
                continue

            C = (
                Ldf.groupby(["barcode", "genome"], sort=False)["L"]
                .sum()
                .rename("C")
                .reset_index()
            )
            N = (
                Ldf.groupby("barcode", sort=False)["read_id"]
                .nunique()
                .rename("n_reads")
                .reset_index()
            )
            C_parts.append(C)
            N_parts.append(N)

            if not wrote_header:
                oh.write("barcode\tread_id\tgenome\tL\tL_amb\n")
                wrote_header = True
            Ldf.to_csv(oh, sep="\t", header=False, index=False)

    C_out = pd.concat(C_parts, ignore_index=True) if C_parts else pd.DataFrame(columns=["barcode", "genome", "C"])
    N_out = pd.concat(N_parts, ignore_index=True) if N_parts else pd.DataFrame(columns=["barcode", "n_reads"])

    return Pass1Outputs(C=C_out, N=N_out, spill_path=spill_path, chunk_id="")


# --------------------------------------------------------------------------------------
# Ambient learning (eta): seed + refine using spill files
# --------------------------------------------------------------------------------------


def _iter_spill_files(spill_root: Path) -> Iterable[Path]:
    for p in sorted(spill_root.glob("L_chunk_*.tsv.gz")):
        yield p


def _compute_eta_from_spills(
    spill_root: Path,
    target_bcs: Sequence[str],
    all_genomes: Sequence[str],
    cfg: MergeConfig,
) -> pd.Series:
    """
    Sum L over target barcodes across all spill files (chunk spills), normalize to eta.

    Expects spill files columns: barcode, read_id, genome, L, L_amb
    """
    target_set = set(map(str, target_bcs))
    mix = {g: 0.0 for g in all_genomes}

    if not target_set:
        eta = pd.Series({g: 1.0 for g in all_genomes}, dtype=float)
        return eta / eta.sum()

    for spill in _iter_spill_files(spill_root):
        for chunk in pd.read_csv(
            spill,
            sep="\t",
            compression="gzip",
            chunksize=int(cfg.pass2_chunksize),
            dtype={"barcode": "string", "read_id": "string", "genome": "string", "L": "float32", "L_amb": "float32"},
        ):
            sub = chunk[chunk["barcode"].isin(target_set)]
            if sub.empty:
                continue
            gsum = sub.groupby("genome", sort=False)["L"].sum()
            for g, v in gsum.items():
                if g in mix:
                    mix[g] += float(v)

    eta = pd.Series(mix, dtype=float) + 1e-12
    return eta / eta.sum()


def _pick_initial_eta_seed(C_all: pd.DataFrame, N_all: pd.DataFrame, cfg: MergeConfig) -> List[str]:
    if C_all.empty:
        return []

    eff = C_all.groupby("barcode", sort=False)["C"].sum().rename("n_effective").reset_index()
    df = eff.merge(N_all, on="barcode", how="left")
    df["n_reads"] = df["n_reads"].fillna(0).astype(int)

    if cfg.empty_reads_max is not None:
        df = df[df["n_reads"] <= int(cfg.empty_reads_max)]
    if df.empty:
        return []

    q = min(max(float(cfg.eta_seed_quantile), 1e-6), 1.0)
    n = max(1, int(math.ceil(q * len(df))))
    return df.sort_values("n_effective", ascending=True)["barcode"].head(n).astype(str).tolist()


def _refine_eta_seed_by_jsd(
    C_all: pd.DataFrame,
    N_all: pd.DataFrame,
    eta: pd.Series,
    cfg: MergeConfig,
) -> List[str]:
    if C_all.empty:
        return []

    mass = C_all.groupby(["barcode", "genome"], sort=False)["C"].sum().reset_index()
    mass["rank"] = mass.groupby("barcode", sort=False)["C"].rank(method="first", ascending=False)
    top = mass[mass["rank"] <= 2].copy()

    top1 = top[top["rank"] == 1].set_index("barcode")["C"]
    top2 = top[top["rank"] == 2].set_index("barcode")["C"]

    eff = mass.groupby("barcode", sort=False)["C"].sum().rename("n_effective")
    stats = pd.DataFrame({"n_effective": eff})
    stats["top1"] = top1
    stats["top2"] = top2
    stats = stats.fillna(0.0)
    stats["p_top1"] = stats["top1"] / (stats["n_effective"] + 1e-12)
    stats["p_top2"] = stats["top2"] / (stats["n_effective"] + 1e-12)
    stats["ratio12"] = stats["p_top1"] / (stats["p_top2"] + 1e-12)

    if not N_all.empty:
        stats["n_reads"] = N_all.set_index("barcode")["n_reads"]
    else:
        stats["n_reads"] = 0

    cand = stats[(stats["p_top1"] <= float(cfg.empty_top1_max)) & (stats["ratio12"] <= float(cfg.empty_ratio12_max))]
    if cfg.empty_reads_max is not None:
        cand = cand[cand["n_reads"] <= int(cfg.empty_reads_max)]
    if cand.empty:
        cand = stats

    jsd_vals: Dict[str, float] = {}
    cand_idx = set(cand.index.astype(str).tolist())
    for bc, sub in mass.groupby("barcode", sort=False):
        bc_s = str(bc)
        if bc_s not in cand_idx:
            continue
        s = sub.set_index("genome")["C"]
        jsd_vals[bc_s] = _barcode_jsd_from_mass(s, eta, normalize=True)

    if not jsd_vals:
        return []

    jsd_ser = pd.Series(jsd_vals, dtype=float)
    tau = jsd_ser.quantile(float(cfg.empty_tau_quantile))
    return jsd_ser[jsd_ser <= tau].index.astype(str).tolist()


def _learn_eta(
    C_all: pd.DataFrame,
    N_all: pd.DataFrame,
    spill_root: Path,
    all_genomes: Sequence[str],
    cfg: MergeConfig,
) -> Tuple[pd.Series, Dict[str, Any]]:
    meta: Dict[str, Any] = {"iters": int(cfg.eta_iters), "seed_sizes": []}

    seed = _pick_initial_eta_seed(C_all, N_all, cfg)
    meta["seed_sizes"].append(len(seed))
    eta = _compute_eta_from_spills(spill_root, seed, list(all_genomes), cfg)

    for _ in range(int(cfg.eta_iters)):
        seed = _refine_eta_seed_by_jsd(C_all, N_all, eta, cfg)
        meta["seed_sizes"].append(len(seed))
        if not seed:
            break
        eta = _compute_eta_from_spills(spill_root, seed, list(all_genomes), cfg)

    return eta, meta


# --------------------------------------------------------------------------------------
# Calling: chunk-spill bulletproof caller (no contiguity assumptions)
# --------------------------------------------------------------------------------------


def _write_call_row(out_handle, bc_s: str, res: Dict[str, Any]) -> None:
    out_handle.write(
        "\t".join(
            [
                bc_s,
                str(res.get("call", "")),
                str(res.get("model", "")),
                str(res.get("genome_1", "")),
                str(res.get("genome_2", "")),
                f"{float(res.get('alpha')):.6g}" if res.get("alpha") is not None else "",
                f"{float(res.get('rho')):.6g}" if res.get("rho") is not None else "",
                str(int(res.get("n_reads", 0))),
                f"{float(res.get('n_effective', 0.0)):.6g}",
                f"{float(res.get('p_top1', 0.0)):.6g}",
                f"{float(res.get('p_top2', 0.0)):.6g}",
                f"{float(res.get('ratio12', 0.0)):.6g}",
                f"{float(res.get('jsd_to_eta', 0.0)):.6g}",
                f"{float(res.get('bic_gap_sd', 0.0)):.6g}",
                f"{float(res.get('doublet_minor_frac', 0.0)):.6g}",
                str(bool(res.get("near_tie_sd", False))),
                f"{float(res.get('bic_empty', 0.0)):.6g}" if res.get("bic_empty") is not None else "",
                f"{float(res.get('bic_best', 0.0)):.6g}" if res.get("bic_best") is not None else "",
            ]
        )
        + "\n"
    )


def _call_from_chunk_spill_streaming(
    spill_path: Path,
    eta: pd.Series,
    topk: Dict[str, List[str]],
    cfg: MergeConfig,
    out_handle,
) -> int:
    """
    Bulletproof: aggregate all rows for each barcode across the entire spill file
    without assuming contiguity.

    Memory is bounded by spill size for one chunk. For typical chunk sizes (e.g. 100–2000 barcodes),
    this is usually fine. If you need stricter bounds, switch this to an on-disk barcode sorter.
    """
    called = 0
    # Accumulate per barcode as lists of small frames (cheap concat at end per barcode).
    acc: Dict[str, List[pd.DataFrame]] = {}

    for chunk in pd.read_csv(
        spill_path,
        sep="\t",
        compression="gzip",
        chunksize=int(cfg.pass2_chunksize),
        dtype={"barcode": "string", "read_id": "string", "genome": "string", "L": "float32", "L_amb": "float32"},
    ):
        if chunk.empty:
            continue
        for bc, sub in chunk.groupby("barcode", sort=False):
            bc_s = str(bc)
            acc.setdefault(bc_s, []).append(sub[["read_id", "genome", "L"]].copy())

    for bc_s, parts in acc.items():
        L_block = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
        cand = topk.get(bc_s, [])
        res = _select_model_for_barcode(L_block, eta, cfg, cand)
        _write_call_row(out_handle, bc_s, res)
        called += 1

    return called


# --------------------------------------------------------------------------------------
# Main command
# --------------------------------------------------------------------------------------


@app.command()
def genotyping(
    assign: str = typer.Option(
        ...,
        help="Input glob for assign filtered tables (e.g. sample/cell_map_ref_chunks/*filtered.tsv.gz)",
    ),
    outdir: Path = typer.Option(Path("merge_out"), help="Output directory"),
    sample: str = typer.Option("sample", help="Sample name"),
    # New shard policy (chunk-defined partitions)
    cell_chunks_glob: Optional[str] = typer.Option(
        None,
        "--cell-chunks-glob",
        help="Glob for cell_map_ref_chunk_*.txt files (barcode lists). If provided, spill/calling is per chunk (bulletproof).",
    ),
    chunk_id_regex: str = typer.Option(
        r"chunk_(\d+)",
        "--chunk-id-regex",
        help=r"Regex with one capturing group for chunk id (applied to both assign filenames and chunk list filenames). Default: r'chunk_(\d+)'.",
    ),
    # Core
    min_reads: int = typer.Option(5, "--min-reads"),
    single_mass_min: float = typer.Option(0.6, "--single-mass-min"),
    ratio_top1_top2_min: float = typer.Option(2.0, "--ratio-top1-top2-min"),
    near_tie_margin: float = typer.Option(2.0, "--near-tie-margin"),
    threads: int = typer.Option(4, "--threads", "-t", min=1, help="Number of workers for Pass 1 (per-file parallelism)"),
    pass1_workers: Optional[int] = typer.Option(
        None,
        "--pass1-workers",
        help="Override Pass 1 worker processes (if set, supersedes --threads).",
    ),
    chunk_rows: int = typer.Option(5_000_000, "--chunk-rows", help="Rows per pandas chunk when reading assign tables"),
    pass2_chunksize: int = typer.Option(200_000, "--pass2-chunksize", help="Chunksize when reading spill files"),
    winner_only: bool = typer.Option(True, "--winner-only/--no-winner-only", help="Winner-only posterior mode"),
    # Promiscuous ambiguous filter (MAPQ-based)
    max_hits: Optional[int] = typer.Option(
        3,
        "--max-hits",
        help="Drop ambiguous groups with >max_hits near-top MAPQ alignments; set to 0 to disable.",
    ),
    hits_delta_mapq: Optional[float] = typer.Option(2.0, "--hits-delta-mapq", help="Near-top MAPQ window (best_MAPQ - delta)"),
    max_rows_per_read_guard: Optional[int] = typer.Option(500, "--max-rows-per-read-guard", help="Guard: drop ambiguous groups with >this rows per (bc,read) early"),
    # fusion, Score weights
    beta: float = typer.Option(1.0, "--beta"),
    w_as: float = typer.Option(1.0, "--w-as"),
    w_mapq: float = typer.Option(1.0, "--w-mapq"),
    w_nm: float = typer.Option(1.0, "--w-nm"),
    ambient_const: float = typer.Option(1e-3, "--ambient-const"),
    # empty gates
    empty_top1_max: float = typer.Option(0.6, "--empty-top1-max"),
    empty_ratio12_max: float = typer.Option(1.5, "--empty-ratio12-max"),
    empty_reads_max: Optional[int] = typer.Option(None, "--empty-reads-max"),
    empty_tau_quantile: float = typer.Option(0.95, "--empty-tau-quantile"),
    # doublet gates
    bic_margin: float = typer.Option(6.0, "--bic-margin"),
    doublet_minor_min: float = typer.Option(0.20, "--doublet-minor-min"),
    # ambient iteration
    eta_iters: int = typer.Option(2, "--eta-iters"),
    eta_seed_quantile: float = typer.Option(0.02, "--eta-seed-quantile"),
    topk_genomes: int = typer.Option(3, "--topk-genomes"),
    # grid search knobs
    alpha_grid: float = typer.Option(0.02, "--alpha-grid"),
    rho_grid: float = typer.Option(0.05, "--rho-grid"),
    max_alpha: float = typer.Option(0.5, "--max-alpha"),
):
    outdir.mkdir(parents=True, exist_ok=True)

    pass1_workers_eff = int(pass1_workers) if pass1_workers is not None else int(threads)

    max_hits_eff = None if (max_hits is None or int(max_hits) <= 0) else int(max_hits)
    hits_delta_mapq_eff = None if max_hits_eff is None else float(hits_delta_mapq)

    cfg = MergeConfig(
        sample=str(sample),
        chunk_rows=int(chunk_rows),
        pass2_chunksize=int(pass2_chunksize),
        winner_only=bool(winner_only),
        pass1_workers=int(pass1_workers_eff),
        # filter
        max_hits=max_hits_eff,
        hits_delta_mapq=hits_delta_mapq_eff,
        max_rows_per_read_guard=(int(max_rows_per_read_guard) if max_rows_per_read_guard is not None else None),
        # fusion / weights
        beta=float(beta),
        w_as=float(w_as),
        w_mapq=float(w_mapq),
        w_nm=float(w_nm),
        ambient_const=float(ambient_const),
        # calling thresholds / gates
        min_reads=int(min_reads),
        single_mass_min=float(single_mass_min),
        ratio_top1_top2_min=float(ratio_top1_top2_min),
        bic_margin=float(bic_margin),
        near_tie_margin=float(near_tie_margin),
        doublet_minor_min=float(doublet_minor_min),
        # empty gates
        empty_top1_max=float(empty_top1_max),
        empty_ratio12_max=float(empty_ratio12_max),
        empty_reads_max=(int(empty_reads_max) if empty_reads_max is not None else None),
        empty_tau_quantile=float(empty_tau_quantile),
        # ambient iteration / candidates
        eta_iters=int(eta_iters),
        eta_seed_quantile=float(eta_seed_quantile),
        topk_genomes=int(topk_genomes),
        # grid
        alpha_grid=float(alpha_grid),
        rho_grid=float(rho_grid),
        max_alpha=float(max_alpha),
    )

    files = [Path(f) for f in glob.glob(assign, recursive=True)]
    if not files:
        raise typer.BadParameter(f"No files matched: {assign}")

    # New policy: chunk-defined spill/calling (recommended)
    chunk_map: Dict[str, set] = {}
    if cell_chunks_glob:
        chunk_map = _load_chunk_barcode_lists(cell_chunks_glob, chunk_id_regex)

    tmp_dir = outdir / f"tmp_{sample}"
    spill_root = tmp_dir / "L_chunk_spill"
    spill_root.mkdir(parents=True, exist_ok=True)

    typer.echo(f"[genotyping] sample={sample} files={len(files)} pass1_workers={pass1_workers_eff}")
    typer.echo(f"[genotyping] filter: max_hits={cfg.max_hits} hits_delta_mapq={cfg.hits_delta_mapq} guard={cfg.max_rows_per_read_guard}")
    typer.echo(f"[genotyping] winner_only={cfg.winner_only} ambient_const={cfg.ambient_const}")
    if cell_chunks_glob:
        typer.echo(f"[genotyping] chunk-policy ENABLED: {len(chunk_map)} chunk lists loaded")
    else:
        typer.echo("[genotyping] chunk-policy DISABLED: no --cell-chunks-glob provided (still uses chunk spills per assign file)")

    # -----------------------
    # Pass 1: per-file parallel spill + counts
    # -----------------------
    typer.echo("[1/4] Pass 1: streaming assign inputs → per-read posteriors + spill (chunk policy)")

    C_list: List[pd.DataFrame] = []
    N_list: List[pd.DataFrame] = []
    spill_paths: List[Path] = []

    def _job(i: int, f: Path) -> Pass1Outputs:
        cid = _infer_chunk_id_from_path(f, chunk_id_regex) or f"{i:04d}"
        bc_set = None
        if cell_chunks_glob:
            if cid not in chunk_map:
                raise ValueError(f"Assign file chunk_id={cid} not found in chunk lists: {f}")
            bc_set = chunk_map[cid]

        spill_path = spill_root / f"L_chunk_{cid}.tsv.gz"
        out = _pass1_process_one_file_chunkspill(f, cfg, spill_path, bc_set)
        out.chunk_id = cid
        return out

    with ProcessPoolExecutor(max_workers=int(pass1_workers_eff)) as ex:
        futs = [ex.submit(_job, i, f) for i, f in enumerate(files)]
        for fut in as_completed(futs):
            out = fut.result()
            spill_paths.append(out.spill_path)
            if not out.C.empty:
                C_list.append(out.C)
            if not out.N.empty:
                N_list.append(out.N)

    C_all = (
        pd.concat(C_list, ignore_index=True).groupby(["barcode", "genome"], sort=False)["C"].sum().reset_index()
        if C_list else pd.DataFrame(columns=["barcode", "genome", "C"])
    )
    N_all = (
        pd.concat(N_list, ignore_index=True).groupby("barcode", sort=False)["n_reads"].sum().reset_index()
        if N_list else pd.DataFrame(columns=["barcode", "n_reads"])
    )

    if C_all.empty:
        raise RuntimeError("Pass 1 produced no mass (C_all empty). Check input tables and schema.")

    all_genomes = C_all["genome"].astype(str).unique().tolist()

    topk: Dict[str, List[str]] = {}
    for bc, sub in C_all.groupby("barcode", sort=False):
        s = sub.sort_values("C", ascending=False)["genome"].astype(str).head(int(cfg.topk_genomes)).tolist()
        topk[str(bc)] = s

    C_all.to_csv(outdir / f"{sample}_C_all.tsv.gz", sep="\t", index=False, compression="gzip")
    N_all.to_csv(outdir / f"{sample}_N_all.tsv.gz", sep="\t", index=False, compression="gzip")
    typer.echo(f"[1/4] wrote: {sample}_C_all.tsv.gz, {sample}_N_all.tsv.gz")
    typer.echo(f"[1/4] spill_root: {spill_root}  spills={len(set(spill_paths))}")

    # -----------------------
    # Pass 2: ambient learning (eta)
    # -----------------------
    typer.echo("[2/4] Pass 2: ambient learning (eta) from chunk spills")
    eta, eta_meta = _learn_eta(C_all, N_all, spill_root, all_genomes, cfg)
    eta_out = outdir / f"{sample}_eta.tsv.gz"
    eta.to_frame("eta").to_csv(eta_out, sep="\t", compression="gzip")
    typer.echo(f"[2/4] eta saved: {eta_out.name}  seed_sizes={eta_meta.get('seed_sizes')}")

    # -----------------------
    # Pass 3: calling (chunk spills) -> genotype calls
    # -----------------------
    typer.echo("[3/4] Pass 3: model selection / calls from chunk spills (bulletproof)")
    calls_path = outdir / f"{sample}_genotype_calls.tsv.gz"
    with gzip.open(calls_path, "wt") as oh:
        oh.write(
            "\t".join(
                [
                    "barcode",
                    "call",
                    "model",
                    "genome_1",
                    "genome_2",
                    "alpha",
                    "rho",
                    "n_reads",
                    "n_effective",
                    "p_top1",
                    "p_top2",
                    "ratio12",
                    "jsd_to_eta",
                    "bic_gap_sd",
                    "doublet_minor_frac",
                    "near_tie_sd",
                    "bic_empty",
                    "bic_best",
                ]
            )
            + "\n"
        )

        called_total = 0
        for spill in sorted(set(spill_paths)):
            if not spill.exists():
                continue
            called_total += _call_from_chunk_spill_streaming(spill, eta, topk, cfg, oh)

    typer.echo(f"[3/4] calls written: {calls_path.name} (emitted rows={called_total})")

    # -----------------------
    # Pass 4: done + pointers
    # -----------------------
    typer.echo("[4/4] Done.")
    typer.echo(f"  - Mass table:   {outdir / f'{sample}_C_all.tsv.gz'}")
    typer.echo(f"  - Reads table:  {outdir / f'{sample}_N_all.tsv.gz'}")
    typer.echo(f"  - Eta:          {eta_out}")
    typer.echo(f"  - Calls:        {calls_path}")
    typer.echo(f"  - Spill root:   {spill_root}")


if __name__ == "__main__":
    app()
