#!/usr/bin/env python3
"""
src/ambientmapper/genotyping.py

ambientmapper genotyping — posterior-aware merge → per-cell genotype calls (+ empty-aware)

Key behaviors (DO NOT change gate design):
- Decision 1 (Empty/Noise): compare BIC_empty vs min(BIC_single, BIC_doublet) using empty_bic_margin,
  and optionally constrain empties by eta-likeness (JSD) + simple structural gates.
- Decision 2 (Singlet vs Doublet): compare BIC_single vs BIC_doublet using bic_margin (ΔBIC gate).
- Purity/strength sublabels: use near_tie_margin (optional) + purity thresholds to label dirty_singlet/weak_doublet.

Shard policy (requested):
- If --cell-chunks-glob points to cell_map_ref_chunk_*.txt barcode lists, we precompute barcode->chunk_id
  and route reads to shards deterministically by chunk_id % shards.
- Otherwise, fall back to hash(barcode) % shards.

IMPORTANT entrypoint separation:
- _run_genotyping(...) is a plain Python function and is the only function that cli.py should call.
- genotyping_cmd(...) is the Typer CLI wrapper that parses flags and forwards them to _run_genotyping.
  This prevents OptionInfo leakage when called programmatically.

Fixes included:
- Pass2 now SKIPS empty gzip shard files (your 38-byte files) instead of crashing with
  pandas.errors.EmptyDataError: No columns to parse from file.
- Pass1 now opens shard gzip outputs lazily (only when writing first rows), avoiding creation of empty files.
- Pass1 multiprocessing uses a picklable top-level worker function (_pass1_worker_job).

Expected inputs:
- assign "filtered" outputs: TSV(.gz) with columns like Read, BC, Genome, AS, MAPQ, NM, assigned_class, ...
- cell chunk lists: one barcode per line, e.g. subset2000_Seedling_cell_map_ref_chunk_0001.txt
"""

from __future__ import annotations

import glob
import gzip
import json
import math
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Iterator

import numpy as np
import pandas as pd
import typer
from pydantic import BaseModel, ConfigDict

app = typer.Typer(add_completion=False, no_args_is_help=True)

# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------


class MergeConfig(BaseModel):
    """
    Runtime config used by _run_genotyping.

    Notes:
    - empty_seed_bic_min is kept as a parameter (CLI-compatible), but is not currently used in eta learning
      because BIC-empty evidence requires read-level blocks. Do not silently reinterpret it.
    """

    model_config = ConfigDict(extra="forbid")

    # Fusion / scoring -> posteriors
    beta: float = 1.0
    w_as: float = 1.0
    w_mapq: float = 1.0
    w_nm: float = 1.0
    ambient_const: float = 1e-3  # ambient pseudocount in posterior denominator

    # Calling thresholds
    min_reads: int = 5
    single_mass_min: float = 0.6
    ratio_top1_top2_min: float = 2.0
    doublet_minor_min: float = 0.20

    # BIC gates (match figure)
    empty_bic_margin: float = 10.0  # Decision 1: empty vs non-empty
    bic_margin: float = 6.0  # Decision 2: singlet vs doublet (ΔBIC gate)
    near_tie_margin: float = 2.0  # sub-gate for dirty/weak labeling only

    # Empty structural gates (used for optional empty call constraints + eta seed refinement)
    empty_top1_max: float = 0.6
    empty_ratio12_max: float = 1.5
    empty_reads_max: Optional[int] = None
    empty_jsd_max: Optional[float] = None
    jsd_normalize: bool = True

    # Ambient learning (eta)
    eta_iters: int = 2
    eta_seed_quantile: float = 0.02
    empty_tau_quantile: float = 0.95
    empty_seed_bic_min: float = 10.0  # currently not used (kept for CLI compatibility)

    # Search grid
    alpha_grid: float = 0.02
    rho_grid: float = 0.05
    max_alpha: float = 0.5
    topk_genomes: int = 3

    # System / IO
    sample: str = "sample"
    shards: int = 32
    chunk_rows: int = 5_000_000
    pass2_chunksize: int = 200_000

    # Posterior mode
    winner_only: bool = True

    # Pass 1 parallelism (files)
    pass1_workers: int = 1

    # Promiscuous ambiguous filter (applies only to assigned_class=="ambiguous")
    max_hits: Optional[int] = None
    hits_delta_mapq: Optional[float] = None
    max_rows_per_read_guard: Optional[int] = 500


# --------------------------------------------------------------------------------------
# Config JSON helpers
# --------------------------------------------------------------------------------------


def _read_cfg_json(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "rt") as f:
        return json.load(f)


def _infer_sample(cfg_json: Dict[str, Any], sample_override: Optional[str]) -> str:
    if sample_override:
        return str(sample_override)
    if cfg_json.get("sample"):
        return str(cfg_json["sample"])
    return "sample"


def _infer_paths(cfg_json: Dict[str, Any], sample: str) -> Dict[str, Path]:
    workdir = cfg_json.get("workdir")
    if not workdir:
        raise ValueError("config JSON missing required key: 'workdir'")
    root = Path(workdir) / sample
    return {
        "root": root,
        "qc": root / "qc",
        "filtered": root / "filtered",
        "chunks": root / "chunks",
        "final": root / "final",
    }


def _infer_assign_glob(d: Dict[str, Path]) -> str:
    candidates = [
        str(d["chunks"] / "cell_map_ref_chunks" / "*_filtered.tsv.gz"),
        str(d["chunks"] / "cell_map_ref_chunks" / "*_filtered.tsv"),
        str(d["chunks"] / "cell_map_ref_chunks" / "*filtered.tsv.gz"),
        str(d["chunks"] / "cell_map_ref_chunks" / "*filtered.tsv"),
    ]
    for pat in candidates:
        if glob.glob(pat):
            return pat
    return str(d["root"] / "**" / "cell_map_ref_chunks" / "*_filtered.tsv.gz")


def _infer_cell_chunks_glob(d: Dict[str, Path]) -> str:
    candidates = [
        str(d["chunks"] / "cell_map_ref_chunks" / "*cell_map_ref_chunk_*.txt"),
        str(d["chunks"] / "cell_map_ref_chunks" / "*chunk_*.txt"),
    ]
    for pat in candidates:
        if glob.glob(pat):
            return pat
    return str(d["root"] / "**" / "cell_map_ref_chunks" / "*chunk_*.txt")


# --------------------------------------------------------------------------------------
# Helpers: schema and coercion
# --------------------------------------------------------------------------------------


def _coerce_assign_schema(df: pd.DataFrame) -> pd.DataFrame:
    m = {"BC": "barcode", "Read": "read_id", "Genome": "genome"}
    out = df.rename(columns=m)

    required = ["barcode", "read_id", "genome", "AS", "MAPQ", "NM"]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"Input is missing required columns: {missing}")

    out["barcode"] = out["barcode"].astype("string")
    out["read_id"] = out["read_id"].astype("string")
    out["genome"] = out["genome"].astype("string")

    for c in ["AS", "MAPQ", "NM"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    if "assigned_class" in out.columns:
        out["assigned_class"] = out["assigned_class"].astype("string")
    else:
        out["assigned_class"] = pd.Series(["ambiguous"] * len(out), dtype="string")

    return out


def _reduce_alignments_to_per_genome(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["barcode", "read_id", "genome"]
    agg = {"AS": "max", "MAPQ": "max", "NM": "min", "assigned_class": "first"}
    return df.groupby(keys, observed=True, sort=False).agg(agg).reset_index()


# --------------------------------------------------------------------------------------
# Promiscuous ambiguous filter (NumPy)
# --------------------------------------------------------------------------------------

def _filter_promiscuous_ambiguous_reads(df: pd.DataFrame, cfg: MergeConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
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
# Core: posterior computation
# --------------------------------------------------------------------------------------


def _compute_read_posteriors(df: pd.DataFrame, cfg: MergeConfig) -> pd.DataFrame:
    df = df[["barcode", "read_id", "genome", "AS", "MAPQ", "NM"]].copy()
    df = df.dropna(subset=["barcode", "read_id", "genome"])

    AS = df["AS"].to_numpy(np.float32, copy=False)
    MQ = df["MAPQ"].to_numpy(np.float32, copy=False)
    NM = df["NM"].to_numpy(np.float32, copy=False)

    rid = df["barcode"].astype(str) + "::" + df["read_id"].astype(str)
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
# Model selection (gate design preserved)
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
    minor_frac = float(min(p_top1, p_top2)) if (p_top1 > 0 and p_top2 > 0) else 0.0

    jsd = _barcode_jsd_from_mass(mass, eta, normalize=bool(cfg.jsd_normalize))

    out: Dict[str, Any] = {
        "n_reads": read_count,
        "n_effective": tot,
        "p_top1": p_top1,
        "p_top2": p_top2,
        "ratio12": ratio12,
        "top_genome": (mass.index[0] if len(mass) else None),
        "jsd_to_eta": jsd,
        "doublet_minor_frac": minor_frac,
    }

    bic_empty = _bic(_loglik_empty(L_block, eta), 0, read_count)
    out["bic_empty"] = float(bic_empty)

    if read_count < int(cfg.min_reads) or not candidate_genomes:
        out.update(
            call="low_reads",
            model=None,
            genome_1=None,
            genome_2=None,
            bic_best=None,
            alpha=None,
            rho=None,
            bic_single=None,
            bic_doublet=None,
            bic_gap_sd=None,
            near_tie_sd=None,
        )
        return out

    alphas = np.arange(0.0, float(cfg.max_alpha) + 1e-9, float(cfg.alpha_grid))

    best_single: Optional[Dict[str, Any]] = None
    for g1 in candidate_genomes:
        for a in alphas:
            bic = _bic(_loglik_for_params(L_block, eta, "single", g1, None, float(a)), 1, read_count)
            if best_single is None or bic < best_single["bic"]:
                best_single = {"model": "single", "g1": g1, "alpha": float(a), "bic": float(bic)}

    best_doublet: Optional[Dict[str, Any]] = None
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

    bic_s = float(best_single["bic"]) if best_single else float("inf")
    bic_d = float(best_doublet["bic"]) if best_doublet else float("inf")
    out["bic_single"] = bic_s if np.isfinite(bic_s) else None
    out["bic_doublet"] = bic_d if np.isfinite(bic_d) else None

    # --------------------------
    # Decision 1: Empty/Noise
    # --------------------------
    bic_best_non_empty = min(bic_s, bic_d)
    empty_by_bic = (bic_empty + float(cfg.empty_bic_margin)) < bic_best_non_empty

    structural_empty = (p_top1 <= float(cfg.empty_top1_max)) and (ratio12 <= float(cfg.empty_ratio12_max))
    if cfg.empty_reads_max is not None:
        structural_empty = structural_empty and (read_count <= int(cfg.empty_reads_max))

    jsd_ok = True
    if cfg.empty_jsd_max is not None:
        jsd_ok = jsd <= float(cfg.empty_jsd_max)

    if empty_by_bic and structural_empty and jsd_ok:
        out.update(
            call="empty",
            model="empty",
            genome_1=None,
            genome_2=None,
            bic_best=float(bic_empty),
            alpha=None,
            rho=None,
            bic_gap_sd=None,
            near_tie_sd=None,
        )
        return out

    # --------------------------
    # Decision 2: Single vs Doublet (ΔBIC gate)
    # --------------------------
    chosen: Optional[Dict[str, Any]] = None
    call_core: str = "ambiguous"

    if np.isfinite(bic_s) and np.isfinite(bic_d):
        if (bic_s + float(cfg.bic_margin)) < bic_d:
            chosen = best_single
            call_core = "single"
        elif (bic_d + float(cfg.bic_margin)) < bic_s:
            chosen = best_doublet
            call_core = "doublet"
        else:
            chosen = best_single if bic_s <= bic_d else best_doublet
            call_core = "ambiguous"
        bic_gap = abs(bic_s - bic_d)
    else:
        chosen = best_single if np.isfinite(bic_s) else best_doublet
        call_core = (
            "single"
            if (chosen and chosen["model"] == "single")
            else ("doublet" if (chosen and chosen["model"] == "doublet") else "ambiguous")
        )
        bic_gap = float("inf")

    out["bic_gap_sd"] = float(bic_gap) if np.isfinite(bic_gap) else None
    out["near_tie_sd"] = bool(np.isfinite(bic_gap) and (bic_gap <= float(cfg.near_tie_margin)))

    if chosen is None:
        out.update(call="uncalled", model=None, genome_1=None, genome_2=None, bic_best=None, alpha=None, rho=None)
        return out

    out["model"] = chosen["model"]
    out["genome_1"] = chosen.get("g1")
    out["genome_2"] = chosen.get("g2")
    out["alpha"] = chosen.get("alpha")
    out["rho"] = chosen.get("rho")
    out["bic_best"] = float(chosen["bic"])

    # --------------------------
    # Purity/strength sublabels
    # --------------------------
    if call_core == "single":
        passes_mass = p_top1 >= float(cfg.single_mass_min)
        passes_ratio = ratio12 >= float(cfg.ratio_top1_top2_min)
        out["call"] = "single_clean" if (passes_mass and passes_ratio and (not out["near_tie_sd"])) else "dirty_singlet"
    elif call_core == "doublet":
        passes_minor = minor_frac >= float(cfg.doublet_minor_min)
        out["call"] = "doublet" if (passes_minor and (not out["near_tie_sd"])) else "weak_doublet"
    else:
        out["call"] = "ambiguous"

    return out


# --------------------------------------------------------------------------------------
# Cell chunk parsing for shard routing
# --------------------------------------------------------------------------------------


def _parse_chunk_id_from_path(p: Path, chunk_id_regex: str) -> Optional[int]:
    m = re.search(chunk_id_regex, p.name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _load_barcode_to_chunk_id(cell_chunks_glob: str, chunk_id_regex: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    files = [Path(x) for x in glob.glob(cell_chunks_glob)]
    for fp in sorted(files):
        cid = _parse_chunk_id_from_path(fp, chunk_id_regex)
        if cid is None:
            continue
        with open(fp, "rt") as f:
            for line in f:
                bc = line.strip()
                if not bc:
                    continue
                if bc not in out:
                    out[bc] = cid
    return out


def _route_shards_for_barcodes(barcodes: pd.Series, cfg: MergeConfig, bc_to_chunk: Optional[Dict[str, int]]) -> np.ndarray:
    bcs = barcodes.astype(str).to_numpy()
    if bc_to_chunk:
        ser = pd.Series(bcs, dtype="string").map(bc_to_chunk)
        miss = ser.isna().to_numpy()
        sid = np.empty(len(bcs), dtype=np.int32)

        cid = ser.fillna(-1).to_numpy(dtype=np.int64)
        ok = cid >= 0
        sid[ok] = (cid[ok] % int(cfg.shards)).astype(np.int32)

        if miss.any():
            h = pd.util.hash_pandas_object(pd.Series(bcs[miss], dtype="string"), index=False).to_numpy(np.uint64)
            sid[miss] = (h % np.uint64(cfg.shards)).astype(np.int32)
        return sid

    h = pd.util.hash_pandas_object(pd.Series(bcs, dtype="string"), index=False).to_numpy(np.uint64)
    return (h % np.uint64(cfg.shards)).astype(np.int32)


# --------------------------------------------------------------------------------------
# Robust shard reading (SKIP empty gz files)
# --------------------------------------------------------------------------------------


def _iter_shard_chunks(shard_path: Path, cfg: MergeConfig) -> Iterator[pd.DataFrame]:
    """
    Yield pandas chunks from a shard_*.tsv.gz file.
    Skips empty gzip files (common when Pass1 created files but wrote no header/rows).
    """
    try:
        sz = shard_path.stat().st_size
    except FileNotFoundError:
        return
    # Your empty shards are ~38 bytes (gzip container only). Anything very small is not a valid TSV header.
    if sz < 64:
        return

    try:
        for chunk in pd.read_csv(
            shard_path,
            sep="\t",
            compression="gzip",
            chunksize=int(cfg.pass2_chunksize),
            dtype={
                "barcode": "string",
                "read_id": "string",
                "genome": "string",
                "L": "float32",
                "L_amb": "float32",
            },
        ):
            if chunk is None or chunk.empty:
                continue
            yield chunk
    except pd.errors.EmptyDataError:
        # Header missing / no columns.
        return


# --------------------------------------------------------------------------------------
# Pass 1: stream assign files -> (C_all, N_all) + shard spill
# --------------------------------------------------------------------------------------


@dataclass
class Pass1Outputs:
    C: pd.DataFrame
    N: pd.DataFrame


def _pass1_worker_job(args: Tuple[int, str, MergeConfig, str, Optional[Dict[str, int]]]) -> Pass1Outputs:
    """
    Top-level worker so it is picklable by multiprocessing.
    args: (worker_index, filepath, cfg, shard_root, bc_to_chunk)
    """
    i, fp_str, cfg, shard_root_str, bc_to_chunk = args
    fp = Path(fp_str)
    shard_root = Path(shard_root_str)
    wdir = shard_root / f"w{i:03d}"
    return _pass1_process_one_file(fp, cfg, wdir, bc_to_chunk)


def _pass1_process_one_file(fp: Path, cfg: MergeConfig, shard_dir: Path, bc_to_chunk: Optional[Dict[str, int]]) -> Pass1Outputs:
    """
    Writes per-read posterior rows to shard gzip files under shard_dir.
    FIX: shard files are opened lazily (only if we actually write rows),
         preventing creation of 38-byte empty gzip files.
    """
    shard_dir.mkdir(parents=True, exist_ok=True)

    handles: Dict[int, gzip.GzipFile] = {}
    header_written = np.zeros(cfg.shards, dtype=bool)

    def _get_handle(sid: int) -> gzip.GzipFile:
        if sid in handles:
            return handles[sid]
        h = gzip.open(shard_dir / f"shard_{sid:02d}.tsv.gz", "at")
        handles[sid] = h
        return h

    C_parts: List[pd.DataFrame] = []
    N_parts: List[pd.DataFrame] = []

    for raw in pd.read_csv(fp, sep="\t", chunksize=int(cfg.chunk_rows), low_memory=False):
        df = _coerce_assign_schema(raw)
        df = _reduce_alignments_to_per_genome(df)
        df = df.dropna(subset=["barcode", "read_id", "genome"])

        df, _ = _filter_promiscuous_ambiguous_reads(df, cfg)
        if df.empty:
            continue

        Ldf = _compute_read_posteriors(df, cfg)
        if Ldf.empty:
            continue

        C_parts.append(Ldf.groupby(["barcode", "genome"], sort=False)["L"].sum().rename("C").reset_index())
        N_parts.append(Ldf.groupby("barcode", sort=False)["read_id"].nunique().rename("n_reads").reset_index())

        sid = _route_shards_for_barcodes(Ldf["barcode"], cfg, bc_to_chunk)
        Ldf = Ldf.assign(_sid=sid)

        # Write per shard (avoid creating files for shards with no rows)
        for i in range(cfg.shards):
            sub = Ldf[Ldf["_sid"] == i]
            if sub.empty:
                continue
            h = _get_handle(i)
            if not header_written[i]:
                h.write("barcode\tread_id\tgenome\tL\tL_amb\n")
                header_written[i] = True
            sub.drop(columns="_sid").to_csv(h, sep="\t", header=False, index=False)

    for h in handles.values():
        h.close()

    C_out = pd.concat(C_parts, ignore_index=True) if C_parts else pd.DataFrame(columns=["barcode", "genome", "C"])
    N_out = pd.concat(N_parts, ignore_index=True) if N_parts else pd.DataFrame(columns=["barcode", "n_reads"])
    return Pass1Outputs(C=C_out, N=N_out)


# --------------------------------------------------------------------------------------
# Ambient learning (eta)
# --------------------------------------------------------------------------------------


def _compute_eta_from_shards(shard_root: Path, target_bcs: Sequence[str], all_genomes: Sequence[str], cfg: MergeConfig) -> pd.Series:
    target_set = set(map(str, target_bcs))
    mix = {g: 0.0 for g in all_genomes}

    if not target_set:
        eta = pd.Series({g: 1.0 for g in all_genomes}, dtype=float)
        return eta / eta.sum()

    for shard in shard_root.rglob("shard_*.tsv.gz"):
        for chunk in _iter_shard_chunks(shard, cfg):
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


def _refine_eta_seed_by_jsd(C_all: pd.DataFrame, N_all: pd.DataFrame, eta: pd.Series, cfg: MergeConfig) -> List[str]:
    if C_all.empty:
        return []

    mass = C_all.groupby(["barcode", "genome"], sort=False)["C"].sum().reset_index()
    eff = mass.groupby("barcode", sort=False)["C"].sum().rename("n_effective")
    top = mass.sort_values(["barcode", "C"], ascending=[True, False]).groupby("barcode", sort=False).head(2)

    top1 = top.groupby("barcode", sort=False)["C"].nth(0)
    top2 = top.groupby("barcode", sort=False)["C"].nth(1).reindex(top1.index).fillna(0.0)

    stats = pd.DataFrame({"n_effective": eff.reindex(top1.index).fillna(0.0)})
    stats["top1"] = top1
    stats["top2"] = top2
    stats["p_top1"] = stats["top1"] / (stats["n_effective"] + 1e-12)
    stats["p_top2"] = stats["top2"] / (stats["n_effective"] + 1e-12)
    stats["ratio12"] = stats["p_top1"] / (stats["p_top2"] + 1e-12)

    if not N_all.empty:
        stats["n_reads"] = N_all.set_index("barcode")["n_reads"].reindex(stats.index).fillna(0).astype(int)
    else:
        stats["n_reads"] = 0

    cand = stats[(stats["p_top1"] <= float(cfg.empty_top1_max)) & (stats["ratio12"] <= float(cfg.empty_ratio12_max))]
    if cfg.empty_reads_max is not None:
        cand = cand[cand["n_reads"] <= int(cfg.empty_reads_max)]
    if cand.empty:
        cand = stats

    jsd_vals: Dict[str, float] = {}
    cand_set = set(cand.index.astype(str).tolist())
    for bc, sub in mass.groupby("barcode", sort=False):
        bc_s = str(bc)
        if bc_s not in cand_set:
            continue
        s = sub.set_index("genome")["C"]
        jsd_vals[bc_s] = _barcode_jsd_from_mass(s, eta, normalize=bool(cfg.jsd_normalize))

    if not jsd_vals:
        return []

    jsd_ser = pd.Series(jsd_vals, dtype=float)
    tau = jsd_ser.quantile(float(cfg.empty_tau_quantile))
    return jsd_ser[jsd_ser <= tau].index.astype(str).tolist()


def _learn_eta(
    C_all: pd.DataFrame,
    N_all: pd.DataFrame,
    shard_root: Path,
    all_genomes: Sequence[str],
    cfg: MergeConfig,
) -> Tuple[pd.Series, Dict[str, Any]]:
    meta: Dict[str, Any] = {"iters": int(cfg.eta_iters), "seed_sizes": []}

    seed = _pick_initial_eta_seed(C_all, N_all, cfg)
    meta["seed_sizes"].append(len(seed))
    eta = _compute_eta_from_shards(shard_root, seed, list(all_genomes), cfg)

    for _ in range(int(cfg.eta_iters)):
        seed = _refine_eta_seed_by_jsd(C_all, N_all, eta, cfg)
        meta["seed_sizes"].append(len(seed))
        if not seed:
            break
        eta = _compute_eta_from_shards(shard_root, seed, list(all_genomes), cfg)

    return eta, meta


# --------------------------------------------------------------------------------------
# Calling: per-shard file (call each barcode once)
# --------------------------------------------------------------------------------------


def _call_from_shard_file(
    shard_path: Path,
    eta: pd.Series,
    topk: Dict[str, List[str]],
    cfg: MergeConfig,
    out_handle,
) -> int:
    buckets: Dict[str, List[pd.DataFrame]] = {}

    for chunk in _iter_shard_chunks(shard_path, cfg):
        for bc, sub in chunk.groupby("barcode", sort=False):
            buckets.setdefault(str(bc), []).append(sub[["read_id", "genome", "L"]].copy())

    called = 0
    for bc_s, parts in buckets.items():
        L_block = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
        cand = topk.get(bc_s, [])
        res = _select_model_for_barcode(L_block, eta, cfg, cand)

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
                    f"{float(res.get('bic_empty', 0.0)):.6g}" if res.get("bic_empty") is not None else "",
                    f"{float(res.get('bic_best', 0.0)):.6g}" if res.get("bic_best") is not None else "",
                    f"{float(res.get('bic_single')):.6g}" if res.get("bic_single") is not None else "",
                    f"{float(res.get('bic_doublet')):.6g}" if res.get("bic_doublet") is not None else "",
                    f"{float(res.get('bic_gap_sd')):.6g}" if res.get("bic_gap_sd") is not None else "",
                    "1" if res.get("near_tie_sd") else "0" if res.get("near_tie_sd") is not None else "",
                    f"{float(res.get('doublet_minor_frac', 0.0)):.6g}",
                ]
            )
            + "\n"
        )
        called += 1

    return called


# --------------------------------------------------------------------------------------
# Core entrypoint (called by cli.py)
# --------------------------------------------------------------------------------------


def _run_genotyping(
    *,
    config: Path,
    assign: Optional[str] = None,
    cell_chunks_glob: Optional[str] = None,
    chunk_id_regex: str = r"chunk_(\d+)",
    outdir: Optional[Path] = None,
    sample: Optional[str] = None,
    # core
    min_reads: Optional[int] = None,
    single_mass_min: Optional[float] = None,
    ratio_top1_top2_min: Optional[float] = None,
    threads: Optional[int] = None,
    pass1_workers: Optional[int] = None,
    shards: Optional[int] = None,
    chunk_rows: Optional[int] = None,
    pass2_chunksize: Optional[int] = None,
    winner_only: Optional[bool] = None,
    # optional read-filter
    max_hits: Optional[int] = None,
    hits_delta_mapq: Optional[float] = None,
    max_rows_per_read_guard: Optional[int] = None,
    # fusion
    beta: Optional[float] = None,
    w_as: Optional[float] = None,
    w_mapq: Optional[float] = None,
    w_nm: Optional[float] = None,
    ambient_const: Optional[float] = None,
    # empty gates
    empty_bic_margin: Optional[float] = None,
    empty_top1_max: Optional[float] = None,
    empty_ratio12_max: Optional[float] = None,
    empty_reads_max: Optional[int] = None,
    empty_seed_bic_min: Optional[float] = None,
    empty_tau_quantile: Optional[float] = None,
    empty_jsd_max: Optional[float] = None,
    jsd_normalize: Optional[bool] = None,
    # doublet gates
    bic_margin: Optional[float] = None,
    doublet_minor_min: Optional[float] = None,
    near_tie_margin: Optional[float] = None,
    # ambient iteration
    eta_iters: Optional[int] = None,
    eta_seed_quantile: Optional[float] = None,
    topk_genomes: Optional[int] = None,
    resume: bool = True,
) -> None:
    cfg_json = _read_cfg_json(Path(config))
    sample_eff = _infer_sample(cfg_json, sample)
    d = _infer_paths(cfg_json, sample_eff)

    outdir_eff = Path(outdir) if outdir is not None else d["final"]
    outdir_eff.mkdir(parents=True, exist_ok=True)

    assign_glob = assign if assign is not None else _infer_assign_glob(d)
    cell_chunks_pat = cell_chunks_glob if cell_chunks_glob is not None else _infer_cell_chunks_glob(d)

    # worker policy: pass1 defaults to threads if not provided
    threads_eff = int(threads) if threads is not None else 1
    pass1_workers_eff = int(pass1_workers) if pass1_workers is not None else int(threads_eff)

    cfg = MergeConfig(sample=str(sample_eff), pass1_workers=int(pass1_workers_eff))

    if shards is not None:
        cfg.shards = int(shards)
    if chunk_rows is not None:
        cfg.chunk_rows = int(chunk_rows)
    if pass2_chunksize is not None:
        cfg.pass2_chunksize = int(pass2_chunksize)
    if winner_only is not None:
        cfg.winner_only = bool(winner_only)

    cfg.max_hits = None if (max_hits is None or int(max_hits) <= 0) else int(max_hits)
    cfg.hits_delta_mapq = None if (cfg.max_hits is None or hits_delta_mapq is None) else float(hits_delta_mapq)
    if max_rows_per_read_guard is not None:
        cfg.max_rows_per_read_guard = int(max_rows_per_read_guard)

    if beta is not None:
        cfg.beta = float(beta)
    if w_as is not None:
        cfg.w_as = float(w_as)
    if w_mapq is not None:
        cfg.w_mapq = float(w_mapq)
    if w_nm is not None:
        cfg.w_nm = float(w_nm)
    if ambient_const is not None:
        cfg.ambient_const = float(ambient_const)

    if min_reads is not None:
        cfg.min_reads = int(min_reads)
    if single_mass_min is not None:
        cfg.single_mass_min = float(single_mass_min)
    if ratio_top1_top2_min is not None:
        cfg.ratio_top1_top2_min = float(ratio_top1_top2_min)

    if empty_bic_margin is not None:
        cfg.empty_bic_margin = float(empty_bic_margin)
    if empty_top1_max is not None:
        cfg.empty_top1_max = float(empty_top1_max)
    if empty_ratio12_max is not None:
        cfg.empty_ratio12_max = float(empty_ratio12_max)
    if empty_reads_max is not None:
        cfg.empty_reads_max = int(empty_reads_max)
    if empty_seed_bic_min is not None:
        cfg.empty_seed_bic_min = float(empty_seed_bic_min)
    if empty_tau_quantile is not None:
        cfg.empty_tau_quantile = float(empty_tau_quantile)
    if empty_jsd_max is not None:
        cfg.empty_jsd_max = float(empty_jsd_max)
    if jsd_normalize is not None:
        cfg.jsd_normalize = bool(jsd_normalize)

    if bic_margin is not None:
        cfg.bic_margin = float(bic_margin)
    if doublet_minor_min is not None:
        cfg.doublet_minor_min = float(doublet_minor_min)
    if near_tie_margin is not None:
        cfg.near_tie_margin = float(near_tie_margin)

    if eta_iters is not None:
        cfg.eta_iters = int(eta_iters)
    if eta_seed_quantile is not None:
        cfg.eta_seed_quantile = float(eta_seed_quantile)
    if topk_genomes is not None:
        cfg.topk_genomes = int(topk_genomes)

    files = [Path(f) for f in glob.glob(assign_glob, recursive=True)]
    if not files:
        raise typer.BadParameter(f"No files matched: {assign_glob}")

    tmp_dir = outdir_eff / f"tmp_{sample_eff}"
    shard_root = tmp_dir / "L_shards_workers"
    shard_root.mkdir(parents=True, exist_ok=True)

    bc_to_chunk: Optional[Dict[str, int]] = None
    chunk_files = glob.glob(cell_chunks_pat)
    if chunk_files:
        bc_to_chunk = _load_barcode_to_chunk_id(cell_chunks_pat, chunk_id_regex)
        typer.echo(
            f"[genotyping] shard policy: chunk-based ({len(bc_to_chunk):,} barcodes from {len(chunk_files)} chunk files)"
        )
    else:
        typer.echo("[genotyping] shard policy: hash(barcode)%shards (no chunk list files found)")

    typer.echo(f"[genotyping] sample={sample_eff} files={len(files)} pass1_workers={pass1_workers_eff} shards={cfg.shards}")
    typer.echo(f"[genotyping] assign_glob={assign_glob}")
    typer.echo(f"[genotyping] cell_chunks_glob={cell_chunks_pat}")
    typer.echo(
        f"[genotyping] filter: max_hits={cfg.max_hits} hits_delta_mapq={cfg.hits_delta_mapq} guard={cfg.max_rows_per_read_guard}"
    )
    typer.echo(f"[genotyping] winner_only={cfg.winner_only} ambient_const={cfg.ambient_const}")
    typer.echo(f"[genotyping] gates: empty_bic_margin={cfg.empty_bic_margin} bic_margin={cfg.bic_margin} near_tie_margin={cfg.near_tie_margin}")
    if cfg.empty_seed_bic_min is not None:
        typer.echo("[genotyping] NOTE: empty_seed_bic_min is currently not used in eta learning in this implementation.")

    C_path = outdir_eff / f"{sample_eff}_C_all.tsv.gz"
    N_path = outdir_eff / f"{sample_eff}_N_all.tsv.gz"
    eta_out = outdir_eff / f"{sample_eff}_eta.tsv.gz"
    calls_path = outdir_eff / f"{sample_eff}_genotype_calls.tsv.gz"

    # -----------------------
    # Pass 1
    # -----------------------
    if resume and C_path.exists() and N_path.exists() and any(shard_root.rglob("shard_*.tsv.gz")):
        typer.echo("[1/4] Pass 1: resume (found C/N + shard spill)")
        C_all = pd.read_csv(C_path, sep="\t", compression="gzip", dtype={"barcode": "string", "genome": "string", "C": "float64"})
        N_all = pd.read_csv(N_path, sep="\t", compression="gzip", dtype={"barcode": "string", "n_reads": "int64"})
    else:
        typer.echo("[1/4] Pass 1: streaming assign inputs → per-read posteriors + spill shards")

        C_list: List[pd.DataFrame] = []
        N_list: List[pd.DataFrame] = []

        args = [(i, str(f), cfg, str(shard_root), bc_to_chunk) for i, f in enumerate(files)]
        with ProcessPoolExecutor(max_workers=int(pass1_workers_eff)) as ex:
            futs = [ex.submit(_pass1_worker_job, a) for a in args]
            for fut in as_completed(futs):
                out = fut.result()
                if not out.C.empty:
                    C_list.append(out.C)
                if not out.N.empty:
                    N_list.append(out.N)

        C_all = (
            pd.concat(C_list, ignore_index=True).groupby(["barcode", "genome"], sort=False)["C"].sum().reset_index()
            if C_list
            else pd.DataFrame(columns=["barcode", "genome", "C"])
        )
        N_all = (
            pd.concat(N_list, ignore_index=True).groupby("barcode", sort=False)["n_reads"].sum().reset_index()
            if N_list
            else pd.DataFrame(columns=["barcode", "n_reads"])
        )

        if C_all.empty:
            raise RuntimeError("Pass 1 produced no mass (C_all empty). Check input tables and schema.")

        C_all.to_csv(C_path, sep="\t", index=False, compression="gzip")
        N_all.to_csv(N_path, sep="\t", index=False, compression="gzip")
        typer.echo(f"[1/4] wrote: {C_path.name}, {N_path.name}")

    all_genomes = C_all["genome"].astype(str).unique().tolist()

    topk: Dict[str, List[str]] = {}
    for bc, sub in C_all.groupby("barcode", sort=False):
        s = sub.sort_values("C", ascending=False)["genome"].astype(str).head(int(cfg.topk_genomes)).tolist()
        topk[str(bc)] = s

    # -----------------------
    # Pass 2
    # -----------------------
    if resume and eta_out.exists():
        typer.echo("[2/4] Pass 2: resume (found eta)")
        eta = pd.read_csv(eta_out, sep="\t", compression="gzip").set_index("Unnamed: 0")["eta"]
        eta.index = eta.index.astype(str)
        eta = eta.astype(float)
        eta = eta / (eta.sum() + 1e-12)
    else:
        typer.echo("[2/4] Pass 2: ambient learning (eta) from shard spill")
        eta, eta_meta = _learn_eta(C_all, N_all, shard_root, all_genomes, cfg)
        eta.to_frame("eta").to_csv(eta_out, sep="\t", compression="gzip")
        typer.echo(f"[2/4] eta saved: {eta_out.name}  seed_sizes={eta_meta.get('seed_sizes')}")

    # -----------------------
    # Pass 3
    # -----------------------
    if resume and calls_path.exists():
        typer.echo("[3/4] Pass 3: resume (found calls)")
    else:
        typer.echo("[3/4] Pass 3: model selection / calls from shards")
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
                        "bic_empty",
                        "bic_best",
                        "bic_single",
                        "bic_doublet",
                        "bic_gap_sd",
                        "near_tie_sd",
                        "doublet_minor_frac",
                    ]
                )
                + "\n"
            )

            called_total = 0
            for shard in sorted(shard_root.rglob("shard_*.tsv.gz")):
                called_total += _call_from_shard_file(shard, eta, topk, cfg, oh)

        typer.echo(f"[3/4] calls written: {calls_path.name} (emitted rows={called_total})")

    # -----------------------
    # Pass 4
    # -----------------------
    typer.echo("[4/4] Done.")
    typer.echo(f"  - Mass table:   {C_path}")
    typer.echo(f"  - Reads table:  {N_path}")
    typer.echo(f"  - Eta:          {eta_out}")
    typer.echo(f"  - Calls:        {calls_path}")
    typer.echo(f"  - Shards root:  {shard_root}")


# --------------------------------------------------------------------------------------
# Typer wrapper (CLI)
# --------------------------------------------------------------------------------------


@app.command("genotyping")
def genotyping_cmd(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
    assign: Optional[str] = typer.Option(None, "--assign", help="Glob to assign outputs. If omitted, inferred."),
    cell_chunks_glob: Optional[str] = typer.Option(
        None, "--cell-chunks-glob", help="Glob to cell chunk barcode lists. If omitted, inferred."
    ),
    chunk_id_regex: str = typer.Option(r"chunk_(\d+)", "--chunk-id-regex", help="Regex to extract chunk id from chunk list filename."),
    outdir: Optional[Path] = typer.Option(None, "--outdir", help="Override output dir (default: <workdir>/<sample>/final)."),
    sample: Optional[str] = typer.Option(None, "--sample", help="Override sample name from config."),
    # core
    min_reads: Optional[int] = typer.Option(None, "--min-reads"),
    single_mass_min: Optional[float] = typer.Option(None, "--single-mass-min"),
    ratio_top1_top2_min: Optional[float] = typer.Option(None, "--ratio-top1-top2-min"),
    threads: Optional[int] = typer.Option(None, "--threads"),
    pass1_workers: Optional[int] = typer.Option(None, "--pass1-workers"),
    shards: Optional[int] = typer.Option(None, "--shards"),
    chunk_rows: Optional[int] = typer.Option(None, "--chunk-rows"),
    pass2_chunksize: Optional[int] = typer.Option(None, "--pass2-chunksize"),
    winner_only: Optional[bool] = typer.Option(None, "--winner-only/--no-winner-only"),
    # optional read-filter
    max_hits: Optional[int] = typer.Option(None, "--max-hits"),
    hits_delta_mapq: Optional[float] = typer.Option(None, "--hits-delta-mapq"),
    max_rows_per_read_guard: Optional[int] = typer.Option(None, "--max-rows-per-read-guard"),
    # fusion
    beta: Optional[float] = typer.Option(None, "--beta"),
    w_as: Optional[float] = typer.Option(None, "--w-as"),
    w_mapq: Optional[float] = typer.Option(None, "--w-mapq"),
    w_nm: Optional[float] = typer.Option(None, "--w-nm"),
    ambient_const: Optional[float] = typer.Option(None, "--ambient-const"),
    # empty gates
    empty_bic_margin: Optional[float] = typer.Option(None, "--empty-bic-margin"),
    empty_top1_max: Optional[float] = typer.Option(None, "--empty-top1-max"),
    empty_ratio12_max: Optional[float] = typer.Option(None, "--empty-ratio12-max"),
    empty_reads_max: Optional[int] = typer.Option(None, "--empty-reads-max"),
    empty_seed_bic_min: Optional[float] = typer.Option(None, "--empty-seed-bic-min"),
    empty_tau_quantile: Optional[float] = typer.Option(None, "--empty-tau-quantile"),
    empty_jsd_max: Optional[float] = typer.Option(None, "--empty-jsd-max"),
    jsd_normalize: Optional[bool] = typer.Option(None, "--jsd-normalize/--no-jsd-normalize"),
    # doublet gates
    bic_margin: Optional[float] = typer.Option(None, "--bic-margin"),
    doublet_minor_min: Optional[float] = typer.Option(None, "--doublet-minor-min"),
    near_tie_margin: Optional[float] = typer.Option(None, "--near-tie-margin"),
    # ambient iteration
    eta_iters: Optional[int] = typer.Option(None, "--eta-iters"),
    eta_seed_quantile: Optional[float] = typer.Option(None, "--eta-seed-quantile"),
    topk_genomes: Optional[int] = typer.Option(None, "--topk-genomes"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
) -> None:
    _run_genotyping(
        config=config,
        assign=assign,
        cell_chunks_glob=cell_chunks_glob,
        chunk_id_regex=chunk_id_regex,
        outdir=outdir,
        sample=sample,
        min_reads=min_reads,
        single_mass_min=single_mass_min,
        ratio_top1_top2_min=ratio_top1_top2_min,
        threads=threads,
        pass1_workers=pass1_workers,
        shards=shards,
        chunk_rows=chunk_rows,
        pass2_chunksize=pass2_chunksize,
        winner_only=winner_only,
        max_hits=max_hits,
        hits_delta_mapq=hits_delta_mapq,
        max_rows_per_read_guard=max_rows_per_read_guard,
        beta=beta,
        w_as=w_as,
        w_mapq=w_mapq,
        w_nm=w_nm,
        ambient_const=ambient_const,
        empty_bic_margin=empty_bic_margin,
        empty_top1_max=empty_top1_max,
        empty_ratio12_max=empty_ratio12_max,
        empty_reads_max=empty_reads_max,
        empty_seed_bic_min=empty_seed_bic_min,
        empty_tau_quantile=empty_tau_quantile,
        empty_jsd_max=empty_jsd_max,
        jsd_normalize=jsd_normalize,
        bic_margin=bic_margin,
        doublet_minor_min=doublet_minor_min,
        near_tie_margin=near_tie_margin,
        eta_iters=eta_iters,
        eta_seed_quantile=eta_seed_quantile,
        topk_genomes=topk_genomes,
        resume=resume,
    )

if __name__ == "__main__":
    app()

  app()
