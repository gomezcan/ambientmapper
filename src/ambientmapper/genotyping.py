#!/usr/bin/env python3
"""
src/ambientmapper/genotyping.py

ambientmapper genotyping — posterior-aware merge → per-cell genotype calls (+ empty-aware)

Key behaviors (DO NOT change gate design):
- Decision 1 (Empty/Noise): compare BIC_empty vs min(BIC_single, BIC_doublet) using empty_bic_margin,
  and optionally constrain empties by eta-likeness (JSD) + simple structural gates.
- Decision 2 (Singlet vs Doublet): compare BIC_single vs BIC_doublet using bic_margin (ΔBIC gate).
- Purity/strength sublabels: use near_tie_margin (optional) + purity thresholds to label dirty_singlet/weak_doublet.

Shard policy (NEW, requested):
- If you provide --cell-chunks-glob pointing to the *cell_map_ref_chunk_XXXX.txt files, we precompute
  barcode->chunk_id and route reads to shards based on chunk_id (deterministic; no barcode appears in >1 chunk).
- If no cell chunk lists are available, we fall back to hash(barcode) % shards.

Calling is made bulletproof:
- For each shard spill file, we accumulate rows per barcode across all read_csv chunks and call each barcode once.
  This is safe because each spill file corresponds to a small set of barcodes under the new shard policy.

CLI compatibility (FIXED):
- The genotyping() command accepts the same kwargs that ambientmapper/cli.py forwards:
  config, assign (optional), outdir (optional), sample (optional), plus all tuning knobs.
- Unknown keyword errors (e.g., empty_bic_margin, assign) are eliminated.

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

    # Calling thresholds
    min_reads: int = 5
    single_mass_min: float = 0.6
    doublet_minor_min: float = 0.20

    # BIC gates (match figure)
    empty_bic_margin: float = 10.0      # Decision 1: empty vs non-empty
    bic_margin: float = 6.0             # Decision 2: singlet vs doublet (ΔBIC >= bic_margin)

    # Optional “near tie” band for dirty/weak labels (sub-gates)
    near_tie_margin: float = 2.0

    # Ratio/purity gate for singlets
    ratio_top1_top2_min: float = 2.0

    # Ambient learning
    eta_iters: int = 2
    eta_seed_quantile: float = 0.02       # initial seed fraction by n_effective
    empty_tau_quantile: float = 0.95      # refine seed by JSD threshold (quantile of "most eta-like")

    # Empty structural gates (used for eta-seed refinement + optional empty call constraints)
    empty_top1_max: float = 0.6
    empty_ratio12_max: float = 1.5
    empty_reads_max: Optional[int] = None

    # Eta seed quality (NEW, used ONLY for selecting eta training seed)
    empty_seed_bic_min: float = 10.0

    # Optional absolute JSD threshold for empty calling (if provided)
    empty_jsd_max: Optional[float] = None
    jsd_normalize: bool = True

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
# Config JSON helpers (minimal; matches your pipeline conventions)
# --------------------------------------------------------------------------------------

def _read_cfg_json(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "rt") as f:
        return json.load(f)

def _infer_sample(cfg_json: Dict[str, Any], sample_override: Optional[str]) -> str:
    if sample_override:
        return str(sample_override)
    # common patterns
    if "sample" in cfg_json and cfg_json["sample"]:
        return str(cfg_json["sample"])
    # fallback: try top-level name-like key
    return "sample"

def _infer_paths(cfg_json: Dict[str, Any], sample: str) -> Dict[str, Path]:
    """
    Matches the structure shown in your traceback: d['root'], d['qc'], d['filtered'], d['chunks'], d['final'].
    """
    workdir = cfg_json.get("workdir", None)
    if not workdir:
        raise ValueError("config JSON missing required key: 'workdir'")
    root = Path(workdir) / sample

    # standard pipeline subdirs
    qc = root / "qc"
    filtered = root / "filtered"
    chunks = root / "chunks"
    final = root / "final"

    # some runs place cell_map_ref_chunks under chunks/
    return {"root": root, "qc": qc, "filtered": filtered, "chunks": chunks, "final": final}

def _infer_assign_glob(d: Dict[str, Path]) -> str:
    # expected assign filtered outputs live inside cell_map_ref_chunks under chunks/
    candidates = [
        str(d["chunks"] / "cell_map_ref_chunks" / "*_filtered.tsv.gz"),
        str(d["chunks"] / "cell_map_ref_chunks" / "*_filtered.tsv"),
        str(d["chunks"] / "cell_map_ref_chunks" / "*filtered.tsv.gz"),
        str(d["chunks"] / "cell_map_ref_chunks" / "*filtered.tsv"),
    ]
    for pat in candidates:
        if glob.glob(pat):
            return pat
    # last resort: search deeper
    deep = str(d["root"] / "**" / "cell_map_ref_chunks" / "*_filtered.tsv.gz")
    return deep

def _infer_cell_chunks_glob(d: Dict[str, Path]) -> str:
    candidates = [
        str(d["chunks"] / "cell_map_ref_chunks" / "*cell_map_ref_chunk_*.txt"),
        str(d["chunks"] / "cell_map_ref_chunks" / "*chunk_*.txt"),
    ]
    for pat in candidates:
        if glob.glob(pat):
            return pat
    deep = str(d["root"] / "**" / "cell_map_ref_chunks" / "*chunk_*.txt")
    return deep


# --------------------------------------------------------------------------------------
# Helpers: schema and coercion
# --------------------------------------------------------------------------------------

def _coerce_assign_schema(df: pd.DataFrame) -> pd.DataFrame:
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
    is_amb = (assigned == "ambiguous")
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
    df = df[["barcode", "read_id", "genome", "AS", "MAPQ", "NM"]].copy()
    df = df.dropna(subset=["barcode", "read_id", "genome"])

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
# Model selection (implements your original gate design)
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

    # Always compute empty BIC
    bic_empty = _bic(_loglik_empty(L_block, eta), 0, read_count)
    out["bic_empty"] = float(bic_empty)

    # if too few reads, stop early
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

    # Fit best singlet/doublet on candidate genomes
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
                        bic = _bic(_loglik_for_params(L_block, eta, "doublet", g1, g2, float(a), float(r)), 2, read_count)
                        if best_doublet is None or bic < best_doublet["bic"]:
                            best_doublet = {"model": "doublet", "g1": g1, "g2": g2, "alpha": float(a), "rho": float(r), "bic": float(bic)}

    bic_s = float(best_single["bic"]) if best_single else float("inf")
    bic_d = float(best_doublet["bic"]) if best_doublet else float("inf")
    out["bic_single"] = bic_s if np.isfinite(bic_s) else None
    out["bic_doublet"] = bic_d if np.isfinite(bic_d) else None

    # ------------------------------------------------------------------
    # Decision 1: Empty/Noise (BIC margin is empty_bic_margin, per design)
    # Optional: also constrain by structural gates and/or absolute JSD threshold.
    # ------------------------------------------------------------------
    bic_best_non_empty = min(bic_s, bic_d)
    empty_by_bic = (bic_empty + float(cfg.empty_bic_margin)) < bic_best_non_empty

    # structural candidate (your Part 3 structural gate idea)
    structural_empty = (p_top1 <= float(cfg.empty_top1_max)) and (ratio12 <= float(cfg.empty_ratio12_max))
    if cfg.empty_reads_max is not None:
        structural_empty = structural_empty and (read_count <= int(cfg.empty_reads_max))

    jsd_ok = True
    if cfg.empty_jsd_max is not None:
        jsd_ok = (jsd <= float(cfg.empty_jsd_max))

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

    # ------------------------------------------------------------------
    # Decision 2: Singlet vs Doublet using bic_margin (ΔBIC gate, per design)
    # ------------------------------------------------------------------
    # Define ΔBIC in the direction “single better than doublet” etc
    # If both exist:
    #  - single if bic_s + bic_margin < bic_d
    #  - doublet if bic_d + bic_margin < bic_s
    #  - else ambiguous between single/doublet
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
        # Only one model exists
        chosen = best_single if np.isfinite(bic_s) else best_doublet
        call_core = "single" if (chosen and chosen["model"] == "single") else ("doublet" if chosen else "ambiguous")
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

    # ------------------------------------------------------------------
    # Purity/strength sublabels (do not change core design; only label)
    # ------------------------------------------------------------------
    if call_core == "single":
        passes_mass = (p_top1 >= float(cfg.single_mass_min))
        passes_ratio = (ratio12 >= float(cfg.ratio_top1_top2_min))
        if passes_mass and passes_ratio and (not out["near_tie_sd"]):
            out["call"] = "single_clean"
        else:
            out["call"] = "dirty_singlet"
    elif call_core == "doublet":
        passes_minor = (minor_frac >= float(cfg.doublet_minor_min))
        if passes_minor and (not out["near_tie_sd"]):
            out["call"] = "doublet"
        else:
            out["call"] = "weak_doublet"
    else:
        out["call"] = "ambiguous"

    return out


# --------------------------------------------------------------------------------------
# NEW: Cell chunk parsing for shard routing
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
    """
    Reads all chunk list files and returns barcode->chunk_id.

    Assumes each barcode appears in exactly one chunk list (your guarantee).
    """
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
                # keep first occurrence; duplicates indicate upstream violation
                if bc not in out:
                    out[bc] = cid
    return out


def _route_shards_for_barcodes(
    barcodes: pd.Series,
    cfg: MergeConfig,
    bc_to_chunk: Optional[Dict[str, int]],
) -> np.ndarray:
    """
    Returns shard id per barcode.
    Priority: chunk-based routing if bc_to_chunk is available; else hash-based.
    """
    bcs = barcodes.astype(str).to_numpy()

    if bc_to_chunk:
        # vectorized lookup via pandas map
        ser = pd.Series(bcs, dtype="string").map(bc_to_chunk)
        # fallback for missing barcodes: hash routing
        miss = ser.isna().to_numpy()
        sid = np.empty(len(bcs), dtype=np.int32)

        # chunk-based: shard = chunk_id % shards
        cid = ser.fillna(-1).to_numpy(dtype=np.int64)
        ok = (cid >= 0)
        sid[ok] = (cid[ok] % int(cfg.shards)).astype(np.int32)

        if miss.any():
            h = pd.util.hash_pandas_object(pd.Series(bcs[miss], dtype="string"), index=False).to_numpy(np.uint64)
            sid[miss] = (h % np.uint64(cfg.shards)).astype(np.int32)

        return sid

    # default: hash-based
    h = pd.util.hash_pandas_object(pd.Series(bcs, dtype="string"), index=False).to_numpy(np.uint64)
    return (h % np.uint64(cfg.shards)).astype(np.int32)


# --------------------------------------------------------------------------------------
# Pass 1: stream assign files -> (C_all, N_all) + shard spill
# --------------------------------------------------------------------------------------

@dataclass
class Pass1Outputs:
    C: pd.DataFrame
    N: pd.DataFrame
    shard_dir: Path


def _pass1_process_one_file(
    fp: Path,
    cfg: MergeConfig,
    shard_dir: Path,
    bc_to_chunk: Optional[Dict[str, int]],
) -> Pass1Outputs:
    shard_dir.mkdir(parents=True, exist_ok=True)

    handles = [gzip.open(shard_dir / f"shard_{i:02d}.tsv.gz", "at") for i in range(cfg.shards)]
    header_written = np.zeros(cfg.shards, dtype=bool)

    C_parts: List[pd.DataFrame] = []
    N_parts: List[pd.DataFrame] = []

    for raw in pd.read_csv(fp, sep="\t", chunksize=int(cfg.chunk_rows), low_memory=False):
        df = _coerce_assign_schema(raw)
        df = _reduce_alignments_to_per_genome(df)
        df = df.dropna(subset=["barcode", "read_id", "genome"])

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

        sid = _route_shards_for_barcodes(Ldf["barcode"], cfg, bc_to_chunk)
        Ldf = Ldf.assign(_sid=sid)

        for i in range(cfg.shards):
            sub = Ldf[Ldf["_sid"] == i]
            if sub.empty:
                continue
            if not header_written[i]:
                handles[i].write("barcode\tread_id\tgenome\tL\tL_amb\n")
                header_written[i] = True
            sub.drop(columns="_sid").to_csv(handles[i], sep="\t", header=False, index=False)

    for h in handles:
        h.close()

    C_out = pd.concat(C_parts, ignore_index=True) if C_parts else pd.DataFrame(columns=["barcode", "genome", "C"])
    N_out = pd.concat(N_parts, ignore_index=True) if N_parts else pd.DataFrame(columns=["barcode", "n_reads"])

    return Pass1Outputs(C=C_out, N=N_out, shard_dir=shard_dir)


# --------------------------------------------------------------------------------------
# Ambient learning (eta)
# --------------------------------------------------------------------------------------

def _compute_eta_from_shards(
    shard_root: Path,
    target_bcs: Sequence[str],
    all_genomes: Sequence[str],
    cfg: MergeConfig,
) -> pd.Series:
    target_set = set(map(str, target_bcs))
    mix = {g: 0.0 for g in all_genomes}

    if not target_set:
        eta = pd.Series({g: 1.0 for g in all_genomes}, dtype=float)
        return eta / eta.sum()

    for shard in shard_root.rglob("shard_*.tsv.gz"):
        for chunk in pd.read_csv(
            shard,
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

    q = float(cfg.eta_seed_quantile)
    q = min(max(q, 1e-6), 1.0)
    n = max(1, int(math.ceil(q * len(df))))
    return df.sort_values("n_effective", ascending=True)["barcode"].head(n).astype(str).tolist()

def _refine_eta_seed_by_jsd_and_bic(
    C_all: pd.DataFrame,
    N_all: pd.DataFrame,
    eta: pd.Series,
    cfg: MergeConfig,
    # Optional: pass precomputed bic evidence for empty-likeness; if not available, we use structural gates only.
    empty_like_bcs: Optional[set] = None,
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
    if empty_like_bcs is not None:
        cand = cand[cand.index.astype(str).isin(empty_like_bcs)]

    if cand.empty:
        cand = stats

    jsd_vals: Dict[str, float] = {}
    for bc, sub in mass.groupby("barcode", sort=False):
        bc_s = str(bc)
        if bc_s not in cand.index.astype(str):
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

    # NOTE: we do not require BIC evidence here because BIC needs read-level blocks.
    # If you want seed-by-BIC, you can add a pre-pass that computes empty-likeness per barcode and pass it in.
    for _ in range(int(cfg.eta_iters)):
        seed = _refine_eta_seed_by_jsd_and_bic(C_all, N_all, eta, cfg, empty_like_bcs=None)
        meta["seed_sizes"].append(len(seed))
        if not seed:
            break
        eta = _compute_eta_from_shards(shard_root, seed, list(all_genomes), cfg)

    return eta, meta


# --------------------------------------------------------------------------------------
# Calling: bulletproof per-shard file (call each barcode once)
# --------------------------------------------------------------------------------------

def _call_from_shard_file(
    shard_path: Path,
    eta: pd.Series,
    topk: Dict[str, List[str]],
    cfg: MergeConfig,
    out_handle,
) -> int:
    """
    Bulletproof:
    - Accumulate all rows per barcode across all read_csv chunks for this shard file.
    - Call each barcode exactly once.
    """
    buckets: Dict[str, List[pd.DataFrame]] = {}

    for chunk in pd.read_csv(
        shard_path,
        sep="\t",
        compression="gzip",
        chunksize=int(cfg.pass2_chunksize),
        dtype={"barcode": "string", "read_id": "string", "genome": "string", "L": "float32", "L_amb": "float32"},
    ):
        if chunk.empty:
            continue
        for bc, sub in chunk.groupby("barcode", sort=False):
            bc_s = str(bc)
            buckets.setdefault(bc_s, []).append(sub[["read_id", "genome", "L"]].copy())

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
# Main command (signature matches cli.py forwarded kwargs)
# --------------------------------------------------------------------------------------

@app.command()
def genotyping(
    # REQUIRED by pipeline wrapper
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
    # Optional overrides / inferred defaults
    assign: Optional[str] = typer.Option(None, "--assign", help="Glob to assign outputs. If omitted, inferred from config/workdir."),
    cell_chunks_glob: Optional[str] = typer.Option(
        None,
        "--cell-chunks-glob",
        help="Glob to cell_map_ref_chunk_*.txt files listing barcodes per chunk. If omitted, inferred from config/workdir. "
             "Used to route shards deterministically by chunk id.",
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
    cfg_json = _read_cfg_json(config)
    sample_eff = _infer_sample(cfg_json, sample)
    d = _infer_paths(cfg_json, sample_eff)

    outdir_eff = Path(outdir) if outdir is not None else d["final"]
    outdir_eff.mkdir(parents=True, exist_ok=True)

    assign_glob = assign if assign is not None else _infer_assign_glob(d)
    cell_chunks_pat = cell_chunks_glob if cell_chunks_glob is not None else _infer_cell_chunks_glob(d)

    # resolve effective worker count
    threads_eff = int(threads) if threads is not None else 1
    pass1_workers_eff = int(pass1_workers) if pass1_workers is not None else int(threads_eff)

    # disable filter unless BOTH are provided and max_hits>0
    max_hits_eff = None if (max_hits is None or int(max_hits) <= 0) else int(max_hits)
    hits_delta_mapq_eff = None if (max_hits_eff is None or hits_delta_mapq is None) else float(hits_delta_mapq)

    # Build cfg from defaults then override with CLI where provided
    cfg = MergeConfig(
        sample=str(sample_eff),
        pass1_workers=int(pass1_workers_eff),
    )

    # apply overrides
    if shards is not None:
        cfg.shards = int(shards)
    if chunk_rows is not None:
        cfg.chunk_rows = int(chunk_rows)
    if pass2_chunksize is not None:
        cfg.pass2_chunksize = int(pass2_chunksize)
    if winner_only is not None:
        cfg.winner_only = bool(winner_only)

    if max_rows_per_read_guard is not None:
        cfg.max_rows_per_read_guard = int(max_rows_per_read_guard)
    cfg.max_hits = max_hits_eff
    cfg.hits_delta_mapq = hits_delta_mapq_eff

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

    # resolve input files
    files = [Path(f) for f in glob.glob(assign_glob, recursive=True)]
    if not files:
        raise typer.BadParameter(f"No files matched: {assign_glob}")

    tmp_dir = outdir_eff / f"tmp_{sample_eff}"
    shard_root = tmp_dir / "L_shards_workers"
    shard_root.mkdir(parents=True, exist_ok=True)

    # load barcode->chunk_id mapping for new shard policy (best-effort)
    bc_to_chunk: Optional[Dict[str, int]] = None
    chunk_files = glob.glob(cell_chunks_pat)
    if chunk_files:
        bc_to_chunk = _load_barcode_to_chunk_id(cell_chunks_pat, chunk_id_regex)
        typer.echo(f"[genotyping] shard policy: chunk-based ({len(bc_to_chunk):,} barcodes from {len(chunk_files)} chunk files)")
    else:
        typer.echo("[genotyping] shard policy: hash(barcode)%shards (no chunk list files found)")

    typer.echo(f"[genotyping] sample={sample_eff} files={len(files)} pass1_workers={pass1_workers_eff} shards={cfg.shards}")
    typer.echo(f"[genotyping] assign_glob={assign_glob}")
    typer.echo(f"[genotyping] filter: max_hits={cfg.max_hits} hits_delta_mapq={cfg.hits_delta_mapq} guard={cfg.max_rows_per_read_guard}")
    typer.echo(f"[genotyping] winner_only={cfg.winner_only} ambient_const={cfg.ambient_const}")
    typer.echo(f"[genotyping] gates: empty_bic_margin={cfg.empty_bic_margin} bic_margin={cfg.bic_margin} near_tie_margin={cfg.near_tie_margin}")

    # -----------------------
    # Pass 1
    # -----------------------
    typer.echo("[1/4] Pass 1: streaming assign inputs → per-read posteriors + spill shards")
    C_list: List[pd.DataFrame] = []
    N_list: List[pd.DataFrame] = []

    def _job(i: int, f: Path) -> Pass1Outputs:
        wdir = shard_root / f"w{i:03d}"
        return _pass1_process_one_file(f, cfg, wdir, bc_to_chunk)

    with ProcessPoolExecutor(max_workers=int(pass1_workers_eff)) as ex:
        futs = [ex.submit(_job, i, f) for i, f in enumerate(files)]
        for fut in as_completed(futs):
            out = fut.result()
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

    # top-k genomes per barcode
    topk: Dict[str, List[str]] = {}
    for bc, sub in C_all.groupby("barcode", sort=False):
        s = sub.sort_values("C", ascending=False)["genome"].astype(str).head(int(cfg.topk_genomes)).tolist()
        topk[str(bc)] = s

    C_all.to_csv(outdir_eff / f"{sample_eff}_C_all.tsv.gz", sep="\t", index=False, compression="gzip")
    N_all.to_csv(outdir_eff / f"{sample_eff}_N_all.tsv.gz", sep="\t", index=False, compression="gzip")
    typer.echo(f"[1/4] wrote: {sample_eff}_C_all.tsv.gz, {sample_eff}_N_all.tsv.gz")

    # -----------------------
    # Pass 2
    # -----------------------
    typer.echo("[2/4] Pass 2: ambient learning (eta) from shard spill")
    eta, eta_meta = _learn_eta(C_all, N_all, shard_root, all_genomes, cfg)
    eta_out = outdir_eff / f"{sample_eff}_eta.tsv.gz"
    eta.to_frame("eta").to_csv(eta_out, sep="\t", compression="gzip")
    typer.echo(f"[2/4] eta saved: {eta_out.name}  seed_sizes={eta_meta.get('seed_sizes')}")

    # -----------------------
    # Pass 3
    # -----------------------
    typer.echo("[3/4] Pass 3: model selection / calls from shards")
    calls_path = outdir_eff / f"{sample_eff}_genotype_calls.tsv.gz"
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
    typer.echo(f"  - Mass table:   {outdir_eff / f'{sample_eff}_C_all.tsv.gz'}")
    typer.echo(f"  - Reads table:  {outdir_eff / f'{sample_eff}_N_all.tsv.gz'}")
    typer.echo(f"  - Eta:          {eta_out}")
    typer.echo(f"  - Calls:        {calls_path}")
    typer.echo(f"  - Shards root:  {shard_root}")


if __name__ == "__main__":
    app()
