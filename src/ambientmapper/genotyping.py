#!/usr/bin/env python3
"""
src/ambientmapper/genotyping.py

ambientmapper genotyping — posterior-aware merge → per-cell genotype calls (+ empty-aware)

This version:
- Matches ambientmapper/cli.py genotyping parameter set (no "unexpected keyword argument" errors).
- Uses empty parameters in the model (empty_bic_margin, empty_seed_bic_min, empty_jsd_max, jsd_normalize).
- Implements chunk-defined spill/calling by default when chunk lists are available:
    - assign outputs are per-chunk (cell_map_ref_chunks/*_filtered.tsv.gz)
    - barcode lists are per-chunk (cell_map_ref_chunks/*_cell_map_ref_chunk_*.txt)
  This makes Pass3 calling "bulletproof": each barcode is called exactly once.
- Falls back to hashed shard spill if chunk lists are not found.

Notes:
- Input expected columns: Read/BC/Genome/AS/MAPQ/NM/assigned_class (gz or plain tsv accepted).
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
# Config model
# --------------------------------------------------------------------------------------


class MergeConfig(BaseModel):
    # Fusion / scoring -> posteriors
    beta: float = 1.0
    w_as: float = 1.0
    w_mapq: float = 1.0
    w_nm: float = 1.0
    ambient_const: float = 1e-3

    # Calling thresholds / gates
    min_reads: int = 5
    single_mass_min: float = 0.6
    ratio_top1_top2_min: float = 2.0
    doublet_minor_min: float = 0.20

    # Margins
    bic_margin: float = 6.0            # single vs doublet confidence (near tie uses near_tie_margin)
    near_tie_margin: float = 2.0       # if |BIC(single)-BIC(doublet)| <= near_tie_margin => near tie
    empty_bic_margin: float = 10.0     # empty override margin: if bic_empty + empty_bic_margin < bic_best => empty

    # Ambient learning
    eta_iters: int = 2
    eta_seed_quantile: float = 0.02
    empty_tau_quantile: float = 0.95   # JSD quantile threshold

    # Empty gate (structure) to restrict eta candidates
    empty_top1_max: float = 0.6
    empty_ratio12_max: float = 1.5
    empty_reads_max: Optional[int] = None

    # Empty seed strengthening
    empty_seed_bic_min: float = 10.0   # require bic_best - bic_empty >= empty_seed_bic_min for eta-seed membership
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

    # Pass 1 parallelism
    pass1_workers: int = 1

    # Promiscuous ambiguous filter (assigned_class=="ambiguous" only)
    max_hits: Optional[int] = None
    hits_delta_mapq: Optional[float] = None
    max_rows_per_read_guard: Optional[int] = 500


# --------------------------------------------------------------------------------------
# Config loader / inference
# --------------------------------------------------------------------------------------


def _read_cfg_json(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "rt") as f:
        cfg = json.load(f)
    return cfg


def _infer_sample(cfg: Dict[str, Any], sample_override: Optional[str]) -> str:
    if sample_override:
        return str(sample_override)
    if "sample" in cfg and cfg["sample"]:
        return str(cfg["sample"])
    # fallback: config file stem
    return "sample"


def _infer_paths(cfg: Dict[str, Any], sample: str) -> Dict[str, Path]:
    """
    Best-effort inference of pipeline directories.
    Prefers cfg['d'] if present (as seen in your traceback).
    """
    d: Dict[str, Path] = {}
    if isinstance(cfg.get("d"), dict):
        for k, v in cfg["d"].items():
            try:
                d[k] = Path(v)
            except Exception:
                pass

    # fallback if d not present
    if not d:
        workdir = Path(cfg.get("workdir", "."))
        d["root"] = workdir / sample
        d["filtered"] = d["root"] / "filtered"
        d["chunks"] = d["root"] / "chunks"
        d["final"] = d["root"] / "final"

    # ensure key existence
    for k in ["root", "filtered", "chunks", "final"]:
        d.setdefault(k, d.get("root", Path(".")) / k)

    return d


def _infer_assign_glob(d: Dict[str, Path]) -> str:
    """
    Default: <filtered>/cell_map_ref_chunks/*_filtered.tsv.gz
    fallback: <chunks>/cell_map_ref_chunks/*_filtered.tsv.gz
    """
    p1 = d["filtered"] / "cell_map_ref_chunks" / "*_filtered.tsv.gz"
    if glob.glob(str(p1)):
        return str(p1)
    p2 = d["chunks"] / "cell_map_ref_chunks" / "*_filtered.tsv.gz"
    if glob.glob(str(p2)):
        return str(p2)
    # last resort: look under root
    p3 = d["root"] / "cell_map_ref_chunks" / "*_filtered.tsv.gz"
    return str(p3)


def _infer_cell_chunks_glob_from_assign(assign_glob: str) -> str:
    """
    Default: same directory as assign glob, matching *_cell_map_ref_chunk_*.txt
    """
    parent = Path(assign_glob).parent
    return str(parent / "*_cell_map_ref_chunk_*.txt")


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
# Promiscuous ambiguous filter (MAPQ-based)
# --------------------------------------------------------------------------------------


def _filter_promiscuous_ambiguous_reads(
    df: pd.DataFrame, cfg: MergeConfig
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
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
# Posterior computation
# --------------------------------------------------------------------------------------


def _compute_read_posteriors(df: pd.DataFrame, cfg: MergeConfig) -> pd.DataFrame:
    df = df[["barcode", "read_id", "genome", "AS", "MAPQ", "NM"]].copy()
    df = df.dropna(subset=["barcode", "read_id", "genome"])
    if df.empty:
        return pd.DataFrame(columns=["barcode", "read_id", "genome", "L", "L_amb"])

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
# Model selection + gates
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
        "jsd_to_eta": _barcode_jsd_from_mass(mass, eta, normalize=bool(cfg.jsd_normalize)),
    }

    out["bic_empty"] = _bic(_loglik_empty(L_block, eta), 0, read_count)

    best_single: Optional[Dict[str, Any]] = None
    best_doublet: Optional[Dict[str, Any]] = None

    if read_count >= cfg.min_reads and candidate_genomes:
        alphas = np.arange(0.0, float(cfg.max_alpha) + 1e-9, float(cfg.alpha_grid))

        for g1 in candidate_genomes:
            for a in alphas:
                bic = _bic(_loglik_for_params(L_block, eta, "single", g1, None, float(a)), 1, read_count)
                if best_single is None or bic < best_single["bic"]:
                    best_single = {"model": "single", "g1": g1, "alpha": float(a), "bic": float(bic)}

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

    # Empty override uses empty_bic_margin (not bic_margin)
    if (out["bic_empty"] + float(cfg.empty_bic_margin)) < float(best_non_empty["bic"]):
        out["call"] = "empty"
        out["doublet_minor_frac"] = 0.0
        out["near_tie_sd"] = False
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
# Chunk policy helpers
# --------------------------------------------------------------------------------------


def _infer_chunk_id(name: str, chunk_id_regex: str) -> Optional[str]:
    rx = re.compile(chunk_id_regex)
    m = rx.search(name)
    return m.group(1) if m else None


def _load_chunk_barcode_lists(cell_chunks_glob: str, chunk_id_regex: str) -> Dict[str, set]:
    out: Dict[str, set] = {}
    rx = re.compile(chunk_id_regex)
    files = sorted(glob.glob(cell_chunks_glob))
    if not files:
        return {}

    for fp in files:
        m = rx.search(Path(fp).name)
        if not m:
            continue
        cid = m.group(1)
        bcs: set = set()
        with open(fp, "rt") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                bcs.add(line.split()[0])
        out[cid] = bcs

    return out


# --------------------------------------------------------------------------------------
# Pass 1 spill variants
# --------------------------------------------------------------------------------------


@dataclass
class Pass1Outputs:
    C: pd.DataFrame
    N: pd.DataFrame
    spill_paths: List[Path]


def _pass1_chunkspill_one_file(
    fp: Path,
    cfg: MergeConfig,
    spill_path: Path,
    chunk_barcodes: Optional[set],
) -> Tuple[pd.DataFrame, pd.DataFrame, Path]:
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

            df, _ = _filter_promiscuous_ambiguous_reads(df, cfg)
            if df.empty:
                continue

            Ldf = _compute_read_posteriors(df, cfg)
            if Ldf.empty:
                continue

            C_parts.append(Ldf.groupby(["barcode", "genome"], sort=False)["L"].sum().rename("C").reset_index())
            N_parts.append(Ldf.groupby("barcode", sort=False)["read_id"].nunique().rename("n_reads").reset_index())

            if not wrote_header:
                oh.write("barcode\tread_id\tgenome\tL\tL_amb\n")
                wrote_header = True
            Ldf.to_csv(oh, sep="\t", header=False, index=False)

    C_out = pd.concat(C_parts, ignore_index=True) if C_parts else pd.DataFrame(columns=["barcode", "genome", "C"])
    N_out = pd.concat(N_parts, ignore_index=True) if N_parts else pd.DataFrame(columns=["barcode", "n_reads"])
    return C_out, N_out, spill_path


def _pass1_hashed_shards_one_file(
    fp: Path,
    cfg: MergeConfig,
    shard_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[Path]]:
    shard_dir.mkdir(parents=True, exist_ok=True)
    handles = [gzip.open(shard_dir / f"shard_{i:02d}.tsv.gz", "at") for i in range(cfg.shards)]
    header_written = np.zeros(cfg.shards, dtype=bool)

    C_parts: List[pd.DataFrame] = []
    N_parts: List[pd.DataFrame] = []

    for raw in pd.read_csv(fp, sep="\t", chunksize=int(cfg.chunk_rows), low_memory=False):
        df = _coerce_assign_schema(raw)
        df = _reduce_alignments_to_per_genome(df)
        df = df.dropna(subset=["barcode", "read_id", "genome"])
        if df.empty:
            continue

        df, _ = _filter_promiscuous_ambiguous_reads(df, cfg)
        if df.empty:
            continue

        Ldf = _compute_read_posteriors(df, cfg)
        if Ldf.empty:
            continue

        C_parts.append(Ldf.groupby(["barcode", "genome"], sort=False)["L"].sum().rename("C").reset_index())
        N_parts.append(Ldf.groupby("barcode", sort=False)["read_id"].nunique().rename("n_reads").reset_index())

        bcs = Ldf["barcode"].astype(str)
        h = pd.util.hash_pandas_object(bcs, index=False).to_numpy(np.uint64)
        sid = (h % np.uint64(cfg.shards)).astype(np.int32)
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
    shard_paths = [shard_dir / f"shard_{i:02d}.tsv.gz" for i in range(cfg.shards)]
    return C_out, N_out, shard_paths


# --------------------------------------------------------------------------------------
# Spill iterators
# --------------------------------------------------------------------------------------


def _iter_spills(spill_root: Path) -> Iterable[Path]:
    yield from sorted(spill_root.glob("L_chunk_*.tsv.gz"))


def _iter_shards(shard_root: Path) -> Iterable[Path]:
    yield from sorted(shard_root.rglob("shard_*.tsv.gz"))


# --------------------------------------------------------------------------------------
# Ambient learning (eta)
# --------------------------------------------------------------------------------------


def _compute_eta_from_files(
    files_iter: Iterable[Path],
    target_bcs: Sequence[str],
    all_genomes: Sequence[str],
    cfg: MergeConfig,
) -> pd.Series:
    target_set = set(map(str, target_bcs))
    mix = {g: 0.0 for g in all_genomes}

    if not target_set:
        eta = pd.Series({g: 1.0 for g in all_genomes}, dtype=float)
        return eta / eta.sum()

    for fp in files_iter:
        for chunk in pd.read_csv(
            fp,
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

    cand_idx = set(cand.index.astype(str).tolist())
    jsd_vals: Dict[str, float] = {}
    for bc, sub in mass.groupby("barcode", sort=False):
        bc_s = str(bc)
        if bc_s not in cand_idx:
            continue
        s = sub.set_index("genome")["C"]
        jsd = _barcode_jsd_from_mass(s, eta, normalize=bool(cfg.jsd_normalize))
        if cfg.empty_jsd_max is not None and jsd > float(cfg.empty_jsd_max):
            continue
        jsd_vals[bc_s] = jsd

    if not jsd_vals:
        return []

    jsd_ser = pd.Series(jsd_vals, dtype=float)
    tau = jsd_ser.quantile(float(cfg.empty_tau_quantile))
    return jsd_ser[jsd_ser <= tau].index.astype(str).tolist()


def _filter_seed_by_bic_competitiveness(
    files_iter: Iterable[Path],
    seed: Sequence[str],
    eta: pd.Series,
    topk: Dict[str, List[str]],
    cfg: MergeConfig,
) -> List[str]:
    """
    Enforce --empty-seed-bic-min:
      keep bc if bic_best - bic_empty >= empty_seed_bic_min
    This is evaluated on full per-read L blocks (streamed from spill files).
    """
    target_set = set(map(str, seed))
    if not target_set:
        return []

    acc: Dict[str, List[pd.DataFrame]] = {}
    for fp in files_iter:
        for chunk in pd.read_csv(
            fp,
            sep="\t",
            compression="gzip",
            chunksize=int(cfg.pass2_chunksize),
            dtype={"barcode": "string", "read_id": "string", "genome": "string", "L": "float32", "L_amb": "float32"},
        ):
            sub = chunk[chunk["barcode"].isin(target_set)]
            if sub.empty:
                continue
            for bc, g in sub.groupby("barcode", sort=False):
                bc_s = str(bc)
                acc.setdefault(bc_s, []).append(g[["read_id", "genome", "L"]].copy())

    kept: List[str] = []
    thresh = float(cfg.empty_seed_bic_min)
    for bc_s, parts in acc.items():
        L_block = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
        cand = topk.get(bc_s, [])
        res = _select_model_for_barcode(L_block, eta, cfg, cand)
        bic_empty = float(res.get("bic_empty", float("inf")))
        bic_best = float(res.get("bic_best", float("inf")))
        if np.isfinite(bic_empty) and np.isfinite(bic_best):
            if (bic_best - bic_empty) >= thresh:
                kept.append(bc_s)
    return kept


def _learn_eta(
    C_all: pd.DataFrame,
    N_all: pd.DataFrame,
    files_iter_factory,
    all_genomes: Sequence[str],
    topk: Dict[str, List[str]],
    cfg: MergeConfig,
) -> Tuple[pd.Series, Dict[str, Any]]:
    meta: Dict[str, Any] = {"iters": int(cfg.eta_iters), "seed_sizes": []}

    seed = _pick_initial_eta_seed(C_all, N_all, cfg)
    meta["seed_sizes"].append(len(seed))

    eta = _compute_eta_from_files(files_iter_factory(), seed, list(all_genomes), cfg)

    for _ in range(int(cfg.eta_iters)):
        seed = _refine_eta_seed_by_jsd(C_all, N_all, eta, cfg)
        # enforce empty_seed_bic_min with current eta
        if seed and cfg.empty_seed_bic_min is not None:
            seed = _filter_seed_by_bic_competitiveness(files_iter_factory(), seed, eta, topk, cfg)
        meta["seed_sizes"].append(len(seed))
        if not seed:
            break
        eta = _compute_eta_from_files(files_iter_factory(), seed, list(all_genomes), cfg)

    return eta, meta


# --------------------------------------------------------------------------------------
# Calling
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
                f"{float(res.get('doublet_minor_frac', 0.0)):.6g}" if res.get("doublet_minor_frac") is not None else "",
                str(bool(res.get("near_tie_sd", False))),
                f"{float(res.get('bic_empty', 0.0)):.6g}" if res.get("bic_empty") is not None else "",
                f"{float(res.get('bic_best', 0.0)):.6g}" if res.get("bic_best") is not None else "",
            ]
        )
        + "\n"
    )


def _call_from_file_streaming(
    fp: Path,
    eta: pd.Series,
    topk: Dict[str, List[str]],
    cfg: MergeConfig,
    out_handle,
) -> int:
    called = 0
    acc: Dict[str, List[pd.DataFrame]] = {}

    for chunk in pd.read_csv(
        fp,
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
# Main command signature: matches cli.py forwarding
# --------------------------------------------------------------------------------------


@app.command()
def genotyping(
    # from cli
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
    assign: Optional[str] = typer.Option(None, "--assign", help="Glob to assign outputs. If omitted, inferred."),
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
    hits_delta_mapq: Optional[int] = typer.Option(None, "--hits-delta-mapq"),
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
    # ambient iteration
    eta_iters: Optional[int] = typer.Option(None, "--eta-iters"),
    eta_seed_quantile: Optional[float] = typer.Option(None, "--eta-seed-quantile"),
    topk_genomes: Optional[int] = typer.Option(None, "--topk-genomes"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),

    # NEW: chunk policy knobs (optional; defaults inferred)
    cell_chunks_glob: Optional[str] = typer.Option(
        None,
        "--cell-chunks-glob",
        help="Glob to per-chunk barcode lists. If omitted, inferred from assign directory.",
    ),
    chunk_id_regex: str = typer.Option(r"chunk_(\d+)", "--chunk-id-regex"),
):
    cfg_json = _read_cfg_json(config)
    sample_eff = _infer_sample(cfg_json, sample)
    d = _infer_paths(cfg_json, sample_eff)

    outdir_eff = Path(outdir) if outdir is not None else Path(d["final"])
    outdir_eff.mkdir(parents=True, exist_ok=True)

    assign_eff = assign if assign is not None else _infer_assign_glob(d)
    files = [Path(f) for f in glob.glob(assign_eff, recursive=True)]
    if not files:
        raise typer.BadParameter(f"No files matched: {assign_eff}")

    # Infer cell_chunks_glob default from assign directory
    cell_chunks_eff = cell_chunks_glob if cell_chunks_glob is not None else _infer_cell_chunks_glob_from_assign(assign_eff)

    # Resolve effective params
    threads_eff = int(threads) if threads is not None else 1
    pass1_workers_eff = int(pass1_workers) if pass1_workers is not None else int(threads_eff)
    shards_eff = int(shards) if shards is not None else 32
    chunk_rows_eff = int(chunk_rows) if chunk_rows is not None else 5_000_000
    pass2_chunksize_eff = int(pass2_chunksize) if pass2_chunksize is not None else 200_000

    # Winner only default True if not specified
    winner_only_eff = bool(winner_only) if winner_only is not None else True

    # Optional read filter: enabled only if both provided and max_hits>0
    max_hits_eff = None if (max_hits is None or int(max_hits) <= 0) else int(max_hits)
    hits_delta_mapq_eff = None if (hits_delta_mapq is None or max_hits_eff is None) else float(hits_delta_mapq)

    cfg = MergeConfig(
        sample=sample_eff,
        shards=shards_eff,
        chunk_rows=chunk_rows_eff,
        pass2_chunksize=pass2_chunksize_eff,
        winner_only=winner_only_eff,
        pass1_workers=pass1_workers_eff,

        max_hits=max_hits_eff,
        hits_delta_mapq=hits_delta_mapq_eff,

        beta=float(beta) if beta is not None else 1.0,
        w_as=float(w_as) if w_as is not None else 1.0,
        w_mapq=float(w_mapq) if w_mapq is not None else 1.0,
        w_nm=float(w_nm) if w_nm is not None else 1.0,
        ambient_const=float(ambient_const) if ambient_const is not None else 1e-3,

        min_reads=int(min_reads) if min_reads is not None else 5,
        single_mass_min=float(single_mass_min) if single_mass_min is not None else 0.6,
        ratio_top1_top2_min=float(ratio_top1_top2_min) if ratio_top1_top2_min is not None else 2.0,
        doublet_minor_min=float(doublet_minor_min) if doublet_minor_min is not None else 0.20,

        bic_margin=float(bic_margin) if bic_margin is not None else 6.0,
        empty_bic_margin=float(empty_bic_margin) if empty_bic_margin is not None else 10.0,

        empty_top1_max=float(empty_top1_max) if empty_top1_max is not None else 0.6,
        empty_ratio12_max=float(empty_ratio12_max) if empty_ratio12_max is not None else 1.5,
        empty_reads_max=int(empty_reads_max) if empty_reads_max is not None else None,
        empty_seed_bic_min=float(empty_seed_bic_min) if empty_seed_bic_min is not None else 10.0,
        empty_tau_quantile=float(empty_tau_quantile) if empty_tau_quantile is not None else 0.95,
        empty_jsd_max=float(empty_jsd_max) if empty_jsd_max is not None else None,
        jsd_normalize=bool(jsd_normalize) if jsd_normalize is not None else True,

        eta_iters=int(eta_iters) if eta_iters is not None else 2,
        eta_seed_quantile=float(eta_seed_quantile) if eta_seed_quantile is not None else 0.02,
        topk_genomes=int(topk_genomes) if topk_genomes is not None else 3,
    )

    # Resume behavior: if outputs exist, skip
    calls_path = outdir_eff / f"{sample_eff}_genotype_calls.tsv.gz"
    eta_out = outdir_eff / f"{sample_eff}_eta.tsv.gz"
    if resume and calls_path.exists() and eta_out.exists():
        typer.echo(f"[genotyping] resume: outputs exist, skipping: {calls_path.name}")
        return

    # Try enabling chunk policy automatically if chunk lists exist
    chunk_map = _load_chunk_barcode_lists(cell_chunks_eff, chunk_id_regex)
    use_chunk_policy = bool(chunk_map)

    tmp_dir = outdir_eff / f"tmp_{sample_eff}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if use_chunk_policy:
        spill_root = tmp_dir / "L_chunk_spill"
        spill_root.mkdir(parents=True, exist_ok=True)
        typer.echo(f"[genotyping] chunk policy ON  chunks={len(chunk_map)}  cell_chunks_glob={cell_chunks_eff}")
    else:
        shard_root = tmp_dir / "L_shards_workers"
        shard_root.mkdir(parents=True, exist_ok=True)
        typer.echo(f"[genotyping] chunk policy OFF (no chunk lists found) -> hashed shards shards={cfg.shards}")

    typer.echo(f"[genotyping] sample={sample_eff} files={len(files)} pass1_workers={pass1_workers_eff}")
    typer.echo(f"[genotyping] winner_only={cfg.winner_only} ambient_const={cfg.ambient_const}")
    typer.echo(f"[genotyping] filter: max_hits={cfg.max_hits} hits_delta_mapq={cfg.hits_delta_mapq}")

    # -----------------------
    # Pass 1
    # -----------------------
    typer.echo("[1/3] Pass 1: stream assign -> C_all/N_all + spill")
    C_list: List[pd.DataFrame] = []
    N_list: List[pd.DataFrame] = []
    spill_files: List[Path] = []

    def _job_chunk(i: int, f: Path):
        cid = _infer_chunk_id(f.name, chunk_id_regex) or f"{i:04d}"
        bc_set = chunk_map.get(cid)
        if bc_set is None:
            raise ValueError(f"chunk_id={cid} (from {f.name}) not present in chunk barcode lists")
        spill_path = spill_root / f"L_chunk_{cid}.tsv.gz"
        return _pass1_chunkspill_one_file(f, cfg, spill_path, bc_set)

    def _job_shard(i: int, f: Path):
        wdir = shard_root / f"w{i:03d}"
        return _pass1_hashed_shards_one_file(f, cfg, wdir)

    with ProcessPoolExecutor(max_workers=int(pass1_workers_eff)) as ex:
        if use_chunk_policy:
            futs = [ex.submit(_job_chunk, i, f) for i, f in enumerate(files)]
        else:
            futs = [ex.submit(_job_shard, i, f) for i, f in enumerate(files)]

        for fut in as_completed(futs):
            C_out, N_out, paths = fut.result()
            if not C_out.empty:
                C_list.append(C_out)
            if not N_out.empty:
                N_list.append(N_out)
            if use_chunk_policy:
                spill_files.append(paths)  # single spill file
            else:
                spill_files.extend(paths)  # list of shard files

    C_all = (
        pd.concat(C_list, ignore_index=True).groupby(["barcode", "genome"], sort=False)["C"].sum().reset_index()
        if C_list else pd.DataFrame(columns=["barcode", "genome", "C"])
    )
    N_all = (
        pd.concat(N_list, ignore_index=True).groupby("barcode", sort=False)["n_reads"].sum().reset_index()
        if N_list else pd.DataFrame(columns=["barcode", "n_reads"])
    )
    if C_all.empty:
        raise RuntimeError("Pass 1 produced no mass (C_all empty). Check inputs and schema.")

    C_all.to_csv(outdir_eff / f"{sample_eff}_C_all.tsv.gz", sep="\t", index=False, compression="gzip")
    N_all.to_csv(outdir_eff / f"{sample_eff}_N_all.tsv.gz", sep="\t", index=False, compression="gzip")

    all_genomes = C_all["genome"].astype(str).unique().tolist()
    topk: Dict[str, List[str]] = {}
    for bc, sub in C_all.groupby("barcode", sort=False):
        topk[str(bc)] = sub.sort_values("C", ascending=False)["genome"].astype(str).head(int(cfg.topk_genomes)).tolist()

    # choose iterator factory for eta learning / seed BIC filtering
    if use_chunk_policy:
        files_iter_factory = lambda: _iter_spills(spill_root)
    else:
        files_iter_factory = lambda: _iter_shards(shard_root)

    # -----------------------
    # Pass 2: eta
    # -----------------------
    typer.echo("[2/3] Pass 2: ambient learning (eta)")
    eta, eta_meta = _learn_eta(C_all, N_all, files_iter_factory, all_genomes, topk, cfg)
    eta.to_frame("eta").to_csv(eta_out, sep="\t", compression="gzip")
    typer.echo(f"[2/3] eta saved: {eta_out.name} seed_sizes={eta_meta.get('seed_sizes')}")

    # -----------------------
    # Pass 3: calls
    # -----------------------
    typer.echo("[3/3] Pass 3: calls")
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
        for fp in sorted(set(spill_files)):
            if fp.exists():
                called_total += _call_from_file_streaming(fp, eta, topk, cfg, oh)

    typer.echo(f"[genotyping] calls written: {calls_path.name} (rows={called_total})")


if __name__ == "__main__":
    app()
