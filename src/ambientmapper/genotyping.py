#!/usr/bin/env python3
"""
src/ambientmapper/genotyping.py

ambientmapper genotyping — posterior-aware merge → per-cell genotype calls (+ empty-aware)

This module consumes per-read, multi-genome evidence exported by `assign` and
emits per-cell genotype calls: {single, doublet, indistinguishable, ambiguous, empty}.
All decisions are data-driven from `assign` outputs (no plate/design priors here).

Two-pass execution
  Pass 1: Stream input TSV(.gz) chunks → compute per-read posteriors L(r,g),
          accumulate expected counts C(b,g) + n_reads(b), and spill per-read L
          into barcode-hashed shard files on disk.
  Pass 2: Walk shard files → group rows per barcode → model selection (empty vs
          single vs doublet with ambient) → write calls.

Key output contract (updated)
  - ALWAYS write one row per barcode that reaches inference inclusion:
      <outdir>/<sample>_cells_calls.tsv.gz
    including call="empty" rows (empties are NOT dropped before writing).

Empty gate update (entropy_norm)
  - Add entropy_norm in [0,1] computed with K_eff = topk_genomes (Option 1).
  - Empty gate uses entropy_norm >= empty_entropy_norm_min.

Notes
  - Winner-only mode requires AS, MAPQ, NM.
  - Probabilistic mode fuses AS/MAPQ/NM + p-value penalty.
  - Empty model compares against best non-empty BIC with conservative gates.
"""
from __future__ import annotations

import glob
import gzip
import hashlib
import math
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import typer
from typer.models import OptionInfo
from pydantic import BaseModel
from tqdm import tqdm

app = typer.Typer(add_completion=False, no_args_is_help=True)

# ------------------------------
# Config
# ------------------------------


class MergeConfig(BaseModel):
    # Fusion
    beta: float = 0.5
    w_as: float = 0.5
    w_mapq: float = 1.0
    w_nm: float = 1.0
    ambient_const: float = 1e-3
    p_eps: float = 1e-3

    # Calling thresholds
    min_reads: int = 100              # minimum reads to fit Single/Doublet (inference inclusion is separate if you add it later)
    single_mass_min: float = 0.7      # purity threshold for Single
    doublet_minor_min: float = 0.20   # minor fraction threshold for Doublet
    bic_margin: float = 6.0           # ΔBIC required to accept Doublet over Single
    near_tie_margin: float = 2.0      # tie band for Indistinguishable
    ratio_top1_top2_min: float = 2.0  # top1/top2 dominance ratio for Single

    # Empty discovery / ambient learning
    eta_iters: int = 2
    eta_seed_quantile: float = 0.02
    empty_bic_margin: float = 10.0
    empty_top1_max: float = 0.6
    empty_ratio12_max: float = 1.5
    empty_entropy_norm_min: float = 0.8  # NEW: normalized entropy gate
    empty_reads_max: Optional[int] = None

    # Search grid
    alpha_grid: float = 0.02
    rho_grid: float = 0.05
    max_alpha: float = 0.5
    topk_genomes: int = 3  # also used as K_eff for entropy_norm

    # System
    sample: str = "sample"
    shards: int = 32
    chunk_rows: int = 5_000_000         # pass1 chunk size
    pass2_chunksize: int = 200_000      # pass2 shard read chunks
    winner_only: bool = True
    pass1_workers: int = 1


# ------------------------------
# I/O & Schema
# ------------------------------

REQUIRED_COLS = {"barcode", "read_id", "genome"}


def _coerce_assign_schema(df: pd.DataFrame) -> pd.DataFrame:
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

    # IDs to string
    for col in ("barcode", "read_id", "genome"):
        if col in out.columns:
            out[col] = out[col].astype(str)

    # Numeric alignment metrics
    for col in ("AS", "MAPQ", "NM"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # p-values (or deciles)
    for pcol in ("p_as", "p_mq"):
        if pcol in out.columns:
            out[pcol] = pd.to_numeric(out[pcol], errors="coerce")
            max_val = out[pcol].max(skipna=True)
            if max_val and max_val > 1.0:
                out[pcol] = out[pcol] / 10.0

    return out


def _reduce_alignments_to_per_genome(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["barcode", "read_id", "genome"]
    have = set(df.columns)
    if not all(k in have for k in keys):
        raise ValueError("Missing required keys after coercion.")

    agg: Dict[str, str] = {}
    if "AS" in have:
        agg["AS"] = "max"
    if "MAPQ" in have:
        agg["MAPQ"] = "max"
    if "NM" in have:
        agg["NM"] = "min"
    if "p_as" in have:
        agg["p_as"] = "min"
    if "p_mq" in have:
        agg["p_mq"] = "min"

    if not agg:
        return df.drop_duplicates(keys)[keys]

    return df.groupby(keys, observed=True).agg(agg).reset_index()


def _write_gzip_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, compression="gzip")


def _pass1_job(i: int, fp: Path, cfg: MergeConfig, worker_root: Path) -> Tuple[int, pd.DataFrame, pd.DataFrame]:
    wd = worker_root / f"w{i:03d}"
    wd.mkdir(parents=True, exist_ok=True)
    C_one, N_one = _pass1_process_one_file(fp, cfg, wd)
    return i, C_one, N_one


# ------------------------------
# Core math
# ------------------------------

def _zscore_series(x: np.ndarray) -> np.ndarray:
    x = x.astype(float, copy=False)

    if np.isnan(x).any():
        med0 = np.nanmedian(x)
        if np.isnan(med0):
            return np.zeros_like(x, dtype=float)
        x = np.where(np.isnan(x), med0, x)

    med = np.median(x)
    mad = np.median(np.abs(x - med))
    if mad < 1e-6:
        s = x.std(ddof=0)
        s = 1.0 if s < 1e-6 else s
        return (x - med) / s
    return (x - med) / (1.4826 * mad)


def _compute_read_posteriors(df: pd.DataFrame, cfg: MergeConfig) -> pd.DataFrame:
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(f"Missing cols: {missing}")

    df = df.copy()
    df["_rid"] = df["barcode"].astype(str) + "::" + df["read_id"].astype(str)

    # -------------------
    # Winner-only (default)
    # -------------------
    if cfg.winner_only:
        for col in ("AS", "MAPQ", "NM"):
            if col not in df.columns:
                raise ValueError("winner_only=True requires AS, MAPQ, NM")

        df["_sort_as"] = -df["AS"].fillna(-9999)
        df["_sort_mapq"] = -df["MAPQ"].fillna(-1)
        df["_sort_nm"] = df["NM"].fillna(9999)

        df = df.sort_values(
            by=["_rid", "_sort_as", "_sort_mapq", "_sort_nm"],
            ascending=True,
        )
        out = df.drop_duplicates(subset=["_rid"], keep="first").copy()

        denom = 1.0 + cfg.ambient_const
        out["L"] = 1.0 / denom
        out["L_amb"] = cfg.ambient_const / denom

        return out[["barcode", "read_id", "genome", "L", "L_amb"]].reset_index(drop=True)

    # -------------------
    # Probabilistic mode
    # -------------------
    df["_rid_code"] = df["_rid"].astype("category").cat.codes
    S_parts: List[np.ndarray] = []

    if "AS" in df.columns:
        S_as = df.groupby("_rid_code", observed=True)["AS"].transform(
            lambda s: _zscore_series(s.to_numpy(float))
        )
        S_parts.append(cfg.w_as * S_as.to_numpy(float))

    if "MAPQ" in df.columns:
        S_mq = df.groupby("_rid_code", observed=True)["MAPQ"].transform(
            lambda s: _zscore_series(s.to_numpy(float))
        )
        S_parts.append(cfg.w_mapq * S_mq.to_numpy(float))

    if "NM" in df.columns:
        S_nm = df.groupby("_rid_code", observed=True)["NM"].transform(
            lambda s: _zscore_series(s.to_numpy(float))
        )
        S_parts.append(-cfg.w_nm * S_nm.to_numpy(float))

    S = np.sum(S_parts, axis=0) if S_parts else np.zeros(len(df), dtype=float)

    # p-value penalty (NaN-safe): missing p treated as 1.0
    gamma = np.ones(len(df), dtype=float)
    if ("p_as" in df.columns) or ("p_mq" in df.columns):
        pa = df["p_as"].to_numpy(float) if "p_as" in df.columns else np.ones(len(df), dtype=float)
        pm = df["p_mq"].to_numpy(float) if "p_mq" in df.columns else np.ones(len(df), dtype=float)
        pa = np.nan_to_num(pa, nan=1.0, posinf=1.0, neginf=1.0)
        pm = np.nan_to_num(pm, nan=1.0, posinf=1.0, neginf=1.0)
        pmin = np.minimum(pa, pm)
        gamma = np.maximum(cfg.p_eps, 1.0 - pmin)

    w = np.exp(cfg.beta * S) * gamma

    df["_tmp_w"] = w
    w_sum = df.groupby("_rid_code", observed=True)["_tmp_w"].transform("sum")
    total = w_sum + cfg.ambient_const

    df["L"] = w / total
    df["L_amb"] = cfg.ambient_const / total

    return df[["barcode", "read_id", "genome", "L", "L_amb"]].reset_index(drop=True)


def _aggregate_expected_counts_from_chunk(Ldf: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    C = (
        Ldf.groupby(["barcode", "genome"], observed=True)["L"]
        .sum()
        .rename("C")
        .reset_index()
    )
    N = (
        Ldf.groupby("barcode", observed=True)["read_id"]
        .nunique()
        .rename("n_reads")
        .reset_index()
    )
    return C, N


def _loglik_empty(L_block: pd.DataFrame, eta: pd.Series) -> float:
    genomes = L_block["genome"].astype(str)
    eta_row = eta.reindex(genomes).fillna(0.0).to_numpy(dtype=float)

    wL = eta_row * L_block["L"].to_numpy(dtype=float)

    tmp = L_block[["read_id"]].copy()
    tmp["wL"] = wL
    s = tmp.groupby("read_id", observed=True)["wL"].sum().to_numpy()
    s = np.clip(s, 1e-12, None)
    return float(np.log(s).sum())


def _loglik_for_params(
    L_block: pd.DataFrame,
    eta: pd.Series,
    model: str,
    g1: str,
    g2: Optional[str],
    alpha: float,
    rho: float = 0.5,
) -> float:
    genomes = L_block["genome"].astype(str)
    eta_row = eta.reindex(genomes).fillna(0.0).to_numpy(dtype=float)

    if model == "single":
        eta_g1 = float(eta.get(g1, 0.0))
        is_g1 = (genomes.to_numpy() == g1)
        w = np.where(is_g1, (1.0 - alpha) + alpha * eta_g1, alpha * eta_row)

    elif model == "doublet":
        if g2 is None:
            raise ValueError("doublet requires g2")
        is_g1 = (genomes.to_numpy() == g1)
        is_g2 = (genomes.to_numpy() == g2)
        mix = np.where(is_g1, rho, np.where(is_g2, 1.0 - rho, 0.0))
        w = (1.0 - alpha) * mix + alpha * eta_row

    else:
        raise ValueError(f"Unknown model {model}")

    tmp = L_block[["read_id"]].copy()
    tmp["wL"] = w * L_block["L"].to_numpy(dtype=float)
    s = tmp.groupby("read_id", observed=True)["wL"].sum().to_numpy()
    s = np.clip(s, 1e-12, None)
    return float(np.log(s).sum())


def _bic(loglik: float, n_params: int, n_reads: int) -> float:
    return -2.0 * loglik + n_params * math.log(max(n_reads, 1))


# ------------------------------
# Model selection / calling
# ------------------------------

def _select_model_for_barcode(
    L_block: pd.DataFrame,
    eta: pd.Series,
    cfg: MergeConfig,
    candidate_genomes: Sequence[str],
) -> Dict[str, Any]:
    read_count = int(L_block["read_id"].nunique())

    # Composition stats from expected counts
    mass = (
        L_block.groupby("genome", observed=True)["L"]
        .sum()
        .sort_values(ascending=False)
    )
    tot = float(mass.sum())
    props = (mass / max(tot, 1e-12)).to_numpy()
    genomes_sorted = mass.index.to_list()

    p1 = float(props[0]) if len(props) > 0 else 0.0
    p2 = float(props[1]) if len(props) > 1 else 0.0
    p3 = float(props[2]) if len(props) > 2 else 0.0
    top3_sum = p1 + p2 + p3

    # entropy (natural log)
    entropy = float(-np.sum(props * np.log(np.clip(props, 1e-12, 1.0)))) if tot > 0 else 0.0

    # entropy_norm using Option 1: K_eff = cfg.topk_genomes
    K_eff = int(cfg.topk_genomes) if int(cfg.topk_genomes) > 0 else 1
    Hmax = math.log(K_eff) if K_eff > 1 else 0.0
    if Hmax > 0:
        entropy_norm = entropy / Hmax
        if entropy_norm < 0.0:
            entropy_norm = 0.0
        elif entropy_norm > 1.0:
            entropy_norm = 1.0
    else:
        entropy_norm = 0.0

    # ratio: if p2==0 => Inf (and will fail empty gate)
    ratio12 = (p1 / p2) if p2 > 0 else float("inf")
    top_genome = genomes_sorted[0] if genomes_sorted else None

    out: Dict[str, Any] = {
        "n_reads": read_count,
        "n_effective": tot,
        "p_top1": p1,
        "p_top2": p2,
        "p_top3": p3,
        "top3_sum": top3_sum,
        "entropy": entropy,
        "entropy_norm": float(entropy_norm),
        "ratio_top1_top2": ratio12,
        "top_genome": top_genome,
        "call": "ambiguous",
        "status_flag": "ambiguous",
        "bic_empty": float("inf"),
        "bic_single": float("inf"),
        "bic_doublet": float("inf"),
        "bic_best": float("inf"),
        "bic_best_non_empty": float("inf"),
        "delta_empty": float("nan"),
        "delta_math": float("nan"),
        "model": None,
        "genome_1": None,
        "genome_2": None,
        "alpha": float("nan"),
        "rho": float("nan"),
        "purity": 0.0,
        "minor": 0.0,
        "indistinguishable_set": "",
    }

    # Empty model (always)
    ll_empty = _loglik_empty(L_block, eta)
    bic_empty = _bic(ll_empty, n_params=0, n_reads=read_count)
    out["bic_empty"] = bic_empty

    # Single/doublet only if enough reads
    evals: List[Dict[str, Any]] = []
    best_single: Optional[Dict[str, Any]] = None
    best_doublet: Optional[Dict[str, Any]] = None

    if read_count >= cfg.min_reads and candidate_genomes:
        alphas = np.arange(0.0, cfg.max_alpha + 1e-9, cfg.alpha_grid)
        rhos = np.arange(0.1, 0.9 + 1e-9, cfg.rho_grid)

        # Single
        for g1 in candidate_genomes:
            best_s = {"bic": float("inf")}
            for a in alphas:
                ll = _loglik_for_params(L_block, eta, "single", g1, None, a)
                bic = _bic(ll, n_params=1, n_reads=read_count)
                if bic < best_s["bic"]:
                    best_s = {"model": "single", "g1": g1, "g2": None, "alpha": a, "rho": 1.0, "bic": bic}
            evals.append(best_s)

        # Doublet
        if len(candidate_genomes) >= 2:
            for i in range(len(candidate_genomes)):
                for j in range(i + 1, len(candidate_genomes)):
                    g1, g2 = candidate_genomes[i], candidate_genomes[j]
                    best_d = {"bic": float("inf")}
                    for a in alphas:
                        for r in rhos:
                            ll = _loglik_for_params(L_block, eta, "doublet", g1, g2, a, r)
                            bic = _bic(ll, n_params=2, n_reads=read_count)
                            if bic < best_d["bic"]:
                                best_d = {"model": "doublet", "g1": g1, "g2": g2, "alpha": a, "rho": r, "bic": bic}
                    evals.append(best_d)

        single_fits = [e for e in evals if e.get("model") == "single"]
        doublet_fits = [e for e in evals if e.get("model") == "doublet"]

        best_single = min(single_fits, key=lambda x: x["bic"]) if single_fits else None
        best_doublet = min(doublet_fits, key=lambda x: x["bic"]) if doublet_fits else None

    out["bic_single"] = best_single["bic"] if best_single else float("inf")
    out["bic_doublet"] = best_doublet["bic"] if best_doublet else float("inf")
    out["delta_math"] = float(out["bic_doublet"] - out["bic_single"])

    # Best non-empty model
    best_non_empty: Optional[Dict[str, Any]]
    if best_single and best_doublet:
        best_non_empty = best_single if best_single["bic"] <= best_doublet["bic"] else best_doublet
    else:
        best_non_empty = best_single or best_doublet

    bic_non_empty = best_non_empty["bic"] if best_non_empty else float("inf")
    out["bic_best_non_empty"] = float(bic_non_empty)
    out["delta_empty"] = float(bic_non_empty - bic_empty) if math.isfinite(bic_non_empty) else float("inf")

    # -------------------------
    # Empty gate (conservative)
    # -------------------------
    is_empty_best = (bic_empty < bic_non_empty)
    bic_margin_ok = (bic_non_empty - bic_empty) >= cfg.empty_bic_margin
    weak_dominance = (p1 <= cfg.empty_top1_max) and (ratio12 <= cfg.empty_ratio12_max)
    diffuse = (entropy_norm >= cfg.empty_entropy_norm_min)
    ceiling_ok = (read_count <= cfg.empty_reads_max) if cfg.empty_reads_max is not None else True

    if is_empty_best and bic_margin_ok and weak_dominance and diffuse and ceiling_ok:
        out.update(
            {
                "call": "empty",
                "status_flag": "empty",
                "bic_best": bic_empty,
                "model": "empty",
                "alpha": 1.0,
                "rho": float("nan"),
            }
        )
        return out

    # If no non-empty fit (e.g., low depth), fall back
    if best_non_empty is None:
        out.update({"call": "ambiguous", "status_flag": "low_depth", "bic_best": bic_empty})
        return out

    # -------------------------
    # Non-empty: compute purity/minor
    # -------------------------
    model = best_non_empty["model"]
    alpha = float(best_non_empty["alpha"])
    rho = float(best_non_empty["rho"])

    if model == "doublet":
        purity = (1.0 - alpha) * max(rho, 1.0 - rho)
        minor = (1.0 - alpha) * min(rho, 1.0 - rho)
    else:
        purity = (1.0 - alpha)
        minor = 0.0

    out.update(
        {
            "model": model,
            "genome_1": best_non_empty.get("g1"),
            "genome_2": best_non_empty.get("g2"),
            "alpha": alpha,
            "rho": rho,
            "purity": float(purity),
            "minor": float(minor),
            "bic_best": float(best_non_empty["bic"]),
        }
    )

    # Strong doublet gate
    is_strong_doublet = (
        model == "doublet"
        and minor >= cfg.doublet_minor_min
        and (out["bic_single"] - out["bic_doublet"]) >= cfg.bic_margin
    )
    if is_strong_doublet:
        out.update({"call": "doublet", "status_flag": "doublet"})
        return out

    # Indistinguishable vs single-by-default
    delta = out["bic_doublet"] - out["bic_single"]
    is_near_tie = (
        len(genomes_sorted) >= 2
        and abs(delta) < cfg.near_tie_margin
        and ratio12 < cfg.ratio_top1_top2_min
        and p1 >= cfg.single_mass_min
    )
    if is_near_tie:
        out.update(
            {
                "call": "indistinguishable",
                "status_flag": "low_confidence",
                "indistinguishable_set": f"{genomes_sorted[0]},{genomes_sorted[1]}",
            }
        )
        return out

    # Single by default
    if (ratio12 >= cfg.ratio_top1_top2_min) and (purity >= cfg.single_mass_min):
        out.update({"call": "single", "status_flag": "single"})
    else:
        out.update({"call": "ambiguous", "status_flag": "ambiguous"})
    return out


# ------------------------------
# Pass drivers
# ------------------------------

def _barcode_to_shard_idx(barcode: str, shards: int) -> int:
    h = hashlib.sha1(barcode.encode("utf-8")).digest()
    return int.from_bytes(h[:4], "little") % max(shards, 1)


def _open_shard_handles(shard_dir: Path, shards: int) -> List[gzip.GzipFile]:
    shard_dir.mkdir(parents=True, exist_ok=True)
    return [gzip.open(shard_dir / f"shard_{i:02d}.tsv.gz", "at") for i in range(shards)]


def _write_L_chunk_to_shards(
    Ldf: pd.DataFrame,
    handles: List[gzip.GzipFile],
    shards: int,
    header_flags: Dict[int, bool],
) -> None:
    cols = ["barcode", "read_id", "genome", "L", "L_amb"]
    Ldf = Ldf[cols].copy()

    for bc, sub in Ldf.groupby("barcode", sort=False, observed=True):
        idx = _barcode_to_shard_idx(str(bc), shards)
        if not header_flags[idx]:
            handles[idx].write("\t".join(cols) + "\n")
            header_flags[idx] = True
        sub.to_csv(handles[idx], sep="\t", header=False, index=False)


def _iter_shard_rows(shard_fp: Path, chunksize: int) -> Iterator[pd.DataFrame]:
    try:
        it = pd.read_csv(
            shard_fp,
            sep="\t",
            compression="gzip",
            chunksize=chunksize,
            dtype={
                "barcode": "string",
                "read_id": "string",
                "genome": "string",
                "L": "string",
                "L_amb": "string",
            },
        )
    except FileNotFoundError:
        return

    for chunk in it:
        chunk = chunk[chunk["barcode"] != "barcode"]
        chunk = chunk.dropna(subset=["barcode", "read_id", "genome"])
        if chunk.empty:
            continue

        chunk["L"] = pd.to_numeric(chunk["L"], errors="coerce")
        chunk["L_amb"] = pd.to_numeric(chunk["L_amb"], errors="coerce")
        chunk = chunk[chunk["L"].notna() & chunk["L_amb"].notna()]
        if chunk.empty:
            continue

        chunk["L"] = chunk["L"].astype("float32")
        chunk["L_amb"] = chunk["L_amb"].astype("float32")
        yield chunk


def _pass1_stream_build(assign_glob: str, cfg: MergeConfig, tmp_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame, Path]:
    files = [Path(p) for p in glob.glob(assign_glob, recursive=True)]
    if not files:
        raise FileNotFoundError(f"No files for {assign_glob}")

    workers = max(1, int(getattr(cfg, "pass1_workers", 1) or 1))
    workers = min(workers, len(files))

    worker_root = tmp_dir / "L_shards_workers"
    worker_root.mkdir(parents=True, exist_ok=True)

    C_parts: List[pd.DataFrame] = []
    N_parts: List[pd.DataFrame] = []

    if workers == 1:
        wd = worker_root / "w000"
        wd.mkdir(parents=True, exist_ok=True)
        for fp in tqdm(files, desc="[pass1] streaming"):
            C_one, N_one = _pass1_process_one_file(fp, cfg, wd)
            C_parts.append(C_one)
            N_parts.append(N_one)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futures = [
                ex.submit(_pass1_job, i, fp, cfg, worker_root)
                for i, fp in enumerate(files)
            ]
            for fut in tqdm(futures, desc=f"[pass1] files (workers={workers})"):
                _, C_one, N_one = fut.result()
                C_parts.append(C_one)
                N_parts.append(N_one)

    if C_parts:
        C_all = (
            pd.concat(C_parts, ignore_index=True)
            .groupby(["barcode", "genome"], observed=True)["C"]
            .sum()
            .reset_index()
        )
    else:
        C_all = pd.DataFrame(columns=["barcode", "genome", "C"])

    if N_parts:
        N_all = (
            pd.concat(N_parts, ignore_index=True)
            .groupby("barcode", observed=True)["n_reads"]
            .sum()
            .reset_index()
        )
    else:
        N_all = pd.DataFrame(columns=["barcode", "n_reads"])

    shard_dir = tmp_dir / "L_shards"
    worker_dirs = sorted([p for p in worker_root.glob("w*") if p.is_dir()])
    _merge_worker_shards(worker_dirs, shard_dir, cfg.shards)

    return C_all, N_all, shard_dir


def _pass1_process_one_file(fp: Path, cfg: MergeConfig, shard_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    handles = _open_shard_handles(shard_dir, cfg.shards)
    headers = {i: False for i in range(cfg.shards)}
    C_parts: List[pd.DataFrame] = []
    N_parts: List[pd.DataFrame] = []

    try:
        it = pd.read_csv(fp, sep="\t", chunksize=cfg.chunk_rows, low_memory=False)
        for raw in it:
            df = _coerce_assign_schema(raw)
            df = _reduce_alignments_to_per_genome(df).dropna(subset=["barcode", "read_id", "genome"])
            if df.empty:
                continue

            Ldf = _compute_read_posteriors(df, cfg)
            C, N = _aggregate_expected_counts_from_chunk(Ldf)
            C_parts.append(C)
            N_parts.append(N)
            _write_L_chunk_to_shards(Ldf, handles, cfg.shards, headers)
    finally:
        for h in handles:
            try:
                h.close()
            except Exception:
                pass

    if C_parts:
        C_one = (
            pd.concat(C_parts, ignore_index=True)
            .groupby(["barcode", "genome"], observed=True)["C"]
            .sum()
            .reset_index()
        )
    else:
        C_one = pd.DataFrame(columns=["barcode", "genome", "C"])

    if N_parts:
        N_one = (
            pd.concat(N_parts, ignore_index=True)
            .groupby("barcode", observed=True)["n_reads"]
            .sum()
            .reset_index()
        )
    else:
        N_one = pd.DataFrame(columns=["barcode", "n_reads"])

    return C_one, N_one


def _merge_worker_shards(worker_dirs: List[Path], out_shard_dir: Path, shards: int) -> None:
    out_shard_dir.mkdir(parents=True, exist_ok=True)
    for si in range(shards):
        out_fp = out_shard_dir / f"shard_{si:02d}.tsv.gz"
        if out_fp.exists():
            out_fp.unlink()

        with open(out_fp, "ab") as w:
            for wd in worker_dirs:
                part = wd / f"shard_{si:02d}.tsv.gz"
                if part.exists() and part.stat().st_size > 0:
                    with open(part, "rb") as r:
                        shutil.copyfileobj(r, w, length=1024 * 1024)


def _pass2_worker(args) -> List[Dict[str, Any]]:
    shard_fp, cfg, topk, eta = args
    rows: List[Dict[str, Any]] = []
    blocks: Dict[str, List[pd.DataFrame]] = {}

    for chunk in _iter_shard_rows(shard_fp, cfg.pass2_chunksize):
        for bc, sub in chunk.groupby("barcode", sort=False, observed=True):
            blocks.setdefault(str(bc), []).append(sub)

    for bc, parts in blocks.items():
        L_block = pd.concat(parts, ignore_index=True)
        cand = topk.get(bc, [])
        if L_block.empty:
            continue
        res = _select_model_for_barcode(L_block, eta, cfg, cand)
        res["barcode"] = bc
        rows.append(res)

    return rows


def _unwrap_optioninfo(x):
    return x.default if isinstance(x, OptionInfo) else x


# ------------------------------
# Main
# ------------------------------

@app.command("genotyping")
def genotyping(
    assign: str = typer.Option(..., help="Path/glob to assign outputs."),
    outdir: Path = typer.Option(Path("merge_out"), help="Output dir."),
    sample: str = typer.Option("sample", help="Sample name."),

    # Core
    min_reads: int = typer.Option(100, help="Min reads to fit Single/Doublet models."),
    single_mass_min: float = typer.Option(0.7, help="Purity threshold for single calls."),
    ratio_top1_top2_min: float = typer.Option(2.0, help="Min top1/top2 dominance ratio for single calls."),
    shards: int = typer.Option(32, help="Number of spill shards."),
    threads: int = typer.Option(1, help="Pass-2 workers."),
    chunk_rows: int = typer.Option(5_000_000, help="Pass-1 input chunksize."),
    pass1_workers: Optional[int] = typer.Option(None, help="Pass-1 workers."),
    pass2_chunksize: int = typer.Option(200_000, help="Pass-2 shard chunksize."),
    winner_only: bool = typer.Option(True, help="Winner-only mode (default)."),

    # Fusion
    beta: float = typer.Option(0.5, help="Softmax temperature (probabilistic mode)."),
    w_as: float = typer.Option(0.5, help="Weight for AS (probabilistic mode)."),
    w_mapq: float = typer.Option(1.0, help="Weight for MAPQ (probabilistic mode)."),
    w_nm: float = typer.Option(1.0, help="Weight for NM penalty (probabilistic mode)."),
    ambient_const: float = typer.Option(1e-3, help="Per-read ambient mass."),

    # Empty gates
    empty_bic_margin: float = typer.Option(10.0, help="Min ΔBIC (non-empty - empty) to call empty."),
    empty_top1_max: float = typer.Option(0.6, help="Max top1 mass to allow empty."),
    empty_ratio12_max: float = typer.Option(1.5, help="Max top1/top2 ratio to allow empty."),
    empty_entropy_norm_min: float = typer.Option(0.8, help="Min entropy_norm to allow empty."),
    empty_reads_max: Optional[int] = typer.Option(None, help="Optional ceiling for empty calls."),

    # Doublet gates
    bic_margin: float = typer.Option(6.0, help="ΔBIC required to accept doublet over single."),
    doublet_minor_min: float = typer.Option(0.20, help="Min minor fraction for doublet."),

    # Ambient iteration
    eta_iters: int = typer.Option(2, help="Iterations for ambient refinement."),
    eta_seed_quantile: float = typer.Option(0.02, help="Seed eta from bottom quantile of n_reads."),
    topk_genomes: int = typer.Option(3, help="Top-K candidate genomes per barcode (also K_eff for entropy_norm)."),
):
    """Run iterative empty-aware genotyping (writes empties into *_cells_calls.tsv.gz)."""
    cfg = MergeConfig(
        sample=sample,
        min_reads=min_reads,
        single_mass_min=single_mass_min,
        ratio_top1_top2_min=ratio_top1_top2_min,
        shards=shards,
        chunk_rows=chunk_rows,
        pass1_workers=int(pass1_workers) if pass1_workers is not None else 1,
        pass2_chunksize=_unwrap_optioninfo(pass2_chunksize),
        winner_only=winner_only,
        beta=beta,
        w_as=w_as,
        w_mapq=w_mapq,
        w_nm=w_nm,
        ambient_const=_unwrap_optioninfo(ambient_const),
        empty_bic_margin=_unwrap_optioninfo(empty_bic_margin),
        empty_top1_max=_unwrap_optioninfo(empty_top1_max),
        empty_ratio12_max=_unwrap_optioninfo(empty_ratio12_max),
        empty_entropy_norm_min=_unwrap_optioninfo(empty_entropy_norm_min),
        empty_reads_max=_unwrap_optioninfo(empty_reads_max),
        bic_margin=_unwrap_optioninfo(bic_margin),
        doublet_minor_min=_unwrap_optioninfo(doublet_minor_min),
        eta_iters=_unwrap_optioninfo(eta_iters),
        eta_seed_quantile=_unwrap_optioninfo(eta_seed_quantile),
        topk_genomes=topk_genomes,
    )

    outdir = outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    tmp_dir = outdir / f"tmp_{sample}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Pass 1
    typer.echo("[1/5] Pass 1: streaming inputs → per-read posteriors + spill shards")
    C_all, N_all, shard_dir = _pass1_stream_build(assign, cfg, tmp_dir)

    # candidate genomes per barcode
    topk: Dict[str, List[str]] = {}
    if not C_all.empty:
        for bc, sub in C_all.groupby("barcode", sort=False, observed=True):
            topk[str(bc)] = (
                sub.sort_values("C", ascending=False)["genome"]
                .head(cfg.topk_genomes)
                .astype(str)
                .tolist()
            )

    all_genomes = sorted(C_all["genome"].unique()) if not C_all.empty else []

    # eta estimation helper (smoothed + reindexed)
    def _compute_eta(target_bcs: Sequence[str]) -> pd.Series:
        mix: Dict[str, float] = {}
        target_set = set(map(str, target_bcs))

        for shard in sorted(shard_dir.glob("*.tsv.gz")):
            for chunk in _iter_shard_rows(shard, cfg.pass2_chunksize):
                sub = chunk[chunk["barcode"].isin(target_set)]
                if sub.empty:
                    continue
                add = sub.groupby("genome", observed=True)["L"].sum()
                for g, v in add.items():
                    mix[str(g)] = mix.get(str(g), 0.0) + float(v)

        eps = 1e-12
        allg = all_genomes if all_genomes else sorted(mix.keys())

        if not allg:
            return pd.Series(dtype=float)

        if not mix:
            eta0 = pd.Series({g: 1.0 / len(allg) for g in allg})
            eta0 = eta0 + eps
            return eta0 / eta0.sum()

        eta_s = pd.Series(mix, dtype=float).reindex(allg).fillna(0.0)
        eta_s = eta_s + eps
        eta_s = eta_s / eta_s.sum()
        return eta_s

    # seed eta from low-depth tail
    typer.echo(f"[2/5] Ambient refinement (eta): seed from bottom quantile={cfg.eta_seed_quantile}")
    if N_all.empty:
        eta = pd.Series({g: 1.0 / len(all_genomes) for g in all_genomes}) if all_genomes else pd.Series(dtype=float)
    else:
        cutoff = float(N_all["n_reads"].quantile(cfg.eta_seed_quantile))
        seed_bcs = N_all[N_all["n_reads"] <= max(cutoff, 10)]["barcode"].astype(str).tolist()
        eta = _compute_eta(seed_bcs)

    shard_files = sorted(shard_dir.glob("*.tsv.gz"))

    # iterative refinement
    typer.echo(f"[3/5] Iterating eta using inferred empties (iters={cfg.eta_iters})")
    for i in range(cfg.eta_iters):
        args = [(fp, cfg, topk, eta) for fp in shard_files]
        rows_iter: List[Dict[str, Any]] = []
        if threads <= 1:
            for a in tqdm(args, desc=f"[eta] iter {i+1}/{cfg.eta_iters}"):
                rows_iter.extend(_pass2_worker(a))
        else:
            with ProcessPoolExecutor(max_workers=threads) as ex:
                for rr in tqdm(ex.map(_pass2_worker, args), total=len(args), desc=f"[eta] iter {i+1}/{cfg.eta_iters}"):
                    rows_iter.extend(rr)

        df_iter = pd.DataFrame(rows_iter)
        empty_bcs = df_iter.loc[df_iter["call"] == "empty", "barcode"].astype(str).tolist()

        if len(empty_bcs) < 10:
            typer.echo("  [warn] Too few empty calls to refine eta; stopping refinement.")
            break

        eta = _compute_eta(empty_bcs)
        eta.to_json(outdir / f"{sample}_eta_iter{i+1}.json")

    eta.to_json(outdir / f"{sample}_eta_final.json")

    # final pass2
    typer.echo("[4/5] Final calling (Pass 2)")
    args = [(fp, cfg, topk, eta) for fp in shard_files]
    rows: List[Dict[str, Any]] = []
    if threads <= 1:
        for a in tqdm(args, desc="[pass2] shards"):
            rows.extend(_pass2_worker(a))
    else:
        with ProcessPoolExecutor(max_workers=threads) as ex:
            for rr in tqdm(ex.map(_pass2_worker, args), total=len(args), desc="[pass2] shards"):
                rows.extend(rr)

    calls = pd.DataFrame(rows)

    # best genome convenience column
    if "top_genome" in calls.columns:
        calls["best_genome"] = calls["top_genome"].fillna(calls.get("genome_1", ""))
    else:
        calls["best_genome"] = calls.get("genome_1", "")

    # Output contract: DO NOT drop empties here
    _write_gzip_df(calls, outdir / f"{sample}_cells_calls.tsv.gz")

    # legacy PASS: exclude empties
    legacy = calls.copy()
    legacy["AssignedGenome"] = legacy["best_genome"].fillna("")
    pass_mask = legacy["call"].isin(["single", "doublet", "indistinguishable"])
    legacy.loc[pass_mask, ["barcode", "AssignedGenome", "call", "purity", "n_reads"]].to_csv(
        outdir / f"{sample}_BCs_PASS_by_mapping.csv",
        index=False,
    )

    # cleanup
    try:
        shutil.rmtree(tmp_dir)
    except Exception:
        pass

    typer.echo("[5/5] Done.")


if __name__ == "__main__":
    app()
