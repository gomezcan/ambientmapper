#!/usr/bin/env python3
"""
src/ambientmapper/genotyping.py

ambientmapper genotyping — posterior-aware merge → per-cell genotype calls (+ empty-aware)

Consumes per-read, multi-genome evidence exported by `assign` and emits per-cell calls.

Two-pass execution
  Pass 1: Stream input TSV(.gz) chunks → compute per-read posteriors L(r,g),
          accumulate expected counts C(b,g) + n_reads(b), and spill per-read L
          into barcode-hashed shard files on disk.
  Pass 2: Walk shard files → group rows per barcode → model selection (empty vs
          single vs doublet with ambient) → write calls.

Key output contract
  - ALWAYS write one row per barcode that reaches inference inclusion:
      <outdir>/<sample>_cells_calls.tsv.gz
    including call="empty" rows (empties are NOT dropped before writing).

Empty gate (updated)
  - Empty gate uses JSD(p_b, eta) <= tau
  - tau is learned from high-confidence empty seeds defined non-circularly by ΔBIC-only empties
    (empty best by BIC AND (BIC_non_empty - BIC_empty) >= empty_seed_bic_min).

Notes
  - Winner-only mode requires AS, MAPQ, NM.
  - Probabilistic mode fuses AS/MAPQ/NM + p-value penalty.
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
from pydantic import BaseModel
from tqdm import tqdm
from typer.models import OptionInfo

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
    min_reads: int = 100
    single_mass_min: float = 0.7
    doublet_minor_min: float = 0.20
    bic_margin: float = 6.0
    near_tie_margin: float = 2.0
    ratio_top1_top2_min: float = 2.0

    # Ambient learning
    eta_iters: int = 2
    eta_seed_quantile: float = 0.02

    # Empty gate (structure + BIC)
    empty_bic_margin: float = 10.0
    empty_top1_max: float = 0.6
    empty_ratio12_max: float = 1.5
    empty_reads_max: Optional[int] = None

    # Empty discovery via JSD-to-ambient
    empty_seed_bic_min: float = 20.0
    empty_tau_quantile: float = 0.95
    empty_jsd_max: Optional[float] = None  # learned tau; if set, used as gate
    jsd_normalize: bool = True  # normalize by log(2) -> [0,1]

    # Search grid
    alpha_grid: float = 0.02
    rho_grid: float = 0.05
    max_alpha: float = 0.5
    topk_genomes: int = 3

    # System
    sample: str = "sample"
    shards: int = 32
    chunk_rows: int = 5_000_000
    pass2_chunksize: int = 200_000
    winner_only: bool = True
    pass1_workers: int = 1
  
    # Optional: drop promiscuous ambiguous reads (disabled unless max_hits is set)
    max_hits: Optional[int] = None        # if None, do not filter
    hits_delta_as: Optional[int] = None   # near-top window for counting hits



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
  
    if "assigned_class" in out.columns:
      out["assigned_class"] = out["assigned_class"].astype(str)

    for col in ("barcode", "read_id", "genome"):
      if col in out.columns:
            out[col] = out[col].astype(str)
          
    for col in ("AS", "MAPQ", "NM"):
      if col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")

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
    if "assigned_class" in have:
        agg["assigned_class"] = "first"

    if not agg:
        return df.drop_duplicates(keys)[keys]

    return df.groupby(keys, observed=True).agg(agg).reset_index()


def _write_gzip_df(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False, compression="gzip")


def _unwrap_optioninfo(x):
    return x.default if isinstance(x, OptionInfo) else x


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

    # Winner-only
    if cfg.winner_only:
        for col in ("AS", "MAPQ", "NM"):
            if col not in df.columns:
                raise ValueError("winner_only=True requires AS, MAPQ, NM")

        df["_sort_as"] = -df["AS"].fillna(-9999)
        df["_sort_mapq"] = -df["MAPQ"].fillna(-1)
        df["_sort_nm"] = df["NM"].fillna(9999)

        df = df.sort_values(by=["_rid", "_sort_as", "_sort_mapq", "_sort_nm"], ascending=True)
        out = df.drop_duplicates(subset=["_rid"], keep="first").copy()

        denom = 1.0 + cfg.ambient_const
        out["L"] = 1.0 / denom
        out["L_amb"] = cfg.ambient_const / denom

        return out[["barcode", "read_id", "genome", "L", "L_amb"]].reset_index(drop=True)

    # Probabilistic
    df["_rid_code"] = df["_rid"].astype("category").cat.codes
    S_parts: List[np.ndarray] = []

    if "AS" in df.columns:
        S_as = df.groupby("_rid_code", observed=True)["AS"].transform(lambda s: _zscore_series(s.to_numpy(float)))
        S_parts.append(cfg.w_as * S_as.to_numpy(float))

    if "MAPQ" in df.columns:
        S_mq = df.groupby("_rid_code", observed=True)["MAPQ"].transform(lambda s: _zscore_series(s.to_numpy(float)))
        S_parts.append(cfg.w_mapq * S_mq.to_numpy(float))

    if "NM" in df.columns:
        S_nm = df.groupby("_rid_code", observed=True)["NM"].transform(lambda s: _zscore_series(s.to_numpy(float)))
        S_parts.append(-cfg.w_nm * S_nm.to_numpy(float))

    S = np.sum(S_parts, axis=0) if S_parts else np.zeros(len(df), dtype=float)

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
    C = Ldf.groupby(["barcode", "genome"], observed=True)["L"].sum().rename("C").reset_index()
    N = Ldf.groupby("barcode", observed=True)["read_id"].nunique().rename("n_reads").reset_index()
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


def _jsd(p: np.ndarray, q: np.ndarray, normalize: bool = True) -> float:
    eps = 1e-12
    p = np.clip(p.astype(float, copy=False), eps, None)
    q = np.clip(q.astype(float, copy=False), eps, None)
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    kl_pm = float(np.sum(p * np.log(p / m)))
    kl_qm = float(np.sum(q * np.log(q / m)))
    jsd = 0.5 * (kl_pm + kl_qm)
    if normalize:
        jsd = jsd / math.log(2.0)
    return jsd


def _barcode_jsd_from_mass(mass: pd.Series, eta: pd.Series, normalize: bool) -> float:
    genomes = sorted(set(map(str, mass.index)).union(set(map(str, eta.index))))
    if not genomes:
        return 0.0
    p = mass.reindex(genomes).fillna(0.0).to_numpy(float)
    q = eta.reindex(genomes).fillna(0.0).to_numpy(float)
    return _jsd(p, q, normalize=normalize)


def _filter_promiscuous_ambiguous_reads(
    df: pd.DataFrame, cfg: MergeConfig
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Drop entire (barcode, read_id) groups *only for ambiguous rows* if they have
    > cfg.max_hits genomes within cfg.hits_delta_as of best AS.

    Disabled unless BOTH cfg.max_hits and cfg.hits_delta_as are set.
    Returns: (filtered_df, stats_dict)
    """
    stats: Dict[str, Any] = {
        "enabled": False,
        "reason": "",
        "rows_in": int(len(df)),
        "rows_out": int(len(df)),
        "amb_rows_in": 0,
        "amb_rows_out": 0,
        "amb_read_groups_in": 0,
        "amb_read_groups_dropped": 0,
    }

    # opt-in only
    if cfg.max_hits is None or cfg.hits_delta_as is None:
        stats["reason"] = "disabled (max_hits or hits_delta_as not set)"
        return df, stats
    if cfg.max_hits <= 0 or cfg.hits_delta_as < 0:
        stats["reason"] = "disabled (invalid thresholds)"
        return df, stats
    if "assigned_class" not in df.columns:
        stats["reason"] = "disabled (missing assigned_class)"
        return df, stats
    if "AS" not in df.columns:
        stats["reason"] = "disabled (missing AS)"
        return df, stats

    stats["enabled"] = True

    cls = df["assigned_class"].astype(str).str.lower()
    amb_mask = (cls == "ambiguous")

    stats["amb_rows_in"] = int(amb_mask.sum())
    if stats["amb_rows_in"] == 0:
        stats["reason"] = "enabled but no ambiguous rows"
        return df, stats

    amb = df.loc[amb_mask, ["barcode", "read_id", "AS"]].copy()
  
    # per ambiguous read group
    grp = amb.groupby(["barcode", "read_id"], observed=True)

    # number of ambiguous read-groups present
    stats["amb_read_groups_in"] = int(grp.ngroups)

    # compute best AS per group
    #best_as = grp["AS"].transform("max")  # remove

    # count near-top hits per group (within delta of best)
    delta = int(cfg.hits_delta_as)
    n_near = grp["AS"].transform(lambda s: int((s >= (s.max() - delta)).sum()))

    # mark groups to drop
    to_drop = (n_near > int(cfg.max_hits))
    if not bool(to_drop.any()):
        stats["reason"] = "enabled but no groups exceeded threshold"
        stats["amb_rows_out"] = stats["amb_rows_in"]
        return df, stats

    drop_keys = amb.loc[to_drop, ["barcode", "read_id"]].drop_duplicates()
    stats["amb_read_groups_dropped"] = int(len(drop_keys))

    # drop only ambiguous rows for those keys
    tmp = df.merge(drop_keys.assign(_drop=True), on=["barcode", "read_id"], how="left")
    drop_row = tmp["_drop"].fillna(False).to_numpy(bool) & amb_mask.to_numpy(bool)

    out = df.loc[~drop_row].reset_index(drop=True)

    # stats out
    cls_out = out["assigned_class"].astype(str).str.lower()
    amb_out = (cls_out == "ambiguous")
    stats["rows_out"] = int(len(out))
    stats["amb_rows_out"] = int(amb_out.sum())
    stats["reason"] = "enabled and applied"

    return out, stats



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

    mass = L_block.groupby("genome", observed=True)["L"].sum().sort_values(ascending=False)
    tot = float(mass.sum())
    props = (mass / max(tot, 1e-12)).to_numpy()
    genomes_sorted = mass.index.to_list()

    jsd_to_eta = _barcode_jsd_from_mass(mass, eta, normalize=cfg.jsd_normalize)

    p1 = float(props[0]) if len(props) > 0 else 0.0
    p2 = float(props[1]) if len(props) > 1 else 0.0
    p3 = float(props[2]) if len(props) > 2 else 0.0
    top3_sum = p1 + p2 + p3

    ratio12 = (p1 / p2) if p2 > 0 else float("inf")
    top_genome = genomes_sorted[0] if genomes_sorted else None

    out: Dict[str, Any] = {
        "n_reads": read_count,
        "n_effective": tot,
        "p_top1": p1,
        "p_top2": p2,
        "p_top3": p3,
        "top3_sum": top3_sum,
        "jsd_to_eta": float(jsd_to_eta),
        "ratio_top1_top2": ratio12,
        "top_genome": top_genome,
        "call": "ambiguous_low_depth",
        "status_flag": "init",
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
    out["bic_empty"] = float(bic_empty)

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
                    best_s = {"model": "single", "g1": g1, "g2": None, "alpha": float(a), "rho": 1.0, "bic": float(bic)}
            evals.append(best_s)

        # Doublet
        if len(candidate_genomes) >= 2:
            for i in range(len(candidate_genomes)):
                for j in range(i + 1, len(candidate_genomes)):
                    g1, g2 = candidate_genomes[i], candidate_genomes[j]
                    best_d = {"bic": float("inf")}
                    for a in alphas:
                        for r in rhos:
                            ll = _loglik_for_params(L_block, eta, "doublet", g1, g2, float(a), float(r))
                            bic = _bic(ll, n_params=2, n_reads=read_count)
                            if bic < best_d["bic"]:
                                best_d = {
                                    "model": "doublet",
                                    "g1": g1,
                                    "g2": g2,
                                    "alpha": float(a),
                                    "rho": float(r),
                                    "bic": float(bic),
                                }
                    evals.append(best_d)

        single_fits = [e for e in evals if e.get("model") == "single"]
        doublet_fits = [e for e in evals if e.get("model") == "doublet"]

        best_single = min(single_fits, key=lambda x: x["bic"]) if single_fits else None
        best_doublet = min(doublet_fits, key=lambda x: x["bic"]) if doublet_fits else None

    out["bic_single"] = float(best_single["bic"]) if best_single else float("inf")
    out["bic_doublet"] = float(best_doublet["bic"]) if best_doublet else float("inf")

    # Best non-empty model
    if best_single and best_doublet:
        best_non_empty = best_single if best_single["bic"] <= best_doublet["bic"] else best_doublet
    else:
        best_non_empty = best_single or best_doublet

    # Low-depth fallback (IMPORTANT: prevents crash)
    if best_non_empty is None:
        out.update(
            {
                "call": "ambiguous_low_depth",
                "status_flag": "low_depth",
                "bic_best": float(bic_empty),
                "bic_best_non_empty": float("inf"),
                "delta_empty": float("inf"),
                "model": "empty" if math.isfinite(bic_empty) else None,
                "alpha": float("nan"),
                "rho": float("nan"),
            }
        )
        return out

    bic_non_empty = float(best_non_empty["bic"])
    out["bic_best_non_empty"] = float(bic_non_empty)
    out["delta_empty"] = float(bic_non_empty - bic_empty) if math.isfinite(bic_non_empty) else float("inf")

    # -------------------------
    # Empty gate (conservative) — uses JSD(p_b, eta) <= tau (if tau is set)
    # -------------------------
    is_empty_best = (bic_empty < bic_non_empty)
    bic_margin_ok = (bic_non_empty - bic_empty) >= cfg.empty_bic_margin
    weak_dominance = (p1 <= cfg.empty_top1_max) and (ratio12 <= cfg.empty_ratio12_max)
    ceiling_ok = (read_count <= cfg.empty_reads_max) if cfg.empty_reads_max is not None else True

    jsd_ok = True
    if cfg.empty_jsd_max is not None and math.isfinite(float(cfg.empty_jsd_max)):
        jsd_ok = (jsd_to_eta <= float(cfg.empty_jsd_max))

    if is_empty_best and bic_margin_ok and weak_dominance and ceiling_ok and jsd_ok:
        out.update(
            {
                "call": "empty",
                "status_flag": "empty",
                "bic_best": float(bic_empty),
                "model": "empty",
                "alpha": 1.0,
                "rho": float("nan"),
            }
        )
        return out

    # -------------------------
    # Non-empty: compute purity/minor
    # -------------------------
    model = str(best_non_empty["model"])
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
            "alpha": float(alpha),
            "rho": float(rho),
            "purity": float(purity),
            "minor": float(minor),
            "bic_best": float(best_non_empty["bic"]),
        }
    )

    # Convenience
    bic_s = float(out["bic_single"])
    bic_d = float(out["bic_doublet"])
    delta_sd = bic_d - bic_s  # negative => doublet preferred
    out["delta_math"] = float(delta_sd)

    # ---- 1) confident doublet ----
    is_doublet_pref = (model == "doublet") and math.isfinite(bic_d) and math.isfinite(bic_s) and (bic_d <= bic_s)
    doublet_margin_ok = (bic_s - bic_d) >= cfg.bic_margin
    doublet_minor_ok = (out["minor"] >= cfg.doublet_minor_min)

    if is_doublet_pref and doublet_margin_ok and doublet_minor_ok:
        out.update({"call": "doublet_confident", "status_flag": "doublet"})
        return out

    # ---- 2) indistinguishable (near tie) ----
    is_near_tie = (
        math.isfinite(bic_s)
        and math.isfinite(bic_d)
        and abs(bic_d - bic_s) <= cfg.near_tie_margin
        and (ratio12 < cfg.ratio_top1_top2_min)
        and (p1 >= cfg.single_mass_min)
    )
    if is_near_tie:
        out.update(
            {
                "call": "indistinguishable",
                "status_flag": "near_tie",
                "indistinguishable_set": f"{genomes_sorted[0]},{genomes_sorted[1]}" if len(genomes_sorted) >= 2 else "",
            }
        )
        return out

    # ---- 3) clean singlet ----
    if (ratio12 >= cfg.ratio_top1_top2_min) and (out["purity"] >= cfg.single_mass_min):
        out.update({"call": "single_clean", "status_flag": "single"})
        return out

    # ---- 4) ambiguous subclasses ----
    if is_doublet_pref and (not doublet_minor_ok or not doublet_margin_ok):
        out.update({"call": "ambiguous_weak_doublet", "status_flag": "weak_doublet"})
        return out

    single_pref = (math.isfinite(bic_s) and (not math.isfinite(bic_d) or (bic_s <= bic_d)))
    if single_pref:
        out.update({"call": "ambiguous_dirty_singlet", "status_flag": "dirty_singlet"})
        return out

    out.update({"call": "ambiguous_low_depth", "status_flag": "other"})
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
        # defensive: in case of concatenated gzip members each with header
        if "barcode" in chunk.columns:
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


def _pass1_process_one_file(fp: Path, cfg: MergeConfig, shard_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    handles = _open_shard_handles(shard_dir, cfg.shards)
    headers = {i: False for i in range(cfg.shards)}
    C_parts: List[pd.DataFrame] = []
    N_parts: List[pd.DataFrame] = []

    filt_tot = {
        "enabled": False,
        "chunks": 0,
        "rows_in": 0,
        "rows_out": 0,
        "amb_rows_in": 0,
        "amb_rows_out": 0,
        "amb_read_groups_in": 0,
        "amb_read_groups_dropped": 0,
    }

    try:
        it = pd.read_csv(fp, sep="\t", chunksize=cfg.chunk_rows, low_memory=False)
        for raw in it:
            df = _coerce_assign_schema(raw)
            df = _reduce_alignments_to_per_genome(df).dropna(subset=["barcode", "read_id", "genome"])
            if df.empty:
                continue

            df, st = _filter_promiscuous_ambiguous_reads(df, cfg)

            filt_tot["chunks"] += 1
            filt_tot["enabled"] = filt_tot["enabled"] or bool(st.get("enabled", False))
            filt_tot["rows_in"] += int(st.get("rows_in", 0))
            filt_tot["rows_out"] += int(st.get("rows_out", 0))
            filt_tot["amb_rows_in"] += int(st.get("amb_rows_in", 0))
            filt_tot["amb_rows_out"] += int(st.get("amb_rows_out", 0))
            filt_tot["amb_read_groups_in"] += int(st.get("amb_read_groups_in", 0))
            filt_tot["amb_read_groups_dropped"] += int(st.get("amb_read_groups_dropped", 0))

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

    if filt_tot["enabled"]:
      amb_in = filt_tot["amb_rows_in"]
      amb_out = filt_tot["amb_rows_out"]
      drop_groups = filt_tot["amb_read_groups_dropped"]
      grp_in = max(filt_tot["amb_read_groups_in"], 1)

      frac_amb_dropped = (amb_in - amb_out) / amb_in if amb_in > 0 else 0.0
      frac_groups_dropped = drop_groups / grp_in if grp_in > 0 else 0.0

    if drop_groups > 0:
        typer.echo(
            f"[max-hits] file={fp.name} ΔAS={cfg.hits_delta_as} max_hits={cfg.max_hits} "
            f"amb_rows {amb_in:,}->{amb_out:,} (drop {frac_amb_dropped:.1%}) "
            f"amb_read_groups dropped {drop_groups:,}/{grp_in:,} ({frac_groups_dropped:.1%})"
        )

    if C_parts:
        C_one = pd.concat(C_parts, ignore_index=True).groupby(["barcode", "genome"], observed=True)["C"].sum().reset_index()
    else:
        C_one = pd.DataFrame(columns=["barcode", "genome", "C"])
    if N_parts:
        N_one = pd.concat(N_parts, ignore_index=True).groupby("barcode", observed=True)["n_reads"].sum().reset_index()
    else:
        N_one = pd.DataFrame(columns=["barcode", "n_reads"])

    return C_one, N_one

def _pass1_job(i: int, fp: Path, cfg: MergeConfig, worker_root: Path) -> Tuple[int, pd.DataFrame, pd.DataFrame]:
  wd = worker_root / f"w{i:03d}"
  wd.mkdir(parents=True, exist_ok=True)
  C_one, N_one = _pass1_process_one_file(fp, cfg, wd)
  return i, C_one, N_one

def _merge_worker_shards(worker_dirs: List[Path], out_shard_dir: Path, shards: int) -> None:
    out_shard_dir.mkdir(parents=True, exist_ok=True)
    for si in range(shards):
        out_fp = out_shard_dir / f"shard_{si:02d}.tsv.gz"
        if out_fp.exists():
            out_fp.unlink()

        # Concatenating gzip members is valid; pandas can read it.
        with open(out_fp, "ab") as w:
            for wd in worker_dirs:
                part = wd / f"shard_{si:02d}.tsv.gz"
                if part.exists() and part.stat().st_size > 0:
                    with open(part, "rb") as r:
                        shutil.copyfileobj(r, w, length=1024 * 1024)


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
            futures = [ex.submit(_pass1_job, i, fp, cfg, worker_root) for i, fp in enumerate(files)]
            for fut in tqdm(futures, desc=f"[pass1] files (workers={workers})"):
                _, C_one, N_one = fut.result()
                C_parts.append(C_one)
                N_parts.append(N_one)

    if C_parts:
        C_all = pd.concat(C_parts, ignore_index=True).groupby(["barcode", "genome"], observed=True)["C"].sum().reset_index()
    else:
        C_all = pd.DataFrame(columns=["barcode", "genome", "C"])

    if N_parts:
        N_all = pd.concat(N_parts, ignore_index=True).groupby("barcode", observed=True)["n_reads"].sum().reset_index()
    else:
        N_all = pd.DataFrame(columns=["barcode", "n_reads"])

    shard_dir = tmp_dir / "L_shards"
    worker_dirs = sorted([p for p in worker_root.glob("w*") if p.is_dir()])
    _merge_worker_shards(worker_dirs, shard_dir, cfg.shards)

    return C_all, N_all, shard_dir


def _pass2_worker(args) -> List[Dict[str, Any]]:
    shard_fp, cfg, topk, eta = args
    rows: List[Dict[str, Any]] = []
    blocks: Dict[str, List[pd.DataFrame]] = {}

    for chunk in _iter_shard_rows(shard_fp, cfg.pass2_chunksize):
        for bc, sub in chunk.groupby("barcode", sort=False, observed=True):
            blocks.setdefault(str(bc), []).append(sub)

    for bc, parts in blocks.items():
        L_block = pd.concat(parts, ignore_index=True)
        if L_block.empty:
            continue
        cand = topk.get(bc, [])
        res = _select_model_for_barcode(L_block, eta, cfg, cand)
        res["barcode"] = bc
        rows.append(res)

    return rows


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
    # filter non-informative reads
    max_hits: Optional[int] = typer.Option(None,
        help="(Optional) Drop promiscuous reads among assigned_class=ambiguous only. "
             "Enabled only if BOTH --max-hits and --hits-delta-as are provided. "
             "A read is dropped if >max-hits genomes fall within (best_AS - hits-delta-as)."),
    hits_delta_as: Optional[int] = typer.Option(None,
        help="(Optional) AS window for counting near-top hits for --max-hits filtering. "
             "Enabled only if BOTH --max-hits and --hits-delta-as are provided."),
    # Fusion
    beta: float = typer.Option(0.5, help="Softmax temperature (probabilistic mode)."),
    w_as: float = typer.Option(0.5, help="Weight for AS (probabilistic mode)."),
    w_mapq: float = typer.Option(1.0, help="Weight for MAPQ (probabilistic mode)."),
    w_nm: float = typer.Option(1.0, help="Weight for NM penalty (probabilistic mode)."),
    ambient_const: float = typer.Option(1e-3, help="Per-read ambient mass."),
    # Empty gates (BIC + structure)
    empty_bic_margin: float = typer.Option(10.0, help="Min ΔBIC (non-empty - empty) to call empty."),
    empty_top1_max: float = typer.Option(0.6, help="Max top1 mass to allow empty."),
    empty_ratio12_max: float = typer.Option(1.5, help="Max top1/top2 ratio to allow empty."),
    empty_reads_max: Optional[int] = typer.Option(None, help="Optional ceiling for empty calls."),
    # Empty (JSD) gates
    empty_seed_bic_min: float = typer.Option(20.0, help="ΔBIC-only empty seeds for learning tau (non-circular)."),
    empty_tau_quantile: float = typer.Option(0.95, help="Quantile of seed JSD used as tau."),
    empty_jsd_max: Optional[float] = typer.Option(None, help="Override tau; if set, do not learn."),
    jsd_normalize: bool = typer.Option(True, help="Normalize JSD by log(2) to [0,1]."),
    # Doublet gates
    bic_margin: float = typer.Option(6.0, help="ΔBIC required to accept doublet over single."),
    doublet_minor_min: float = typer.Option(0.20, help="Min minor fraction for doublet."),
    # Ambient iteration
    eta_iters: int = typer.Option(2, help="Iterations for ambient refinement."),
    eta_seed_quantile: float = typer.Option(0.02, help="Seed eta from bottom quantile of n_reads."),
    topk_genomes: int = typer.Option(3, help="Top-K candidate genomes per barcode."),
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
        empty_reads_max=_unwrap_optioninfo(empty_reads_max),
        empty_seed_bic_min=_unwrap_optioninfo(empty_seed_bic_min),
        empty_tau_quantile=_unwrap_optioninfo(empty_tau_quantile),
        empty_jsd_max=_unwrap_optioninfo(empty_jsd_max),
        jsd_normalize=_unwrap_optioninfo(jsd_normalize),
        bic_margin=_unwrap_optioninfo(bic_margin),
        doublet_minor_min=_unwrap_optioninfo(doublet_minor_min),
        eta_iters=_unwrap_optioninfo(eta_iters),
        eta_seed_quantile=_unwrap_optioninfo(eta_seed_quantile),
        topk_genomes=topk_genomes,
        max_hits=max_hits,
        hits_delta_as=hits_delta_as,
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
        return eta_s / eta_s.sum()

    # seed eta from low-depth tail
    typer.echo(f"[2/5] Ambient refinement (eta): seed from bottom quantile={cfg.eta_seed_quantile}")
    if N_all.empty:
        eta = pd.Series({g: 1.0 / len(all_genomes) for g in all_genomes}) if all_genomes else pd.Series(dtype=float)
    else:
        cutoff = float(N_all["n_reads"].quantile(cfg.eta_seed_quantile))
        seed_bcs = N_all[N_all["n_reads"] <= max(cutoff, 10)]["barcode"].astype(str).tolist()
        eta = _compute_eta(seed_bcs)

    shard_files = sorted(shard_dir.glob("*.tsv.gz"))

    # iterative refinement using ΔBIC-only empty seeds
    typer.echo(f"[3/5] Iterating eta using ΔBIC-only empty seeds (iters={cfg.eta_iters})")
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
        if df_iter.empty:
            typer.echo("  [warn] Empty results during eta refinement; stopping.")
            break

        df_iter["bic_non_empty"] = np.minimum(df_iter["bic_single"].to_numpy(float), df_iter["bic_doublet"].to_numpy(float))
        df_iter["delta_empty"] = df_iter["bic_non_empty"] - df_iter["bic_empty"]

        empty_bcs = df_iter.loc[
            (df_iter["bic_empty"] < df_iter["bic_non_empty"]) &
            (df_iter["delta_empty"] >= cfg.empty_seed_bic_min),
            "barcode"
        ].astype(str).tolist()

        if len(empty_bcs) < 10:
            typer.echo("  [warn] Too few empty SEEDS to refine eta; stopping refinement.")
            break

        eta = _compute_eta(empty_bcs)
        eta.to_json(outdir / f"{sample}_eta_iter{i+1}.json")

    eta.to_json(outdir / f"{sample}_eta_final.json")

    # Learn tau if not overridden
    if cfg.empty_jsd_max is None:
        typer.echo("[3.5/5] Learning empty JSD threshold (tau) from ΔBIC-only empty seeds")
        args = [(fp, cfg, topk, eta) for fp in shard_files]
        rows_seed: List[Dict[str, Any]] = []

        if threads <= 1:
            for a in tqdm(args, desc="[tau] shards"):
                rows_seed.extend(_pass2_worker(a))
        else:
            with ProcessPoolExecutor(max_workers=threads) as ex:
                for rr in tqdm(ex.map(_pass2_worker, args), total=len(args), desc="[tau] shards"):
                    rows_seed.extend(rr)

        df_seed = pd.DataFrame(rows_seed)
        if not df_seed.empty:
            df_seed["bic_non_empty"] = np.minimum(df_seed["bic_single"].to_numpy(float), df_seed["bic_doublet"].to_numpy(float))
            df_seed["delta_empty"] = df_seed["bic_non_empty"] - df_seed["bic_empty"]

            seeds = df_seed[
                (df_seed["bic_empty"] < df_seed["bic_non_empty"]) &
                (df_seed["delta_empty"] >= cfg.empty_seed_bic_min) &
                (df_seed["jsd_to_eta"].notna())
            ].copy()

            if cfg.empty_reads_max is not None:
                seeds = seeds[seeds["n_reads"] <= cfg.empty_reads_max]

            if len(seeds) >= 50:
                tau = float(seeds["jsd_to_eta"].quantile(cfg.empty_tau_quantile))
                cfg.empty_jsd_max = tau
                (outdir / f"{sample}_empty_jsd_tau.json").write_text(
                    pd.Series(
                        {"tau": tau, "quantile": cfg.empty_tau_quantile, "seed_bic_min": cfg.empty_seed_bic_min}
                    ).to_json()
                )
                typer.echo(f"  learned tau={tau:.4g} from n={len(seeds)} seed empties")
            else:
                typer.echo(f"  [warn] Too few seed empties to learn tau (n={len(seeds)}). Proceeding without JSD gate.")
        else:
            typer.echo("  [warn] Seed pass produced empty dataframe; proceeding without JSD gate.")

    # Final pass2
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

    _write_gzip_df(calls, outdir / f"{sample}_cells_calls.tsv.gz")

    # legacy PASS: keep “real cells” + dirty singlets (for decontam)
    legacy = calls.copy()
    legacy["AssignedGenome"] = legacy["best_genome"].fillna("")

    pass_mask = legacy["call"].isin(
        ["single_clean", "doublet_confident", "indistinguishable", "ambiguous_dirty_singlet"]
    )

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
