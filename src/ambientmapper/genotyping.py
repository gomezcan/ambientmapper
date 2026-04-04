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

Fixes included (based on your traceback):
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
import os
import shutil
import subprocess
import numpy as np
import pandas as pd
import typer

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Iterator, Callable
from pydantic import BaseModel, ConfigDict

try:
    import duckdb as _duckdb
    _HAS_DUCKDB = True
except ImportError:
    _duckdb = None  # type: ignore
    _HAS_DUCKDB = False

app = typer.Typer(add_completion=False, no_args_is_help=True)

# Environment keys to keep when spawning subprocesses (sort, gzip, zcat, bash).
# Full SLURM environments can exceed ARG_MAX and cause E2BIG / "Argument list too long".
_SUBPROCESS_KEEP_ENV_KEYS = frozenset({
    "PATH", "HOME", "USER", "TMPDIR", "TEMP", "TMP",
    "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH",
    "CONDA_PREFIX", "CONDA_DEFAULT_ENV", "CONDA_EXE",
})

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

    # Pass 1.5 merge control
    pass15_use_external_sort: bool = True
    pass15_sort_mem: str = "4G"       # passed to sort -S
    pass15_tmpdir: Optional[str] = None  # passed to sort -T (defaults to merged_root)

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
    topk_reclass: bool = False  # Enable Pass 2.75 top-K re-classification

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
    
    # Reliability weights (applied to L and L_amb)
    w_confident: float = 1.0
    w_ambiguous: float = 1.0

    # DuckDB acceleration
    pass1_duckdb: bool = True   # Use DuckDB for Pass 1 file reading + reduction
    pass2_duckdb: bool = True   # Use DuckDB for Pass 2 eta computation

    # Read quality filters (applied before posterior computation)
    min_mapq_genotyping: int = 0    # MAPQ < this → exclude row (0 = disabled)
    max_xa_genotyping: int = -1     # XAcount > this → exclude row (-1 = disabled)

    # Cross-mapping-aware singlet (Level 2)
    xmap_chi_threshold: float = 0.05   # χ(g1) above which Level 2 activates
    xmap_phi_min_reads: int = 500      # min reads for D(g1) subset
    xmap_phi_iters: int = 2            # φ refinement iterations
    xmap_phi_iter_eps: float = 0.01    # convergence tolerance
    xmap_phi_ratio12_min: float = 2.0  # ratio12 gate for D(g1)
    xmap_enabled: bool = True          # set False via --no-xmap to disable


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

    if "XAcount" in out.columns:
        out["XAcount"] = pd.to_numeric(out["XAcount"], errors="coerce").fillna(0).astype(int)

    return out


def _reduce_alignments_to_per_genome(df: pd.DataFrame) -> pd.DataFrame:
    keys = ["barcode", "read_id", "genome"]
    def _agg_assigned_class(s: pd.Series) -> str:
        # robust: if any row is ambiguous, treat the genome-hit as ambiguous
        x = set(s.dropna().astype(str).tolist())
        return "ambiguous" if "ambiguous" in x else (next(iter(x)) if x else "ambiguous")

    agg: Dict[str, Callable] = {
        "AS": "max",
        "MAPQ": "max",
        "NM": "min",
        "assigned_class": _agg_assigned_class,
    }
    if "XAcount" in df.columns:
        agg["XAcount"] = "max"
    return df.groupby(keys, observed=True, sort=False).agg(agg).reset_index()

def _reclass_within_topk(df: pd.DataFrame, cfg: MergeConfig) -> pd.DataFrame:
    """Re-classify reads as winner/ambiguous within the top-K genome context.

    Uses a three-tier priority system:

    Tier 1 — Original winner:  If a read has exactly 1 genome with
        ``assigned_class == "winner"`` in the top-K set, preserve it.
        The assign step already proved this read belongs to that genome.

    Tier 2 — Rescued flag:  For reads not resolved by Tier 1, count genomes
        with ``assigned_class == "rescued"`` (spatial proximity to a winner
        in the same barcode/genome, set by ``_friend_rescue`` in assign).
        If exactly 1 top-K genome is rescued → promote to winner.
        If 0 or 2+ → fall through (non-discriminative).

    Tier 3 — Composite score:  For remaining reads, compute
        ``score = AS * w_as + MAPQ * w_mapq - NM * w_nm``
        (same formula as ``_compute_read_posteriors``).
        If exactly 1 genome has the best score → winner; ties → ambiguous.
    """
    if df.empty or not all(c in df.columns for c in ("AS", "MAPQ", "NM")):
        return df

    rid = df["barcode"].astype(str) + "::" + df["read_id"].astype(str)
    codes = rid.astype("category")
    cat_codes = codes.cat.codes.to_numpy(np.int32)

    n_reads = int(cat_codes.max()) + 1 if cat_codes.size else 0
    if n_reads == 0:
        return df

    orig_class = df["assigned_class"].fillna("ambiguous").astype(str).to_numpy()

    # --- Tier 1: Original winner status ---
    is_winner = (orig_class == "winner").astype(np.int32)
    n_winner = np.zeros(n_reads, dtype=np.int32)
    np.add.at(n_winner, cat_codes, is_winner)
    tier1_resolved = (n_winner[cat_codes] == 1)

    # --- Tier 2: Rescued flag (per-genome spatial proximity evidence) ---
    is_rescued = (orig_class == "rescued").astype(np.int32)
    n_rescued = np.zeros(n_reads, dtype=np.int32)
    np.add.at(n_rescued, cat_codes, is_rescued)
    tier2_resolved = ~tier1_resolved & (n_rescued[cat_codes] == 1)

    # --- Tier 3: Composite score (AS + MAPQ - NM) ---
    as_arr = df["AS"].to_numpy(np.float32)
    mq_arr = df["MAPQ"].to_numpy(np.float32)
    nm_arr = df["NM"].to_numpy(np.float32)
    score = as_arr * float(cfg.w_as) + mq_arr * float(cfg.w_mapq) - nm_arr * float(cfg.w_nm)

    needs_tier3 = ~(tier1_resolved | tier2_resolved)
    t3_score = np.where(needs_tier3, score, -np.inf)
    best_score = np.full(n_reads, -np.inf, dtype=np.float32)
    np.maximum.at(best_score, cat_codes, t3_score)

    at_best = (needs_tier3 & (score == best_score[cat_codes])).astype(np.int32)
    n_at_best = np.zeros(n_reads, dtype=np.int32)
    np.add.at(n_at_best, cat_codes, at_best)
    tier3_winner = needs_tier3 & (score == best_score[cat_codes]) & (n_at_best[cat_codes] == 1)

    # --- Combine all tiers ---
    is_final_winner = (
        (tier1_resolved & (is_winner == 1))        # Tier 1: original winner row
        | (tier2_resolved & (is_rescued == 1))      # Tier 2: single rescued row
        | tier3_winner                               # Tier 3: unique best score
    )
    new_class = np.where(is_final_winner, "winner", "ambiguous")

    df = df.copy()
    df["assigned_class"] = new_class
    return df


def shlex_quote(s: str) -> str:
    import shlex
    return shlex.quote(s)

def _merge_worker_shards_pass15(
    shard_root: Path,
    shards: int,
    *,
    force: bool = False,
    sort_mem: str = "4G",
    tmpdir: Optional[Path] = None,
) -> Path:
    """
    Merge worker shards (w*/shard_XX.tsv.gz) -> merged/shard_XX.tsv.gz, globally sorted.

    Sorting key: barcode (col1), read_id (col2), genome (col3).
    Assumes every worker shard has a header line (written lazily only if rows exist).
    Output shard has a single header, followed by globally sorted rows.

    Returns merged_root path.
    """
    merged_root = shard_root / "merged"
    merged_root.mkdir(parents=True, exist_ok=True)
    marker = merged_root / "_MERGED_SORTED_v1.txt"

    # --- RECOVERY / RESUME LOGIC ---
    if not any(shard_root.glob("w*/shard_*.tsv.gz")):
        if any(p.stat().st_size >= 64 for p in merged_root.glob("shard_*.tsv.gz")):
            if not marker.exists():
                marker.write_text("v1\n")
            return merged_root
        raise RuntimeError("Pass 1.5: No worker shards and no valid merged shards found.")

    if marker.exists() and not force:
        if any(p.stat().st_size >= 64 for p in merged_root.glob("shard_*.tsv.gz")):
            return merged_root

    # Tool availability checks
    sort_bin = shutil.which("sort")
    gzip_bin = shutil.which("gzip")
    zcat_bin = shutil.which("zcat") or shutil.which("gzcat")
    bash = shutil.which("bash") or "/bin/bash"
    if sort_bin is None or gzip_bin is None or zcat_bin is None:
        raise RuntimeError("External tools (sort, gzip, zcat/gzcat) not found in PATH.")

    tmpdir_eff = Path(tmpdir) if tmpdir is not None else merged_root
    tmpdir_eff.mkdir(parents=True, exist_ok=True)

    # Sorting command (body only)
    sort_cmd = [
        sort_bin,
        "-t", "\t",
        "-k1,1",
        "-k2,2",
        "-k3,3",
        "-S", str(sort_mem),
        "-T", str(tmpdir_eff),
    ]

    # Environment: keep only essential vars to avoid ARG_MAX / E2BIG on HPC (SLURM envs can be huge)
    env = {k: v for k, v in os.environ.items() if k in _SUBPROCESS_KEEP_ENV_KEYS}
    env["LC_ALL"] = "C"

    # Merge each shard id independently
    for sid in range(int(shards)):
        inputs = sorted(shard_root.glob(f"w*/shard_{sid:02d}.tsv.gz"))
        if not inputs:
            continue

        out_path = merged_root / f"shard_{sid:02d}.tsv.gz"
        if out_path.exists() and not force:
            continue

        # Filter out trivially empty gzip containers (~38 bytes) or similar
        inputs_eff = [p for p in inputs if p.exists() and p.stat().st_size >= 64]
        if not inputs_eff:
            if out_path.exists():
                out_path.unlink()
            continue

        # Validate gzip integrity — skip corrupt shards with a warning
        valid_inputs: List[Path] = []
        for p in inputs_eff:
            rc = subprocess.call(
                [gzip_bin, "-t", str(p)], env=env,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )
            if rc == 0:
                valid_inputs.append(p)
            else:
                typer.echo(f"WARNING: skipping corrupt shard {p} (gzip -t rc={rc})")
        inputs_eff = valid_inputs
        if not inputs_eff:
            if out_path.exists():
                out_path.unlink()
            continue

        # 1) Read header from first non-empty input
        first = str(inputs_eff[0])
        header_cmd = f'{zcat_bin} {shlex_quote(first)} | head -n 1 || true'
        header = subprocess.check_output([bash, "-lc", header_cmd], env=env, text=True)
        header = header.rstrip("\n") + "\n"
        if header.strip() == "":
            # Defensive: if header retrieval failed, avoid producing malformed output
            raise RuntimeError(f"Pass 1.5: failed to read header for shard_{sid:02d} from {first}")

        # 2) Write header as gzip member (member 1)
        try:
            with open(out_path, "wb") as out_fh:
                pH = subprocess.Popen([gzip_bin, "-c"], stdin=subprocess.PIPE, stdout=out_fh, env=env)
                assert pH.stdin is not None
                pH.stdin.write(header.encode("utf-8"))
                pH.stdin.close()
                rcH = pH.wait()
            if rcH != 0:
                raise RuntimeError(f"Pass 1.5 header gzip failed: rc={rcH}")
        except Exception:
            if out_path.exists():
                out_path.unlink()
            raise

        # 3) Stream BODY rows (no headers) from all inputs -> sort -> gzip append (member 2)
        # Write file list to a temp file to avoid ARG_MAX / E2BIG when there are
        # thousands of worker shards (one per input file).
        filelist_path = merged_root / f"_filelist_shard_{sid:02d}.tmp"
        filelist_path.write_text("\n".join(str(p) for p in inputs_eff) + "\n")

        # NOTE: zcat|tail may get SIGPIPE if downstream finishes early; we tolerate rc1==141
        cmd = f"""
set -euo pipefail
while IFS= read -r f; do
  {zcat_bin} "$f" | tail -n +2
done < {shlex_quote(str(filelist_path))}
""".strip()

        try:
            with open(out_path, "ab") as out_fh:
                p1 = subprocess.Popen([bash, "-lc", cmd], stdout=subprocess.PIPE, env=env)
                p2 = subprocess.Popen(sort_cmd, stdin=p1.stdout, stdout=subprocess.PIPE, env=env)
                p3 = subprocess.Popen([gzip_bin, "-c"], stdin=p2.stdout, stdout=out_fh, env=env)

                assert p1.stdout is not None
                assert p2.stdout is not None
                p1.stdout.close()
                p2.stdout.close()

                rc3 = p3.wait()
                rc2 = p2.wait()
                rc1 = p1.wait()

            ok_rc1 = (0, 141)  # 141 == 128 + SIGPIPE(13)
            if rc1 not in ok_rc1 or rc2 != 0 or rc3 != 0:
                raise RuntimeError(
                    f"Pass 1.5 failed for shard_{sid:02d}: rc1={rc1} rc2={rc2} rc3={rc3}"
                )
        except Exception:
            # cleanup partial output
            if out_path.exists():
                out_path.unlink()
            raise
        finally:
            if filelist_path.exists():
                filelist_path.unlink()

    marker.write_text("v1\n")
    return merged_root


def _compute_N_all_from_merged_shards(merged_root: Path, cfg: MergeConfig) -> pd.DataFrame:
    """
    Exact per-barcode unique read_id counts from merged shards.
    Requires merged shards sorted by (barcode, read_id, genome).

    This is exact and avoids chunk-boundary double counting.
    """
    rows: List[Tuple[str, int]] = []

    for shard in sorted(merged_root.glob("shard_*.tsv.gz")):
        prev_bc: Optional[str] = None
        prev_rid: Optional[str] = None
        cur_count = 0

        for chunk in _iter_shard_chunks(shard, cfg):
            if chunk is None or chunk.empty:
                continue

            b = chunk["barcode"].astype(str).to_numpy()
            r = chunk["read_id"].astype(str).to_numpy()

            # determine "new read" boundaries within the chunk
            new_bc = np.empty(len(b), dtype=bool)
            new_bc[0] = True
            new_bc[1:] = b[1:] != b[:-1]

            new_rid = np.empty(len(r), dtype=bool)
            new_rid[0] = True
            new_rid[1:] = r[1:] != r[:-1]

            new_read = new_bc | new_rid

            # fix the first row of this chunk based on the previous chunk tail
            if prev_bc is not None:
                if b[0] == prev_bc and r[0] == prev_rid:
                    new_read[0] = False
                else:
                    new_read[0] = True

            # accumulate counts per barcode, flushing when barcode changes
            # we iterate only over barcode change points (not every row)
            change_idx = np.flatnonzero(new_bc)
            for k in range(len(change_idx)):
                start = int(change_idx[k])
                end = int(change_idx[k + 1]) if k + 1 < len(change_idx) else len(b)

                bc_k = b[start]

                # flush previous barcode when moving to a new one
                if prev_bc is not None and bc_k != prev_bc:
                    rows.append((prev_bc, int(cur_count)))
                    cur_count = 0

                # count new reads in this barcode segment
                cur_count += int(new_read[start:end].sum())

                prev_bc = bc_k
                prev_rid = r[end - 1]

        # flush last barcode seen in this shard
        if prev_bc is not None:
            rows.append((prev_bc, int(cur_count)))

    if not rows:
        return pd.DataFrame(columns=["barcode", "n_reads"])

    out = pd.DataFrame(rows, columns=["barcode", "n_reads"])
    # Each barcode routes to exactly one shard, but be defensive:
    out = out.groupby("barcode", sort=False, as_index=False)["n_reads"].sum()
    out["barcode"] = out["barcode"].astype("string")
    out["n_reads"] = out["n_reads"].astype("int64")
    return out

# --------------------------------------------------------------------------------------
# Promiscuous ambiguous filter (NumPy)
# --------------------------------------------------------------------------------------

def _filter_promiscuous_ambiguous_reads(df: pd.DataFrame, cfg: MergeConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    if cfg.max_hits is None or cfg.hits_delta_mapq is None:
        return df, {"enabled": False}

    assigned = df["assigned_class"].fillna("").astype(str).to_numpy()
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


def _filter_read_quality(df: pd.DataFrame, cfg: MergeConfig) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Exclude reads below MAPQ threshold or with within-genome multi-mapping (XA > max)."""
    if cfg.min_mapq_genotyping <= 0 and cfg.max_xa_genotyping < 0:
        return df, {"enabled": False}

    n_before = len(df)
    mask = np.ones(n_before, dtype=bool)

    n_dropped_mapq = 0
    if cfg.min_mapq_genotyping > 0:
        mapq = df["MAPQ"].to_numpy(dtype=np.float32)
        mapq_fail = mapq < cfg.min_mapq_genotyping
        n_dropped_mapq = int(mapq_fail.sum())
        mask &= ~mapq_fail

    n_dropped_xa = 0
    if cfg.max_xa_genotyping >= 0 and "XAcount" in df.columns:
        xa = df["XAcount"].to_numpy(dtype=np.float32)
        xa_fail = xa > cfg.max_xa_genotyping
        n_dropped_xa = int(xa_fail.sum())
        mask &= ~xa_fail

    n_after = int(mask.sum())
    return df.loc[mask].copy(), {
        "enabled": True,
        "n_before": n_before,
        "n_after": n_after,
        "n_dropped_mapq": n_dropped_mapq,
        "n_dropped_xa": n_dropped_xa,
    }


# --------------------------------------------------------------------------------------
# Core: posterior computation
# --------------------------------------------------------------------------------------

def _compute_read_posteriors(df: pd.DataFrame, cfg: MergeConfig) -> pd.DataFrame:
    df = df[["barcode", "read_id", "genome", "AS", "MAPQ", "NM", "assigned_class"]].copy()
    df = df.dropna(subset=["barcode", "read_id", "genome"])

    AS = df["AS"].to_numpy(np.float32, copy=False)
    MQ = df["MAPQ"].to_numpy(np.float32, copy=False)
    NM = df["NM"].to_numpy(np.float32, copy=False)

    rid = df["barcode"].astype(str) + "::" + df["read_id"].astype(str)
    codes = rid.astype("category").cat.codes.to_numpy(np.int32)

    score = (AS * float(cfg.w_as)) + (MQ * float(cfg.w_mapq)) - (NM * float(cfg.w_nm))
    # Read-level weight: ambiguous reads contribute less (does not drop any reads)
    cls = df["assigned_class"].fillna("").astype(str).to_numpy()
    w_row = np.where(cls == "ambiguous", float(cfg.w_ambiguous), float(cfg.w_confident)).astype(np.float32)
    # Ensure per-read weight is "ambiguous if any ambiguous hit" (max over rows for that read)
    n_codes0 = int(codes.max()) + 1 if codes.size else 0
    w_read = np.zeros(n_codes0, dtype=np.float32)
    np.maximum.at(w_read, codes, w_row)
  
    if cfg.winner_only:
        df["_code"] = codes
        df["_score"] = score
        df = df.sort_values(["_code", "_score"], ascending=[True, False]).drop_duplicates("_code", keep="first")
        # pick read-weight for the chosen winner row
        codes2 = (df["barcode"].astype(str) + "::" + df["read_id"].astype(str)).astype("category").cat.codes.to_numpy(np.int32)
        w2 = w_read[codes2].astype(np.float32, copy=False)
        df["L"] = (1.0 * w2).astype(np.float32)
        df["L_amb"] = (float(cfg.ambient_const) * w2).astype(np.float32)
        df["w_read"] = w2
        return df[["barcode", "read_id", "genome", "L", "L_amb", "w_read"]]        
                
    n_codes = int(codes.max()) + 1 if codes.size else 0
    max_per = np.full(n_codes, -np.inf, dtype=np.float32)
    np.maximum.at(max_per, codes, score)
    s = score - max_per[codes]

    w = np.exp(np.clip(float(cfg.beta) * s, -50, 50)).astype(np.float32)
    denom = np.bincount(codes, weights=w).astype(np.float32) + float(cfg.ambient_const)
    L = (w / denom[codes]).astype(np.float32, copy=False)
    Lamb = (float(cfg.ambient_const) / denom[codes]).astype(np.float32, copy=False)
    wr = w_read[codes].astype(np.float32, copy=False)
    
    df["L"] = L * wr
    df["L_amb"] = Lamb * wr
    df["w_read"] = wr
    return df[["barcode", "read_id", "genome", "L", "L_amb", "w_read"]]
  
# --------------------------------------------------------------------------------------
# Likelihood / BIC / JSD
# --------------------------------------------------------------------------------------

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


def _precompute_per_read_arrays(
    L_block: pd.DataFrame,
    eta: pd.Series,
    candidate_genomes: Sequence[str],
    all_genomes_for_xmap: Optional[Sequence[str]] = None,
) -> Tuple[int, float, np.ndarray, Dict[str, np.ndarray]]:
    """
    Precompute exact per-read aggregates used by the current likelihood functions.

    Returns:
      n_reads: number of unique read_id in this barcode
      n_reads_eff: sum of per-read weights (diagnostic)
      E: shape (n_reads,) where E[r] = sum_g eta[g] * L_{r,g}
      Lg: dict genome->array shape (n_reads,) where Lg[g][r] = sum_{rows with genome==g, read_id==r} L
           If all_genomes_for_xmap is provided, Lg includes ALL genomes (not just candidates).
    """
    # Factorize read_id once (stable order within this L_block)
    read_codes, _ = pd.factorize(L_block["read_id"], sort=False)
    n_reads = int(read_codes.max()) + 1 if len(read_codes) else 0
    if n_reads == 0:
        return 0, 0.0, np.zeros(0, dtype=np.float64), {g: np.zeros(0, dtype=np.float64) for g in candidate_genomes}

    # Base arrays
    L = L_block["L"].to_numpy(dtype=np.float64, copy=False)
    gn = L_block["genome"].astype(str).to_numpy()
    # per-read weight (same for all rows of the same read_id); use max for safety
    wr_row = (
        L_block["w_read"].to_numpy(dtype=np.float64, copy=False)
        if "w_read" in L_block.columns
        else np.ones(len(L_block), dtype=np.float64)
    )
    wr = np.zeros(n_reads, dtype=np.float64)
    np.maximum.at(wr, read_codes, wr_row)
    n_reads_eff = float(wr.sum())

    # E_r = sum eta_g * L_{r,g} (exactly what _loglik_empty_from_E computes)
    eta_row = eta.reindex(pd.Index(gn)).fillna(0.0).to_numpy(dtype=np.float64, copy=False)
    E = np.bincount(read_codes, weights=L * eta_row, minlength=n_reads).astype(np.float64, copy=False)

    # Lg_r: per-genome per-read likelihoods
    # When xmap is active, build for ALL genomes; otherwise only candidates (top-k)
    genomes_to_build = list(all_genomes_for_xmap) if all_genomes_for_xmap else list(candidate_genomes)
    Lg: Dict[str, np.ndarray] = {}
    for g in genomes_to_build:
        m = (gn == g)
        if not np.any(m):
            Lg[g] = np.zeros(n_reads, dtype=np.float64)
        else:
            Lg[g] = np.bincount(read_codes[m], weights=L[m], minlength=n_reads).astype(np.float64, copy=False)
    # Ensure candidate genomes are always present (may not be in all_genomes_for_xmap if data is sparse)
    for g in candidate_genomes:
        if g not in Lg:
            m = (gn == g)
            if not np.any(m):
                Lg[g] = np.zeros(n_reads, dtype=np.float64)
            else:
                Lg[g] = np.bincount(read_codes[m], weights=L[m], minlength=n_reads).astype(np.float64, copy=False)
    return n_reads, n_reads_eff, E, Lg

def _loglik_empty_from_E(E: np.ndarray) -> float:
    # loglik_empty = sum_r log(E_r)
    return float(np.log(np.clip(E, 1e-12, None)).sum())

def _best_single_from_arrays(
    n_reads: int,
    E: np.ndarray,
    Lg: Dict[str, np.ndarray],
    cfg: MergeConfig,
    candidate_genomes: Sequence[str],
    phi_matrix: Optional[pd.DataFrame] = None,
    chi: Optional[pd.Series] = None,
) -> Optional[Dict[str, Any]]:
    if n_reads <= 0 or not candidate_genomes:
        return None

    alphas = np.arange(0.0, float(cfg.max_alpha) + 1e-9, float(cfg.alpha_grid), dtype=np.float64)
    logn = math.log(max(n_reads, 1))

    # Pre-build L_wide matrix (n_reads × G_available) for xmap dot product
    # Only needed if phi_matrix is provided and any candidate has chi > threshold
    xmap_L_wide: Optional[np.ndarray] = None
    xmap_genome_order: Optional[List[str]] = None
    if phi_matrix is not None and chi is not None:
        xmap_genome_order = list(phi_matrix.index)  # target genomes (rows)
        xmap_L_wide = np.column_stack([
            Lg.get(g, np.zeros(n_reads, dtype=np.float64))
            for g in xmap_genome_order
        ])  # shape: (n_reads, G)

    best: Optional[Dict[str, Any]] = None
    for g1 in candidate_genomes:
        # Decide whether to use Level 2 cross-mapping for this g1
        use_xmap = (
            phi_matrix is not None
            and chi is not None
            and g1 in chi.index
            and chi[g1] > cfg.xmap_chi_threshold
            and g1 in phi_matrix.columns
            and xmap_L_wide is not None
            and xmap_genome_order is not None
        )

        if use_xmap:
            phi_vec = phi_matrix[g1].reindex(xmap_genome_order).fillna(0.0).to_numpy()
            L1 = xmap_L_wide @ phi_vec  # L*_g1(r) = Σ_g φ_{g|g1} · L_g(r)
        else:
            L1 = Lg[g1]

        # Vectorized over alpha: s[a, r] = (1-a)*L1[r] + a*E[r]
        s = (1.0 - alphas)[:, None] * L1[None, :] + alphas[:, None] * E[None, :]
        ll = np.log(np.clip(s, 1e-12, None)).sum(axis=1)  # shape (n_alpha,)
        bic = (-2.0 * ll) + (1.0 * logn)                  # 1 parameter for single

        k = int(np.argmin(bic))
        bic_k = float(bic[k])
        if best is None or bic_k < best["bic"]:
            best = {
                "model": "single", "g1": g1, "alpha": float(alphas[k]), "bic": bic_k,
                "xmap_activated": use_xmap,
                "chi_g1": float(chi[g1]) if (chi is not None and g1 in chi.index) else None,
            }

    return best


def _best_doublet_from_arrays(
    n_reads: int,
    E: np.ndarray,
    Lg: Dict[str, np.ndarray],
    cfg: MergeConfig,
    candidate_genomes: Sequence[str],
) -> Optional[Dict[str, Any]]:
    if n_reads <= 0 or len(candidate_genomes) < 2:
        return None

    alphas = np.arange(0.0, float(cfg.max_alpha) + 1e-9, float(cfg.alpha_grid), dtype=np.float64)
    rhos = np.arange(0.1, 0.9 + 1e-9, float(cfg.rho_grid), dtype=np.float64)
    logn = math.log(max(n_reads, 1))

    best: Optional[Dict[str, Any]] = None
    for i in range(len(candidate_genomes)):
        for j in range(i + 1, len(candidate_genomes)):
            g1, g2 = candidate_genomes[i], candidate_genomes[j]
            L1 = Lg[g1]
            L2 = Lg[g2]

            # base_mix[rho, r] = rho*L1[r] + (1-rho)*L2[r]
            base_mix = rhos[:, None] * L1[None, :] + (1.0 - rhos)[:, None] * L2[None, :]

            # Loop over alpha (small, e.g. 26), vectorize over rho (e.g. 17) and reads
            for a in alphas:
                s = (1.0 - a) * base_mix + a * E[None, :]
                ll_rho = np.log(np.clip(s, 1e-12, None)).sum(axis=1)  # per-rho loglik
                bic_rho = (-2.0 * ll_rho) + (2.0 * logn)              # 2 params for doublet

                k = int(np.argmin(bic_rho))
                bic_k = float(bic_rho[k])
                if best is None or bic_k < best["bic"]:
                    best = {
                        "model": "doublet",
                        "g1": g1,
                        "g2": g2,
                        "alpha": float(a),
                        "rho": float(rhos[k]),
                        "bic": bic_k,
                    }

    return best

# --------------------------------------------------------------------------------------
# Model selection (gate design preserved)
# --------------------------------------------------------------------------------------

def _select_model_for_barcode(
    L_block: pd.DataFrame,
    eta: pd.Series,
    cfg: MergeConfig,
    candidate_genomes: Sequence[str],
    phi_matrix: Optional[pd.DataFrame] = None,
    chi: Optional[pd.Series] = None,
) -> Dict[str, Any]:
    # Determine if any candidate genome needs xmap (Level 2)
    need_xmap = (
        phi_matrix is not None
        and chi is not None
        and cfg.xmap_enabled
        and any(
            g1 in chi.index and chi[g1] > cfg.xmap_chi_threshold
            for g1 in candidate_genomes
        )
    )
    all_genomes_for_xmap = list(phi_matrix.index) if need_xmap and phi_matrix is not None else None

    # Precompute per-read arrays (exactly preserves loglik math)
    n_reads, n_reads_eff, E, Lg = _precompute_per_read_arrays(
        L_block, eta, candidate_genomes,
        all_genomes_for_xmap=all_genomes_for_xmap,
    )
    read_count = int(n_reads)

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
        "n_reads_eff": float(n_reads_eff),
        "p_top1": p_top1,
        "p_top2": p_top2,
        "ratio12": ratio12,
        "top_genome": (mass.index[0] if len(mass) else None),
        "jsd_to_eta": jsd,
        "doublet_minor_frac": minor_frac,
        "purity_best": None,  # filled after choosing model (1 - alpha)
    }
    # Empty BIC from precomputed E    
    bic_empty = _bic(_loglik_empty_from_E(E), 0, read_count)
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
      
    # Best single/doublet using vectorized NumPy grids (same grid, same BIC)
    best_single = _best_single_from_arrays(
        read_count, E, Lg, cfg, candidate_genomes,
        phi_matrix=phi_matrix if need_xmap else None,
        chi=chi if need_xmap else None,
    )
    best_doublet = _best_doublet_from_arrays(read_count, E, Lg, cfg, candidate_genomes)
      
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
    if chosen.get("alpha") is not None:
        out["purity_best"] = float(1.0 - float(chosen["alpha"]))

    # Level 2 cross-mapping info (from best_single if singlet won, else from best_single anyway for diagnostics)
    out["xmap_activated"] = best_single.get("xmap_activated", False) if best_single else False
    out["chi_g1"] = best_single.get("chi_g1") if best_single else None

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
                "w_read": "float32",
            },
        ):
            if chunk is None or chunk.empty:
                continue
            yield chunk
    except (pd.errors.EmptyDataError, OSError):
        # Header missing / no columns.
        return


# --------------------------------------------------------------------------------------
# Pass 1: stream assign files -> (C_all, N_all) + shard spill
# --------------------------------------------------------------------------------------

@dataclass
class Pass1Outputs:
    C: pd.DataFrame
    N: pd.DataFrame


def _pass1_read_and_reduce_duckdb(
    fp: Path,
    min_mapq: int = 0,
    max_xa: int = -1,
) -> pd.DataFrame:
    """
    DuckDB fast path: read an assign output TSV(.gz) and perform the
    _reduce_alignments_to_per_genome() reduction in a single SQL pass.

    Returns a DataFrame with columns:
      barcode, read_id, genome, AS, MAPQ, NM, assigned_class[, XAcount]
    identical in schema to _coerce_assign_schema() + _reduce_alignments_to_per_genome().
    """
    import duckdb

    # Sniff header to handle BC/barcode, Read/read_id, Genome/genome variants
    opener = gzip.open if str(fp).endswith(".gz") else open
    with opener(fp, "rt") as f:
        header = f.readline().strip().split("\t")
    bc_col = "BC" if "BC" in header else "barcode"
    read_col = '"Read"' if "Read" in header else "read_id"
    genome_col = '"Genome"' if "Genome" in header else "genome"
    # AS is a SQL keyword — always quote it
    as_col = '"AS"'
    has_xa = "XAcount" in header

    path_str = str(fp).replace("'", "''")
    compression = "'gzip'" if str(fp).endswith(".gz") else "'none'"

    # Build WHERE clause for MAPQ and XA filtering
    where_parts = []
    if min_mapq > 0:
        where_parts.append(f"CAST(MAPQ AS FLOAT) >= {min_mapq}")
    if max_xa >= 0 and has_xa:
        where_parts.append(f"CAST(XAcount AS INT) <= {max_xa}")
    where_clause = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

    xa_select = ",\n        MAX(CAST(XAcount AS INT)) AS XAcount" if has_xa else ""

    sql = f"""
    SELECT
        CAST({bc_col} AS VARCHAR) AS barcode,
        CAST({read_col} AS VARCHAR) AS read_id,
        CAST({genome_col} AS VARCHAR) AS genome,
        MAX(CAST({as_col} AS FLOAT)) AS "AS",
        MAX(CAST(MAPQ AS FLOAT)) AS MAPQ,
        MIN(CAST(NM AS FLOAT)) AS NM,
        CASE
            WHEN SUM(CASE WHEN assigned_class = 'ambiguous' THEN 1 ELSE 0 END) > 0
            THEN 'ambiguous'
            ELSE FIRST(assigned_class)
        END AS assigned_class{xa_select}
    FROM read_csv(
        '{path_str}',
        delim='\\t',
        header=true,
        compression={compression},
        ignore_errors=true
    )
    {where_clause}
    GROUP BY {bc_col}, {read_col}, {genome_col}
    """

    con = duckdb.connect()
    con.execute("SET threads TO 2")
    result = con.execute(sql).fetchdf()
    con.close()

    for col in ["barcode", "read_id", "genome", "assigned_class"]:
        if col in result.columns:
            result[col] = result[col].astype("string")

    return result


def _pass1_worker_job(args) -> Pass1Outputs:
    """
    Top-level worker so it is picklable by multiprocessing.
    args: (worker_index, filepath, cfg, shard_root, bc_to_chunk, use_duckdb)
    """
    i, fp_str, cfg_dict, shard_root_str, bc_to_chunk, use_duckdb = args
    cfg = MergeConfig(**cfg_dict)
    fp = Path(fp_str)
    shard_root = Path(shard_root_str)
    wdir = shard_root / f"w{i:03d}"
    return _pass1_process_one_file(fp, cfg, wdir, bc_to_chunk, use_duckdb=use_duckdb)

def _pass1_process_one_file(
    fp: Path, cfg: MergeConfig, shard_dir: Path,
    bc_to_chunk: Optional[Dict[str, int]], use_duckdb: bool = False,
) -> Pass1Outputs:
    """
    Writes per-read posterior rows to shard gzip files under shard_dir.
    FIX: shard files are opened lazily (only if we actually write rows),
         preventing creation of 38-byte empty gzip files.

    Two paths:
    - DuckDB (use_duckdb=True): single SQL query for file read + reduction,
      then process result in memory-safe chunks.
    - Python (use_duckdb=False): chunked pandas reader with carry-over buffer
      to prevent (barcode, read_id) groups from splitting across chunk boundaries.
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

    def _posterior_and_shard(df: pd.DataFrame) -> None:
        """Filter, compute posteriors, accumulate C, write to shards."""
        nonlocal C_parts
        if df is None or df.empty:
            return
        df, _ = _filter_promiscuous_ambiguous_reads(df, cfg)
        if df.empty:
            return
        df, _ = _filter_read_quality(df, cfg)
        if df.empty:
            return
        Ldf = _compute_read_posteriors(df, cfg)
        if Ldf.empty:
            return
        C_parts.append(Ldf.groupby(["barcode", "genome"], sort=False)["L"].sum().rename("C").reset_index())
        sid = _route_shards_for_barcodes(Ldf["barcode"], cfg, bc_to_chunk)
        Ldf = Ldf.assign(_sid=sid)
        for i in range(cfg.shards):
            sub = Ldf[Ldf["_sid"] == i]
            if sub.empty:
                continue
            h = _get_handle(i)
            if not header_written[i]:
                h.write("barcode\tread_id\tgenome\tL\tL_amb\tw_read\n")
                header_written[i] = True
            sub2 = sub.drop(columns="_sid")
            sub2 = sub2.sort_values(["barcode", "read_id", "genome"], kind="mergesort")
            sub2.to_csv(h, sep="\t", header=False, index=False)

    try:
        if use_duckdb and _HAS_DUCKDB:
            # --- DuckDB path: single SQL query for read + reduction ---
            df_reduced = _pass1_read_and_reduce_duckdb(
                fp,
                min_mapq=cfg.min_mapq_genotyping,
                max_xa=cfg.max_xa_genotyping,
            )
            df_reduced = df_reduced.dropna(subset=["barcode", "read_id", "genome"])

            # Process in memory-safe chunks (no carry-over needed — GROUP BY is complete)
            chunk_sz = int(cfg.chunk_rows)
            for start in range(0, len(df_reduced), chunk_sz):
                block = df_reduced.iloc[start:start + chunk_sz]
                _posterior_and_shard(block)
        else:
            # --- Original pandas chunked path ---
            carry: Optional[pd.DataFrame] = None

            def _process_block(df_block: pd.DataFrame) -> None:
                if df_block is None or df_block.empty:
                    return
                df = _reduce_alignments_to_per_genome(df_block)
                df = df.dropna(subset=["barcode", "read_id", "genome"])
                _posterior_and_shard(df)

            for raw in pd.read_csv(fp, sep="\t", chunksize=int(cfg.chunk_rows), low_memory=False):
                df0 = _coerce_assign_schema(raw)
                if carry is not None and not carry.empty:
                    df0 = pd.concat([carry, df0], ignore_index=True)
                if df0.empty:
                    carry = None
                    continue
                last_bc = str(df0["barcode"].iloc[-1])
                last_rid = str(df0["read_id"].iloc[-1])
                mask_carry = (df0["barcode"].astype(str) == last_bc) & (df0["read_id"].astype(str) == last_rid)
                carry = df0.loc[mask_carry].copy()
                df_main = df0.loc[~mask_carry].copy()
                _process_block(df_main)
            if carry is not None and not carry.empty:
                _process_block(carry)
    finally:
        for h in handles.values():
            try:
                fpath = h.name
                h.close()
                # Force NFS flush so Pass 1.5 reads complete data
                fd = os.open(fpath, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            except Exception:
                pass

    C_out = pd.concat(C_parts, ignore_index=True) if C_parts else pd.DataFrame(columns=["barcode", "genome", "C"])
    N_out = pd.DataFrame(columns=["barcode", "n_reads"])  # placeholder; will be computed exactly post-merge
    return Pass1Outputs(C=C_out, N=N_out)


# --------------------------------------------------------------------------------------
# Pass 2.75: Top-K read re-classification (worker infrastructure)
# --------------------------------------------------------------------------------------

_PASS275_TOPK: Optional[Dict[str, List[str]]] = None
_PASS275_CFG: Optional[MergeConfig] = None
_PASS275_BC_TO_CHUNK: Optional[Dict[str, int]] = None
_PASS275_USE_DUCKDB: bool = False


def _pass275_init_worker(
    topk_path: str,
    cfg_dict: Dict[str, Any],
    bc_to_chunk: Optional[Dict[str, int]],
    use_duckdb: bool,
) -> None:
    """Initializer for Pass 2.75 worker pool — loads topk from disk once."""
    global _PASS275_TOPK, _PASS275_CFG, _PASS275_BC_TO_CHUNK, _PASS275_USE_DUCKDB
    _PASS275_TOPK = _load_topk_tsv(Path(topk_path))
    _PASS275_CFG = MergeConfig(**cfg_dict)
    _PASS275_BC_TO_CHUNK = bc_to_chunk
    _PASS275_USE_DUCKDB = use_duckdb


def _pass275_worker_job(args) -> pd.DataFrame:
    """
    Top-level picklable worker for Pass 2.75.
    args: (worker_index, filepath_str, shard_root_str)
    Returns: C_part DataFrame [barcode, genome, C].
    """
    i, fp_str, shard_root_str = args
    global _PASS275_TOPK, _PASS275_CFG, _PASS275_BC_TO_CHUNK, _PASS275_USE_DUCKDB
    assert _PASS275_TOPK is not None and _PASS275_CFG is not None
    fp = Path(fp_str)
    shard_root = Path(shard_root_str)
    wdir = shard_root / f"w{i:03d}"
    return _pass275_reclass_one_file(
        fp, _PASS275_CFG, wdir, _PASS275_BC_TO_CHUNK,
        _PASS275_TOPK, use_duckdb=_PASS275_USE_DUCKDB,
    )


def _pass275_read_and_reduce_duckdb(
    fp: Path,
    topk: Dict[str, List[str]],
    min_mapq: int = 0,
    max_xa: int = -1,
) -> pd.DataFrame:
    """
    DuckDB fast path for Pass 2.75: read assign TSV, perform per-genome
    reduction, and INNER JOIN with topk to keep only top-K rows.
    Returns DataFrame identical in schema to _pass1_read_and_reduce_duckdb
    but restricted to top-K genomes per barcode.
    """
    import duckdb
    import pyarrow as pa

    # Build topk long-form table for the JOIN
    bc_list: List[str] = []
    gn_list: List[str] = []
    for bc, genomes in topk.items():
        for g in genomes:
            bc_list.append(bc)
            gn_list.append(g)
    topk_arrow = pa.table({
        "tk_barcode": pa.array(bc_list, type=pa.string()),
        "tk_genome": pa.array(gn_list, type=pa.string()),
    })

    # Sniff header (same logic as _pass1_read_and_reduce_duckdb)
    opener = gzip.open if str(fp).endswith(".gz") else open
    with opener(fp, "rt") as f:
        header = f.readline().strip().split("\t")
    bc_col = "BC" if "BC" in header else "barcode"
    read_col = '"Read"' if "Read" in header else "read_id"
    genome_col = '"Genome"' if "Genome" in header else "genome"
    as_col = '"AS"'
    has_xa = "XAcount" in header

    path_str = str(fp).replace("'", "''")
    compression = "'gzip'" if str(fp).endswith(".gz") else "'none'"

    where_parts: List[str] = []
    if min_mapq > 0:
        where_parts.append(f"CAST(MAPQ AS FLOAT) >= {min_mapq}")
    if max_xa >= 0 and has_xa:
        where_parts.append(f"CAST(XAcount AS INT) <= {max_xa}")
    where_clause = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""

    xa_select = ",\n        MAX(CAST(XAcount AS INT)) AS XAcount" if has_xa else ""

    sql = f"""
    SELECT a.*
    FROM (
        SELECT
            CAST({bc_col} AS VARCHAR) AS barcode,
            CAST({read_col} AS VARCHAR) AS read_id,
            CAST({genome_col} AS VARCHAR) AS genome,
            MAX(CAST({as_col} AS FLOAT)) AS "AS",
            MAX(CAST(MAPQ AS FLOAT)) AS MAPQ,
            MIN(CAST(NM AS FLOAT)) AS NM,
            CASE
                WHEN SUM(CASE WHEN assigned_class = 'ambiguous' THEN 1 ELSE 0 END) > 0
                THEN 'ambiguous'
                ELSE FIRST(assigned_class)
            END AS assigned_class{xa_select}
        FROM read_csv(
            '{path_str}',
            delim='\\t',
            header=true,
            compression={compression},
            ignore_errors=true
        )
        {where_clause}
        GROUP BY {bc_col}, {read_col}, {genome_col}
    ) a
    INNER JOIN topk_tbl t
        ON a.barcode = t.tk_barcode AND a.genome = t.tk_genome
    """

    con = duckdb.connect()
    con.execute("SET threads TO 2")
    con.register("topk_tbl", topk_arrow)
    result = con.execute(sql).fetchdf()
    con.close()

    for col in ["barcode", "read_id", "genome", "assigned_class"]:
        if col in result.columns:
            result[col] = result[col].astype("string")

    return result


def _pass275_reclass_one_file(
    fp: Path,
    cfg: MergeConfig,
    shard_dir: Path,
    bc_to_chunk: Optional[Dict[str, int]],
    topk: Dict[str, List[str]],
    use_duckdb: bool = False,
) -> pd.DataFrame:
    """
    Re-read one assign chunk file, restrict to top-K genomes per barcode,
    recompute posteriors, write to shards.

    Structurally mirrors _pass1_process_one_file but adds a top-K genome
    filter before posterior computation.

    Returns: C_part DataFrame [barcode, genome, C].
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

    def _topk_filter(df: pd.DataFrame) -> pd.DataFrame:
        """Filter DataFrame to keep only rows where genome is in topk[barcode]."""
        if df.empty:
            return df
        # Build lookup set for barcodes present in this chunk
        bcs_in_chunk = df["barcode"].astype(str).unique()
        topk_pairs: set = set()
        for bc in bcs_in_chunk:
            for g in topk.get(bc, ()):
                topk_pairs.add((bc, g))
        if not topk_pairs:
            return df.iloc[:0]
        bc_arr = df["barcode"].astype(str).to_numpy()
        gn_arr = df["genome"].astype(str).to_numpy()
        keep = np.array(
            [(bc_arr[i], gn_arr[i]) in topk_pairs for i in range(len(df))],
            dtype=bool,
        )
        return df.loc[keep].copy()

    def _posterior_and_shard(df: pd.DataFrame) -> None:
        """Top-K filter, quality filters, compute posteriors, write to shards."""
        nonlocal C_parts
        if df is None or df.empty:
            return
        # --- Top-K genome filter (the key difference from Pass 1) ---
        df = _topk_filter(df)
        if df.empty:
            return
        df, _ = _filter_promiscuous_ambiguous_reads(df, cfg)
        if df.empty:
            return
        df, _ = _filter_read_quality(df, cfg)
        if df.empty:
            return
        # --- Re-classify winner/ambiguous within top-K context ---
        # Three-tier: (1) original winner, (2) rescued flag, (3) composite score
        df = _reclass_within_topk(df, cfg)
        Ldf = _compute_read_posteriors(df, cfg)
        if Ldf.empty:
            return
        C_parts.append(Ldf.groupby(["barcode", "genome"], sort=False)["L"].sum().rename("C").reset_index())
        sid = _route_shards_for_barcodes(Ldf["barcode"], cfg, bc_to_chunk)
        Ldf = Ldf.assign(_sid=sid)
        for i in range(cfg.shards):
            sub = Ldf[Ldf["_sid"] == i]
            if sub.empty:
                continue
            h = _get_handle(i)
            if not header_written[i]:
                h.write("barcode\tread_id\tgenome\tL\tL_amb\tw_read\n")
                header_written[i] = True
            sub2 = sub.drop(columns="_sid")
            sub2 = sub2.sort_values(["barcode", "read_id", "genome"], kind="mergesort")
            sub2.to_csv(h, sep="\t", header=False, index=False)

    try:
        if use_duckdb and _HAS_DUCKDB:
            # DuckDB path: read + reduce + top-K JOIN in a single SQL query
            df_reduced = _pass275_read_and_reduce_duckdb(
                fp, topk,
                min_mapq=cfg.min_mapq_genotyping,
                max_xa=cfg.max_xa_genotyping,
            )
            df_reduced = df_reduced.dropna(subset=["barcode", "read_id", "genome"])
            chunk_sz = int(cfg.chunk_rows)
            for start in range(0, len(df_reduced), chunk_sz):
                block = df_reduced.iloc[start:start + chunk_sz]
                _posterior_and_shard(block)
        else:
            # Pandas chunked path (mirrors Pass 1)
            carry: Optional[pd.DataFrame] = None

            def _process_block(df_block: pd.DataFrame) -> None:
                if df_block is None or df_block.empty:
                    return
                df = _reduce_alignments_to_per_genome(df_block)
                df = df.dropna(subset=["barcode", "read_id", "genome"])
                _posterior_and_shard(df)

            for raw in pd.read_csv(fp, sep="\t", chunksize=int(cfg.chunk_rows), low_memory=False):
                df0 = _coerce_assign_schema(raw)
                if carry is not None and not carry.empty:
                    df0 = pd.concat([carry, df0], ignore_index=True)
                if df0.empty:
                    carry = None
                    continue
                last_bc = str(df0["barcode"].iloc[-1])
                last_rid = str(df0["read_id"].iloc[-1])
                mask_carry = (df0["barcode"].astype(str) == last_bc) & (df0["read_id"].astype(str) == last_rid)
                carry = df0.loc[mask_carry].copy()
                df_main = df0.loc[~mask_carry].copy()
                _process_block(df_main)
            if carry is not None and not carry.empty:
                _process_block(carry)
    finally:
        for h in handles.values():
            try:
                fpath = h.name
                h.close()
                fd = os.open(fpath, os.O_RDONLY)
                try:
                    os.fsync(fd)
                finally:
                    os.close(fd)
            except Exception:
                pass

    C_out = pd.concat(C_parts, ignore_index=True) if C_parts else pd.DataFrame(columns=["barcode", "genome", "C"])
    return C_out


# --------------------------------------------------------------------------------------
# Pass 3 parallelism support (globals per worker process)
# --------------------------------------------------------------------------------------

_PASS3_ETA: Optional[pd.Series] = None
_PASS3_TOPK: Optional[Dict[str, List[str]]] = None
_PASS3_CFG: Optional[MergeConfig] = None
_PASS3_PHI: Optional[pd.DataFrame] = None
_PASS3_CHI: Optional[pd.Series] = None

def _write_topk_tsv(topk: Dict[str, List[str]], path: Path) -> None:
    """
    Write topk mapping once so worker processes can load it from disk
    without pickling a huge dict to every process.
    Format: barcode \t genome1 \t genome2 \t ... (variable columns)
    """
    with gzip.open(path, "wt") if str(path).endswith(".gz") else open(path, "wt") as oh:
        for bc, gs in topk.items():
            oh.write(str(bc))
            for g in gs:
                oh.write("\t")
                oh.write(str(g))
            oh.write("\n")


def _load_topk_tsv(path: Path) -> Dict[str, List[str]]:
    topk: Dict[str, List[str]] = {}
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            bc = parts[0]
            gs = [p for p in parts[1:] if p]
            topk[bc] = gs
    return topk


def _pass3_init_worker(
    eta_path: str,
    topk_path: str,
    cfg_dict: Dict[str, Any],
    phi_path: Optional[str] = None,
    chi_path: Optional[str] = None,
) -> None:
    """
    Initializer run once per worker process.
    Loads eta + topk + optional phi/chi into process globals to avoid reloading for every shard.
    """
    global _PASS3_ETA, _PASS3_TOPK, _PASS3_CFG, _PASS3_PHI, _PASS3_CHI

    # eta
    eta_df = pd.read_csv(eta_path, sep="\t", compression="gzip")
    # eta saved by: eta.to_frame("eta").to_csv(...)
    # so first column is index-like; handle robustly
    idx_col = eta_df.columns[0]
    _PASS3_ETA = eta_df.set_index(idx_col)["eta"].astype(float)
    _PASS3_ETA.index = _PASS3_ETA.index.astype(str)
    _PASS3_ETA = _PASS3_ETA / (_PASS3_ETA.sum() + 1e-12)

    # topk
    _PASS3_TOPK = _load_topk_tsv(Path(topk_path))

    # cfg
    _PASS3_CFG = MergeConfig(**cfg_dict)

    # phi / chi (Level 2 cross-mapping)
    _PASS3_PHI = None
    _PASS3_CHI = None
    if phi_path is not None and Path(phi_path).exists():
        phi_df = pd.read_csv(phi_path, sep="\t", index_col=0, compression="gzip")
        phi_df.index = phi_df.index.astype(str)
        phi_df.columns = phi_df.columns.astype(str)
        _PASS3_PHI = phi_df
    if chi_path is not None and Path(chi_path).exists():
        chi_df = pd.read_csv(chi_path, sep="\t")
        _PASS3_CHI = chi_df.set_index("genome")["chi"].astype(float)
        _PASS3_CHI.index = _PASS3_CHI.index.astype(str)


def _pass3_process_one_shard(shard_path_str: str, out_part_path_str: str) -> Tuple[str, int]:
    """
    Worker job: process one shard file and write a part file (no header).
    Returns (shard_path, called_count).
    """
    global _PASS3_ETA, _PASS3_TOPK, _PASS3_CFG, _PASS3_PHI, _PASS3_CHI
    assert _PASS3_ETA is not None and _PASS3_TOPK is not None and _PASS3_CFG is not None

    shard_path = Path(shard_path_str)
    out_part = Path(out_part_path_str)

    called = 0
    with gzip.open(out_part, "wt") as oh:
        called += _call_from_shard_file(
            shard_path=shard_path,
            eta=_PASS3_ETA,
            topk=_PASS3_TOPK,
            cfg=_PASS3_CFG,
            out_handle=oh,
            phi_matrix=_PASS3_PHI,
            chi=_PASS3_CHI,
        )
    return (shard_path.name, int(called))


# --------------------------------------------------------------------------------------
# Ambient learning (eta)
# --------------------------------------------------------------------------------------

def _compute_eta_from_shards(shard_root: Path, target_bcs: Sequence[str], all_genomes: Sequence[str], cfg: MergeConfig) -> pd.Series:
    target_set = set(map(str, target_bcs))
    mix = {g: 0.0 for g in all_genomes}

    if not target_set:
        eta = pd.Series({g: 1.0 for g in all_genomes}, dtype=float)
        return eta / eta.sum()

    #for shard in shard_root.rglob("shard_*.tsv.gz"):
    for shard in (shard_root / "merged").glob("shard_*.tsv.gz"):
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


def _compute_eta_from_shards_duckdb(
    shard_root: Path,
    target_bcs: Sequence[str],
    all_genomes: Sequence[str],
    cfg: MergeConfig,
) -> pd.Series:
    """DuckDB fast path: read all merged shards and aggregate L by genome in one SQL pass."""
    import duckdb
    import pyarrow as pa

    target_set = set(map(str, target_bcs))
    if not target_set:
        eta = pd.Series({g: 1.0 for g in all_genomes}, dtype=float)
        return eta / eta.sum()

    merged_glob = str(shard_root / "merged" / "shard_*.tsv.gz").replace("'", "''")

    con = duckdb.connect()
    con.execute("SET threads TO 4")

    con.register(
        "_target_bcs",
        pa.table({"barcode": pa.array(sorted(target_set), type=pa.string())}),
    )

    sql = f"""
    SELECT
        CAST(genome AS VARCHAR) AS genome,
        SUM(CAST(L AS DOUBLE)) AS L_sum
    FROM read_csv(
        '{merged_glob}',
        delim='\\t',
        header=true,
        compression='gzip',
        columns={{
            'barcode': 'VARCHAR',
            'read_id': 'VARCHAR',
            'genome': 'VARCHAR',
            'L': 'FLOAT',
            'L_amb': 'FLOAT',
            'w_read': 'FLOAT'
        }}
    )
    WHERE barcode IN (SELECT barcode FROM _target_bcs)
    GROUP BY genome
    """

    result = con.execute(sql).fetchdf()
    con.close()

    mix = {g: 0.0 for g in all_genomes}
    for _, row in result.iterrows():
        g = str(row["genome"])
        if g in mix:
            mix[g] += float(row["L_sum"])

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

    _eta_fn = (
        _compute_eta_from_shards_duckdb
        if cfg.pass2_duckdb and _HAS_DUCKDB
        else _compute_eta_from_shards
    )

    seed = _pick_initial_eta_seed(C_all, N_all, cfg)
    meta["seed_sizes"].append(len(seed))
    eta = _eta_fn(shard_root, seed, list(all_genomes), cfg)

    for _ in range(int(cfg.eta_iters)):
        seed = _refine_eta_seed_by_jsd(C_all, N_all, eta, cfg)
        meta["seed_sizes"].append(len(seed))
        if not seed:
            break
        eta = _eta_fn(shard_root, seed, list(all_genomes), cfg)

    return eta, meta


# --------------------------------------------------------------------------------------
# Cross-mapping profile (Level 2)
# --------------------------------------------------------------------------------------

def _compute_phi_matrix(
    C_all: pd.DataFrame,
    N_all: pd.DataFrame,
    all_genomes: Sequence[str],
    cfg: MergeConfig,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Estimate per-genome cross-mapping profiles φ_{g|g1} from step 4 outputs.

    Returns
    -------
    phi_df : pd.DataFrame
        Shape (G, G). Index = target genomes. Columns = anchor genomes.
        phi_df[g1][g] = φ_{g|g1}.  Each column sums to 1.0.
    chi : pd.Series
        Index = anchor genomes. Values = χ(g1) = max_{g≠g1} φ_{g|g1}.
    """
    genome_list = list(all_genomes)
    G = len(genome_list)

    # Step 1: Pivot C_all to wide (barcode × genome)
    C_wide = C_all.pivot(index="barcode", columns="genome", values="C").fillna(0.0)
    # Ensure all genomes present as columns
    for g in genome_list:
        if g not in C_wide.columns:
            C_wide[g] = 0.0
    C_wide = C_wide[genome_list]

    # N_all: barcode → n_reads (may be barcode-only or barcode×genome)
    if "genome" in N_all.columns:
        n_reads = N_all.groupby("barcode")["n_reads"].sum()
    else:
        n_reads = N_all.set_index("barcode")["n_reads"]
    n_reads = n_reads.reindex(C_wide.index).fillna(0)

    # Step 2: top_genome and ratio12 from C_wide
    C_vals = C_wide.to_numpy()
    top2_idx = np.argsort(-C_vals, axis=1)[:, :2]  # indices of top-2
    rows = np.arange(len(C_vals))
    top1_vals = C_vals[rows, top2_idx[:, 0]]
    top2_vals = C_vals[rows, top2_idx[:, 1]]
    ratio12 = top1_vals / (top2_vals + 1e-12)
    top_genome = np.array(genome_list)[top2_idx[:, 0]]

    n_reads_arr = n_reads.to_numpy()

    # Build N_wide for off-diagonal correction (C_g - N_g)
    # N_wide: barcode × genome with unique-read counts per genome
    if "genome" in N_all.columns and "n_reads" in N_all.columns:
        N_wide = N_all.pivot(index="barcode", columns="genome", values="n_reads").fillna(0.0)
    else:
        # Fallback: uniform split (rare, only if N_all has no genome column)
        N_wide = pd.DataFrame(0.0, index=C_wide.index, columns=genome_list)
    for g in genome_list:
        if g not in N_wide.columns:
            N_wide[g] = 0.0
    N_wide = N_wide.reindex(C_wide.index).fillna(0.0)[genome_list]

    C_np = C_wide.to_numpy()
    N_np = N_wide.to_numpy()

    # Step 3: Iterative φ estimation
    phi_cols: Dict[str, Optional[np.ndarray]] = {}
    for _iteration in range(max(1, cfg.xmap_phi_iters)):
        phi_cols = {}
        for gi, g1 in enumerate(genome_list):
            # Build D(g1): top_genome==g1 AND n_reads>=threshold AND ratio12>=threshold
            mask = (
                (top_genome == g1)
                & (n_reads_arr >= cfg.xmap_phi_min_reads)
                & (ratio12 >= cfg.xmap_phi_ratio12_min)
            )
            n_D = int(mask.sum())
            if n_D < 10:
                phi_cols[g1] = None
                continue

            D_C = C_np[mask]   # (n_D, G)
            D_N = N_np[mask]   # (n_D, G)

            # Raw scores
            s = np.zeros(G, dtype=np.float64)
            for gj in range(G):
                if gj == gi:
                    s[gj] = D_C[:, gj].mean()  # diagonal: full C
                else:
                    s[gj] = np.maximum(D_C[:, gj] - D_N[:, gj], 0.0).mean()  # off-diag: C-N

            total = s.sum()
            if total <= 0:
                phi_cols[g1] = None
                continue
            phi_cols[g1] = s / total

    # Step 4: Assemble phi_df and chi
    phi_data = {}
    for g1 in genome_list:
        if phi_cols.get(g1) is not None:
            phi_data[g1] = pd.Series(phi_cols[g1], index=genome_list)
        else:
            # Fallback: identity (all mass on diagonal)
            identity = np.zeros(G, dtype=np.float64)
            identity[genome_list.index(g1)] = 1.0
            phi_data[g1] = pd.Series(identity, index=genome_list)

    phi_df = pd.DataFrame(phi_data, index=genome_list)  # (G × G)

    # Compute chi: max off-diagonal per column
    chi_vals = {}
    for g1 in genome_list:
        col = phi_df[g1].copy()
        col[g1] = 0.0  # exclude diagonal
        chi_vals[g1] = col.max()
    chi = pd.Series(chi_vals)

    # Validation
    col_sums = phi_df.sum(axis=0)
    if abs(col_sums - 1.0).max() > 0.01:
        typer.echo(f"[xmap] WARNING: phi columns do not sum to 1.0 (max deviation={abs(col_sums - 1.0).max():.4f})")

    return phi_df, chi


# --------------------------------------------------------------------------------------
# Calling: per-shard file (call each barcode once)
# --------------------------------------------------------------------------------------

def _call_from_shard_file(
    shard_path: Path,
    eta: pd.Series,
    topk: Dict[str, List[str]],
    cfg: MergeConfig,
    out_handle,
    phi_matrix: Optional[pd.DataFrame] = None,
    chi: Optional[pd.Series] = None,
) -> int:
    """
    Stream calls from a shard file without materializing all barcodes in memory.

    Requires shard rows to be grouped by barcode. This is satisfied if Pass 1 writes
    each shard with rows sorted by ["barcode","read_id","genome"].
    """
    called = 0

    cur_bc: Optional[str] = None
    parts: List[pd.DataFrame] = []

    def _flush() -> None:
        nonlocal called, cur_bc, parts
        if cur_bc is None or not parts:
            return
        L_block = pd.concat(parts, ignore_index=True) if len(parts) > 1 else parts[0]
        cand = topk.get(cur_bc, [])
        res = _select_model_for_barcode(L_block, eta, cfg, cand,
                                        phi_matrix=phi_matrix, chi=chi)

        out_handle.write(
            "\t".join(
                [
                    cur_bc,
                    str(res.get("call", "")),
                    str(res.get("model", "")),
                    str(res.get("genome_1", "")),
                    str(res.get("genome_2", "")),
                    f"{float(res.get('alpha')):.6g}" if res.get("alpha") is not None else "",
                    f"{float(res.get('rho')):.6g}" if res.get("rho") is not None else "",
                    str(int(res.get("n_reads", 0))),
                    f"{float(res.get('n_reads_eff', 0.0)):.6g}",
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
                    f"{float(res.get('purity_best')):.6g}" if res.get("purity_best") is not None else "",
                    "1" if res.get("xmap_activated") else "0",
                    f"{float(res.get('chi_g1')):.6g}" if res.get("chi_g1") is not None else "",
                ]
            )
            + "\n"
        )
        called += 1
        parts = []

    for chunk in _iter_shard_chunks(shard_path, cfg):
        # keep only needed columns and ensure order is preserved        
        chunk2 = chunk[["barcode", "read_id", "genome", "L", "L_amb", "w_read"]]
        
        # --- OPTIMIZED MONOTONIC CHECK ---
        b = chunk2["barcode"]
        # Only perform the expensive check if there's actually a transition to check
        if b.nunique() > 1:
            if not b.is_monotonic_increasing:
                raise RuntimeError(
                   f"{shard_path.name} is not sorted by barcode; cannot stream safely. "
                    "This usually happens if you pointed Pass 3 to worker shards (w*/) "
                    "instead of merged shards. Delete shards and rerun Pass 1.5."
                    )
        # ---------------------------------
        # chunk is expected sorted by barcode; groupby preserves order with sort=False
        for bc, sub in chunk2.groupby("barcode", sort=False):
            bc_s = str(bc)            
            sub2 = sub[["read_id", "genome", "L", "w_read"]].copy()
            
            if cur_bc is None:
                cur_bc = bc_s
                parts = [sub2]
                continue

            if bc_s == cur_bc:
                parts.append(sub2)
            else:
                _flush()
                cur_bc = bc_s
                parts = [sub2]

    _flush()
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
    pass3_workers: Optional[int] = None,
    shards: Optional[int] = None,
    chunk_rows: Optional[int] = None,
    pass2_chunksize: Optional[int] = None,
    winner_only: Optional[bool] = None,
    # model
    alpha_grid: Optional[float] = None,
    rho_grid: Optional[float] = None,
    max_alpha: Optional[float] = None,
    # optional read-filter
    max_hits: Optional[int] = None,
    hits_delta_mapq: Optional[float] = None,
    max_rows_per_read_guard: Optional[int] = None,
    # read quality filters
    min_mapq_genotyping: Optional[int] = None,
    max_xa_genotyping: Optional[int] = None,
    # fusion
    beta: Optional[float] = None,
    w_as: Optional[float] = None,
    w_mapq: Optional[float] = None,
    w_nm: Optional[float] = None,
    ambient_const: Optional[float] = None,
    w_confident: Optional[float] = None,
    w_ambiguous: Optional[float] = None,
  
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
    topk_reclass: Optional[bool] = None,
    # fixed-eta override
    eta_file: Optional[Path] = None,
    # DuckDB acceleration
    pass1_duckdb: Optional[bool] = None,
    pass2_duckdb: Optional[bool] = None,
    # Cross-mapping (Level 2)
    xmap_chi_threshold: Optional[float] = None,
    xmap_phi_min_reads: Optional[int] = None,
    xmap_phi_iters: Optional[int] = None,
    xmap_phi_file: Optional[Path] = None,
    xmap_enabled: Optional[bool] = None,
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
    
    # worker policy: pass3 defaults to threads if not provided
    pass3_workers_eff = int(pass3_workers) if pass3_workers is not None else int(threads_eff)
  
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
    if min_mapq_genotyping is not None:
        cfg.min_mapq_genotyping = int(min_mapq_genotyping)
    if max_xa_genotyping is not None:
        cfg.max_xa_genotyping = int(max_xa_genotyping)

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
    if w_confident is not None:
        cfg.w_confident = float(w_confident)
    if w_ambiguous is not None:
        cfg.w_ambiguous = float(w_ambiguous)

    if alpha_grid is not None:
        cfg.alpha_grid = float(alpha_grid)
    if rho_grid is not None:
        cfg.rho_grid = float(rho_grid)
    if max_alpha is not None:
        cfg.max_alpha = float(max_alpha)

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
    if topk_reclass is not None:
        cfg.topk_reclass = bool(topk_reclass)
    if pass1_duckdb is not None:
        cfg.pass1_duckdb = bool(pass1_duckdb)
    if pass2_duckdb is not None:
        cfg.pass2_duckdb = bool(pass2_duckdb)

    # Cross-mapping (Level 2)
    if xmap_chi_threshold is not None:
        cfg.xmap_chi_threshold = float(xmap_chi_threshold)
    if xmap_phi_min_reads is not None:
        cfg.xmap_phi_min_reads = int(xmap_phi_min_reads)
    if xmap_phi_iters is not None:
        cfg.xmap_phi_iters = int(xmap_phi_iters)
    if xmap_enabled is not None:
        cfg.xmap_enabled = bool(xmap_enabled)

    files = [Path(f) for f in glob.glob(assign_glob, recursive=True)]
    if not files:
        raise typer.BadParameter(f"No files matched: {assign_glob}")

    if cfg.alpha_grid <= 0 or cfg.rho_grid <= 0:
        raise ValueError("alpha_grid and rho_grid must be > 0")
    if not (0 < cfg.max_alpha <= 1.0):
        raise ValueError("max_alpha must be in (0,1]")

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
    typer.echo(f"[genotyping] weights: w_confident={cfg.w_confident} w_ambiguous={cfg.w_ambiguous}")
    typer.echo(f"[genotyping] gates: empty_bic_margin={cfg.empty_bic_margin} bic_margin={cfg.bic_margin} near_tie_margin={cfg.near_tie_margin}")
    typer.echo(f"[genotyping] grid: alpha_grid={cfg.alpha_grid} rho_grid={cfg.rho_grid} max_alpha={cfg.max_alpha}")
    use_pass1_duckdb = bool(cfg.pass1_duckdb) and _HAS_DUCKDB
    use_pass2_duckdb = bool(cfg.pass2_duckdb) and _HAS_DUCKDB
    typer.echo(
        f"[genotyping] pass1_duckdb={'on' if use_pass1_duckdb else 'off (fallback)'} "
        f"pass2_duckdb={'on' if use_pass2_duckdb else 'off (fallback)'}"
    )
    if cfg.empty_seed_bic_min is not None:
        typer.echo("[genotyping] NOTE: empty_seed_bic_min is currently not used in eta learning in this implementation.")

    C_path = outdir_eff / f"{sample_eff}_C_all.tsv.gz"
    N_path = outdir_eff / f"{sample_eff}_N_all.tsv.gz"
    eta_out = outdir_eff / f"{sample_eff}_eta.tsv.gz"
    calls_path = outdir_eff / f"{sample_eff}_cells_calls.tsv.gz"

    # -----------------------
    # Pass 1
    # -----------------------
    marker = shard_root / "_SHARD_FORMAT_v2_sorted_by_barcode.txt"

    # Clean stale shards from previous runs when resume is disabled
    if not resume:
        for wdir in shard_root.glob("w*"):
            if wdir.is_dir():
                shutil.rmtree(wdir)
        merged_dir = shard_root / "merged"
        if merged_dir.exists():
            shutil.rmtree(merged_dir)
        if marker.exists():
            marker.unlink()
        # Also clean Pass 2.75 reclass shards
        reclass_dir = tmp_dir / "L_shards_workers_reclass"
        if reclass_dir.exists():
            shutil.rmtree(reclass_dir)
        for p in (C_path, N_path):
            if p.exists():
                p.unlink()

    marker_ok = False
    if marker.exists():
        try:
            marker_ok = marker.read_text().strip() == "v2"
        except Exception:
            marker_ok = False

    has_worker_shards = any(shard_root.glob("w*/shard_*.tsv.gz"))
    has_merged_shards = any((shard_root / "merged").glob("shard_*.tsv.gz"))
    if resume and C_path.exists() and N_path.exists() and marker_ok and (has_worker_shards or has_merged_shards):    
        typer.echo("[1/4] Pass 1: resume (found C/N + shard spill + marker v2)")
        C_all = pd.read_csv(C_path, 
                            sep="\t", 
                            compression="gzip", 
                            dtype={"barcode": "string", "genome": "string", "C": "float64"})
        N_all = pd.read_csv(N_path,
                            sep="\t",
                            compression="gzip", dtype={"barcode": "string", "n_reads": "int64"})
              
    else:
        typer.echo(
            f"[1/4] Pass 1: streaming assign inputs → per-read posteriors + spill shards"
            f"{' [duckdb]' if use_pass1_duckdb else ''}"
        )

        C_list: List[pd.DataFrame] = []
        N_list: List[pd.DataFrame] = []

        cfg_dict = cfg.model_dump()
        args = [(i, str(f), cfg_dict, str(shard_root), bc_to_chunk, use_pass1_duckdb) for i, f in enumerate(files)]
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
        
        # Only write marker when shards were (re)generated by this code path
        marker.write_text("v2\n")

    all_genomes = C_all["genome"].astype(str).unique().tolist()

    topk: Dict[str, List[str]] = {}
    for bc, sub in C_all.groupby("barcode", sort=False):
        s = sub.sort_values("C", ascending=False)["genome"].astype(str).head(int(cfg.topk_genomes)).tolist()
        topk[str(bc)] = s
    
    # -----------------------
    # Pass 1.5: merge worker shards -> merged shards (globally sorted)
    # -----------------------
    typer.echo("[1.5/4] Pass 1.5: merging worker shards -> merged/ (globally sorted)")
    
    merged_path = shard_root / "merged"
    force_merge = not resume

    # If the user explicitly chose --no-resume, wipe the existing merged shards
    # to ensure a clean start.
    if force_merge and merged_path.exists():
        typer.echo(f"  - force=True: cleaning existing merged directory: {merged_path}")
        shutil.rmtree(merged_path)

    # Define the temp directory for the external sort
    pass15_tmp = Path(cfg.pass15_tmpdir) if cfg.pass15_tmpdir else (tmp_dir / "pass15_tmp")
    pass15_tmp.mkdir(parents=True, exist_ok=True)
  
    merged_root = _merge_worker_shards_pass15(
        shard_root=shard_root,
        shards=cfg.shards,
        force=force_merge,
        sort_mem=str(cfg.pass15_sort_mem),
        tmpdir=pass15_tmp,  # keeps sort temp out of merged/
    )
    
    shards_list = sorted(merged_root.glob("shard_*.tsv.gz"))
    if not shards_list:
        raise RuntimeError(f"Pass 1.5 produced no merged shards under {merged_root}")

    # -----------------------
    # Pass 1.75: exact N_all from merged shards (Fix #2)
    # -----------------------
    if resume and N_path.exists():
        # Optional: keep existing N if you trust it; but safest is to recompute when C was regenerated.
        typer.echo("[1.75/4] N_all: using existing N_all (resume)")
        N_all = pd.read_csv(N_path, sep="\t", compression="gzip", dtype={"barcode": "string", "n_reads": "int64"})
    else:
        typer.echo("[1.75/4] N_all: computing exact unique read counts from merged shards")
        N_all = _compute_N_all_from_merged_shards(merged_root, cfg)
        N_all.to_csv(N_path, sep="\t", index=False, compression="gzip")
        typer.echo(f"[1.75/4] wrote exact: {N_path.name}")

    # -----------------------
    # Pass 2
    # -----------------------
    if eta_file is not None:
        # Fixed-eta mode: load pre-computed profile, skip learning entirely
        typer.echo(f"[2/4] Pass 2: fixed-eta mode — loading from {eta_file}")
        eta_df = pd.read_csv(eta_file, sep="\t")
        idx_col = eta_df.columns[0]
        eta = eta_df.set_index(idx_col)["eta"].astype(float)
        eta.index = eta.index.astype(str)
        missing = set(all_genomes) - set(eta.index)
        if missing:
            raise ValueError(
                f"--eta-file is missing genomes present in config: {missing}"
            )
        extra = set(eta.index) - set(all_genomes)
        if extra:
            typer.echo(f"  WARNING: --eta-file has extra genomes (ignored): {extra}", err=True)
        eta = eta.reindex(all_genomes).fillna(0.0)
        s = eta.sum()
        if abs(s - 1.0) > 0.01:
            typer.echo(f"  WARNING: --eta-file values sum to {s:.4f} (expected 1.0); renormalizing.", err=True)
        eta = eta / (s + 1e-12)
        eta.to_frame("eta").to_csv(eta_out, sep="\t", compression="gzip")
        typer.echo(f"  eta saved: {eta_out.name}")
    elif resume and eta_out.exists():
        typer.echo("[2/4] Pass 2: resume (found eta)")
        eta_df = pd.read_csv(eta_out, sep="\t", compression="gzip")
        idx_col = eta_df.columns[0]
        eta = eta_df.set_index(idx_col)["eta"]
        eta.index = eta.index.astype(str)
        eta = eta.astype(float)
        eta = eta / (eta.sum() + 1e-12)
    else:
        typer.echo(
            f"[2/4] Pass 2: ambient learning (eta) from shard spill"
            f"{' [duckdb]' if use_pass2_duckdb else ''}"
        )
        eta, eta_meta = _learn_eta(C_all, N_all, shard_root, all_genomes, cfg)
        eta.to_frame("eta").to_csv(eta_out, sep="\t", compression="gzip")
        typer.echo(f"[2/4] eta saved: {eta_out.name}  seed_sizes={eta_meta.get('seed_sizes')}")

    # -----------------------
    # Pass 2.5: Cross-mapping profile (Level 2)
    # -----------------------
    phi_out = outdir_eff / f"{sample_eff}_phi_matrix.tsv.gz"
    chi_out = outdir_eff / f"{sample_eff}_chi.tsv"
    phi_path_str: Optional[str] = None
    chi_path_str: Optional[str] = None

    if cfg.xmap_enabled:
        if resume and phi_out.exists() and chi_out.exists():
            typer.echo("[xmap] resume (found phi/chi)")
            phi_path_str = str(phi_out)
            chi_path_str = str(chi_out)
        else:
            typer.echo("[xmap] estimating cross-mapping profile φ")
            phi_matrix, chi = _compute_phi_matrix(C_all, N_all, all_genomes, cfg)
            phi_matrix.to_csv(phi_out, sep="\t", compression="gzip")
            chi_df = pd.DataFrame({
                "genome": chi.index,
                "chi": chi.values,
                "level2_activated": chi.values > cfg.xmap_chi_threshold,
            })
            chi_df.to_csv(chi_out, sep="\t", index=False)
            phi_path_str = str(phi_out)
            chi_path_str = str(chi_out)
            for g1 in all_genomes:
                act = chi.get(g1, 0) > cfg.xmap_chi_threshold
                typer.echo(f"[xmap] {g1}: chi={chi.get(g1, 0):.3f} -> Level2={'ON' if act else 'off'}")

    # -----------------------
    # Pass 2.75: Top-K read re-classification (optional)
    # -----------------------
    reclass_merged_root: Optional[Path] = None

    if cfg.topk_reclass and int(cfg.topk_genomes) >= 2:
        reclass_shard_root = tmp_dir / "L_shards_workers_reclass"
        reclass_merged_candidate = reclass_shard_root / "merged"
        reclass_marker = reclass_shard_root / "_RECLASS_v1.txt"
        reclass_C_path = outdir_eff / f"{sample_eff}_C_all_reclass.tsv.gz"
        reclass_N_path = outdir_eff / f"{sample_eff}_N_all_reclass.tsv.gz"

        if resume and reclass_marker.exists() and reclass_C_path.exists() and reclass_N_path.exists():
            typer.echo("[2.75] Pass 2.75: resume (found re-classified shards)")
            C_all = pd.read_csv(reclass_C_path, sep="\t", compression="gzip",
                                dtype={"barcode": "string", "genome": "string", "C": "float64"})
            N_all = pd.read_csv(reclass_N_path, sep="\t", compression="gzip",
                                dtype={"barcode": "string", "n_reads": "int64"})
            reclass_merged_root = reclass_merged_candidate
        else:
            typer.echo(
                f"[2.75] Pass 2.75: re-classifying reads with top-K={cfg.topk_genomes} "
                f"per barcode ({len(files)} assign files, {len(topk):,} barcodes)"
            )
            reclass_shard_root.mkdir(parents=True, exist_ok=True)

            # Write topk for workers
            topk_path_275 = reclass_shard_root / f"{sample_eff}_topk.tsv.gz"
            _write_topk_tsv(topk, topk_path_275)

            cfg_dict_275 = cfg.model_dump()

            args275 = [
                (i, str(f), str(reclass_shard_root))
                for i, f in enumerate(files)
            ]

            C_list_275: List[pd.DataFrame] = []
            with ProcessPoolExecutor(
                max_workers=max(1, int(pass1_workers_eff)),
                initializer=_pass275_init_worker,
                initargs=(str(topk_path_275), cfg_dict_275, bc_to_chunk, use_pass1_duckdb),
            ) as ex:
                futs = [ex.submit(_pass275_worker_job, a) for a in args275]
                for fut in as_completed(futs):
                    C_part = fut.result()
                    if not C_part.empty:
                        C_list_275.append(C_part)

            # Aggregate new C_all
            if C_list_275:
                C_all = (
                    pd.concat(C_list_275, ignore_index=True)
                    .groupby(["barcode", "genome"], sort=False)["C"].sum().reset_index()
                )
            else:
                C_all = pd.DataFrame(columns=["barcode", "genome", "C"])

            if C_all.empty:
                typer.echo("[2.75] WARNING: re-classification produced no mass. Falling back to original shards.", err=True)
            else:
                C_all.to_csv(reclass_C_path, sep="\t", index=False, compression="gzip")

                # Merge re-classified worker shards
                typer.echo("[2.75] merging re-classified worker shards")
                reclass_pass15_tmp = tmp_dir / "pass275_sort_tmp"
                reclass_pass15_tmp.mkdir(parents=True, exist_ok=True)
                reclass_merged_root = _merge_worker_shards_pass15(
                    shard_root=reclass_shard_root,
                    shards=cfg.shards,
                    force=True,
                    sort_mem=str(cfg.pass15_sort_mem),
                    tmpdir=reclass_pass15_tmp,
                )

                # Recompute N_all from new merged shards
                typer.echo("[2.75] computing N_all from re-classified shards")
                N_all = _compute_N_all_from_merged_shards(reclass_merged_root, cfg)
                N_all.to_csv(reclass_N_path, sep="\t", index=False, compression="gzip")

                reclass_marker.write_text("v1\n")
                typer.echo(
                    f"[2.75] done: {len(C_all):,} (barcode, genome) entries, "
                    f"{N_all['n_reads'].sum():,} total reads"
                )

    # -----------------------
    # Pass 3: model selection / calls
    # -----------------------

    if resume and calls_path.exists():
        typer.echo("[3/4] Pass 3: resume (found calls)")
    else:
        # Use re-classified shards if Pass 2.75 ran, otherwise original
        if reclass_merged_root is not None:
            merged_root = reclass_merged_root
            typer.echo("[3/4] Pass 3: model selection from re-classified shards (parallel)")
        else:
            merged_root = shard_root / "merged"
            typer.echo("[3/4] Pass 3: model selection / calls from shards (parallel)")
        shards_list = sorted(merged_root.glob("shard_*.tsv.gz"))

        if not shards_list:
            raise RuntimeError(f"Pass 3: no shard_*.tsv.gz found under {merged_root}")
        
        pass3_tmp = tmp_dir / "pass3_parts"
        pass3_tmp.mkdir(parents=True, exist_ok=True)
        
        # Write topk once so workers can load from disk
        topk_path = pass3_tmp / f"{sample_eff}_topk.tsv.gz"
        if not topk_path.exists():
            _write_topk_tsv(topk, topk_path)

        # eta already saved at eta_out; ensure it exists
        if not eta_out.exists():
            raise RuntimeError(f"Pass 3: expected eta file missing: {eta_out}")

        # Make cfg pickling cheap/robust
        cfg_dict = cfg.model_dump()

        # Use threads_eff cores unless pass3_workers overrides; Pass 3 wants full CPU:
        pass3_workers = max(1, int(pass3_workers_eff))

        # Launch shard workers
        part_paths: Dict[str, Path] = {}
        jobs = []
        for shard in shards_list:
            part = pass3_tmp / (shard.name.replace(".tsv.gz", "") + ".calls.tsv.gz")
            part_paths[shard.name] = part
            jobs.append((str(shard), str(part)))

        called_by_shard: Dict[str, int] = {}
        with ProcessPoolExecutor(
            max_workers=pass3_workers,
            initializer=_pass3_init_worker,
            initargs=(str(eta_out), str(topk_path), cfg_dict, phi_path_str, chi_path_str),
        ) as ex:
            futs = [ex.submit(_pass3_process_one_shard, shard_str, part_str) for shard_str, part_str in jobs]
            for fut in as_completed(futs):
                shard_name, n_called = fut.result()
                called_by_shard[shard_name] = int(n_called)    

        # Deterministic concatenation in shard-sorted order into calls_path
        header = "\t".join(
            [
                "barcode",
                "call",
                "model",
                "genome_1",
                "genome_2",
                "alpha",
                "rho",
                "n_reads",
                "n_reads_eff",
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
                "purity_best",
                "xmap_activated",
                "chi_g1",
            ]
        )
        
        called_total = 0
        with gzip.open(calls_path, "wt") as out_final:
            out_final.write(header + "\n")
            for shard in shards_list:
                part = part_paths[shard.name]
                if not part.exists():
                    raise RuntimeError(f"Pass 3: missing part output for {shard.name}: {part}")
                with gzip.open(part, "rt") as pf:
                    # parts have no header; just copy lines
                    for line in pf:
                        out_final.write(line)
                called_total += int(called_by_shard.get(shard.name, 0))

        calls_df = pd.read_csv(calls_path, sep="\t", compression="gzip", usecols=["barcode"], dtype={"barcode":"string"})
        if calls_df["barcode"].duplicated().any():
            raise RuntimeError("Calls contain duplicate barcodes (unexpected after merged shards). Check Pass 1.5 merge and shard routing.")
      
        typer.echo(f"[3/4] calls written: {calls_path.name} (emitted rows={called_total})")
        typer.echo(f"[3/4] pass3_workers={pass3_workers} shards_processed={len(shards_list)}")
        

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
    pass3_workers: Optional[int] = typer.Option(None, "--pass3-workers"),
    shards: Optional[int] = typer.Option(None, "--shards"),
    chunk_rows: Optional[int] = typer.Option(None, "--chunk-rows"),
    pass2_chunksize: Optional[int] = typer.Option(None, "--pass2-chunksize"),
    winner_only: Optional[bool] = typer.Option(None, "--winner-only/--no-winner-only"),
    # optional read-filter
    max_hits: Optional[int] = typer.Option(None, "--max-hits"),
    hits_delta_mapq: Optional[float] = typer.Option(None, "--hits-delta-mapq"),
    max_rows_per_read_guard: Optional[int] = typer.Option(None, "--max-rows-per-read-guard"),
    # model
    alpha_grid: Optional[float] = typer.Option(None, "--alpha-grid", help="Alpha step for grid search (default 0.02)."),
    rho_grid: Optional[float] = typer.Option(None, "--rho-grid", help="Rho step for doublet grid search (default 0.05)."),
    max_alpha: Optional[float] = typer.Option(None, "--max-alpha", help="Max alpha for grid search (default 0.5)."),
  
    # fusion
    beta: Optional[float] = typer.Option(None, "--beta"),
    w_as: Optional[float] = typer.Option(None, "--w-as"),
    w_mapq: Optional[float] = typer.Option(None, "--w-mapq"),
    w_nm: Optional[float] = typer.Option(None, "--w-nm"),
    ambient_const: Optional[float] = typer.Option(None, "--ambient-const"),
    w_confident: Optional[float] = typer.Option(None, "--w-confident", help="Weight for non-ambiguous reads (default 1.0)."),
    w_ambiguous: Optional[float] = typer.Option(None, "--w-ambiguous", help="Weight for ambiguous reads (default 0.1)."),
  
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
    topk_reclass: Optional[bool] = typer.Option(
        None, "--topk-reclass/--no-topk-reclass",
        help="Re-classify reads within top-K genomes per barcode (Pass 2.75). "
             "Improves singlet detection for closely related genomes.",
    ),
    # DuckDB acceleration
    pass1_duckdb: Optional[bool] = typer.Option(
        None, "--pass1-duckdb/--pass1-no-duckdb",
        help="Use DuckDB for Pass 1 file reading (default: on). Falls back if duckdb unavailable.",
    ),
    pass2_duckdb: Optional[bool] = typer.Option(
        None, "--pass2-duckdb/--pass2-no-duckdb",
        help="Use DuckDB for Pass 2 eta computation (default: on). Falls back if duckdb unavailable.",
    ),
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
        pass3_workers=pass3_workers,
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
        w_confident=w_confident,
        w_ambiguous=w_ambiguous,
        #
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
        topk_reclass=topk_reclass,
        pass1_duckdb=pass1_duckdb,
        pass2_duckdb=pass2_duckdb,
        resume=resume,
    )


if __name__ == "__main__":
    app()
