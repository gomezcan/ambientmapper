#!/usr/bin/env python3
"""
src/ambientmapper/decontam.py

ambientmapper decontam v2 — model-informed purification + read drop list + post-clean cell validation

Goals
  1) Read-level cleaning: emit a drop list for BAM filtering.
  2) Conservative handling of ambiguous/indistinguishable, design-aware rescue when available.
  3) Mode-independence: dropping driven by assignment evidence (winner / scores / p-values), not genotyping internals.
  4) Genome-agnostic + scalable: supports 2..N genomes; outputs barnyard-like pre/post summaries without a combined genome.
  5) Parallelized across assignment chunk files (per-file workers → merge parts).

Key idea
  Decontam decides an AllowedSet(barcode) and then drops confident off-set reads, with an optional “Safe Keep”
  guard to avoid over-stripping homologous/shared regions.

Outputs
  - {sample}_reads_to_drop.tsv.gz
  - {sample}_barcode_policy.tsv.gz
  - {sample}_cells_calls.decontam.tsv.gz   (cells_calls + policy + post-clean metrics + optional reclassify)
  - Barnyard-like:
      {sample}_pre_barcode_genome_counts.tsv.gz
      {sample}_post_barcode_genome_counts.tsv.gz
      {sample}_pre_barcode_composition.tsv.gz
      {sample}_post_barcode_composition.tsv.gz
      {sample}_pre_contamination_bins.tsv.gz (if design)
      {sample}_post_contamination_bins.tsv.gz (if design)

Assumptions about assignment inputs
  The assignment TSVs contain per-read, per-genome rows and typically an `assigned_class` column with "winner"
  rows. If no winner labels exist, decontam can compute winners using AS/MAPQ/NM ordering.

Parallelization
  Runs one worker per assignment chunk file; each worker writes its own drop-list part and returns partial
  pre/post counters; the master merges them.

"""

from __future__ import annotations

import gzip
import json
import glob
import math
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Iterable

import numpy as np
import pandas as pd
import typer

from .data import get_default_layout_path
from .utils import (
    load_sample_to_wells,
    load_barcode_layout,
    build_well_to_sample,
    build_barcode_to_sample,
)

app = typer.Typer(
    help="Model-informed barcode + read-level decontamination decisions (v2).",
    add_completion=False,
    invoke_without_command=True,
    no_args_is_help=True,
)

# -------------------------
# Helpers: sample inference
# -------------------------
def _infer_sample_name_from_cells_calls(cells_calls: Path) -> str:
    bn = cells_calls.name
    return bn.replace("_cells_calls.tsv.gz", "").replace("_cells_calls.tsv", "")


def _infer_sample_root_from_cells_calls(cells_calls: Path) -> Optional[Path]:
    """
    Expect: <workdir>/<sample>/final/<sample>_cells_calls.tsv(.gz)
    Return: <workdir>/<sample> or None.
    """
    p = cells_calls.resolve()
    if p.parent.name != "final":
        return None
    sample_root = p.parent.parent
    return sample_root if sample_root.exists() else None


def _default_assign_glob(sample_root: Path, sample_name: str) -> str:
    return str(sample_root / "cell_map_ref_chunks" / f"{sample_name}_chunk*_filtered.tsv.gz")


# -------------------------
# Helpers: robust column picking + string normalization
# -------------------------
def _col_first(df: pd.DataFrame, cols: list[str]) -> Optional[str]:
    for c in cols:
        if c in df.columns:
            return c
    return None


def _norm_str(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() == "nan":
        return ""
    return s


# -------------------------
# Design key logic
# -------------------------
def _design_key(barcode: str, mode: str, n: int) -> str:
    """
    Extract the barcode key used for design matching.

    mode:
      - before-dash : take part before first '-'
      - full        : entire string
      - last        : last n characters of before-dash part
      - first       : first n characters of before-dash part

    Normalizes by removing separators like '_' ':' '.' that may appear in layout files.
    """
    s = str(barcode).strip()
    if mode == "full":
        base = s
    else:
        base = s.split("-")[0]

    base = base.replace("_", "").replace(":", "").replace(".", "")

    if mode == "before-dash":
        return base
    if n <= 0:
        return base
    if mode == "last":
        return base[-n:]
    if mode == "first":
        return base[:n]
    return base


def _build_bckey_to_expected(
    layout_path: Path,
    design_path: Path,
    *,
    design_bc_mode: str,
    design_bc_n: int,
) -> Dict[str, str]:
    """
    Build bc_key -> expected_genome map.

    Supports:
      (1) TSV with header: barcode  expected_genome
      (2) Pool-ranges design file: genome <tab> wells (A1-4,B1-4,...), using a 96-well layout.
          In this mode, sample names are treated as expected genomes.
    """
    design_path = Path(design_path)
    layout_path = Path(layout_path)

    # Direct barcode mapping TSV
    df = None
    try:
        df = pd.read_csv(design_path, sep="\t", dtype=str)
    except Exception:
        df = None

    if df is not None and {"barcode", "expected_genome"}.issubset(df.columns):
        df = df[["barcode", "expected_genome"]].astype(str)
        df["barcode"] = df["barcode"].map(lambda x: _design_key(x, design_bc_mode, design_bc_n))
        out = dict(zip(df["barcode"], df["expected_genome"]))
        return {k: _norm_str(v) for k, v in out.items() if _norm_str(k)}

    # Pool ranges + layout
    sample_to_wells = load_sample_to_wells(design_path)
    well_to_sample = build_well_to_sample(sample_to_wells)
    well_to_barcode = load_barcode_layout(layout_path)
    bc_to_sample = build_barcode_to_sample(well_to_barcode, well_to_sample)

    out: Dict[str, str] = {}
    for bc, sample in bc_to_sample.items():
        k = _design_key(bc, design_bc_mode, design_bc_n)
        out[str(k)] = _norm_str(sample)
    return out


# -------------------------
# AllowedSet policy
# -------------------------
@dataclass(frozen=True)
class Policy:
    barcode: str
    bc_key: str
    call: str
    model: str
    genome_1: str
    genome_2: str
    expected: str
    allowed_set: Tuple[str, ...]  # sorted unique
    strategy_used: str
    flags: Tuple[str, ...]  # sorted unique flags
    keep_preclean: bool  # e.g., ambiguous_policy=drop, empty, unknown_in_design_strict


def _parse_allowed_set(s: str) -> Tuple[str, ...]:
    if not s:
        return tuple()
    parts = [p.strip() for p in str(s).split(",") if p.strip()]
    parts = sorted(set(parts))
    return tuple(parts)


def _allowed_set_to_str(allowed: Tuple[str, ...]) -> str:
    return ",".join(allowed)


def _call_bucket(call: str, model: str) -> str:
    c = _norm_str(call)
    m = _norm_str(model)
    if c == "empty" or m == "empty":
        return "empty"
    if c == "doublet_confident" or m == "doublet":
        return "doublet"
    if c == "single_clean" or m == "single":
        return "single"
    if c == "indistinguishable":
        return "indistinguishable"
    if c.startswith("ambiguous"):
        return "ambiguous"
    # includes ambiguous_low_depth style or other subclasses
    return "ambiguous"


def _compute_policy_for_row(
    *,
    bc_full: str,
    bc_key: str,
    call: str,
    model: str,
    g1: str,
    g2: str,
    indist_set: str,
    expected: str,
    has_design: bool,
    strict_design_drop_mismatch: bool,
    ambiguous_policy: str,   # drop|design_rescue|top1_rescue|top12_rescue
    doublet_policy: str,     # top12|expected
    indist_policy: str,      # top12|expected
) -> Policy:
    flags = set()
    bucket = _call_bucket(call, model)

    expected = _norm_str(expected)
    g1 = _norm_str(g1)
    g2 = _norm_str(g2)
    call = _norm_str(call) or "unknown"
    model = _norm_str(model)

    # design mismatch handling
    keep_preclean = True
    strategy = "agnostic"

    if has_design:
        strategy = "design"
        if not expected:
            flags.add("unknown_in_design")
            if strict_design_drop_mismatch:
                keep_preclean = False
                flags.add("drop_unknown_in_design")
        else:
            # assess mismatch vs inferred genomes
            if expected == g1:
                flags.add("expected_is_g1")
            elif expected and g2 and expected == g2:
                flags.add("expected_is_g2")
            elif expected:
                # if indistinguishable_set includes expected, don't mark mismatch
                ind = set(_parse_allowed_set(indist_set))
                if expected not in ind:
                    flags.add("design_mismatch")

    allowed: Tuple[str, ...] = tuple()

    # Empty always dropped
    if bucket == "empty":
        keep_preclean = False
        flags.add("empty")
        return Policy(
            barcode=bc_full,
            bc_key=bc_key,
            call=call,
            model=model,
            genome_1=g1,
            genome_2=g2,
            expected=expected,
            allowed_set=tuple(),
            strategy_used="empty",
            flags=tuple(sorted(flags)),
            keep_preclean=False,
        )

    # Ambiguous policy gate (prevents hallucination)
    if bucket == "ambiguous":
        strategy = f"{strategy}|ambiguous:{ambiguous_policy}"
        if ambiguous_policy == "drop":
            keep_preclean = False
            flags.add("ambiguous_drop")
            allowed = tuple()
        elif ambiguous_policy == "design_rescue":
            if has_design and expected:
                allowed = (expected,)
                flags.add("ambiguous_design_rescue")
            else:
                keep_preclean = False
                flags.add("ambiguous_no_design_drop")
                allowed = tuple()
        elif ambiguous_policy == "top1_rescue":
            allowed = (g1,) if g1 else tuple()
            flags.add("ambiguous_top1_rescue")
        elif ambiguous_policy == "top12_rescue":
            s = set([x for x in (g1, g2) if x])
            allowed = tuple(sorted(s))
            flags.add("ambiguous_top12_rescue")
        else:
            # fallback: drop
            keep_preclean = False
            flags.add("ambiguous_policy_unknown_drop")
            allowed = tuple()

        return Policy(
            barcode=bc_full,
            bc_key=bc_key,
            call=call,
            model=model,
            genome_1=g1,
            genome_2=g2,
            expected=expected,
            allowed_set=allowed,
            strategy_used=strategy,
            flags=tuple(sorted(flags)),
            keep_preclean=keep_preclean,
        )

    # Indistinguishable
    if bucket == "indistinguishable":
        strategy = f"{strategy}|indist:{indist_policy}"
        if has_design and expected and indist_policy == "expected":
            allowed = (expected,)
        else:
            ind = _parse_allowed_set(indist_set)
            if ind:
                allowed = ind
            else:
                s = set([x for x in (g1, g2) if x])
                allowed = tuple(sorted(s))
        return Policy(
            barcode=bc_full,
            bc_key=bc_key,
            call=call,
            model=model,
            genome_1=g1,
            genome_2=g2,
            expected=expected,
            allowed_set=allowed,
            strategy_used=strategy,
            flags=tuple(sorted(flags)),
            keep_preclean=keep_preclean,
        )

    # Doublet
    if bucket == "doublet":
        strategy = f"{strategy}|doublet:{doublet_policy}"
        if has_design and expected and doublet_policy == "expected":
            allowed = (expected,)
        else:
            s = set([x for x in (g1, g2) if x])
            allowed = tuple(sorted(s))
        return Policy(
            barcode=bc_full,
            bc_key=bc_key,
            call=call,
            model=model,
            genome_1=g1,
            genome_2=g2,
            expected=expected,
            allowed_set=allowed,
            strategy_used=strategy,
            flags=tuple(sorted(flags)),
            keep_preclean=keep_preclean,
        )

    # Single (or other non-ambiguous, non-doublet)
    # If design exists and expected is known, enforce expected; else use genome_1
    strategy = f"{strategy}|single"
    if has_design and expected:
        allowed = (expected,)
    else:
        allowed = (g1,) if g1 else tuple()

    return Policy(
        barcode=bc_full,
        bc_key=bc_key,
        call=call,
        model=model,
        genome_1=g1,
        genome_2=g2,
        expected=expected,
        allowed_set=allowed,
        strategy_used=strategy,
        flags=tuple(sorted(flags)),
        keep_preclean=keep_preclean,
    )


# -------------------------
# Barnyard summaries
# -------------------------
def _write_barnyard_summaries(
    out_dir: Path,
    sample_name: str,
    prefix: str,
    counts_map: Dict[Tuple[str, str], int],
    expected_map: Dict[str, str],
):
    """
    Generates:
      1) *_barcode_genome_counts.tsv.gz  (long)
      2) *_barcode_composition.tsv.gz
      3) *_contamination_bins.tsv.gz   (only if expected_map is non-empty)
    """
    if not counts_map:
        return

    rows = [{"barcode": bc, "genome": gn, "n_winner_reads": int(n)} for (bc, gn), n in counts_map.items()]
    df_counts = pd.DataFrame.from_records(rows)

    out_counts = out_dir / f"{sample_name}_{prefix}_barcode_genome_counts.tsv.gz"
    df_counts.to_csv(out_counts, sep="\t", index=False, compression="gzip")

    # Composition
    totals = (
        df_counts.groupby("barcode", as_index=False)["n_winner_reads"]
        .sum()
        .rename(columns={"n_winner_reads": "total_reads"})
    )

    top_df = (
        df_counts.sort_values(["barcode", "n_winner_reads"], ascending=[True, False])
        .drop_duplicates("barcode")[["barcode", "genome", "n_winner_reads"]]
        .rename(columns={"genome": "top_genome", "n_winner_reads": "top_reads"})
    )

    comp = totals.merge(top_df, on="barcode", how="left")
    comp["top_frac"] = comp["top_reads"] / comp["total_reads"].replace(0, np.nan)

    # expected genome
    if expected_map:
        comp["expected_genome"] = comp["barcode"].map(expected_map).map(_norm_str)
        comp.loc[comp["expected_genome"] == "", "expected_genome"] = np.nan
    else:
        comp["expected_genome"] = np.nan

    # expected_reads
    exp_counts = (
        df_counts.rename(columns={"n_winner_reads": "expected_reads"})
        .merge(comp[["barcode", "expected_genome"]], on="barcode", how="left")
    )
    exp_counts = exp_counts[exp_counts["genome"] == exp_counts["expected_genome"]]
    exp_reads = exp_counts.groupby("barcode", as_index=False)["expected_reads"].sum()

    comp = comp.merge(exp_reads, on="barcode", how="left")
    comp["expected_reads"] = comp["expected_reads"].fillna(0).astype(int)

    comp["expected_frac"] = comp["expected_reads"] / comp["total_reads"].replace(0, np.nan)
    comp.loc[comp["expected_genome"].isna(), "expected_frac"] = np.nan
    comp["contamination_rate"] = 1.0 - comp["expected_frac"]

    out_comp = out_dir / f"{sample_name}_{prefix}_barcode_composition.tsv.gz"
    comp.to_csv(out_comp, sep="\t", index=False, compression="gzip")

    # Bins (design-known only)
    if expected_map:
        valid_comp = comp[(comp["expected_genome"].notna()) & (comp["expected_genome"].astype(str) != "")].copy()
        if valid_comp.empty:
            return

        bins = [0, 0.01, 0.05, 0.10, 0.20, 0.50, 1.0000001]
        labels = ["0-1%", "1-5%", "5-10%", "10-20%", "20-50%", "50-100%"]
        valid_comp["bin_label"] = pd.cut(
            valid_comp["contamination_rate"],
            bins=bins,
            labels=labels,
            include_lowest=True,
            right=False,
        )

        summary = (
            valid_comp.groupby(["expected_genome", "bin_label"], dropna=False, observed=False)
            .size()
            .reset_index(name="n_barcodes")
        )

        out_bins = out_dir / f"{sample_name}_{prefix}_contamination_bins.tsv.gz"
        summary.to_csv(out_bins, sep="\t", index=False, compression="gzip")


# -------------------------
# Assignment processing: winner + safe keep
# -------------------------
def _has_cols(df: pd.DataFrame, cols: Iterable[str]) -> bool:
    have = set(df.columns)
    return all(c in have for c in cols)


def _compute_winner_rows(
    chunk: pd.DataFrame,
    *,
    read_id_col: str,
    barcode_col: str,
    genome_col: str,
    class_col: str,
) -> pd.Series:
    """
    Return boolean mask for winner rows.
    If class_col exists, uses (assigned_class == "winner").
    Else computes winners using AS/MAPQ/NM ordering:
        max AS, then max MAPQ, then min NM.
    """
    if class_col in chunk.columns:
        return chunk[class_col].astype(str).eq("winner")

    # compute winner by ordering; requires AS, MAPQ, NM
    if not _has_cols(chunk, ["AS", "MAPQ", "NM"]):
        raise ValueError("No class_col winners and missing AS/MAPQ/NM; cannot compute winners.")
    rid = chunk[barcode_col].astype(str) + "::" + chunk[read_id_col].astype(str)

    # sort so best comes first
    tmp = chunk.copy()
    tmp["_rid"] = rid
    tmp["_AS"] = pd.to_numeric(tmp["AS"], errors="coerce").fillna(-999999)
    tmp["_MAPQ"] = pd.to_numeric(tmp["MAPQ"], errors="coerce").fillna(-999999)
    tmp["_NM"] = pd.to_numeric(tmp["NM"], errors="coerce").fillna(999999)

    tmp = tmp.sort_values(by=["_rid", "_AS", "_MAPQ", "_NM"], ascending=[True, False, False, True])
    first_idx = tmp.drop_duplicates("_rid", keep="first").index
    mask = chunk.index.isin(first_idx)
    return pd.Series(mask, index=chunk.index)


def _is_confident_winner(
    chunk: pd.DataFrame,
    *,
    is_winner: pd.Series,
    p_as_col: str,
    decontam_alpha: Optional[float],
    require_p_as: bool,
) -> pd.Series:
    """
    Confidence:
      - baseline: winner rows
      - optional override: p_as <= alpha marks confident winner (even if not winner)
      - if require_p_as: only those with p_as<=alpha are confident (winner label ignored)
    """
    if decontam_alpha is None or p_as_col not in chunk.columns:
        return is_winner

    p_raw = chunk[p_as_col]
    p_vals = pd.to_numeric(p_raw, errors="coerce")
    has_p = p_vals.notna()
    is_override = has_p & (p_vals <= float(decontam_alpha))

    if require_p_as:
        return is_override
    # if p present and <= alpha => confident; if p missing => use winner label
    return is_override | ((~has_p) & is_winner)


def _safe_keep_mask_as(
    chunk: pd.DataFrame,
    *,
    bc_full: pd.Series,
    read_id: pd.Series,
    genome: pd.Series,
    allowed1: pd.Series,
    allowed2: pd.Series,
    safe_keep_delta_as: Optional[int],
    safe_keep_mapq_min: Optional[int],
    safe_keep_nm_max: Optional[int],
) -> pd.Series:
    """
    Safe keep (AS margin based):
      Keep read if it maps to any allowed genome with AS within delta of best disallowed AS.
    This is evaluated per (barcode, read_id).

    Implementation assumes AS exists. If missing, returns all-False.
    """
    if safe_keep_delta_as is None:
        return pd.Series(False, index=chunk.index)

    if "AS" not in chunk.columns:
        return pd.Series(False, index=chunk.index)

    rid = bc_full.astype(str) + "::" + read_id.astype(str)
    AS = pd.to_numeric(chunk["AS"], errors="coerce").fillna(-999999).astype(float)

    is_allowed = genome.astype(str).eq(allowed1.fillna("").astype(str))
    if allowed2 is not None:
        is_allowed = is_allowed | genome.astype(str).eq(allowed2.fillna("").astype(str))

    # Optional constraints on the allowed hit
    if safe_keep_mapq_min is not None and "MAPQ" in chunk.columns:
        MAPQ = pd.to_numeric(chunk["MAPQ"], errors="coerce").fillna(-999999)
        is_allowed = is_allowed & (MAPQ >= int(safe_keep_mapq_min))

    if safe_keep_nm_max is not None and "NM" in chunk.columns:
        NM = pd.to_numeric(chunk["NM"], errors="coerce").fillna(999999)
        is_allowed = is_allowed & (NM <= int(safe_keep_nm_max))

    # per rid best allowed AS
    best_allowed_as = pd.Series(-999999.0, index=chunk.index)
    if is_allowed.any():
        best_allowed_as = AS.where(is_allowed).groupby(rid, observed=True).transform("max")

    # per rid best disallowed AS (disallowed = not in {allowed1, allowed2})
    is_disallowed = ~(
        genome.astype(str).eq(allowed1.fillna("").astype(str)) |
        genome.astype(str).eq(allowed2.fillna("").astype(str) if allowed2 is not None else "")
    )
    best_disallowed_as = AS.where(is_disallowed).groupby(rid, observed=True).transform("max")

    # mark rows belonging to reads where best_allowed_as >= best_disallowed_as - delta
    delta = float(safe_keep_delta_as)
    safe_keep_read = (best_allowed_as >= (best_disallowed_as - delta))

    # safe_keep is a per-row mask; we use it later as a per-read property by re-grouping
    return safe_keep_read.fillna(False)


# -------------------------
# Per-file worker
# -------------------------
def _process_one_assign_file(
    fp: Path,
    *,
    out_part: Path,
    valid_barcodes: set[str],
    policy_by_barcode: Dict[str, Policy],
    bc_key_mode: str,
    bc_key_n: int,
    chunksize: int,
    read_id_col: str,
    barcode_col: str,
    genome_col: str,
    class_col: str,
    p_as_col: str,
    decontam_alpha: Optional[float],
    require_p_as: bool,
    safe_keep_delta_as: Optional[int],
    safe_keep_mapq_min: Optional[int],
    safe_keep_nm_max: Optional[int],
) -> Tuple[Counter, Counter, Counter, Counter, int]:
    """
    Returns:
      pre_counts[(barcode, genome)] += n_confident_winner
      post_counts[(barcode, genome)] += n_confident_winner_surviving
      post_total_reads[barcode] += n_confident_winner_surviving (sum across genomes)
      post_allowed_reads[barcode] += n_confident_winner_surviving where genome ∈ AllowedSet
      n_drop_rows
    """
    pre_counts: Counter = Counter()
    post_counts: Counter = Counter()
    post_total_reads: Counter = Counter()
    post_allowed_reads: Counter = Counter()
    n_drop_rows = 0

    out_part.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_part, "wt") as fh:
        # no header here; master writes one header once
        for chunk in pd.read_csv(fp, sep="\t", chunksize=chunksize, dtype=str, low_memory=False):
            required = {read_id_col, barcode_col, genome_col}
            missing = required - set(chunk.columns)
            if missing:
                raise ValueError(f"File {fp.name} missing columns: {sorted(missing)}")

            bc_full = chunk[barcode_col].astype(str)
            mask_valid = bc_full.isin(valid_barcodes)

            # Policies per barcode: allowed1/allowed2 + keep_preclean
            # For invalid barcodes (not in calls), skip counting and dropping.
            pol = bc_full.map(lambda b: policy_by_barcode.get(str(b)))
            keep_preclean = pol.map(lambda p: bool(p.keep_preclean) if p else False)

            allowed1 = pol.map(lambda p: p.allowed_set[0] if p and len(p.allowed_set) >= 1 else "")
            allowed2 = pol.map(lambda p: p.allowed_set[1] if p and len(p.allowed_set) >= 2 else "")

            # Winner rows and confidence
            is_winner = _compute_winner_rows(
                chunk,
                read_id_col=read_id_col,
                barcode_col=barcode_col,
                genome_col=genome_col,
                class_col=class_col,
            )
            is_confident = _is_confident_winner(
                chunk,
                is_winner=is_winner,
                p_as_col=p_as_col,
                decontam_alpha=decontam_alpha,
                require_p_as=require_p_as,
            )

            # Only evaluate confident-winner rows for pre/post counts & drop decisions
            genome = chunk[genome_col].astype(str)
            read_id = chunk[read_id_col].astype(str)

            # Safe keep: evaluate per read (barcode+read_id) using AS margin
            safe_keep = _safe_keep_mask_as(
                chunk,
                bc_full=bc_full,
                read_id=read_id,
                genome=genome,
                allowed1=allowed1,
                allowed2=allowed2,
                safe_keep_delta_as=safe_keep_delta_as,
                safe_keep_mapq_min=safe_keep_mapq_min,
                safe_keep_nm_max=safe_keep_nm_max,
            )

            # mismatch among confident winners (winner genome is this row's genome on winner row)
            # drop only if: valid barcode + keep_preclean + confident winner + winner genome not in allowed set + not safe-kept
            is_allowed = genome.eq(allowed1) | genome.eq(allowed2)
            no_allowed = (allowed1 == "") & (allowed2 == "")
            mismatch_conf = mask_valid & keep_preclean & is_confident & (~no_allowed) & (~is_allowed)

            # Safe keep is computed on all rows; we want safe-keep property on the read:
            # if any row of that read is safe_keep True => safe_keep_read True.
            rid = bc_full + "::" + read_id
            safe_keep_read = safe_keep.groupby(rid, observed=True).transform("max").fillna(False)

            # apply safe keep to mismatch_conf rows
            drop_mismatch = mismatch_conf & (~safe_keep_read)

            # Also drop everything for barcodes not kept preclean (empty/ambiguous drop/unknown-in-design strict)
            drop_barcode = mask_valid & (~keep_preclean)
            # But only confident winners produce actual rows in drop list; for BAM filtering we still want
            # to drop reads irrespective of genome row, so we drop based on winner rows only to avoid duplicates.
            drop_read_row = is_confident & (drop_barcode | drop_mismatch)

            # PRE counts: confident winner rows for valid barcodes (regardless of keep_preclean)
            mask_pre = mask_valid & is_confident
            if mask_pre.any():
                vc = pd.DataFrame({"barcode": bc_full[mask_pre], "genome": genome[mask_pre]}).value_counts()
                for (b, g), n in vc.items():
                    pre_counts[(str(b), str(g))] += int(n)

            # POST counts: confident winner rows for valid barcodes that are NOT dropped
            mask_post = mask_valid & is_confident & (~drop_read_row)
            if mask_post.any():
                vc2 = pd.DataFrame({"barcode": bc_full[mask_post], "genome": genome[mask_post]}).value_counts()
                for (b, g), n in vc2.items():
                    post_counts[(str(b), str(g))] += int(n)

                # also totals and allowed totals per barcode for post-clean gating
                # total post winners:
                tot = bc_full[mask_post].value_counts()
                for b, n in tot.items():
                    post_total_reads[str(b)] += int(n)
                # allowed post winners:
                allowed_post_mask = mask_post & (genome.eq(allowed1) | genome.eq(allowed2))
                if allowed_post_mask.any():
                    tot2 = bc_full[allowed_post_mask].value_counts()
                    for b, n in tot2.items():
                        post_allowed_reads[str(b)] += int(n)

            # Emit drop list rows (winner/confident rows only) to avoid duplicates
            if drop_read_row.any():
                reason = pd.Series("", index=chunk.index, dtype=object)
                reason.loc[drop_barcode & is_confident] = "drop_barcode_policy"
                reason.loc[drop_mismatch & is_confident] = "mismatch_winner"

                p_raw = chunk[p_as_col] if p_as_col in chunk.columns else pd.Series("", index=chunk.index)

                out_df = pd.DataFrame(
                    {
                        "read_id": read_id,
                        "barcode": bc_full,
                        "bc_key": bc_full.map(lambda x: _design_key(x, bc_key_mode, bc_key_n)),
                        "allowed_set": (allowed1.fillna("").astype(str) + ("," + allowed2.fillna("").astype(str))).str.strip(","),
                        "winner_genome": genome,
                        "p_as": p_raw.fillna("").astype(str),
                        "reason": reason,
                    }
                )
                n_this = int(drop_read_row.sum())
                n_drop_rows += n_this
                out_df.loc[drop_read_row].to_csv(fh, sep="\t", header=False, index=False)

    return pre_counts, post_counts, post_total_reads, post_allowed_reads, n_drop_rows


# -------------------------
# Post-clean gating + reclassify
# -------------------------
def _post_clean_call(
    *,
    pre_call: str,
    keep_post: bool,
    post_top1: str,
    post_top1_frac: float,
    post_top2: str,
    post_top2_frac: float,
    reclassify_after_clean: bool,
    reclassify_doublet_minor_frac: float,
    reclassify_single_purity_min: float,
) -> str:
    if not keep_post:
        return "empty_postclean"
    if not reclassify_after_clean:
        return pre_call or "kept_postclean"

    # Use post composition to update high-level label
    # If post has a clear dominant genome:
    if post_top1 and post_top1_frac >= reclassify_single_purity_min:
        # if pre-call was doublet but minor vanished
        if str(pre_call).startswith("doublet") and (post_top2_frac < reclassify_doublet_minor_frac):
            return "single_postclean_from_doublet"
        return "single_postclean"

    # Otherwise, ambiguous post-clean
    return "ambiguous_postclean"


def _compute_post_composition_from_counts(
    post_counts: Dict[Tuple[str, str], int]
) -> Dict[str, Dict[str, float]]:
    """
    Return per-barcode:
      total
      top1 genome + frac
      top2 genome + frac
    """
    by_bc: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for (bc, gn), n in post_counts.items():
        by_bc[str(bc)].append((str(gn), int(n)))

    out: Dict[str, Dict[str, float]] = {}
    for bc, items in by_bc.items():
        items_sorted = sorted(items, key=lambda x: x[1], reverse=True)
        total = float(sum(n for _, n in items_sorted)) or 0.0
        top1, n1 = (items_sorted[0] if items_sorted else ("", 0))
        top2, n2 = (items_sorted[1] if len(items_sorted) > 1 else ("", 0))
        out[bc] = {
            "post_total_reads": total,
            "post_top1_genome": top1,
            "post_top1_frac": (n1 / total) if total > 0 else 0.0,
            "post_top2_genome": top2,
            "post_top2_frac": (n2 / total) if total > 0 else 0.0,
        }
    return out


# -------------------------
# Main command
# -------------------------
@app.callback(invoke_without_command=True)
def decontam_cmd(
    cells_calls: Path = typer.Option(..., exists=True, readable=True, help="*_cells_calls.tsv(.gz) from genotyping."),
    out_dir: Path = typer.Option(..., help="Output directory for decontam artifacts."),

    # Parallel
    threads: int = typer.Option(1, "--threads", help="Parallel workers over assignment files."),
    tmp_dir: Optional[Path] = typer.Option(None, "--tmp-dir", help="Temp directory for per-file drop parts."),

    # Design / Layout
    design_file: Optional[Path] = typer.Option(None, "--design-file", readable=True, help="Optional design file. Enables design-aware mode."),
    layout_file: str = typer.Option("DEFAULT", "--layout-file", help='Layout path for well-range design files, or "DEFAULT".'),
    strict_design_drop_mismatch: bool = typer.Option(
        True,
        "--strict-design-drop-mismatch/--no-strict-design-drop-mismatch",
        help="If strict, drop barcodes whose bc_key is not present in the design (unknown wells).",
    ),

    # Policies
    ambiguous_policy: str = typer.Option(
        "design_rescue",
        "--ambiguous-policy",
        help="Ambiguous handling: drop|design_rescue|top1_rescue|top12_rescue",
    ),
    doublet_policy: str = typer.Option(
        "top12",
        "--doublet-policy",
        help="Doublet handling: top12 (keep both) | expected (in-silico sort to expected if design known).",
    ),
    indist_policy: str = typer.Option(
        "top12",
        "--indist-policy",
        help="Indistinguishable handling: top12 | expected (if design known).",
    ),

    # Post-clean retention thresholds
    min_reads_post_clean: int = typer.Option(
        100,
        "--min-reads-post-clean",
        help="Keep barcode post-clean only if allowed/expected genomes have >= this many post-clean confident winner reads.",
    ),
    min_allowed_frac_post_clean: float = typer.Option(
        0.9,
        "--min-allowed-frac-post-clean",
        help="Keep barcode post-clean only if (post_allowed_reads/post_total_reads) >= this threshold (if total>0).",
    ),

    # Reclassify after cleaning
    reclassify_after_clean: bool = typer.Option(
        False,
        "--reclassify-after-clean/--no-reclassify-after-clean",
        help="If enabled, compute call_postclean based on post-clean composition.",
    ),
    reclassify_doublet_minor_frac: float = typer.Option(
        0.05,
        "--reclassify-doublet-minor-frac",
        help="If post top2 fraction < this, doublet can reclassify to single_postclean_from_doublet.",
    ),
    reclassify_single_purity_min: float = typer.Option(
        0.7,
        "--reclassify-single-purity-min",
        help="Post-clean top1 fraction threshold for single_postclean.",
    ),

    # Assignment inputs
    assignments: Optional[Path] = typer.Option(None, "--assignments", exists=True, readable=True),
    assign_glob: Optional[str] = typer.Option(None, "--assign-glob"),

    # Columns
    read_id_col: str = typer.Option("Read", "--read-id-col"),
    barcode_col: str = typer.Option("BC", "--barcode-col"),
    genome_col: str = typer.Option("Genome", "--genome-col"),
    class_col: str = typer.Option("assigned_class", "--class-col"),
    p_as_col: str = typer.Option("p_as", "--p-as-col"),

    # Winner confidence logic
    decontam_alpha: Optional[float] = typer.Option(None, "--decontam-alpha", help="If set, p_as <= alpha can define confidence."),
    require_p_as: bool = typer.Option(False, "--require-p-as", help="If set, requires p_as <= alpha for confidence."),

    # Safe keep (AS margin based)
    safe_keep_delta_as: Optional[int] = typer.Option(
        3, "--safe-keep-delta-as",
        help="Safe keep: if best allowed AS within delta of best disallowed AS, keep read (even if winner is disallowed). Set to 0/None to disable.",
    ),
    safe_keep_mapq_min: Optional[int] = typer.Option(
        None, "--safe-keep-mapq-min",
        help="Optional: require allowed hit MAPQ >= this to qualify for safe keep.",
    ),
    safe_keep_nm_max: Optional[int] = typer.Option(
        None, "--safe-keep-nm-max",
        help="Optional: require allowed hit NM <= this to qualify for safe keep.",
    ),

    # Misc
    sample_name: Optional[str] = typer.Option(None, "--sample-name"),
    chunksize: int = typer.Option(1_000_000, "--chunksize"),
    design_bc_mode: str = typer.Option("before-dash", "--design-bc-mode"),
    design_bc_n: int = typer.Option(10, "--design-bc-n"),
):
    """
    Decontam v2:
      - Builds AllowedSet per barcode from genotyping calls + optional design.
      - Streams assignment evidence and writes a read drop list.
      - Produces pre/post barnyard summaries and post-clean barcode gating.
      - Optional post-clean reclassification.
      - Parallelized over assignment chunk files.
    """
    # -------------------------
    # Setup
    # -------------------------
    cells_calls = cells_calls.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if sample_name is None:
        sample_name = _infer_sample_name_from_cells_calls(cells_calls)
    sample_root = _infer_sample_root_from_cells_calls(cells_calls)

    if design_file is not None:
        design_file = Path(design_file).expanduser().resolve()
        if not design_file.exists():
            raise FileNotFoundError(f"--design-file not found: {design_file}")

    if assignments is None and (assign_glob is None or not assign_glob.strip()):
        if sample_root is not None:
            assign_glob = _default_assign_glob(sample_root, sample_name)
            typer.echo(f"[decontam] auto --assign-glob: {assign_glob}")

    # temp dir for parts
    tmp_dir = (tmp_dir or (out_dir / "tmp_decontam")).expanduser().resolve()
    drops_dir = tmp_dir / "drops"
    drops_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------
    # Load cell calls
    # -------------------------
    typer.echo(f"[decontam] Loading cell calls: {cells_calls}")
    calls = pd.read_csv(cells_calls, sep="\t", dtype=str)

    if "barcode" not in calls.columns:
        raise ValueError(f"{cells_calls} missing required column: 'barcode'")

    # genotyping output columns (robustly)
    col_call = _col_first(calls, ["call", "Call"])
    col_model = _col_first(calls, ["model", "Model"])
    col_g1 = _col_first(calls, ["genome_1", "genome1", "g1", "best_genome", "top_genome"])
    col_g2 = _col_first(calls, ["genome_2", "genome2", "g2"])
    col_ind = _col_first(calls, ["indistinguishable_set", "indist_set", "indistinguishable"])
    if col_call is None:
        raise ValueError(f"{cells_calls} missing 'call' column (required).")
    if col_g1 is None:
        raise ValueError(f"{cells_calls} missing a top/best genome column (need genome_1 or best/top genome).")

    calls["barcode"] = calls["barcode"].astype(str)
    calls["bc_key"] = calls["barcode"].map(lambda x: _design_key(x, design_bc_mode, design_bc_n))

    valid_barcodes = set(calls["barcode"].unique())

    # -------------------------
    # Design expected map (bc_key -> expected)
    # -------------------------
    bckey_to_expected: Dict[str, str] = {}
    has_design = False
    if design_file is not None:
        typer.echo(f"[decontam] Design-aware mode: {design_file}")
        has_design = True
        if layout_file == "DEFAULT":
            layout_path = get_default_layout_path()
        else:
            layout_path = Path(layout_file).expanduser().resolve()
            if not layout_path.exists():
                raise FileNotFoundError(f"--layout-file not found: {layout_path}")

        bckey_to_expected = _build_bckey_to_expected(
            layout_path,
            design_file,
            design_bc_mode=design_bc_mode,
            design_bc_n=design_bc_n,
        )
        if not bckey_to_expected:
            raise ValueError(f"Design parsing produced an empty bc_key->expected map: {design_file}")
    else:
        typer.echo("[decontam] No design file provided: agnostic mode.")

    # expected map per barcode for plotting/bins
    barcode_to_expected_map: Dict[str, str] = {}
    if has_design:
        barcode_to_expected_map = {b: _norm_str(bckey_to_expected.get(k, "")) for b, k in zip(calls["barcode"], calls["bc_key"])}
    else:
        # agnostic expected for plotting = genome_1
        barcode_to_expected_map = {b: _norm_str(g) for b, g in zip(calls["barcode"], calls[col_g1])}

    # -------------------------
    # Build per-barcode Policy and write policy table
    # -------------------------
    policy_rows: List[dict] = []
    policy_by_barcode: Dict[str, Policy] = {}

    for row in calls.itertuples(index=False):
        bc_full = _norm_str(getattr(row, "barcode", ""))
        bc_key = _norm_str(getattr(row, "bc_key", ""))
        call = _norm_str(getattr(row, col_call, ""))
        model = _norm_str(getattr(row, col_model, "")) if col_model else ""
        g1 = _norm_str(getattr(row, col_g1, ""))
        g2 = _norm_str(getattr(row, col_g2, "")) if col_g2 else ""
        indist = _norm_str(getattr(row, col_ind, "")) if col_ind else ""

        expected = _norm_str(bckey_to_expected.get(bc_key, "")) if has_design else ""

        pol = _compute_policy_for_row(
            bc_full=bc_full,
            bc_key=bc_key,
            call=call,
            model=model,
            g1=g1,
            g2=g2,
            indist_set=indist,
            expected=expected,
            has_design=has_design,
            strict_design_drop_mismatch=strict_design_drop_mismatch,
            ambiguous_policy=ambiguous_policy,
            doublet_policy=doublet_policy,
            indist_policy=indist_policy,
        )
        policy_by_barcode[bc_full] = pol

        policy_rows.append(
            {
                "barcode": pol.barcode,
                "bc_key": pol.bc_key,
                "call": pol.call,
                "model": pol.model,
                "genome_1": pol.genome_1,
                "genome_2": pol.genome_2,
                "expected_genome": pol.expected,
                "allowed_set": _allowed_set_to_str(pol.allowed_set),
                "keep_preclean": bool(pol.keep_preclean),
                "strategy_used": pol.strategy_used,
                "flags": ";".join(pol.flags),
            }
        )

    policy_path = out_dir / f"{sample_name}_barcode_policy.tsv.gz"
    pd.DataFrame.from_records(policy_rows).to_csv(policy_path, sep="\t", index=False, compression="gzip")
    typer.echo(f"[decontam] Wrote {policy_path}")

    # -------------------------
    # Identify assignment files
    # -------------------------
    input_files: List[Path] = []
    if assignments is not None:
        input_files.append(Path(assignments).expanduser().resolve())

    if assign_glob is not None:
        assign_glob = assign_glob.strip() or None

    if assign_glob:
        found = sorted(glob.glob(assign_glob, recursive=True))
        if not found:
            raise FileNotFoundError(
                f"[decontam] No assignment files matched: {assign_glob}\n"
                f"Expected files like: {sample_name}_chunk*_filtered.tsv.gz under cell_map_ref_chunks/."
            )
        input_files.extend([Path(f).resolve() for f in found])

    input_files = sorted(set(input_files))
    if not input_files:
        typer.echo("[decontam] No assignment files found. Skipping read-level logic.")

    # -------------------------
    # Parallel per-file processing
    # -------------------------
    pre_counts: Counter = Counter()
    post_counts: Counter = Counter()
    post_total_reads: Counter = Counter()
    post_allowed_reads: Counter = Counter()
    drop_total = 0

    reads_to_drop_path: Optional[Path] = None

    if input_files:
        threads = max(1, int(threads))
        typer.echo(f"[decontam] Processing {len(input_files)} assignment files (threads={threads})")

        def _part_name(fp: Path) -> Path:
            return drops_dir / f"{fp.stem}.reads_to_drop.part.tsv.gz"

        part_paths: List[Path] = []

        # dispatch
        if threads == 1:
            for fp in input_files:
                part = _part_name(fp)
                part_paths.append(part)
                pc, qc, pt, pa, nd = _process_one_assign_file(
                    fp,
                    out_part=part,
                    valid_barcodes=valid_barcodes,
                    policy_by_barcode=policy_by_barcode,
                    bc_key_mode=design_bc_mode,
                    bc_key_n=design_bc_n,
                    chunksize=chunksize,
                    read_id_col=read_id_col,
                    barcode_col=barcode_col,
                    genome_col=genome_col,
                    class_col=class_col,
                    p_as_col=p_as_col,
                    decontam_alpha=decontam_alpha,
                    require_p_as=require_p_as,
                    safe_keep_delta_as=(None if safe_keep_delta_as is None or safe_keep_delta_as <= 0 else safe_keep_delta_as),
                    safe_keep_mapq_min=safe_keep_mapq_min,
                    safe_keep_nm_max=safe_keep_nm_max,
                )
                pre_counts.update(pc)
                post_counts.update(qc)
                post_total_reads.update(pt)
                post_allowed_reads.update(pa)
                drop_total += nd
                typer.echo(f"  done: {fp.name} (drop_rows={nd})")
        else:
            with ProcessPoolExecutor(max_workers=threads) as ex:
                futs = {}
                for fp in input_files:
                    part = _part_name(fp)
                    part_paths.append(part)
                    fut = ex.submit(
                        _process_one_assign_file,
                        fp,
                        out_part=part,
                        valid_barcodes=valid_barcodes,
                        policy_by_barcode=policy_by_barcode,
                        bc_key_mode=design_bc_mode,
                        bc_key_n=design_bc_n,
                        chunksize=chunksize,
                        read_id_col=read_id_col,
                        barcode_col=barcode_col,
                        genome_col=genome_col,
                        class_col=class_col,
                        p_as_col=p_as_col,
                        decontam_alpha=decontam_alpha,
                        require_p_as=require_p_as,
                        safe_keep_delta_as=(None if safe_keep_delta_as is None or safe_keep_delta_as <= 0 else safe_keep_delta_as),
                        safe_keep_mapq_min=safe_keep_mapq_min,
                        safe_keep_nm_max=safe_keep_nm_max,
                    )
                    futs[fut] = fp

                for fut in as_completed(futs):
                    fp = futs[fut]
                    pc, qc, pt, pa, nd = fut.result()
                    pre_counts.update(pc)
                    post_counts.update(qc)
                    post_total_reads.update(pt)
                    post_allowed_reads.update(pa)
                    drop_total += nd
                    typer.echo(f"  done: {fp.name} (drop_rows={nd})")

        # merge parts into final gzip (single header)
        reads_to_drop_path = out_dir / f"{sample_name}_reads_to_drop.tsv.gz"
        with gzip.open(reads_to_drop_path, "wt") as wh:
            wh.write("read_id\tbarcode\tbc_key\tallowed_set\twinner_genome\tp_as\treason\n")
        with open(reads_to_drop_path, "ab") as w:
            for part in sorted(part_paths):
                if part.exists() and part.stat().st_size > 0:
                    with open(part, "rb") as r:
                        shutil.copyfileobj(r, w, length=1024 * 1024)

        typer.echo(f"[decontam] Wrote {reads_to_drop_path} (drop_rows_total={drop_total})")

    # -------------------------
    # Barnyard summaries pre/post
    # -------------------------
    typer.echo("[decontam] Writing barnyard-like summary tables...")
    _write_barnyard_summaries(out_dir, sample_name, "pre", pre_counts, barcode_to_expected_map)
    _write_barnyard_summaries(out_dir, sample_name, "post", post_counts, barcode_to_expected_map)

    # -------------------------
    # Post-clean gating + optional reclassify; write merged cells_calls output
    # -------------------------
    post_comp = _compute_post_composition_from_counts(post_counts)

    # Build augmented calls table with 1:1 row correspondence
    out_calls = calls.copy()

    # add policy columns
    out_calls["allowed_set"] = out_calls["barcode"].map(lambda b: _allowed_set_to_str(policy_by_barcode[str(b)].allowed_set) if str(b) in policy_by_barcode else "")
    out_calls["expected_genome"] = out_calls["barcode"].map(lambda b: barcode_to_expected_map.get(str(b), ""))
    out_calls["keep_preclean"] = out_calls["barcode"].map(lambda b: bool(policy_by_barcode[str(b)].keep_preclean) if str(b) in policy_by_barcode else False)
    out_calls["policy_flags"] = out_calls["barcode"].map(lambda b: ";".join(policy_by_barcode[str(b)].flags) if str(b) in policy_by_barcode else "")
    out_calls["policy_strategy"] = out_calls["barcode"].map(lambda b: policy_by_barcode[str(b)].strategy_used if str(b) in policy_by_barcode else "")

    # post metrics
    def _get_post(b: str, k: str, default):
        return post_comp.get(str(b), {}).get(k, default)

    out_calls["post_total_reads"] = out_calls["barcode"].map(lambda b: int(post_total_reads.get(str(b), 0)))
    out_calls["post_allowed_reads"] = out_calls["barcode"].map(lambda b: int(post_allowed_reads.get(str(b), 0)))
    out_calls["post_allowed_frac"] = out_calls.apply(
        lambda r: (float(r["post_allowed_reads"]) / float(r["post_total_reads"])) if float(r["post_total_reads"]) > 0 else 0.0,
        axis=1,
    )

    out_calls["post_top1_genome"] = out_calls["barcode"].map(lambda b: _get_post(b, "post_top1_genome", ""))
    out_calls["post_top1_frac"] = out_calls["barcode"].map(lambda b: float(_get_post(b, "post_top1_frac", 0.0)))
    out_calls["post_top2_genome"] = out_calls["barcode"].map(lambda b: _get_post(b, "post_top2_genome", ""))
    out_calls["post_top2_frac"] = out_calls["barcode"].map(lambda b: float(_get_post(b, "post_top2_frac", 0.0)))

    # post-clean keep/drop
    # - require keep_preclean True
    # - require post_allowed_reads >= min_reads_post_clean
    # - require post_allowed_frac >= min_allowed_frac_post_clean (only when post_total_reads>0)
    def _keep_post(row) -> bool:
        if not bool(row["keep_preclean"]):
            return False
        allowed_ok = int(row["post_allowed_reads"]) >= int(min_reads_post_clean)
        if int(row["post_total_reads"]) <= 0:
            return False
        frac_ok = float(row["post_allowed_frac"]) >= float(min_allowed_frac_post_clean)
        return bool(allowed_ok and frac_ok)

    out_calls["keep_postclean"] = out_calls.apply(_keep_post, axis=1)

    # reason
    def _reason(row) -> str:
        if not bool(row["keep_preclean"]):
            # empty/ambiguous drop/unknown-in-design strict
            return "drop_preclean_policy"
        if int(row["post_total_reads"]) <= 0:
            return "post_total_reads=0"
        if int(row["post_allowed_reads"]) < int(min_reads_post_clean):
            return f"post_allowed_reads<{min_reads_post_clean}"
        if float(row["post_allowed_frac"]) < float(min_allowed_frac_post_clean):
            return f"post_allowed_frac<{min_allowed_frac_post_clean}"
        return "keep"

    out_calls["postclean_reason"] = out_calls.apply(_reason, axis=1)

    # call_postclean (optional reclassify)
    out_calls["call_postclean"] = out_calls.apply(
        lambda r: _post_clean_call(
            pre_call=_norm_str(getattr(r, col_call, "")) if col_call else "",
            keep_post=bool(r["keep_postclean"]),
            post_top1=_norm_str(r["post_top1_genome"]),
            post_top1_frac=float(r["post_top1_frac"]),
            post_top2=_norm_str(r["post_top2_genome"]),
            post_top2_frac=float(r["post_top2_frac"]),
            reclassify_after_clean=bool(reclassify_after_clean),
            reclassify_doublet_minor_frac=float(reclassify_doublet_minor_frac),
            reclassify_single_purity_min=float(reclassify_single_purity_min),
        ),
        axis=1,
    )

    out_calls_path = out_dir / f"{sample_name}_cells_calls.decontam.tsv.gz"
    out_calls.to_csv(out_calls_path, sep="\t", index=False, compression="gzip")
    typer.echo(f"[decontam] Wrote {out_calls_path}")

    # -------------------------
    # Params JSON
    # -------------------------
    params_path = out_dir / f"{sample_name}_decontam_params.json"
    params = {
        "cells_calls": str(cells_calls),
        "out_dir": str(out_dir),
        "threads": int(threads),
        "tmp_dir": str(tmp_dir),
        "design_file": str(design_file) if design_file else None,
        "layout_file": layout_file,
        "strict_design_drop_mismatch": bool(strict_design_drop_mismatch),
        "ambiguous_policy": ambiguous_policy,
        "doublet_policy": doublet_policy,
        "indist_policy": indist_policy,
        "min_reads_post_clean": int(min_reads_post_clean),
        "min_allowed_frac_post_clean": float(min_allowed_frac_post_clean),
        "reclassify_after_clean": bool(reclassify_after_clean),
        "reclassify_doublet_minor_frac": float(reclassify_doublet_minor_frac),
        "reclassify_single_purity_min": float(reclassify_single_purity_min),
        "assignments": str(assignments) if assignments else None,
        "assign_glob": assign_glob,
        "read_id_col": read_id_col,
        "barcode_col": barcode_col,
        "genome_col": genome_col,
        "class_col": class_col,
        "p_as_col": p_as_col,
        "decontam_alpha": decontam_alpha,
        "require_p_as": bool(require_p_as),
        "safe_keep_delta_as": safe_keep_delta_as,
        "safe_keep_mapq_min": safe_keep_mapq_min,
        "safe_keep_nm_max": safe_keep_nm_max,
        "chunksize": int(chunksize),
        "reads_to_drop": str(reads_to_drop_path) if reads_to_drop_path else None,
        "policy_path": str(policy_path),
    }
    with params_path.open("w") as f:
        json.dump(params, f, indent=2)
    typer.echo(f"[decontam] Wrote {params_path}")

    # -------------------------
    # Cleanup tmp (best-effort)
    # -------------------------
    try:
        shutil.rmtree(tmp_dir)
    except Exception:
        pass

    typer.echo("[decontam] Done.")


if __name__ == "__main__":
    app()
