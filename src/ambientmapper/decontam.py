#!/usr/bin/env python3
"""
src/ambientmapper/decontam.py
ambientmapper decontam v2 — model-informed purification (AllowedSet) + read drop list + pre/post barnyard summaries

Goals
  1) Read-level cleaning: emit a drop list for BAM filtering.
  2) Conservative handling of ambiguous/indistinguishable, design-aware rescue when available.
  3) Mode-independence: dropping driven by assignment evidence (winner / scores / p-values), not genotyping internals.
  4) Genome-agnostic + scalable: supports 2..N genomes; outputs barnyard-like pre/post summaries without a combined genome.
  5) Parallelized across assignment chunk files (per-file workers → merge parts).

Key features
- AllowedSet logic per barcode: policies for doublets/indistinguishable (top12 | top1 | expected)
- Ambiguous handling modes: drop | design_rescue | top1_rescue | top12_rescue
- Post-clean barcode keep gate using:
    (1) min_reads_post_clean
    (2) min_allowed_frac_post_clean
- Parallel processing over many assignment chunk files
- Safe gzip-part merging: workers write per-file parts; master writes one header and concatenates member gz streams
- Barnyard summaries pre/post based on "confident winner" evidence from assignment files

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
  - The assignment TSVs contain per-read, per-genome rows and typically an `assigned_class` column with "winner"
  - rows. If no winner labels exist, decontam can compute winners using AS/MAPQ/NM ordering
  
Notes
- This module is assignment-driven for read-level decisions (mode-independent w.r.t genotyping).
- "winner evidence" here is `assigned_class == "winner"` plus optional `p_as <= decontam_alpha` override.
"""
from __future__ import annotations

import gzip
import glob
import json
import shutil
from collections import Counter, defaultdict

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

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
    help="ambientmapper decontam v2 — model-informed purification: AllowedSet + read drops + post-clean gating",
    add_completion=False,
    invoke_without_command=True,
    no_args_is_help=True,
)

# -------------------------
# Helpers
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


def _norm_str(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() == "nan":
        return ""
    return s


def _col_first(df: pd.DataFrame, cols: list[str]) -> Optional[str]:
    for c in cols:
        if c in df.columns:
            return c
    return None


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


def _parse_allowed_set_str(s: str) -> Set[str]:
    s = _norm_str(s)
    if not s:
        return set()
    return {x for x in (t.strip() for t in s.split(",")) if x}


def _allowed_set_to_str(S: Set[str]) -> str:
    if not S:
        return ""
    return ",".join(sorted(S))


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
# Policies
# -------------------------
@dataclass
class PolicyRow:
    barcode: str
    bc_key: str
    call: str
    model: str
    genome_1: str
    genome_2: str
    expected_genome: str
    allowed_set: Set[str]
    allowed_set_str: str
    action: str  # keep_cleaned | drop_barcode
    reason: str
    flags: str


def _is_doublet_like(call: str, model: str) -> bool:
    call = _norm_str(call)
    model = _norm_str(model)
    if model == "doublet":
        return True
    return call in {"doublet_confident", "ambiguous_weak_doublet"}


def _is_single_like(call: str, model: str) -> bool:
    call = _norm_str(call)
    model = _norm_str(model)
    if model == "single":
        return True
    return call in {"single_clean", "ambiguous_dirty_singlet"}


def _is_indist(call: str) -> bool:
    return _norm_str(call) == "indistinguishable"


def _is_empty(call: str, model: str) -> bool:
    return _norm_str(call) == "empty" or _norm_str(model) == "empty"


def _compute_allowed_set(
    *,
    has_design: bool,
    expected: str,
    call: str,
    model: str,
    g1: str,
    g2: str,
    indist_set: str,
    ambiguous_policy: str,
    doublet_policy: str,
    indist_policy: str,
) -> Tuple[Set[str], str, str, str]:
    """
    Returns: (AllowedSet, action, reason, flags)
    action: keep_cleaned or drop_barcode
    """
    expected = _norm_str(expected)
    call = _norm_str(call)
    model = _norm_str(model)
    g1 = _norm_str(g1)
    g2 = _norm_str(g2)
    flags: List[str] = []

    if _is_empty(call, model):
        return set(), "drop_barcode", "call_empty", "empty"

    indS = _parse_allowed_set_str(indist_set)

    # DESIGN-AWARE, expected known
    if has_design and expected:
        if expected == g2 and expected != g1:
            flags.append("expected_is_top2")
        if expected not in {g1, g2} and (not indS or expected not in indS):
            flags.append("design_mismatch")

        if call.startswith("ambiguous") and ambiguous_policy == "drop":
            return set(), "drop_barcode", "ambiguous_policy_drop", ";".join(flags) or ""

        if _is_doublet_like(call, model):
            if doublet_policy == "top12":
                S = {x for x in [g1, g2] if x} or {expected}
                return S, "keep_cleaned", "doublet_top12", ";".join(flags) or ""
            return {expected}, "keep_cleaned", "doublet_expected_only", ";".join(flags) or ""

        if _is_indist(call):
            if indist_policy == "top12":
                S = indS or {x for x in [g1, g2] if x} or {expected}
                return S, "keep_cleaned", "indist_top12", ";".join(flags) or ""
            return {expected}, "keep_cleaned", "indist_expected_only", ";".join(flags) or ""

        return {expected}, "keep_cleaned", "design_expected_only", ";".join(flags) or ""

    # DESIGN PRESENT BUT expected unknown
    if has_design and not expected:
        if ambiguous_policy in {"drop", "design_rescue"}:
            return set(), "drop_barcode", "unknown_expected_drop", "unknown_in_design"
        if ambiguous_policy == "top12_rescue":
            S = {x for x in [g1, g2] if x} or ({g1} if g1 else set())
            return S, ("keep_cleaned" if S else "drop_barcode"), "unknown_expected_top12_rescue", "unknown_in_design"
        if ambiguous_policy == "top1_rescue":
            S = {g1} if g1 else set()
            return S, ("keep_cleaned" if S else "drop_barcode"), "unknown_expected_top1_rescue", "unknown_in_design"
        return set(), "drop_barcode", "unknown_expected_drop", "unknown_in_design"

    # NO DESIGN
    if call.startswith("ambiguous"):
        if ambiguous_policy == "drop":
            return set(), "drop_barcode", "ambiguous_policy_drop", ""
        if ambiguous_policy == "design_rescue":
            return set(), "drop_barcode", "ambiguous_no_design_drop", ""
        if ambiguous_policy == "top12_rescue":
            S = {x for x in [g1, g2] if x} or ({g1} if g1 else set())
            return S, ("keep_cleaned" if S else "drop_barcode"), "ambiguous_top12_rescue", ""
        if ambiguous_policy == "top1_rescue":
            S = {g1} if g1 else set()
            return S, ("keep_cleaned" if S else "drop_barcode"), "ambiguous_top1_rescue", ""
        return set(), "drop_barcode", "ambiguous_policy_drop", ""

    if _is_doublet_like(call, model):
        if doublet_policy == "top12":
            S = {x for x in [g1, g2] if x}
            return S, ("keep_cleaned" if S else "drop_barcode"), "doublet_top12", ""
        S = {g1} if g1 else set()
        return S, ("keep_cleaned" if S else "drop_barcode"), "doublet_fallback_top1", ""

    if _is_indist(call):
        if indist_policy == "top12":
            S = indS or {x for x in [g1, g2] if x}
            return S, ("keep_cleaned" if S else "drop_barcode"), "indist_top12", ""
        S = {g1} if g1 else set()
        return S, ("keep_cleaned" if S else "drop_barcode"), "indist_fallback_top1", ""

    if _is_single_like(call, model):
        S = {g1} if g1 else set()
        return S, ("keep_cleaned" if S else "drop_barcode"), "single_top1", ""

    return set(), "drop_barcode", "unknown_call_drop", ""


# -------------------------
# Barnyard summaries
# -------------------------
def _write_barnyard_summaries(
    out_dir: Path,
    sample_name: str,
    prefix: str,
    counts_map: Dict[Tuple[str, str], int],
    expected_map: Dict[str, str],
) -> None:
    if not counts_map:
        return

    rows = [{"barcode": bc, "genome": gn, "n_winner_reads": int(n)} for (bc, gn), n in counts_map.items()]
    df_counts = pd.DataFrame.from_records(rows)

    out_counts = out_dir / f"{sample_name}_{prefix}_barcode_genome_counts.tsv.gz"
    df_counts.to_csv(out_counts, sep="\t", index=False, compression="gzip")

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

    if expected_map:
        comp["expected_genome"] = comp["barcode"].map(expected_map).map(_norm_str)
        comp.loc[comp["expected_genome"] == "", "expected_genome"] = np.nan
    else:
        comp["expected_genome"] = np.nan

    exp_counts = (
        df_counts.rename(columns={"n_winner_reads": "expected_reads"})
        .merge(comp[["barcode", "expected_genome"]], on="barcode", how="left")
    )
    exp_counts = exp_counts[exp_counts["genome"] == exp_counts["expected_genome"]]
    exp_reads = exp_counts.groupby("barcode", as_index=False)["expected_reads"].sum()

    comp = comp.merge(exp_reads, on="barcode", how="left")
    comp["expected_reads"] = comp["expected_reads"].fillna(0).astype(int)

    comp["expected_frac"] = comp["expected_reads"] / comp["total_reads"]
    comp.loc[comp["expected_genome"].isna(), "expected_frac"] = np.nan
    comp["contamination_rate"] = 1.0 - comp["expected_frac"]

    out_comp = out_dir / f"{sample_name}_{prefix}_barcode_composition.tsv.gz"
    comp.to_csv(out_comp, sep="\t", index=False, compression="gzip")

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
# Read-level processing
# -------------------------
def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _pick_winner_per_read(df: pd.DataFrame, rid_col: str) -> pd.DataFrame:
    """
    Winner per read (rid_col) using:
      AS desc, MAPQ desc, NM asc.
    If a column is missing, that key is neutral.
    Returns one row per rid_col.
    """
    tmp = df.copy()

    if "AS" in tmp.columns:
        tmp["_sort_as"] = -_coerce_numeric(tmp["AS"]).fillna(-999999)
    else:
        tmp["_sort_as"] = 0

    if "MAPQ" in tmp.columns:
        tmp["_sort_mapq"] = -_coerce_numeric(tmp["MAPQ"]).fillna(-1)
    else:
        tmp["_sort_mapq"] = 0

    if "NM" in tmp.columns:
        tmp["_sort_nm"] = _coerce_numeric(tmp["NM"]).fillna(999999)
    else:
        tmp["_sort_nm"] = 0

    tmp = tmp.sort_values([rid_col, "_sort_as", "_sort_mapq", "_sort_nm"], ascending=True)
    win = tmp.drop_duplicates(subset=[rid_col], keep="first").copy()
    return win

def _pick_winner_using_assigned_class_or_scores(
    chunk: pd.DataFrame,
    *,
    rid_col: str,
    genome_col: str,
    class_col: str,
) -> pd.DataFrame:
    # If assigned_class present: use it
    if class_col in chunk.columns:
        w = chunk[chunk[class_col].astype(str) == "winner"].copy()
        if not w.empty:
            # if multiple winners per read, tie-break by scores
            w = w.rename(columns={genome_col: "genome"})
            return _pick_winner_per_read(w, rid_col).rename(columns={"genome": "winner_genome"})
        # fall back to score-based if no winner rows found

    # fallback: score-based across all rows
    tmp = chunk.rename(columns={genome_col: "genome"}).copy()
    return _pick_winner_per_read(tmp, rid_col).rename(columns={"genome": "winner_genome"})


def _process_one_assign_file(
    fp: Path,
    *,
    out_part: Path,
    valid_barcodes: Set[str],
    bc_key_mode: str,
    bc_key_n: int,
    allowed_pairs_df: pd.DataFrame,          # optional debug only
    policy_df: pd.DataFrame,                # must include: barcode, allowed_set, action
    barcodes_to_drop: Set[str],
    has_design: bool,
    chunksize: int,
    read_id_col: str,
    barcode_col: str,
    genome_col: str,
    class_col: str,
    p_as_col: str,
    decontam_alpha: Optional[float],
    require_p_as: bool,
    safe_keep_delta_as: int,
    safe_keep_mapq_min: Optional[int],
    safe_keep_nm_max: Optional[int],
    **_ignored: Any,
) -> Tuple[Counter, Counter, int]:
    """
    Process one assignment TSV(.gz) and write per-read drops for this file shard.

    Patches applied:
      1) Confidence is computed on the WINNER ROW (never by grabbing an arbitrary p_as per _rid).
      2) Policy merge cannot collide: allowed_set is merged onto chunk (so it propagates into win);
         action is merged onto win (allowed_set not merged again).
      3) All keep/drop masks are computed AFTER all merges, using numpy arrays to avoid any
         index-alignment surprises.
      4) safe_keep merges are guarded to not change row count; per-_rid summaries are de-duplicated.
    """
    pre_counts: Counter = Counter()
    post_counts: Counter = Counter()
    n_drop_rows = 0

    # -------------------------
    # Optional debug allowed-pairs (bc_key x genome)
    # -------------------------
    ap = None
    if allowed_pairs_df is not None and not allowed_pairs_df.empty:
        ap = allowed_pairs_df.copy()
        need = {"bc_key", "genome"}
        miss = need - set(ap.columns)
        if miss:
            raise ValueError(f"allowed_pairs_df missing columns: {sorted(miss)}")
        if "is_allowed" not in ap.columns:
            ap["is_allowed"] = 1
        ap = ap[["bc_key", "genome", "is_allowed"]].drop_duplicates().copy()

    # -------------------------
    # Policy table
    # -------------------------
    if policy_df is None or policy_df.empty:
        raise ValueError("policy_df is required and cannot be empty.")

    need_pol = {"barcode", "allowed_set", "action"}
    miss_pol = need_pol - set(policy_df.columns)
    if miss_pol:
        raise ValueError(f"policy_df missing required columns: {sorted(miss_pol)}")

    pol = (
        policy_df[["barcode", "allowed_set", "action"]]
        .drop_duplicates(subset=["barcode"], keep="first")
        .copy()
    )
    pol["barcode"] = pol["barcode"].astype(str)
    pol["allowed_set"] = pol["allowed_set"].fillna("").astype(str)
    pol["action"] = pol["action"].fillna("keep_cleaned").astype(str)

    pol_action = pol[["barcode", "action"]].copy()

    # exact-token membership for comma-separated allowed_set
    def _token_in_allowed_set(genome: str, aset: str) -> bool:
        if not aset:
            return False
        # exact tokens; no substring matches
        toks = [t for t in aset.split(",") if t]
        return genome in toks

    out_part.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_part, "wt") as fh:
        fh.write("read_id\tbarcode\tbc_key\tallowed_set\twinner_genome\tp_as\treason\n")

        for chunk in pd.read_csv(fp, sep="\t", chunksize=chunksize, dtype=str, low_memory=False):
            required = {read_id_col, barcode_col, genome_col}
            miss = required - set(chunk.columns)
            if miss:
                raise ValueError(f"File {fp.name} missing columns: {sorted(miss)}")

            # normalize and filter
            chunk = chunk.copy()
            chunk["barcode"] = chunk[barcode_col].astype(str)
            mask_valid = chunk["barcode"].isin(valid_barcodes)
            if not bool(mask_valid.any()):
                continue
            chunk = chunk.loc[mask_valid].copy()

            chunk["read_id"] = chunk[read_id_col].astype(str)
            chunk["bc_key"] = chunk["barcode"].map(lambda x: _design_key(x, bc_key_mode, bc_key_n))
            chunk["_rid"] = chunk["barcode"] + "::" + chunk["read_id"]

            # ---- PATCH: attach allowed_set onto CHUNK (propagates into win) ----
            chunk = chunk.merge(pol[["barcode", "allowed_set"]], on="barcode", how="left")
            chunk["allowed_set"] = chunk["allowed_set"].fillna("").astype(str)

            # optional debug: bc_key x genome membership (not used for decisions)
            if has_design and ap is not None:
                tmp_ap = ap.rename(columns={"genome": genome_col}).copy()
                chunk = chunk.merge(tmp_ap, on=["bc_key", genome_col], how="left")
                chunk["is_allowed_pair"] = pd.to_numeric(chunk["is_allowed"], errors="coerce").fillna(0).astype(int)
                chunk.drop(columns=["is_allowed"], inplace=True, errors="ignore")
            else:
                chunk["is_allowed_pair"] = 0

            # -------------------------
            # Winner selection (prefer assigned_class if present)
            # win should retain columns from the chosen winner row, including allowed_set and p_as_col
            # -------------------------
            win = _pick_winner_using_assigned_class_or_scores(
                chunk,
                rid_col="_rid",
                genome_col=genome_col,
                class_col=class_col,
            ).copy()

            # Ensure identity cols exist
            for col in ["_rid", "barcode", "read_id", "bc_key"]:
                if col not in win.columns:
                    if col in ("barcode", "read_id") and "_rid" in win.columns:
                        parts = win["_rid"].astype(str).str.split("::", n=1, expand=True)
                        if col == "barcode":
                            win["barcode"] = parts[0]
                        else:
                            win["read_id"] = parts[1] if parts.shape[1] > 1 else ""
                    elif col == "bc_key" and "barcode" in win.columns:
                        win["bc_key"] = win["barcode"].map(lambda x: _design_key(x, bc_key_mode, bc_key_n))
                    elif col == "_rid" and ("barcode" in win.columns and "read_id" in win.columns):
                        win["_rid"] = win["barcode"].astype(str) + "::" + win["read_id"].astype(str)
                    else:
                        raise ValueError(f"Winner table missing required column and cannot recover: {col}")

            # Ensure allowed_set exists on win (should, because chunk had it)
            if "allowed_set" not in win.columns:
                win["allowed_set"] = ""
            win["allowed_set"] = win["allowed_set"].fillna("").astype(str)

            # ---- PATCH: p_as comes from the WINNER row (never arbitrary first row per _rid) ----
            if "p_as" in win.columns:
                # already normalized somewhere upstream
                pass
            elif p_as_col in win.columns:
                win = win.rename(columns={p_as_col: "p_as"})
            else:
                win["p_as"] = ""

            # numeric for confidence; string for output
            win["_p_as_val"] = pd.to_numeric(win["p_as"], errors="coerce")
            win["p_as"] = win["p_as"].fillna("").astype(str)

            # ---- PATCH: merge ACTION only (avoid allowed_set collisions) ----
            win = win.merge(pol_action, on="barcode", how="left", validate="m:1")
            win["action"] = win["action"].fillna("keep_cleaned").astype(str)

            # -------------------------
            # Confidence ON WINNER
            # -------------------------
            has_class = class_col in chunk.columns
            if decontam_alpha is None:
                # if no alpha provided, treat winners as confident (your original behavior)
                win["is_confident"] = 1
            else:
                alpha = float(decontam_alpha)
                has_p = win["_p_as_val"].notna()
                is_conf = has_p & (win["_p_as_val"] <= alpha)
                if not require_p_as:
                    # if p missing, still accept winner as confident
                    is_conf = is_conf | (~has_p)
                win["is_confident"] = is_conf.astype(int)

                # If you want to force “must be assign-winner” when assigned_class exists,
                # you can add an additional gate here; leaving off for robustness.

            # Drop whole barcode?
            win["drop_barcode"] = (win["barcode"].isin(barcodes_to_drop)) | (win["action"] != "keep_cleaned")

            # -------------------------
            # Winner allowed (membership-based, robust)
            # -------------------------
            allowed_set_arr = win["allowed_set"].fillna("").astype(str).to_numpy()
            winner_arr = win["winner_genome"].fillna("").astype(str).to_numpy()

            winner_allowed_arr = np.fromiter(
                (_token_in_allowed_set(w, aset) for w, aset in zip(winner_arr, allowed_set_arr)),
                dtype=bool,
                count=len(win),
            )
            win["winner_allowed"] = winner_allowed_arr.astype(int)

            # -------------------------
            # safe_keep (membership-based across all rows)
            # -------------------------
            have_scores = all(c in chunk.columns for c in ["AS", "MAPQ", "NM"])
            safe_keep_enabled = have_scores and (safe_keep_delta_as is not None) and (int(safe_keep_delta_as) > 0)

            if safe_keep_enabled:
                tmp = chunk[["_rid", genome_col, "allowed_set", "AS", "MAPQ", "NM"]].copy()
                tmp["AS"] = _coerce_numeric(tmp["AS"])
                tmp["MAPQ"] = _coerce_numeric(tmp["MAPQ"])
                tmp["NM"] = _coerce_numeric(tmp["NM"])

                g_arr = tmp[genome_col].fillna("").astype(str).to_numpy()
                a_arr = tmp["allowed_set"].fillna("").astype(str).to_numpy()

                tmp["is_allowed_row"] = np.fromiter(
                    (_token_in_allowed_set(g, aset) for g, aset in zip(g_arr, a_arr)),
                    dtype=bool,
                    count=len(tmp),
                ).astype(int)

                best_allowed_as = (
                    tmp[tmp["is_allowed_row"] == 1]
                    .groupby("_rid", observed=True)["AS"]
                    .max()
                    .rename("best_allowed_as")
                    .reset_index()
                    .drop_duplicates("_rid", keep="first")
                )
                best_disallowed_as = (
                    tmp[tmp["is_allowed_row"] == 0]
                    .groupby("_rid", observed=True)["AS"]
                    .max()
                    .rename("best_disallowed_as")
                    .reset_index()
                    .drop_duplicates("_rid", keep="first")
                )

                n0 = len(win)
                win = win.merge(best_allowed_as, on="_rid", how="left")
                win = win.merge(best_disallowed_as, on="_rid", how="left")
                if len(win) != n0:
                    raise ValueError(f"safe_keep merges changed row count: {n0} -> {len(win)}")

                ok = (
                    win["best_allowed_as"].notna()
                    & win["best_disallowed_as"].notna()
                    & (win["best_allowed_as"] >= (win["best_disallowed_as"] - float(safe_keep_delta_as)))
                )

                if safe_keep_mapq_min is not None:
                    best_allowed_mapq = (
                        tmp[tmp["is_allowed_row"] == 1]
                        .groupby("_rid", observed=True)["MAPQ"]
                        .max()
                        .rename("best_allowed_mapq")
                        .reset_index()
                        .drop_duplicates("_rid", keep="first")
                    )
                    n0 = len(win)
                    win = win.merge(best_allowed_mapq, on="_rid", how="left")
                    if len(win) != n0:
                        raise ValueError(f"safe_keep MAPQ merge changed row count: {n0} -> {len(win)}")
                    ok = ok & win["best_allowed_mapq"].notna() & (win["best_allowed_mapq"] >= float(safe_keep_mapq_min))

                if safe_keep_nm_max is not None:
                    best_allowed_nm = (
                        tmp[tmp["is_allowed_row"] == 1]
                        .groupby("_rid", observed=True)["NM"]
                        .min()
                        .rename("best_allowed_nm")
                        .reset_index()
                        .drop_duplicates("_rid", keep="first")
                    )
                    n0 = len(win)
                    win = win.merge(best_allowed_nm, on="_rid", how="left")
                    if len(win) != n0:
                        raise ValueError(f"safe_keep NM merge changed row count: {n0} -> {len(win)}")
                    ok = ok & win["best_allowed_nm"].notna() & (win["best_allowed_nm"] <= float(safe_keep_nm_max))

                win["safe_keep"] = ok.fillna(False)
            else:
                win["safe_keep"] = False

            # -------------------------
            # FINAL keep/drop masks (computed at the end; numpy to avoid index issues)
            # -------------------------
            is_conf_arr = win["is_confident"].fillna(0).to_numpy().astype(int) == 1
            drop_bc_arr = win["drop_barcode"].fillna(False).to_numpy().astype(bool)
            safe_keep_arr = win["safe_keep"].fillna(False).to_numpy().astype(bool)

            keep_mask = (
                (~drop_bc_arr)
                & is_conf_arr
                & (winner_allowed_arr | safe_keep_arr)
            )
            drop_mask = is_conf_arr & (~keep_mask)

            # PRE counts: confident winners (not filtered by allowedness)
            # (this is for diagnostics / before-vs-after)
            if is_conf_arr.any():
                vc = win.loc[is_conf_arr, ["barcode", "winner_genome"]].value_counts()
                for (b, g), n in vc.items():
                    pre_counts[(str(b), str(g))] += int(n)

            # POST counts: kept winners
            if keep_mask.any():
                vc2 = win.loc[keep_mask, ["barcode", "winner_genome"]].value_counts()
                for (b, g), n in vc2.items():
                    post_counts[(str(b), str(g))] += int(n)

            # Write drop list: only confident winners that are NOT kept
            if drop_mask.any():
                reason = np.where(
                    win.loc[drop_mask, "drop_barcode"].to_numpy(bool),
                    "drop_barcode",
                    "mismatch_winner",
                )
                out = pd.DataFrame(
                    {
                        "read_id": win.loc[drop_mask, "read_id"].astype(str),
                        "barcode": win.loc[drop_mask, "barcode"].astype(str),
                        "bc_key": win.loc[drop_mask, "bc_key"].astype(str),
                        "allowed_set": win.loc[drop_mask, "allowed_set"].astype(str),
                        "winner_genome": win.loc[drop_mask, "winner_genome"].astype(str),
                        "p_as": win.loc[drop_mask, "p_as"].astype(str),
                        "reason": reason,
                    }
                )
                n_this = int(drop_mask.sum())
                n_drop_rows += n_this
                out.to_csv(fh, sep="\t", header=False, index=False)

    return pre_counts, post_counts, n_drop_rows



# -------------------------
# Post-clean gating
# -------------------------
def _compute_post_metrics(
    calls_barcodes: pd.DataFrame,  # columns: barcode
    post_counts: Dict[Tuple[str, str], int],
    barcode_to_allowedset: Dict[str, Set[str]],
    min_reads_post_clean: int,
    min_allowed_frac_post_clean: float,
) -> pd.DataFrame:
    total_post: Dict[str, int] = defaultdict(int)
    allowed_post: Dict[str, int] = defaultdict(int)

    for (bc, gn), n in post_counts.items():
        n = int(n)
        total_post[bc] += n
        if gn in barcode_to_allowedset.get(bc, set()):
            allowed_post[bc] += n

    rows = []
    for bc in calls_barcodes["barcode"].astype(str).tolist():
        S = barcode_to_allowedset.get(bc, set())
        tr = int(total_post.get(bc, 0))
        ar = int(allowed_post.get(bc, 0))
        frac = (ar / tr) if tr > 0 else 0.0

        keep = (ar >= int(min_reads_post_clean)) and (frac >= float(min_allowed_frac_post_clean))
        if keep:
            reason = "keep"
        elif ar < int(min_reads_post_clean):
            reason = f"allowed_reads_post<{min_reads_post_clean}"
        else:
            reason = f"allowed_frac_post<{min_allowed_frac_post_clean}"

        rows.append(
            {
                "barcode": bc,
                "post_total_reads": tr,
                "post_allowed_reads": ar,
                "post_allowed_frac": float(frac),
                "keep_postclean": bool(keep),
                "drop_reason": reason,
            }
        )

    return pd.DataFrame.from_records(rows)


# -------------------------
# Main (runs as ambientmapper decontam ...)
# -------------------------
@app.callback()
def decontam(
    cells_calls: Path = typer.Option(..., exists=True, readable=True, help="*_cells_calls.tsv(.gz) from genotyping."),
    out_dir: Path = typer.Option(..., help="Output directory for decontam artifacts."),

    # Parallelism
    threads: int = typer.Option(1, "--threads", help="Parallel workers across assignment chunk files."),
    tmp_dir: Optional[Path] = typer.Option(None, "--tmp-dir", help="Temp directory for drop parts (default: out_dir/tmp_decontam)."),

    # Design / Layout
    design_file: Optional[Path] = typer.Option(None, "--design-file", readable=True, help="Optional design file. Enables design-aware mode."),
    layout_file: str = typer.Option("DEFAULT", "--layout-file", help='Layout path for well-range design files, or "DEFAULT".'),
    strict_design_drop_mismatch: bool = typer.Option(
        True,
        "--strict-design-drop-mismatch/--no-strict-design-drop-mismatch",
        help="If strict, drop barcodes whose expected genome is mismatch in the design (unknown wells).",
    ),

    # Policies
    ambiguous_policy: str = typer.Option(
        "design_rescue",
        "--ambiguous-policy",
        help="Ambiguous policy: drop|design_rescue|top1_rescue|top12_rescue",
    ),
    doublet_policy: str = typer.Option(
        "top12",
        "--doublet-policy",
        help="Doublet policy: top12 (rescue) | expected (design only; in-silico sorting)",
    ),
    indist_policy: str = typer.Option(
        "top12",
        "--indist-policy",
        help="Indistinguishable policy: top12 | expected (design only)",
    ),

    # Post-clean barcode gate
    min_reads_post_clean: int = typer.Option(
        100,
        "--min-reads-post-clean",
        help="Keep barcode only if it has >= this many post-clean allowed winner reads.",
    ),
    min_allowed_frac_post_clean: float = typer.Option(
        0.90,
        "--min-allowed-frac-post-clean",
        help="Keep barcode only if post-clean allowed fraction >= this threshold.",
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

    # Winner confidence
    decontam_alpha: Optional[float] = typer.Option(None, "--decontam-alpha", help="Optional p_as threshold to treat reads as confident winners."),
    require_p_as: bool = typer.Option(False, "--require-p-as", help="If set, only p_as<=alpha reads are considered confident."),

    # Safe keep (homology guard)
    safe_keep_delta_as: int = typer.Option(0, "--safe-keep-delta-as", help="Safe-keep if best allowed AS is within this delta of best disallowed AS (0 disables)."),
    safe_keep_mapq_min: Optional[int] = typer.Option(None, "--safe-keep-mapq-min", help="Optional MAPQ minimum for safe-kept allowed hit."),
    safe_keep_nm_max: Optional[int] = typer.Option(None, "--safe-keep-nm-max", help="Optional NM maximum for safe-kept allowed hit."),

    # Misc
    sample_name: Optional[str] = typer.Option(None, "--sample-name"),
    chunksize: int = typer.Option(1_000_000, "--chunksize"),
    design_bc_mode: str = typer.Option("before-dash", "--design-bc-mode"),
    design_bc_n: int = typer.Option(10, "--design-bc-n"),
) -> None:
    cells_calls = cells_calls.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if tmp_dir is None:
        tmp_dir = out_dir / "tmp_decontam"
    tmp_dir = tmp_dir.expanduser().resolve()
    tmp_dir.mkdir(parents=True, exist_ok=True)

    if sample_name is None:
        sample_name = _infer_sample_name_from_cells_calls(cells_calls)
    sample_root = _infer_sample_root_from_cells_calls(cells_calls)

    # Design
    has_design = False
    bckey_to_expected: Dict[str, str] = {}
    if design_file is not None:
        design_file = Path(design_file).expanduser().resolve()
        if not design_file.exists():
            raise FileNotFoundError(f"--design-file not found: {design_file}")
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

    # infer assign glob if not provided
    if assignments is None and (assign_glob is None or not str(assign_glob).strip()):
        if sample_root is not None:
            assign_glob = _default_assign_glob(sample_root, sample_name)
            typer.echo(f"[decontam] auto --assign-glob: {assign_glob}")

    # Load calls
    typer.echo(f"[decontam] Loading cell calls: {cells_calls}")
    calls = pd.read_csv(cells_calls, sep="\t", dtype=str)
    if "barcode" not in calls.columns:
        raise ValueError(f"{cells_calls} missing required column: barcode")

    calls["barcode"] = calls["barcode"].astype(str)
    calls["bc_key"] = calls["barcode"].map(lambda x: _design_key(x, design_bc_mode, design_bc_n))

    call_col = _col_first(calls, ["call"]) or "call"
    model_col = _col_first(calls, ["model"]) or "model"
    g1_col = _col_first(calls, ["genome_1", "genome1", "top_genome", "best_genome"]) or "genome_1"
    g2_col = _col_first(calls, ["genome_2", "genome2"]) or "genome_2"
    indist_col = _col_first(calls, ["indistinguishable_set"]) or "indistinguishable_set"

    for c in [call_col, model_col, g1_col, g2_col, indist_col]:
        if c not in calls.columns:
            calls[c] = ""

    valid_barcodes: Set[str] = set(calls["barcode"].unique())

    # Build policy + allowed pairs
    policy_rows: List[Dict[str, Any]] = []
    barcode_to_allowedset: Dict[str, Set[str]] = {}
    barcode_to_expected: Dict[str, str] = {}
    bc_key_to_drop: Set[str] = set()

    for row in calls.itertuples(index=False):
        bc_full = _norm_str(getattr(row, "barcode", ""))
        bc_key = _norm_str(getattr(row, "bc_key", ""))

        callv = _norm_str(getattr(row, call_col, ""))
        modelv = _norm_str(getattr(row, model_col, ""))
        g1v = _norm_str(getattr(row, g1_col, ""))
        g2v = _norm_str(getattr(row, g2_col, ""))
        indv = _norm_str(getattr(row, indist_col, ""))

        expected = ""
        if has_design and bc_key:
            expected = _norm_str(bckey_to_expected.get(bc_key, ""))
        barcode_to_expected[bc_full] = expected

        if has_design and strict_design_drop_mismatch and (not expected):
            allowed, action, reason, flags = set(), "drop_barcode", "unknown_in_design_strict", "unknown_in_design"
        else:
            allowed, action, reason, flags = _compute_allowed_set(
                has_design=has_design,
                expected=expected,
                call=callv,
                model=modelv,
                g1=g1v,
                g2=g2v,
                indist_set=indv,
                ambiguous_policy=ambiguous_policy,
                doublet_policy=doublet_policy,
                indist_policy=indist_policy,
            )

        barcode_to_allowedset[bc_full] = allowed
        if action == "drop_barcode" and bc_key:
            barcodes_to_drop.add(bc_full)

        policy_rows.append(
            {
                "barcode": bc_full,
                "bc_key": bc_key,
                "call": callv,
                "model": modelv,
                "genome_1": g1v,
                "genome_2": g2v,
                "expected_genome": expected,
                "allowed_set": _allowed_set_to_str(allowed),
                "action": action,
                "reason": reason,
                "flags": flags,
            }
        )

    df_policy = pd.DataFrame.from_records(policy_rows)
    df_policy.to_csv(out_dir / f"{sample_name}_barcode_policy.tsv.gz", sep="\t", index=False, compression="gzip")

    allowed_pairs: List[Tuple[str, str]] = []
    for bc, bc_key, allowed_str, action in df_policy[["barcode", "bc_key", "allowed_set", "action"]].itertuples(index=False):
        if action != "keep_cleaned":
            continue
        bc_key = _norm_str(bc_key)
        if not bc_key:
            continue
        for g in _parse_allowed_set_str(allowed_str):
            allowed_pairs.append((bc_key, g))

    allowed_pairs_df = pd.DataFrame.from_records(allowed_pairs, columns=["bc_key", "genome"])
    allowed_pairs_df = allowed_pairs_df.drop_duplicates(["bc_key","genome"]).copy()

    if allowed_pairs_df.empty:
        allowed_pairs_df = pd.DataFrame(columns=["bc_key", "genome"])
    allowed_pairs_df["is_allowed"] = 1

    allowed_set_cache = (
        allowed_pairs_df.groupby("bc_key")["genome"]
        .apply(lambda s: ",".join(sorted(set(map(str, s)))))
        .to_dict()
        if not allowed_pairs_df.empty
        else {}
    )

    # Assignment inputs
    input_files: List[Path] = []
    if assignments is not None:
        input_files.append(Path(assignments).expanduser().resolve())

    if assign_glob is not None:
        assign_glob = str(assign_glob).strip() or None
    if assign_glob:
        found = sorted(glob.glob(assign_glob, recursive=True))
        if not found:
            raise FileNotFoundError(f"[decontam] No assignment files matched: {assign_glob}")
        input_files.extend([Path(f).resolve() for f in found])

    input_files = sorted(set(input_files))
    if not input_files:
        raise FileNotFoundError("[decontam] No assignment inputs found. Provide --assignments or --assign-glob.")

    # Parallel per-file
    threads = max(1, int(threads))
    drops_dir = tmp_dir / "drop_parts"
    drops_dir.mkdir(parents=True, exist_ok=True)

    def _part_path(fp: Path) -> Path:
        return drops_dir / f"{fp.stem}.reads_to_drop.part.tsv.gz"

    part_paths: List[Path] = []
    pre_counts: Counter = Counter()
    post_counts: Counter = Counter()
    drop_total = 0

    typer.echo(f"[decontam] Processing {len(input_files)} assignment files (threads={threads})")

    if threads == 1:
        for fp in input_files:
            part = _part_path(fp)
            part_paths.append(part)
            pc, qc, nd = _process_one_assign_file(
                fp,
                out_part=part,
                valid_barcodes=valid_barcodes,
                bc_key_mode=design_bc_mode,
                bc_key_n=design_bc_n,
                allowed_pairs_df=allowed_pairs_df,
                allowed_set_cache=allowed_set_cache,
                bc_key_to_drop=bc_key_to_drop,
                chunksize=chunksize,
                read_id_col=read_id_col,
                barcode_col=barcode_col,
                genome_col=genome_col,
                class_col=class_col,
                p_as_col=p_as_col,
                decontam_alpha=decontam_alpha,
                require_p_as=require_p_as,
                safe_keep_delta_as=safe_keep_delta_as,
                safe_keep_mapq_min=safe_keep_mapq_min,
                safe_keep_nm_max=safe_keep_nm_max,
                policy_df=df_policy,
            )
            pre_counts.update(pc)
            post_counts.update(qc)
            drop_total += nd
            typer.echo(f"  done: {fp.name} (drop_rows={nd})")
    else:
        with ProcessPoolExecutor(max_workers=threads) as ex:
            futs = {}
            for fp in input_files:
                part = _part_path(fp)
                part_paths.append(part)
                fut = ex.submit(
                    _process_one_assign_file,
                    fp,
                    out_part=part,
                    valid_barcodes=valid_barcodes,
                    bc_key_mode=design_bc_mode,
                    bc_key_n=design_bc_n,
                    allowed_pairs_df=allowed_pairs_df,
                    allowed_set_cache=allowed_set_cache,
                    bc_key_to_drop=bc_key_to_drop,
                    chunksize=chunksize,
                    read_id_col=read_id_col,
                    barcode_col=barcode_col,
                    genome_col=genome_col,
                    class_col=class_col,
                    p_as_col=p_as_col,
                    decontam_alpha=decontam_alpha,
                    require_p_as=require_p_as,
                    safe_keep_delta_as=safe_keep_delta_as,
                    safe_keep_mapq_min=safe_keep_mapq_min,
                    safe_keep_nm_max=safe_keep_nm_max,
                    policy_df=df_policy,
                )
                futs[fut] = fp

            for fut in as_completed(futs):
                fp = futs[fut]
                pc, qc, nd = fut.result()
                pre_counts.update(pc)
                post_counts.update(qc)
                drop_total += nd
                typer.echo(f"  done: {fp.name} (drop_rows={nd})")

    # Merge parts into final reads_to_drop (single header)
    reads_to_drop_path = out_dir / f"{sample_name}_reads_to_drop.tsv.gz"
    with gzip.open(reads_to_drop_path, "wt") as out_fh:
        out_fh.write("read_id\tbarcode\tbc_key\tallowed_set\twinner_genome\tp_as\treason\n")

    with gzip.open(reads_to_drop_path, "at") as out_fh:
        for part in sorted(part_paths):
            if not part.exists() or part.stat().st_size == 0:
                continue
            with gzip.open(part, "rt") as in_fh:
                first = True
                for line in in_fh:
                    if first:
                        first = False
                        continue  # skip part header
                    out_fh.write(line)

    typer.echo(f"[decontam] Wrote {reads_to_drop_path} (drop_rows_total={drop_total})")

    # Barnyard summaries
    typer.echo("[decontam] Writing barnyard-like summary tables...")
    _write_barnyard_summaries(out_dir, sample_name, "pre", pre_counts, barcode_to_expected)
    _write_barnyard_summaries(out_dir, sample_name, "post", post_counts, barcode_to_expected)

    # Post-clean metrics
    df_post = _compute_post_metrics(
        calls_barcodes=calls[["barcode"]].copy(),
        post_counts=post_counts,
        barcode_to_allowedset=barcode_to_allowedset,
        min_reads_post_clean=min_reads_post_clean,
        min_allowed_frac_post_clean=min_allowed_frac_post_clean,
    )
    df_post.to_csv(out_dir / f"{sample_name}_barcode_postclean.tsv.gz", sep="\t", index=False, compression="gzip")

    # Append into calls and write combined file
    calls_out = calls.merge(
        df_policy[["barcode", "allowed_set", "expected_genome", "action", "reason", "flags"]],
        on="barcode",
        how="left",
    )
    calls_out = calls_out.merge(df_post, on="barcode", how="left")

    out_calls = out_dir / f"{sample_name}_cells_calls.decontam.tsv.gz"
    calls_out.to_csv(out_calls, sep="\t", index=False, compression="gzip")
    typer.echo(f"[decontam] Wrote {out_calls}")

    # Params JSON
    params = {
        "cells_calls": str(cells_calls),
        "out_dir": str(out_dir),
        "design_file": str(design_file) if design_file else None,
        "layout_file": layout_file,
        "strict_design_drop_mismatch": strict_design_drop_mismatch,
        "ambiguous_policy": ambiguous_policy,
        "doublet_policy": doublet_policy,
        "indist_policy": indist_policy,
        "min_reads_post_clean": int(min_reads_post_clean),
        "min_allowed_frac_post_clean": float(min_allowed_frac_post_clean),
        "assignments": str(assignments) if assignments else None,
        "assign_glob": assign_glob,
        "threads": int(threads),
        "chunksize": int(chunksize),
        "read_id_col": read_id_col,
        "barcode_col": barcode_col,
        "genome_col": genome_col,
        "class_col": class_col,
        "p_as_col": p_as_col,
        "decontam_alpha": decontam_alpha,
        "require_p_as": bool(require_p_as),
        "safe_keep_delta_as": int(safe_keep_delta_as),
        "safe_keep_mapq_min": safe_keep_mapq_min,
        "safe_keep_nm_max": safe_keep_nm_max,
        "design_bc_mode": design_bc_mode,
        "design_bc_n": int(design_bc_n),
        "reads_to_drop": str(reads_to_drop_path),
    }
    params_path = out_dir / f"{sample_name}_decontam_params.json"
    params_path.write_text(json.dumps(params, indent=2))
    typer.echo(f"[decontam] Wrote {params_path}")

    # Cleanup tmp
    try:
        shutil.rmtree(tmp_dir)
    except Exception:
        pass

    typer.echo("[decontam] Done.")


if __name__ == "__main__":
    app()
