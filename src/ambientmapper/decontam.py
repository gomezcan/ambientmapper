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
import json
import glob
import shutil
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Iterable, Any

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
    help="Generate barcode + read-level decontamination decisions (AllowedSet v2).",
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


def _col_first(df: pd.DataFrame, cols: List[str]) -> Optional[str]:
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


def _parse_set(s: str) -> List[str]:
    s = _norm_str(s)
    if not s:
        return []
    s = s.replace(";", ",")
    out = []
    for t in s.split(","):
        t = t.strip()
        if t:
            out.append(t)
    return out


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


def _allowed_from_policy(
    *,
    call: str,
    g1: str,
    g2: str,
    indist_set: str,
    expected: str,
    has_design: bool,
    doublet_policy: str,
    indist_policy: str,
    ambiguous_policy: str,
) -> Tuple[List[str], str]:
    """
    Returns (allowed_list, note).

    Policies:
      doublet_policy: top12 | top1 | expected
      indist_policy : top12 | top1 | expected
      ambiguous_policy: drop | design_rescue | top1_rescue | top12_rescue
    """
    call = _norm_str(call)
    g1 = _norm_str(g1)
    g2 = _norm_str(g2)
    expected = _norm_str(expected)
    indist_list = _parse_set(indist_set)

    is_empty = (call == "empty")
    is_doublet = call.startswith("doublet")
    is_indist = (call == "indistinguishable")
    is_singlelike = (call == "single_clean") or (call == "single")
    is_ambig = call.startswith("ambiguous") or (call in {"ambiguous_low_depth", "ambiguous_dirty_singlet"})

    if is_empty:
        return ([], "empty_allowed_empty")

    if is_singlelike:
        if has_design and expected:
            return ([expected], "single_expected")
        return ([g1] if g1 else [], "single_top1")

    if is_doublet:
        if doublet_policy == "expected":
            if has_design and expected:
                return ([expected], "doublet_expected")
            return ([g1] if g1 else [], "doublet_expected_fallback_top1")
        if doublet_policy == "top1":
            return ([g1] if g1 else [], "doublet_top1")
        # top12
        out: List[str] = []
        if g1:
            out.append(g1)
        if g2 and g2 != g1:
            out.append(g2)
        return (out, "doublet_top12")

    if is_indist:
        if indist_policy == "expected":
            if has_design and expected:
                return ([expected], "indist_expected")
            return ([g1] if g1 else [], "indist_expected_fallback_top1")
        if indist_policy == "top1":
            return ([g1] if g1 else [], "indist_top1")
        # top12
        if indist_list:
            return (indist_list, "indist_set")
        out2: List[str] = []
        if g1:
            out2.append(g1)
        if g2 and g2 != g1:
            out2.append(g2)
        return (out2, "indist_top12")

    if is_ambig:
        if ambiguous_policy == "drop":
            return ([], "ambiguous_drop")
        if ambiguous_policy == "design_rescue":
            if has_design and expected:
                return ([expected], "ambiguous_design_rescue")
            return ([], "ambiguous_design_rescue_no_design")
        if ambiguous_policy == "top12_rescue":
            out3: List[str] = []
            if g1:
                out3.append(g1)
            if g2 and g2 != g1:
                out3.append(g2)
            return (out3, "ambiguous_top12_rescue")
        # top1_rescue (default)
        return ([g1] if g1 else [], "ambiguous_top1_rescue")

    # Fallback: prefer design if present
    if has_design and expected:
        return ([expected], "fallback_expected")
    return ([g1] if g1 else [], "fallback_top1")


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
    comp["top_frac"] = comp["top_reads"] / comp["total_reads"]

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

    comp["expected_frac"] = comp["expected_reads"] / comp["total_reads"]
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


def _write_post_keepdrop(
    out_dir: Path,
    sample_name: str,
    *,
    expected_map: Dict[str, str],
    allowedset_map: Dict[str, Tuple[str, ...]],
    post_counts: Dict[Tuple[str, str], int],
    min_reads_post_clean: int,
    min_allowed_frac_post_clean: float,
):
    """
    Post-clean barcode keep/drop based on *post-clean* winner reads in allowed/expected genomes.

    Writes: {sample}_post_barcode_keepdrop.tsv.gz
      barcode
      expected_genome
      allowed_set
      allowed_reads_post
      total_reads_post
      allowed_frac_post
      keep
      reason
    """
    total_post: Dict[str, int] = defaultdict(int)
    by_bc_genome: Dict[str, Dict[str, int]] = defaultdict(dict)

    for (bc, gn), n in post_counts.items():
        n = int(n)
        total_post[bc] += n
        by_bc_genome[bc][gn] = by_bc_genome[bc].get(gn, 0) + n

    rows = []
    all_barcodes = set(expected_map.keys()) | set(allowedset_map.keys()) | set(total_post.keys())

    for bc in sorted(all_barcodes):
        exp = _norm_str(expected_map.get(bc, ""))
        allowed_tup = allowedset_map.get(bc, tuple())
        allowed_str = ",".join(allowed_tup) if allowed_tup else ""
        tr = int(total_post.get(bc, 0))

        # Define allowed reads:
        # - if expected genome exists (design-aware): use expected reads as the primary metric
        # - else: sum over allowed set
        if exp:
            ar = int(by_bc_genome.get(bc, {}).get(exp, 0))
            metric_note = "expected"
        else:
            ar = 0
            if allowed_tup:
                for g in allowed_tup:
                    ar += int(by_bc_genome.get(bc, {}).get(g, 0))
            metric_note = "allowedset"

        frac = (ar / tr) if tr > 0 else 0.0
        keep = (ar >= int(min_reads_post_clean)) and (frac >= float(min_allowed_frac_post_clean))

        if tr == 0:
            reason = "post_total_reads=0"
        elif ar < int(min_reads_post_clean):
            reason = f"allowed_reads_post<{min_reads_post_clean}"
        elif frac < float(min_allowed_frac_post_clean):
            reason = f"allowed_frac_post<{min_allowed_frac_post_clean}"
        else:
            reason = "keep"

        rows.append(
            {
                "barcode": bc,
                "expected_genome": exp if exp else "",
                "allowed_set": allowed_str,
                "allowed_metric": metric_note,
                "allowed_reads_post": int(ar),
                "total_reads_post": int(tr),
                "allowed_frac_post": float(frac),
                "keep": bool(keep),
                "reason": reason,
            }
        )

    out_path = out_dir / f"{sample_name}_post_barcode_keepdrop.tsv.gz"
    pd.DataFrame.from_records(rows).to_csv(out_path, sep="\t", index=False, compression="gzip")
    return out_path


# -------------------------
# Parallel read-level processing
# -------------------------


def _is_nan_like(x: Any) -> bool:
    try:
        return isinstance(x, float) and np.isnan(x)
    except Exception:
        return False


def _tuple_series_to_str(series: pd.Series) -> pd.Series:
    def f(t):
        if t is None or _is_nan_like(t):
            return ""
        if isinstance(t, tuple):
            return ",".join(t)
        return ""
    return series.map(f)


def _process_one_assign_file(
    fp: Path,
    *,
    out_part: Path,
    valid_barcodes: set[str],
    allowed_set_by_key: Dict[str, Tuple[str, ...]],
    drop_bckeys: set[str],
    design_bc_mode: str,
    design_bc_n: int,
    chunksize: int,
    read_id_col: str,
    barcode_col: str,
    genome_col: str,
    class_col: str,
    p_as_col: str,
    decontam_alpha: Optional[float],
    require_p_as: bool,
) -> Tuple[Counter, Counter, int]:
    """
    Worker: process one assignment file, write drop-part rows WITHOUT header, return partial counters.
    """
    pre_counts: Counter = Counter()
    post_counts: Counter = Counter()
    n_drop_rows = 0

    out_part.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_part, "wt") as fh:
        for chunk in pd.read_csv(fp, sep="\t", chunksize=chunksize, dtype=str, low_memory=False):
            required = {read_id_col, barcode_col, genome_col, class_col}
            missing = required - set(chunk.columns)
            if missing:
                raise ValueError(f"File {fp.name} missing columns: {sorted(missing)}")

            bc_full = chunk[barcode_col].astype(str)
            mask_valid = bc_full.isin(valid_barcodes)

            cls = chunk[class_col].astype(str)
            is_winner_default = cls.eq("winner")

            has_p = (p_as_col in chunk.columns)
            p_raw = chunk[p_as_col] if has_p else pd.Series(pd.NA, index=chunk.index)

            # Confidence
            if decontam_alpha is not None and has_p:
                p_vals = pd.to_numeric(p_raw, errors="coerce")
                haspv = p_vals.notna()
                is_winner_override = haspv & (p_vals <= float(decontam_alpha))
                if require_p_as:
                    is_confident = is_winner_override
                else:
                    is_confident = is_winner_override | ((~haspv) & is_winner_default)
            else:
                is_confident = is_winner_default

            winner_genome = chunk[genome_col].astype(str)

            bc_key = bc_full.map(lambda x: _design_key(x, design_bc_mode, design_bc_n))
            allowed_tup = bc_key.map(allowed_set_by_key)

            mask_drop_barcode = bc_key.isin(drop_bckeys)
            mask_no_allowed = allowed_tup.isna() | allowed_tup.map(lambda t: (t is None) or _is_nan_like(t) or (isinstance(t, tuple) and len(t) == 0))

            # membership
            at_list = allowed_tup.tolist()
            wg_list = winner_genome.tolist()
            in_allowed = []
            for w, t in zip(wg_list, at_list):
                if t is None or _is_nan_like(t):
                    in_allowed.append(False)
                elif isinstance(t, tuple):
                    in_allowed.append(str(w) in t)
                else:
                    in_allowed.append(False)
            mask_in_allowed = pd.Series(in_allowed, index=chunk.index)

            # Drop logic:
            # - drop all confident winners for bc_key flagged drop_barcode
            # - drop all confident winners when allowed set is empty (ambiguous drop or empty)
            # - drop confident winners that are not in allowed set
            mask_mismatch = (~mask_drop_barcode) & (~mask_no_allowed) & (~mask_in_allowed)
            mask_drop_read = is_confident & (mask_drop_barcode | mask_no_allowed | mask_mismatch)

            # PRE counts: confident winners only
            mask_pre = mask_valid & is_confident
            if mask_pre.any():
                vc = pd.DataFrame({"barcode": bc_full[mask_pre], "genome": winner_genome[mask_pre]}).value_counts()
                for (b, g), n in vc.items():
                    pre_counts[(str(b), str(g))] += int(n)

            # POST counts: confident winners that survive filtering
            mask_post = mask_valid & is_confident & (~mask_drop_read)
            if mask_post.any():
                vc2 = pd.DataFrame({"barcode": bc_full[mask_post], "genome": winner_genome[mask_post]}).value_counts()
                for (b, g), n in vc2.items():
                    post_counts[(str(b), str(g))] += int(n)

            # Write dropped reads rows (no header in part)
            if mask_drop_read.any():
                reasons = pd.Series("", index=chunk.index, dtype=object)
                reasons.loc[mask_drop_barcode & is_confident] = "drop_barcode"
                reasons.loc[mask_no_allowed & is_confident] = "no_allowed_set"
                reasons.loc[mask_mismatch & is_confident] = "mismatch_winner"

                out_df = pd.DataFrame(
                    {
                        "read_id": chunk[read_id_col].astype(str),
                        "barcode": bc_full,
                        "bc_key": bc_key,
                        "allowed_set": _tuple_series_to_str(allowed_tup),
                        "winner_genome": winner_genome,
                        "p_as": p_raw.fillna("").astype(str),
                        "reason": reasons,
                    }
                )
                n_this = int(mask_drop_read.sum())
                n_drop_rows += n_this
                out_df.loc[mask_drop_read].to_csv(fh, sep="\t", header=False, index=False)

    return pre_counts, post_counts, n_drop_rows


# -------------------------
# Main command
# -------------------------


@app.callback(invoke_without_command=True)
def decontam_cmd(
    cells_calls: Path = typer.Option(..., exists=True, readable=True, help="*_cells_calls.tsv(.gz) from genotyping."),
    out_dir: Path = typer.Option(..., help="Output directory for decontam artifacts."),

    # Parallelism
    threads: int = typer.Option(1, "--threads", help="Parallel workers over assignment chunk files."),
    tmp_dir: Optional[Path] = typer.Option(None, "--tmp-dir", help="Temp dir for per-file drop parts (default: out_dir/tmp_decontam)."),

    # Design / Layout
    design_file: Optional[Path] = typer.Option(None, "--design-file", readable=True, help="Optional design file. Enables design-aware mode."),
    layout_file: str = typer.Option("DEFAULT", "--layout-file", help='Layout path for well-range design files, or "DEFAULT".'),
    strict_design_drop_mismatch: bool = typer.Option(
        True,
        "--strict-design-drop-mismatch/--no-strict-design-drop-mismatch",
        help="If strict, drop barcodes whose bc_key is not present in the design (unknown wells).",
    ),

    # Policies
    doublet_policy: str = typer.Option("top12", "--doublet-policy", help="AllowedSet for doublets: top12|top1|expected"),
    indist_policy: str = typer.Option("top12", "--indist-policy", help="AllowedSet for indistinguishable: top12|top1|expected"),
    ambiguous_policy: str = typer.Option(
        "design_rescue",
        "--ambiguous-policy",
        help="Ambiguous handling: drop|design_rescue|top1_rescue|top12_rescue",
    ),

    # Post-clean keep gate
    min_reads_post_clean: int = typer.Option(
        100,
        "--min-reads-post-clean",
        help="Keep barcode only if allowed/expected post-clean winner reads >= this.",
    ),
    min_allowed_frac_post_clean: float = typer.Option(
        0.90,
        "--min-allowed-frac-post-clean",
        help="Keep barcode only if allowed_reads_post / total_reads_post >= this.",
    ),

    # Assignment inputs
    assignments: Optional[Path] = typer.Option(None, "--assignments", exists=True, readable=True, help="Optional single assignment file."),
    assign_glob: Optional[str] = typer.Option(None, "--assign-glob", help="Glob for assignment chunk files."),

    # Columns
    read_id_col: str = typer.Option("Read", "--read-id-col", help="Read id column in assignment files."),
    barcode_col: str = typer.Option("BC", "--barcode-col", help="Barcode column in assignment files."),
    genome_col: str = typer.Option("Genome", "--genome-col", help="Genome column in assignment files."),
    class_col: str = typer.Option("assigned_class", "--class-col", help="Assigned class column (expects 'winner')."),
    p_as_col: str = typer.Option("p_as", "--p-as-col", help="Optional p-value column for override confidence."),

    # Winner confidence logic
    decontam_alpha: Optional[float] = typer.Option(None, "--decontam-alpha", help="If set, reads with p_as<=alpha are considered confident."),
    require_p_as: bool = typer.Option(False, "--require-p-as", help="If set, only p_as<=alpha reads are considered confident."),

    # Misc
    sample_name: Optional[str] = typer.Option(None, "--sample-name"),
    chunksize: int = typer.Option(1_000_000, "--chunksize"),
    design_bc_mode: str = typer.Option("before-dash", "--design-bc-mode"),
    design_bc_n: int = typer.Option(10, "--design-bc-n"),
):
    """
    Generate decontamination decisions (barcode + read level) + barnyard-like summary TSVs.

    Outputs
      - {sample}_reads_to_drop.tsv.gz
      - {sample}_barcode_policy.tsv.gz
      - {sample}_pre_barcode_genome_counts.tsv.gz
      - {sample}_post_barcode_genome_counts.tsv.gz
      - {sample}_pre_barcode_composition.tsv.gz
      - {sample}_post_barcode_composition.tsv.gz
      - {sample}_pre_contamination_bins.tsv.gz (if design)
      - {sample}_post_contamination_bins.tsv.gz (if design)
      - {sample}_post_barcode_keepdrop.tsv.gz
      - {sample}_decontam_params.json
    """
    # Normalize policy values
    doublet_policy = _norm_str(doublet_policy).lower()
    indist_policy = _norm_str(indist_policy).lower()
    ambiguous_policy = _norm_str(ambiguous_policy).lower()

    if doublet_policy not in {"top12", "top1", "expected"}:
        raise ValueError(f"--doublet-policy must be one of top12|top1|expected (got {doublet_policy})")
    if indist_policy not in {"top12", "top1", "expected"}:
        raise ValueError(f"--indist-policy must be one of top12|top1|expected (got {indist_policy})")
    if ambiguous_policy not in {"drop", "design_rescue", "top1_rescue", "top12_rescue"}:
        raise ValueError(f"--ambiguous-policy must be one of drop|design_rescue|top1_rescue|top12_rescue (got {ambiguous_policy})")

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

    # Temp dir
    tmp_dir = (tmp_dir or (out_dir / "tmp_decontam")).expanduser().resolve()
    drops_dir = tmp_dir / "drops"
    drops_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1) Load per-cell calls
    # ------------------------------------------------------------------
    typer.echo(f"[decontam] Loading cell calls: {cells_calls}")
    calls = pd.read_csv(cells_calls, sep="\t", dtype=str, low_memory=False)

    if "barcode" not in calls.columns:
        raise ValueError(f"{cells_calls} missing required column: 'barcode'")
    if "call" not in calls.columns:
        raise ValueError(f"{cells_calls} missing required column: 'call'")

    g1_col = _col_first(calls, ["genome_1", "top_genome", "best_genome", "top1_genome"])
    g2_col = _col_first(calls, ["genome_2", "top2_genome"])
    if g1_col is None:
        raise ValueError(
            f"{cells_calls} missing a top1 column. Expected one of: genome_1, top_genome, best_genome, top1_genome"
        )

    calls["barcode"] = calls["barcode"].astype(str)
    calls["bc_key"] = calls["barcode"].map(lambda x: _design_key(x, design_bc_mode, design_bc_n))

    # Valid barcodes set to bound counting/memory
    valid_barcodes = set(calls["barcode"].unique())

    # ------------------------------------------------------------------
    # 2) Build design map (optional)
    # ------------------------------------------------------------------
    bckey_to_expected: Optional[Dict[str, str]] = None
    if design_file is not None:
        typer.echo(f"[decontam] Design-aware: {design_file}")
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
        typer.echo("[decontam] Agnostic (no design)")

    has_design = bckey_to_expected is not None

    # ------------------------------------------------------------------
    # 3) Build AllowedSet policy per barcode
    #    - allowed_set_by_key: bc_key -> tuple(genomes)
    #    - allowed_set_by_barcode: barcode -> tuple(genomes) (for keep/drop and reporting)
    #    - expected_map: barcode -> expected genome (if design), else ""
    # ------------------------------------------------------------------
    allowed_set_by_key: Dict[str, Tuple[str, ...]] = {}
    allowed_set_by_barcode: Dict[str, Tuple[str, ...]] = {}
    expected_map: Dict[str, str] = {}
    drop_bckeys: set[str] = set()

    policy_rows: List[dict] = []

    for row in calls.itertuples(index=False):
        bc_full = _norm_str(getattr(row, "barcode", ""))
        bc_key = _norm_str(getattr(row, "bc_key", ""))
        call_val = _norm_str(getattr(row, "call", ""))

        g1 = _norm_str(getattr(row, g1_col, ""))
        g2 = _norm_str(getattr(row, g2_col, "")) if g2_col else ""

        indist_set = ""
        if "indistinguishable_set" in calls.columns:
            indist_set = _norm_str(getattr(row, "indistinguishable_set", ""))

        expected = ""
        if has_design and bc_key:
            expected = _norm_str(bckey_to_expected.get(bc_key, ""))

        # strict design mismatch: unknown well key => drop barcode entirely
        if has_design and strict_design_drop_mismatch and (not expected):
            drop_bckeys.add(bc_key)

        allowed_list, note = _allowed_from_policy(
            call=call_val,
            g1=g1,
            g2=g2,
            indist_set=indist_set,
            expected=expected,
            has_design=has_design,
            doublet_policy=doublet_policy,
            indist_policy=indist_policy,
            ambiguous_policy=ambiguous_policy,
        )

        allowed_tuple = tuple(sorted(set([x for x in allowed_list if _norm_str(x)])))

        # Store maps
        if bc_key:
            allowed_set_by_key[bc_key] = allowed_tuple
        allowed_set_by_barcode[bc_full] = allowed_tuple
        expected_map[bc_full] = expected if expected else ""

        policy_rows.append(
            {
                "barcode": bc_full,
                "bc_key": bc_key,
                "call": call_val,
                "genome_1": g1,
                "genome_2": g2,
                "expected_genome": expected if expected else "",
                "allowed_set": ",".join(allowed_tuple),
                "doublet_policy": doublet_policy,
                "indist_policy": indist_policy,
                "ambiguous_policy": ambiguous_policy,
                "strict_design_drop_mismatch": bool(strict_design_drop_mismatch),
                "drop_barcode": bool((bc_key in drop_bckeys) if bc_key else False),
                "note": note,
            }
        )

    policy_path = out_dir / f"{sample_name}_barcode_policy.tsv.gz"
    pd.DataFrame.from_records(policy_rows).to_csv(policy_path, sep="\t", index=False, compression="gzip")
    typer.echo(f"[decontam] Wrote {policy_path}")

    # ------------------------------------------------------------------
    # 4) Identify assignment files
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # 5) Read-level decisions + barnyard pre/post counts (confident winners)
    #    Parallel per file: write drop parts + merge counts
    # ------------------------------------------------------------------
    reads_to_drop_path: Optional[Path] = None
    pre_counts: Counter = Counter()
    post_counts: Counter = Counter()
    drop_total = 0

    if input_files:
        threads = max(1, int(threads))
        typer.echo(f"[decontam] Processing {len(input_files)} assignment files (threads={threads})")

        def _part_path(fp: Path) -> Path:
            # ensure uniqueness; fp.stem is enough for *_chunkNN_filtered.tsv.gz
            return drops_dir / f"{fp.stem}.reads_to_drop.part.tsv.gz"

        part_paths: List[Path] = []

        if threads == 1:
            for fp in input_files:
                part = _part_path(fp)
                part_paths.append(part)
                pc, qc, nd = _process_one_assign_file(
                    fp,
                    out_part=part,
                    valid_barcodes=valid_barcodes,
                    allowed_set_by_key=allowed_set_by_key,
                    drop_bckeys=drop_bckeys,
                    design_bc_mode=design_bc_mode,
                    design_bc_n=design_bc_n,
                    chunksize=chunksize,
                    read_id_col=read_id_col,
                    barcode_col=barcode_col,
                    genome_col=genome_col,
                    class_col=class_col,
                    p_as_col=p_as_col,
                    decontam_alpha=decontam_alpha,
                    require_p_as=require_p_as,
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
                        allowed_set_by_key=allowed_set_by_key,
                        drop_bckeys=drop_bckeys,
                        design_bc_mode=design_bc_mode,
                        design_bc_n=design_bc_n,
                        chunksize=chunksize,
                        read_id_col=read_id_col,
                        barcode_col=barcode_col,
                        genome_col=genome_col,
                        class_col=class_col,
                        p_as_col=p_as_col,
                        decontam_alpha=decontam_alpha,
                        require_p_as=require_p_as,
                    )
                    futs[fut] = fp

                for fut in as_completed(futs):
                    fp = futs[fut]
                    pc, qc, nd = fut.result()
                    pre_counts.update(pc)
                    post_counts.update(qc)
                    drop_total += nd
                    typer.echo(f"  done: {fp.name} (drop_rows={nd})")

        # Merge gzip parts safely: one header + concat gzip members
        reads_to_drop_path = out_dir / f"{sample_name}_reads_to_drop.tsv.gz"
        header = "read_id\tbarcode\tbc_key\tallowed_set\twinner_genome\tp_as\treason\n"
        with gzip.open(reads_to_drop_path, "wt") as wtxt:
            wtxt.write(header)

        # append parts as raw gzip members
        with open(reads_to_drop_path, "ab") as wbin:
            for part in sorted(part_paths):
                if part.exists() and part.stat().st_size > 0:
                    with open(part, "rb") as rbin:
                        shutil.copyfileobj(rbin, wbin, length=1024 * 1024)

        typer.echo(f"[decontam] Wrote {reads_to_drop_path} (drop_rows_total={drop_total})")

        # cleanup tmp parts (leave tmp_dir if user set it explicitly)
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 6) Barnyard summaries pre/post
    # ------------------------------------------------------------------
    typer.echo("[decontam] Writing barnyard-like summary tables...")
    _write_barnyard_summaries(out_dir, sample_name, "pre", pre_counts, expected_map)
    _write_barnyard_summaries(out_dir, sample_name, "post", post_counts, expected_map)

    # ------------------------------------------------------------------
    # 7) Post-clean keep/drop (uses expected if design else allowed_set)
    # ------------------------------------------------------------------
    keepdrop_path = _write_post_keepdrop(
        out_dir,
        sample_name,
        expected_map=expected_map,
        allowedset_map=allowed_set_by_barcode,
        post_counts=post_counts,
        min_reads_post_clean=min_reads_post_clean,
        min_allowed_frac_post_clean=min_allowed_frac_post_clean,
    )
    typer.echo(f"[decontam] Wrote {keepdrop_path}")

    # ------------------------------------------------------------------
    # 8) Params JSON
    # ------------------------------------------------------------------
    params_path = out_dir / f"{sample_name}_decontam_params.json"
    params = {
        "cells_calls": str(cells_calls),
        "out_dir": str(out_dir),
        "design_file": str(design_file) if design_file else None,
        "layout_file": layout_file,
        "strict_design_drop_mismatch": bool(strict_design_drop_mismatch),
        "doublet_policy": doublet_policy,
        "indist_policy": indist_policy,
        "ambiguous_policy": ambiguous_policy,
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
        "decontam_alpha": float(decontam_alpha) if decontam_alpha is not None else None,
        "require_p_as": bool(require_p_as),
        "reads_to_drop": str(reads_to_drop_path) if reads_to_drop_path else None,
        "cells_calls_columns_used": {
            "g1_col": g1_col,
            "g2_col": g2_col,
            "has_indistinguishable_set": bool("indistinguishable_set" in calls.columns),
        },
        "design_bc_mode": design_bc_mode,
        "design_bc_n": int(design_bc_n),
    }
    with params_path.open("w") as f:
        json.dump(params, f, indent=2)
    typer.echo(f"[decontam] Wrote {params_path}")

    typer.echo("[decontam] Done.")


if __name__ == "__main__":
    app()
