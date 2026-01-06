# src/ambientmapper/decontam.py
from __future__ import annotations

import gzip
import json
import glob
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Optional, List, Tuple

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
    help="Generate barcode + read-level decontamination decisions.",
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

    # expected genome (normalize empty -> NaN)
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
    post_counts: Dict[Tuple[str, str], int],
    min_reads_post_clean: int,
):
    """
    Post-clean barcode keep/drop based on *post-clean* winner reads in the expected genome.

    Writes: {sample}_post_barcode_keepdrop.tsv.gz
      barcode
      expected_genome
      expected_reads_post
      total_reads_post
      keep
      reason
    """
    # Build post totals and expected reads without building a large dataframe
    total_post: Dict[str, int] = defaultdict(int)
    expected_post: Dict[str, int] = defaultdict(int)

    for (bc, gn), n in post_counts.items():
        n = int(n)
        total_post[bc] += n
        exp = _norm_str(expected_map.get(bc, ""))
        if exp and gn == exp:
            expected_post[bc] += n

    # Include all barcodes that have an expected genome (even if 0 post reads)
    rows = []
    for bc, exp in expected_map.items():
        exp = _norm_str(exp)
        if not exp:
            continue
        er = int(expected_post.get(bc, 0))
        tr = int(total_post.get(bc, 0))
        keep = er >= int(min_reads_post_clean)
        reason = "keep" if keep else f"expected_reads_post<{min_reads_post_clean}"
        rows.append(
            {
                "barcode": bc,
                "expected_genome": exp,
                "expected_reads_post": er,
                "total_reads_post": tr,
                "keep": bool(keep),
                "reason": reason,
            }
        )

    out_path = out_dir / f"{sample_name}_post_barcode_keepdrop.tsv.gz"
    pd.DataFrame.from_records(rows).to_csv(out_path, sep="\t", index=False, compression="gzip")
    return out_path


@app.callback(invoke_without_command=True)
def decontam_cmd(
    cells_calls: Path = typer.Option(..., exists=True, readable=True, help="*_cells_calls.tsv(.gz) from genotyping."),
    out_dir: Path = typer.Option(..., help="Output directory for decontam artifacts."),

    # Design / Layout
    design_file: Optional[Path] = typer.Option(None, "--design-file", readable=True, help="Optional design file. Enables design-aware mode."),
    layout_file: str = typer.Option("DEFAULT", "--layout-file", help='Layout path for well-range design files, or "DEFAULT".'),
    strict_design_drop_mismatch: bool = typer.Option(
        True,
        "--strict-design-drop-mismatch/--no-strict-design-drop-mismatch",
        help="If strict, drop barcodes whose bc_key is not present in the design (unknown wells).",
    ),

    # Post-clean barcode retention threshold (applied AFTER computing post-clean counts)
    min_reads_post_clean: int = typer.Option(
        10,
        "--min-reads-post-clean",
        help="After decontam, keep barcode only if expected/allowed genome has >= this many post-clean winner reads.",
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

    # Winner logic
    decontam_alpha: Optional[float] = typer.Option(None, "--decontam-alpha"),
    require_p_as: bool = typer.Option(False, "--require-p-as"),

    # Misc
    sample_name: Optional[str] = typer.Option(None, "--sample-name"),
    chunksize: int = typer.Option(1_000_000, "--chunksize"),
    design_bc_mode: str = typer.Option("before-dash", "--design-bc-mode"),
    design_bc_n: int = typer.Option(10, "--design-bc-n"),
):
    """
    Generate decontamination decisions (barcode + read level) + barnyard-like summary TSVs.

    Key rules:
      - bc_full = cell identity (counts/stats per cell)
      - bc_key  = well identity for design matching ONLY
      - design mismatch => keep barcode; drop mismatching *winner* reads (keep reads matching expected)
      - unknown-in-design => optionally drop entire barcode (strict flag)

    Barnyard summaries:
      - pre_* computed from confident winners (is_confident)
      - post_* computed from confident winners that survive decontam drop masks
      - post barcode retention table uses expected_reads_post (post-clean evidence)
    """
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

    # ------------------------------------------------------------------
    # 1) Load per-cell calls
    # ------------------------------------------------------------------
    typer.echo(f"[decontam] Loading cell calls: {cells_calls}")
    calls = pd.read_csv(cells_calls, sep="\t", dtype=str)

    if "barcode" not in calls.columns:
        raise ValueError(f"{cells_calls} missing required column: 'barcode'")

    best_col = _col_first(calls, ["best_genome", "top_genome", "genome_top1", "genome_1"])
    if best_col is None:
        raise ValueError(
            f"{cells_calls} missing a best-genome column. Expected one of: best_genome, top_genome, genome_top1, genome_1"
        )

    calls["barcode"] = calls["barcode"].astype(str)
    calls["bc_key"] = calls["barcode"].map(lambda x: _design_key(x, design_bc_mode, design_bc_n))

    # Valid barcodes set to limit growth of counters (keeps memory bounded to called barcodes)
    valid_barcodes = set(calls["barcode"].unique())

    # ------------------------------------------------------------------
    # 1b) Build design map (optional)
    # ------------------------------------------------------------------
    bckey_to_expected: Optional[Dict[str, str]] = None
    if design_file is not None:
        typer.echo(f"[decontam] Strategy A: Design-Aware ({design_file})")
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
        typer.echo(f"[decontam] Strategy B: Agnostic (Allowed = {best_col})")

    # ------------------------------------------------------------------
    # 2) Per-barcode allowed genome decision
    #    - Read filtering uses bc_key -> allowed genome
    #    - Summary/retention is per bc_full
    # ------------------------------------------------------------------
    allowed_genome_map: Dict[str, str] = {}  # bc_key -> allowed_genome
    barcodes_to_drop_set: set[str] = set()  # bc_key to drop entirely

    barcode_summary_rows: List[dict] = []
    per_barcode_rules_rows: List[dict] = []

    # expected_map for composition/keepdrop is per bc_full:
    # - design-aware: expected genome (from bc_key)
    # - agnostic: allowed genome == best_col per barcode
    barcode_to_expected_map: Dict[str, str] = {}

    for row in calls.itertuples(index=False):
        bc_full = _norm_str(getattr(row, "barcode", ""))
        bc_key = _norm_str(getattr(row, "bc_key", ""))
        best = _norm_str(getattr(row, best_col, ""))

        expected: Optional[str] = None
        allowed: str = ""
        action = "keep_cleaned"
        notes = ""

        if not bc_key:
            action = "drop_barcode"
            notes = "empty_bc_key"

        elif bckey_to_expected is not None:
            expected = _norm_str(bckey_to_expected.get(bc_key))
            barcode_to_expected_map[bc_full] = expected  # may be ""

            if not expected:
                if strict_design_drop_mismatch:
                    action = "drop_barcode"
                    notes = "unknown_in_design"
                else:
                    action = "keep_cleaned"
                    allowed = ""
                    notes = "unknown_in_design_keep"
            else:
                allowed = expected
                notes = "design_expected_enforced"

        else:
            # agnostic mode
            expected = best
            barcode_to_expected_map[bc_full] = best
            if not best:
                allowed = ""
                notes = "no_best_genome_keep"
            else:
                allowed = best
                notes = f"agnostic_{best_col}"

        if action == "drop_barcode":
            barcodes_to_drop_set.add(bc_key)
        elif allowed:
            allowed_genome_map[bc_key] = allowed
            per_barcode_rules_rows.append({"barcode": bc_full, "bc_key": bc_key, "allowed_genome": allowed})

        barcode_summary_rows.append(
            {
                "barcode": bc_full,
                "bc_key": bc_key,
                "best_genome_col": best_col,
                "best_genome": best,
                "expected_genome": expected,
                "allowed_genome": allowed,
                "action": action,
                "notes": notes,
                "strategy": "A" if (bckey_to_expected is not None) else "B",
            }
        )

    # ------------------------------------------------------------------
    # 3) Identify assignment files
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
    # 4) Read-level decisions + barnyard pre/post counts (winner-only)
    # ------------------------------------------------------------------
    reads_to_drop_path: Optional[Path] = None

    # Accumulators: (barcode, genome) -> n_winner_reads
    pre_counts: Dict[Tuple[str, str], int] = Counter()
    post_counts: Dict[Tuple[str, str], int] = Counter()

    if input_files:
        reads_to_drop_path = out_dir / f"{sample_name}_reads_to_drop.tsv.gz"
        typer.echo(f"[decontam] Processing {len(input_files)} assignment files...")

        with gzip.open(reads_to_drop_path, "wt") as fh:
            fh.write("read_id\tbarcode\tbc_key\tallowed_genome\twinner_genome\tp_as\treason\n")

            for fp in input_files:
                typer.echo(f"  -> Streaming {fp.name}")

                for chunk in pd.read_csv(fp, sep="\t", chunksize=chunksize, dtype=str):
                    required = {read_id_col, barcode_col, genome_col, class_col}
                    missing = required - set(chunk.columns)
                    if missing:
                        raise ValueError(f"File {fp.name} missing columns: {sorted(missing)}")

                    bc_full_series = chunk[barcode_col].astype(str)
                    mask_valid = bc_full_series.isin(valid_barcodes)

                    # Winner/confidence logic
                    class_series = chunk[class_col].astype(str)
                    is_winner_default = class_series.eq("winner")

                    has_p_col = p_as_col in chunk.columns
                    p_raw = chunk[p_as_col] if has_p_col else pd.Series(pd.NA, index=chunk.index)

                    if decontam_alpha is not None and has_p_col:
                        p_vals = pd.to_numeric(p_raw, errors="coerce")
                        has_p = p_vals.notna()
                        is_winner_override = has_p & (p_vals <= float(decontam_alpha))
                        if require_p_as:
                            is_confident = is_winner_override
                        else:
                            is_confident = is_winner_override | ((~has_p) & is_winner_default)
                    else:
                        is_confident = is_winner_default

                    winner_genome = chunk[genome_col].astype(str)

                    # Filtering masks
                    bc_key_series = bc_full_series.map(lambda x: _design_key(x, design_bc_mode, design_bc_n))
                    allowed_series = bc_key_series.map(allowed_genome_map)

                    mask_drop_barcode = bc_key_series.isin(barcodes_to_drop_set)
                    mask_no_allowed = allowed_series.isna() | allowed_series.astype(str).eq("")
                    mask_mismatch = (~mask_drop_barcode) & (~mask_no_allowed) & (winner_genome != allowed_series)

                    mask_mismatch_conf = mask_mismatch & is_confident
                    mask_drop_read = mask_drop_barcode | mask_mismatch_conf

                    # PRE counts: confident winners only
                    mask_pre = mask_valid & is_confident
                    if mask_pre.any():
                        vc = pd.DataFrame(
                            {"barcode": bc_full_series[mask_pre], "genome": winner_genome[mask_pre]}
                        ).value_counts()
                        for (bc, gn), n in vc.items():
                            pre_counts[(bc, gn)] += int(n)

                    # POST counts: confident winners that survive filtering
                    mask_post = mask_valid & is_confident & (~mask_drop_read)
                    if mask_post.any():
                        vc2 = pd.DataFrame(
                            {"barcode": bc_full_series[mask_post], "genome": winner_genome[mask_post]}
                        ).value_counts()
                        for (bc, gn), n in vc2.items():
                            post_counts[(bc, gn)] += int(n)

                    # Write dropped reads
                    if mask_drop_read.any():
                        reasons = pd.Series("", index=chunk.index, dtype=object)
                        reasons.loc[mask_drop_barcode] = "drop_barcode"
                        reasons.loc[mask_mismatch_conf] = "mismatch_winner"

                        out_df = pd.DataFrame(
                            {
                                "read_id": chunk[read_id_col].astype(str),
                                "barcode": bc_full_series,
                                "bc_key": bc_key_series,
                                "allowed_genome": allowed_series.fillna(""),
                                "winner_genome": winner_genome,
                                "p_as": p_raw.fillna("").astype(str),
                                "reason": reasons,
                            }
                        )
                        out_df.loc[mask_drop_read].to_csv(fh, sep="\t", header=False, index=False)

        typer.echo(f"[decontam] Wrote {reads_to_drop_path}")

    # ------------------------------------------------------------------
    # 5) Write barnyard summaries (pre/post) and post-clean keep/drop table
    # ------------------------------------------------------------------
    typer.echo("[decontam] Writing barnyard-like summary tables...")
    _write_barnyard_summaries(out_dir, sample_name, "pre", pre_counts, barcode_to_expected_map)
    _write_barnyard_summaries(out_dir, sample_name, "post", post_counts, barcode_to_expected_map)

    # Post-clean barcode retention threshold (true post-clean evidence)
    keepdrop_path = _write_post_keepdrop(
        out_dir,
        sample_name,
        expected_map=barcode_to_expected_map,
        post_counts=post_counts,
        min_reads_post_clean=min_reads_post_clean,
    )
    typer.echo(f"[decontam] Wrote {keepdrop_path}")

    # ------------------------------------------------------------------
    # 6) Write standard decontam artifacts
    # ------------------------------------------------------------------
    barcode_summary_path = out_dir / f"{sample_name}_barcode_summary.tsv.gz"
    per_barcode_rules_path = out_dir / f"{sample_name}_per_barcode_rules.tsv.gz"
    params_path = out_dir / f"{sample_name}_decontam_params.json"

    pd.DataFrame.from_records(barcode_summary_rows).to_csv(
        barcode_summary_path, sep="\t", index=False, compression="gzip"
    )
    pd.DataFrame.from_records(per_barcode_rules_rows).to_csv(
        per_barcode_rules_path, sep="\t", index=False, compression="gzip"
    )

    params = {
        "cells_calls": str(cells_calls),
        "out_dir": str(out_dir),
        "design_file": str(design_file) if design_file else None,
        "layout_file": layout_file,
        "strict_design_drop_mismatch": strict_design_drop_mismatch,
        "min_reads_post_clean": int(min_reads_post_clean),
        "assignments": str(assignments) if assignments else None,
        "assign_glob": assign_glob,
        "read_id_col": read_id_col,
        "barcode_col": barcode_col,
        "genome_col": genome_col,
        "class_col": class_col,
        "p_as_col": p_as_col,
        "decontam_alpha": decontam_alpha,
        "require_p_as": require_p_as,
        "chunksize": chunksize,
        "reads_to_drop": str(reads_to_drop_path) if reads_to_drop_path else None,
        "best_col": best_col,
        "design_bc_mode": design_bc_mode,
        "design_bc_n": design_bc_n,
    }
    with params_path.open("w") as f:
        json.dump(params, f, indent=2)

    typer.echo(f"[decontam] Wrote {barcode_summary_path}")
    typer.echo(f"[decontam] Wrote {per_barcode_rules_path}")
    typer.echo(f"[decontam] Wrote {params_path}")
    typer.echo("[decontam] Done.")
