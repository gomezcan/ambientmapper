# src/ambientmapper/decontam.py
from __future__ import annotations

import gzip
import json
import glob
from pathlib import Path
from typing import Dict, Optional, List

import pandas as pd
import typer

from .data import get_default_layout_path
from .utils import (
    load_sample_to_wells,
    load_barcode_layout,
    build_well_to_sample,
    build_barcode_to_sample,
)

app = typer.Typer(help="Generate barcode + read-level decontamination decisions.")


def _bc_seq(barcode: str) -> str:
    """Extract sequence part of barcode (remove suffix -Sample, -1, etc)."""
    return str(barcode).split("-")[0]


def _build_bcseq_to_expected_genome(
    layout_path: Path,
    design_path: Path,
) -> Dict[str, str]:
    """
    Build bc_seq -> expected_genome map.

    Supports:
      (1) TSV with header: barcode  expected_genome
      (2) Pool ranges file: genome <tab> wells (A1-4,B1-4,...), using a 96-well layout
    """
    design_path = Path(design_path)
    layout_path = Path(layout_path)

    # Try direct barcode mapping format first
    try:
        df = pd.read_csv(design_path, sep="\t", dtype=str)
    except Exception:
        df = None

    if df is not None and {"barcode", "expected_genome"}.issubset(df.columns):
        df = df[["barcode", "expected_genome"]].astype(str)
        df["barcode"] = df["barcode"].map(_bc_seq)
        return dict(zip(df["barcode"], df["expected_genome"]))

    # Fallback: pool ranges + layout
    sample_to_wells = load_sample_to_wells(design_path)
    well_to_sample = build_well_to_sample(sample_to_wells)
    well_to_barcode = load_barcode_layout(layout_path)
    bc_to_sample = build_barcode_to_sample(well_to_barcode, well_to_sample)
    return {str(k): str(v) for k, v in bc_to_sample.items()}


@app.command("run")
def decontam_cmd(
    cells_calls: Path = typer.Option(
        ...,
        exists=True,
        readable=True,
        help="*_cells_calls.tsv(.gz) from genotyping.",
    ),
    out_dir: Path = typer.Option(
        ...,
        help="Output directory for decontam artifacts.",
    ),
    # Design / Layout
    design_file: Optional[Path] = typer.Option(
        None,
        "--design-file",
        exists=True,
        readable=True,
        help="Optional design file. If provided enables design-aware mode.",
    ),
    layout_file: str = typer.Option(
        "DEFAULT",
        "--layout-file",
        help='Layout path for well-range design files, or "DEFAULT" (bundled).',
    ),
    strict_design_drop_mismatch: bool = typer.Option(
        True,
        "--strict-design-drop-mismatch/--no-strict-design-drop-mismatch",
        help="If strict, drop whole barcode when expected genome is not in top3.",
    ),
    # Assignment inputs
    assignments: Optional[Path] = typer.Option(
        None,
        "--assignments",
        exists=True,
        readable=True,
        help="Single assignment chunk path (tsv/tsv.gz).",
    ),
    assign_glob: Optional[str] = typer.Option(
        None,
        "--assign-glob",
        help="Glob for assign chunks (e.g. 'SM2/cell_map_ref_chunks/SM2_chunk*_filtered.tsv.gz').",
    ),
    # Column overrides (assign outputs)
    read_id_col: str = typer.Option("Read", "--read-id-col", help="Column for Read ID"),
    barcode_col: str = typer.Option("BC", "--barcode-col", help="Column for Barcode"),
    genome_col: str = typer.Option("Genome", "--genome-col", help="Column for Genome"),
    class_col: str = typer.Option(
        "assigned_class",
        "--class-col",
        help="Column for classification (winner/ambiguous).",
    ),
    p_as_col: str = typer.Option("p_as", "--p-as-col", help="Column for assignment p-value"),
    # Logic configuration
    decontam_alpha: Optional[float] = typer.Option(
        None,
        "--decontam-alpha",
        help="If set, redefine 'winner' using p_as <= alpha (where p_as exists).",
    ),
    require_p_as: bool = typer.Option(
        False,
        "--require-p-as",
        help="If True and --decontam-alpha is set, treat missing p_as as non-winner.",
    ),
    # Misc
    sample_name: Optional[str] = typer.Option(None, "--sample-name", help="Output prefix."),
    chunksize: int = typer.Option(1_000_000, "--chunksize", help="Pandas chunk size."),
):
    """
    Generate decontamination decisions (barcode + read level).

    Outputs:
      - <sample>_barcode_summary.tsv.gz
      - <sample>_barcodes_to_drop.tsv.gz
      - <sample>_per_barcode_rules.tsv.gz
      - <sample>_reads_to_drop.tsv.gz  (only if assignment files are provided)
      - <sample>_decontam_params.json
    """
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if sample_name is None:
        bn = cells_calls.name
        sample_name = bn.replace("_cells_calls.tsv.gz", "").replace("_cells_calls.tsv", "")

    # ------------------------------------------------------------------
    # 1) Load per-barcode calls & Build Design Map
    # ------------------------------------------------------------------
    typer.echo(f"[decontam] Loading cell calls: {cells_calls}")
    calls = pd.read_csv(cells_calls, sep="\t")

    need_calls = {"barcode", "genome_top1", "genome_top2", "genome_top3"}
    miss = need_calls - set(calls.columns)
    if miss:
        raise ValueError(f"{cells_calls} missing columns: {sorted(miss)}")

    calls["barcode"] = calls["barcode"].astype(str)
    calls["bc_seq"] = calls["barcode"].map(_bc_seq)

    bcseq_to_expected: Optional[Dict[str, str]] = None
    if design_file is not None:
        typer.echo(f"[decontam] Strategy A: Design-Aware ({design_file})")

        if layout_file == "DEFAULT":
            layout_path = get_default_layout_path()
        else:
            layout_path = Path(layout_file).expanduser().resolve()
            if not layout_path.exists():
                raise FileNotFoundError(f"--layout-file not found: {layout_path}")

        bcseq_to_expected = _build_bcseq_to_expected_genome(layout_path, design_file)
        if not bcseq_to_expected:
            raise ValueError(f"Design parsing produced an empty bc->expected map: {design_file}")
    else:
        typer.echo("[decontam] Strategy B: Agnostic (Allowed = Top1)")

    # ------------------------------------------------------------------
    # 2) Per-barcode allowed genome decision (keyed by bc_seq)
    # ------------------------------------------------------------------
    allowed_genome_map: Dict[str, str] = {}
    barcodes_to_drop_set: set[str] = set()

    barcode_summary_rows = []
    barcodes_to_drop_rows = []
    per_barcode_rules_rows = []

    for row in calls.itertuples(index=False):
        bc_full = str(row.barcode)
        bc_seq_val = str(row.bc_seq)

        g1 = str(row.genome_top1)
        g2 = str(row.genome_top2)
        g3 = str(row.genome_top3)

        expected: Optional[str] = None
        allowed: Optional[str] = None
        action = "keep_cleaned"
        notes = ""

        if bcseq_to_expected is not None:
            expected = bcseq_to_expected.get(bc_seq_val)
            if expected is None:
                action = "drop_barcode"
                notes = "unknown_in_design"
            else:
                if expected == g1:
                    allowed = expected
                    notes = "design_match_top1"
                elif expected == g2:
                    allowed = expected
                    notes = "design_match_top2"
                elif expected == g3:
                    allowed = expected
                    notes = "design_match_top3"
                else:
                    if strict_design_drop_mismatch:
                        action = "drop_barcode"
                        notes = "design_mismatch_strict"
                    else:
                        allowed = expected
                        notes = "design_mismatch_lenient"
        else:
            allowed = g1
            notes = "agnostic_top1"

        if action == "drop_barcode":
            barcodes_to_drop_set.add(bc_seq_val)
            barcodes_to_drop_rows.append(
                {"barcode": bc_full, "bc_seq": bc_seq_val, "reason": notes}
            )
        elif allowed is not None:
            allowed_genome_map[bc_seq_val] = str(allowed)
            per_barcode_rules_rows.append(
                {"barcode": bc_full, "bc_seq": bc_seq_val, "drop_genome_ne": str(allowed)}
            )

        barcode_summary_rows.append(
            {
                "barcode": bc_full,
                "bc_seq": bc_seq_val,
                "expected_genome": expected,
                "allowed_genome": allowed,
                "action": action,
                "notes": notes,
                "strategy": "A" if (bcseq_to_expected is not None) else "B",
            }
        )

    # ------------------------------------------------------------------
    # 3) Identify assignment files
    # ------------------------------------------------------------------
    input_files: List[Path] = []

    if assignments:
        input_files.append(Path(assignments).expanduser().resolve())

    if assign_glob:
        assign_glob = assign_glob.strip() or None
    if assign_glob:
        found = sorted(glob.glob(assign_glob, recursive=True))
        if not found:
            typer.echo(f"[decontam] Warning: No files matched glob: {assign_glob}", err=True)
        input_files.extend([Path(f).resolve() for f in found])

    input_files = sorted(set(input_files))

    if not input_files:
        typer.echo("[decontam] No assignment files found. Skipping read-level logic.")

    # ------------------------------------------------------------------
    # 4) Process read assignments → reads_to_drop.tsv.gz
    # ------------------------------------------------------------------
    reads_to_drop_path: Optional[Path] = None
    reads_total_map: Dict[str, int] = {}
    reads_dropped_map: Dict[str, int] = {}

    warned_missing_p_as = False

    if input_files:
        reads_to_drop_path = out_dir / f"{sample_name}_reads_to_drop.tsv.gz"
        typer.echo(f"[decontam] Processing {len(input_files)} assignment files...")
        with gzip.open(reads_to_drop_path, "wt") as fh:
            fh.write("read_id\tbarcode\tbc_seq\tallowed_genome\twinner_genome\tp_as\treason\n")

            for fp in input_files:
                typer.echo(f"  -> Streaming {fp.name}")

                for chunk in pd.read_csv(fp, sep="\t", chunksize=chunksize, dtype=str):
                    required_in_chunk = {read_id_col, barcode_col, genome_col, class_col}
                    missing_cols = required_in_chunk - set(chunk.columns)
                    if missing_cols:
                        raise ValueError(f"File {fp.name} missing columns: {sorted(missing_cols)}")

                    bc_full_series = chunk[barcode_col].astype(str)
                    bc_seq_series = bc_full_series.map(_bc_seq)

                    allowed_series = bc_seq_series.map(allowed_genome_map)
                    mask_drop_barcode = bc_seq_series.isin(barcodes_to_drop_set) | allowed_series.isna()

                    winner_genome = chunk[genome_col].astype(str)
                    mask_mismatch = (~mask_drop_barcode) & (winner_genome != allowed_series)

                    # Winner definition
                    class_series = chunk[class_col].astype(str)
                    is_winner_default = (class_series == "winner")

                    has_p_col = (p_as_col in chunk.columns)
                    p_raw = chunk[p_as_col] if has_p_col else pd.Series(pd.NA, index=chunk.index)

                    if decontam_alpha is not None and not has_p_col and not warned_missing_p_as:
                        typer.echo(
                            f"[decontam] Warning: --decontam-alpha set but column '{p_as_col}' not found "
                            f"in {fp.name}; falling back to '{class_col}' only for those files.",
                            err=True,
                        )
                        warned_missing_p_as = True

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

                    mask_mismatch_conf = mask_mismatch & is_confident
                    mask_drop_read = mask_drop_barcode | mask_mismatch_conf

                    # Totals always
                    counts_total = bc_seq_series.value_counts()
                    for b, c in counts_total.items():
                        reads_total_map[b] = reads_total_map.get(b, 0) + int(c)

                    if not mask_drop_read.any():
                        continue

                    counts_dropped = bc_seq_series[mask_drop_read].value_counts()
                    for b, c in counts_dropped.items():
                        reads_dropped_map[b] = reads_dropped_map.get(b, 0) + int(c)

                    reasons = pd.Series("", index=chunk.index, dtype=object)
                    reasons.loc[mask_drop_barcode] = "drop_barcode"

                    if decontam_alpha is not None and has_p_col:
                        p_vals2 = pd.to_numeric(p_raw, errors="coerce")
                        has_p2 = p_vals2.notna()
                        mask_p_conf = mask_mismatch_conf & has_p2 & (p_vals2 <= float(decontam_alpha))
                        mask_class_conf = mask_mismatch_conf & ~mask_p_conf
                        reasons.loc[mask_p_conf] = "mismatch_winner_pvalue"
                        reasons.loc[mask_class_conf] = "mismatch_winner_class"
                    else:
                        reasons.loc[mask_mismatch_conf] = "mismatch_winner"

                    out_df = pd.DataFrame(
                        {
                            "read_id": chunk[read_id_col].astype(str),
                            "barcode": bc_full_series,
                            "bc_seq": bc_seq_series,
                            "allowed_genome": allowed_series.fillna(""),
                            "winner_genome": winner_genome,
                            "p_as": p_raw.fillna("").astype(str),
                            "reason": reasons,
                        }
                    )

                    out_df.loc[mask_drop_read].to_csv(fh, sep="\t", header=False, index=False)

        typer.echo(f"[decontam] Wrote {reads_to_drop_path}")

    # ------------------------------------------------------------------
    # 5) Finalize summary stats
    # ------------------------------------------------------------------
    barcode_summary = pd.DataFrame.from_records(barcode_summary_rows)
    barcode_summary["reads_total"] = barcode_summary["bc_seq"].map(reads_total_map).fillna(0).astype(int)
    barcode_summary["reads_dropped"] = barcode_summary["bc_seq"].map(reads_dropped_map).fillna(0).astype(int)

    mask_nz = barcode_summary["reads_total"] > 0
    barcode_summary["fraction_removed"] = 0.0
    barcode_summary.loc[mask_nz, "fraction_removed"] = (
        barcode_summary.loc[mask_nz, "reads_dropped"] / barcode_summary.loc[mask_nz, "reads_total"]
    )

    # ------------------------------------------------------------------
    # 6) Write artifacts
    # ------------------------------------------------------------------
    barcode_summary_path = out_dir / f"{sample_name}_barcode_summary.tsv.gz"
    barcodes_to_drop_path = out_dir / f"{sample_name}_barcodes_to_drop.tsv.gz"
    per_barcode_rules_path = out_dir / f"{sample_name}_per_barcode_rules.tsv.gz"
    params_path = out_dir / f"{sample_name}_decontam_params.json"

    barcode_summary.to_csv(barcode_summary_path, sep="\t", index=False)
    pd.DataFrame.from_records(barcodes_to_drop_rows).to_csv(barcodes_to_drop_path, sep="\t", index=False)
    pd.DataFrame.from_records(per_barcode_rules_rows).to_csv(per_barcode_rules_path, sep="\t", index=False)

    params = {
        "cells_calls": str(cells_calls),
        "out_dir": str(out_dir),
        "design_file": str(design_file) if design_file else None,
        "layout_file": layout_file,
        "strict_design_drop_mismatch": strict_design_drop_mismatch,
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
    }
    with params_path.open("w") as f:
        json.dump(params, f, indent=2)

    typer.echo(f"[decontam] Wrote {barcode_summary_path}")
    typer.echo(f"[decontam] Wrote {barcodes_to_drop_path}")
    typer.echo(f"[decontam] Wrote {per_barcode_rules_path}")
    typer.echo(f"[decontam] Wrote {params_path}")
    typer.echo("[decontam] Done.")
