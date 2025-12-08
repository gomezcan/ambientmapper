# ambientmapper/decontam.py

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
import typer

from .utils import PlateDesign
from .data import get_default_layout_path

app = typer.Typer(help="AmbientMapper decontamination decision engine.")


@app.command("decontam")
def decontam_cmd(
    cells_calls: Path = typer.Option(..., exists=True, help="Per-barcode calls TSV/TSV.GZ."),
    out_dir: Path = typer.Option(..., help="Output directory for decontam results."),
    design_file: Optional[Path] = typer.Option(
        None, "--design-file", exists=True, help="Optional design TSV for design-aware mode."
    ),
    layout_file: str = typer.Option(
        "DEFAULT",
        "--layout-file",
        help='Layout file path, or "DEFAULT" to use bundled 96-well layout.',
    ),
    strict_design_drop_mismatch: bool = typer.Option(
        True,
        help="If True, barcodes whose expected genome != top1 genome are dropped entirely.",
    ),
    sample_name: Optional[str] = typer.Option(
        None,
        help="Optional sample name; default is inferred from cells_calls basename.",
    ),
):
    """
    Generate per-barcode decontamination decisions.

    Outputs:
      - <sample>_barcode_summary.tsv.gz
      - <sample>_barcodes_to_drop.tsv.gz
      - <sample>_per_barcode_rules.tsv.gz
      - <sample>_decontam_params.json
    """
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if sample_name is None:
        sample_name = cells_calls.name.replace("_cells_calls.tsv.gz", "").replace(
            "_cells_calls.tsv", ""
        )

    # ------------------------------------------------------------------ #
    # 1. Load calls
    # ------------------------------------------------------------------ #
    calls = pd.read_csv(cells_calls, sep="\t")
    required_cols = {
        "barcode",
        "genome_top1",
        "mass_top1",
        "genome_top2",
        "mass_top2",
        "genome_top3",
        "mass_top3",
    }
    missing = required_cols - set(calls.columns)
    if missing:
        raise ValueError(
            f"{cells_calls} missing required columns: {', '.join(sorted(missing))}"
        )

    # ------------------------------------------------------------------ #
    # 2. Determine strategy and load design map if needed
    # ------------------------------------------------------------------ #
    design_map: Optional[PlateDesign] = None
    if design_file is not None:
        typer.echo(f"[decontam] Design file provided: {design_file}")
        typer.echo("[decontam] Using Strategy A: design-aware decontamination")

        layout_path = (
            Path(layout_file)
            if layout_file != "DEFAULT"
            else get_default_layout_path()
        )
        design_map = PlateDesign(layout_path=layout_path, design_path=design_file)
    else:
        typer.echo("[decontam] No design file. Using Strategy B: agnostic mode")

    # Containers
    barcode_summary_rows = []
    barcodes_to_drop_rows = []
    per_barcode_rules_rows = []

    # ------------------------------------------------------------------ #
    # 3. Iterate barcodes
    # ------------------------------------------------------------------ #
    for row in calls.itertuples():
        barcode = row.barcode
        G1, M1 = row.genome_top1, row.mass_top1
        G2, M2 = row.genome_top2, row.mass_top2
        G3, M3 = row.genome_top3, row.mass_top3

        strategy = "A" if design_map is not None else "B"
        action = "keep_cleaned"
        expected_genome = None
        is_doublet_candidate = False
        is_highly_ambiguous = False
        notes = ""

        if design_map is not None:
            # Strategy A: design-aware
            expected_genome = design_map.get_expected_genome(barcode)

            if expected_genome is None:
                # Unknown barcode in design
                action = "drop_barcode"
                notes = "expected_genome_missing"
                barcodes_to_drop_rows.append(
                    {
                        "barcode": barcode,
                        "reason": "unknown_in_design",
                    }
                )
            else:
                # A1: expected == top1
                if expected_genome == G1:
                    # keep barcode; keep only reads from expected genome
                    per_barcode_rules_rows.append(
                        {
                            "barcode": barcode,
                            # meaning: drop reads from genomes != this one
                            "drop_genome_ne": expected_genome,
                        }
                    )
                else:
                    # A2/A3/A4: mismatch of some kind
                    if strict_design_drop_mismatch:
                        action = "drop_barcode"
                        notes = f"design_mismatch: expected={expected_genome},top1={G1}"
                        barcodes_to_drop_rows.append(
                            {
                                "barcode": barcode,
                                "reason": "design_mismatch",
                            }
                        )
                    else:
                        # lenient mode: keep barcode but restrict to expected genome
                        per_barcode_rules_rows.append(
                            {
                                "barcode": barcode,
                                "drop_genome_ne": expected_genome,
                            }
                        )
        else:
            # Strategy B: design-agnostic
            # keep barcode; keep only reads from top1 genome
            per_barcode_rules_rows.append(
                {"barcode": barcode, "drop_genome_ne": G1}
            )

            # Basic ambiguity flags (for QC)
            if M2 > 0 and M1 > 0:
                ratio = M1 / M2 if M2 > 0 else float("inf")
                if ratio < 2.0:
                    is_doublet_candidate = True
                    notes = f"doublet_candidate:top1={G1},top2={G2},ratio={ratio:.2f}"
            # Example "highly ambiguous" criterion: more than 3 genomes above small mass
            # You can adjust this based on full per-genome distribution if available.
            # Here we just show a placeholder flag using top3 masses.
            n_nontrivial = sum(m > 0.05 for m in [M1, M2, M3])
            if n_nontrivial > 2:
                is_highly_ambiguous = True
                if notes:
                    notes += ";"
                notes += "highly_ambiguous_top3"

        # For now we do not compute a true fraction_removed (requires per-genome read counts).
        fraction_removed = None

        barcode_summary_rows.append(
            {
                "barcode": barcode,
                "expected_genome": expected_genome,
                "genome_top1": G1,
                "mass_top1": M1,
                "genome_top2": G2,
                "mass_top2": M2,
                "genome_top3": G3,
                "mass_top3": M3,
                "strategy": strategy,
                "action": action,
                "fraction_removed": fraction_removed,
                "is_doublet_candidate": is_doublet_candidate,
                "is_highly_ambiguous": is_highly_ambiguous,
                "notes": notes,
            }
        )

    # ------------------------------------------------------------------ #
    # 4. Save outputs
    # ------------------------------------------------------------------ #
    barcode_summary = pd.DataFrame.from_records(barcode_summary_rows)
    barcodes_to_drop = pd.DataFrame.from_records(barcodes_to_drop_rows)
    per_barcode_rules = pd.DataFrame.from_records(per_barcode_rules_rows)

    barcode_summary_path = out_dir / f"{sample_name}_barcode_summary.tsv.gz"
    barcodes_to_drop_path = out_dir / f"{sample_name}_barcodes_to_drop.tsv.gz"
    per_barcode_rules_path = out_dir / f"{sample_name}_per_barcode_rules.tsv.gz"
    params_path = out_dir / f"{sample_name}_decontam_params.json"

    barcode_summary.to_csv(barcode_summary_path, sep="\t", index=False)
    if not barcodes_to_drop.empty:
        barcodes_to_drop.to_csv(barcodes_to_drop_path, sep="\t", index=False)
    else:
        # write empty with header
        barcodes_to_drop.to_csv(barcodes_to_drop_path, sep="\t", index=False)

    per_barcode_rules.to_csv(per_barcode_rules_path, sep="\t", index=False)

    params = {
        "cells_calls": str(cells_calls),
        "design_file": str(design_file) if design_file is not None else None,
        "layout_file": layout_file,
        "strict_design_drop_mismatch": strict_design_drop_mismatch,
        "sample_name": sample_name,
    }
    with params_path.open("w") as fh:
        json.dump(params, fh, indent=2)

    typer.echo(f"[decontam] Wrote {barcode_summary_path}")
    typer.echo(f"[decontam] Wrote {barcodes_to_drop_path}")
    typer.echo(f"[decontam] Wrote {per_barcode_rules_path}")
    typer.echo(f"[decontam] Wrote {params_path}")
