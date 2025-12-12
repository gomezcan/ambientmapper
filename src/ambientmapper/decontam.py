# ambientmapper/decontam.py
from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import typer

from .data import get_default_layout_path
from .utils import (
    load_sample_to_wells,
    load_barcode_layout,
    build_well_to_sample,
    build_barcode_to_sample,
)

app = typer.Typer(help="AmbientMapper decontamination decision engine.")


def _build_bcseq_to_expected_genome(
    layout_path: Path,
    design_path: Path,
) -> Dict[str, str]:
    """
    Build bc_seq -> expected_genome map.

    Two supported formats for design_path:

    1) Direct mapping TSV with header:
         barcode  expected_genome
         GTTACCCTGTACAATC   B73
         ...

       In this case, layout_path is ignored.

    2) Pool ranges format (no header), e.g.:
         B73   A1-4,B1-4,C1-4,...
         At    A9-12,B9-12,...

       Combined with the 96-well layout, this yields:
         well_id        -> tn5_bc
         sample/genome  -> wells
         tn5_bc         -> sample/genome
    """
    design_path = Path(design_path)
    layout_path = Path(layout_path)

    # Try direct "barcode / expected_genome" format first.
    try:
        df = pd.read_csv(design_path, sep="\t")
    except Exception:
        df = None

    if df is not None and {"barcode", "expected_genome"}.issubset(df.columns):
        df = df[["barcode", "expected_genome"]].astype(str)
        return dict(zip(df["barcode"], df["expected_genome"]))

    # Fallback: pool ranges + layout
    sample_to_wells = load_sample_to_wells(design_path)
    well_to_sample = build_well_to_sample(sample_to_wells)
    well_to_barcode = load_barcode_layout(layout_path)
    bc_to_sample = build_barcode_to_sample(well_to_barcode, well_to_sample)
    # Here "sample" is your genome ID (e.g. B73, At)
    return bc_to_sample


@app.command("decontam")
def decontam_cmd(
    cells_calls: Path = typer.Option(
        ...,
        exists=True,
        help="Per-barcode calls TSV/TSV.GZ.",
    ),
    out_dir: Path = typer.Option(
        ...,
        help="Output directory for decontam results.",
    ),
    design_file: Optional[Path] = typer.Option(
        None,
        "--design-file",
        exists=True,
        help=(
            "Optional design file. "
            "If provided, enables design-aware mode using the bundled 96-well layout "
            "or a custom layout. Supported formats:\n"
            "  (1) TSV with 'barcode' and 'expected_genome' columns.\n"
            "  (2) Pool ranges text file, e.g. "
            "'B73\\tA1-4,B1-4,...', 'At\\tA9-12,...'."
        ),
    ),
    layout_file: str = typer.Option(
        "DEFAULT",
        "--layout-file",
        help=(
            'Layout file path, or "DEFAULT" to use bundled 96-well Tn5 layout. '
            "Used only if --design-file is provided and is in pool-ranges format."
        ),
    ),
    assignments: Optional[Path] = typer.Option(
        None,
        "--assignments",
        exists=True,
        help=(
            "Optional per-read assignments TSV(.gz) with at least columns "
            "'barcode', 'read_id', 'genome'. When provided, a "
            "<sample>_reads_to_drop.tsv.gz file is generated and "
            "barcode summaries are updated with per-barcode read counts and "
            "fraction_removed."
        ),
    ),
    chunksize: int = typer.Option(
        1_000_000,
        help="Chunk size (rows) for streaming --assignments TSV.",
    ),
    strict_design_drop_mismatch: bool = typer.Option(
        True,
        help=(
            "In design-aware mode, if expected genome is not among "
            "top1–top3 genomes for a barcode, drop the barcode entirely. "
            "If False, keep the barcode but restrict it to the expected genome."
        ),
    ),
    sample_name: Optional[str] = typer.Option(
        None,
        help=(
            "Optional sample name; default is inferred from cells_calls basename "
            "(prefix before '_cells_calls.tsv[.gz]')."
        ),
    ),
):
    """
    Generate decontamination decisions.

    Outputs:
      - <sample>_barcode_summary.tsv.gz
      - <sample>_barcodes_to_drop.tsv.gz
      - <sample>_per_barcode_rules.tsv.gz
      - <sample>_decontam_params.json
      - <sample>_reads_to_drop.tsv.gz   (only if --assignments is provided)
    """
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if sample_name is None:
        bn = cells_calls.name
        sample_name = (
            bn.replace("_cells_calls.tsv.gz", "")
            .replace("_cells_calls.tsv", "")
        )

    # ------------------------------------------------------------------
    # 1. Load per-barcode calls
    # ------------------------------------------------------------------
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
            f"{cells_calls} missing required columns: "
            f"{', '.join(sorted(missing))}"
        )

    # ------------------------------------------------------------------
    # 2. Determine strategy & build expected-genome map if needed
    # ------------------------------------------------------------------
    bcseq_to_expected: Optional[Dict[str, str]] = None
    if design_file is not None:
        typer.echo(f"[decontam] Design file provided: {design_file}")
        typer.echo("[decontam] Using Strategy A: design-aware decontamination")

        if layout_file == "DEFAULT":
            layout_path = get_default_layout_path()
        else:
            layout_path = Path(layout_file)

        bcseq_to_expected = _build_bcseq_to_expected_genome(
            layout_path=layout_path,
            design_path=design_file,
        )
    else:
        typer.echo("[decontam] No design file. Using Strategy B: agnostic mode")

    # Containers
    barcode_summary_rows = []
    barcodes_to_drop_rows = []
    per_barcode_rules_rows = []

    # Internal state for downstream read-level decontam
    barcodes_to_drop_set: set[str] = set()
    allowed_genome_map: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # 3. Iterate barcodes and define allowed genome per barcode
    # ------------------------------------------------------------------
    for row in calls.itertuples():
        barcode = row.barcode
        bc_seq = str(barcode).split("-")[0]

        G1, M1 = row.genome_top1, row.mass_top1
        G2, M2 = row.genome_top2, row.mass_top2
        G3, M3 = row.genome_top3, row.mass_top3

        strategy = "A" if bcseq_to_expected is not None else "B"
        action = "keep_cleaned"
        expected_genome: Optional[str] = None
        allowed_genome: Optional[str] = None

        is_doublet_candidate = False
        is_highly_ambiguous = False
        notes = ""

        # -------------------------
        # Strategy A: design-aware
        # -------------------------
        if bcseq_to_expected is not None:
            expected_genome = bcseq_to_expected.get(bc_seq)

            if expected_genome is None:
                # Unknown barcode (Tn5) in design
                action = "drop_barcode"
                notes = "expected_genome_missing"
                barcodes_to_drop_rows.append(
                    {
                        "barcode": barcode,
                        "reason": "unknown_in_design",
                    }
                )
                barcodes_to_drop_set.add(barcode)
            else:
                # Check match against top1–top3
                if expected_genome == G1:
                    allowed_genome = expected_genome
                    notes = f"design_match_top1:{G1}"
                elif expected_genome == G2:
                    allowed_genome = expected_genome
                    notes = f"design_match_top2:top1={G1},top2={G2}"
                elif expected_genome == G3:
                    allowed_genome = expected_genome
                    notes = (
                        f"design_match_top3:top1={G1},top2={G2},top3={G3}"
                    )
                else:
                    # Expected genome not in top3
                    if strict_design_drop_mismatch:
                        action = "drop_barcode"
                        notes = (
                            f"design_mismatch_strict:"
                            f"expected={expected_genome},"
                            f"top1={G1},top2={G2},top3={G3}"
                        )
                        barcodes_to_drop_rows.append(
                            {
                                "barcode": barcode,
                                "reason": "design_mismatch",
                            }
                        )
                        barcodes_to_drop_set.add(barcode)
                    else:
                        # Lenient: keep barcode, restrict to expected genome
                        allowed_genome = expected_genome
                        notes = (
                            f"design_mismatch_lenient:"
                            f"expected={expected_genome},"
                            f"top1={G1},top2={G2},top3={G3}"
                        )

        # -------------------------
        # Strategy B: design-agnostic
        # -------------------------
        else:
            # Keep barcode; allowed genome = top1
            allowed_genome = G1

        # Record rule if we are keeping the barcode and have an allowed genome
        if action != "drop_barcode" and allowed_genome is not None:
            per_barcode_rules_rows.append(
                {
                    "barcode": barcode,
                    # Semantic: keep only this genome, drop others.
                    "drop_genome_ne": allowed_genome,
                }
            )
            allowed_genome_map[str(barcode)] = str(allowed_genome)

        # Basic ambiguity flags (QC)
        if M2 > 0 and M1 > 0:
            ratio = M1 / M2 if M2 > 0 else float("inf")
            if ratio < 2.0:
                is_doublet_candidate = True
                if notes:
                    notes += ";"
                notes += (
                    f"doublet_candidate:top1={G1},top2={G2},ratio={ratio:.2f}"
                )

        # Example ambiguous flag: more than 2 genomes with non-trivial mass
        n_nontrivial = sum(m > 0.05 for m in [M1, M2, M3])
        if n_nontrivial > 2:
            is_highly_ambiguous = True
            if notes:
                notes += ";"
            notes += "highly_ambiguous_top3"

        # Placeholder; updated later if --assignments is provided
        fraction_removed = None

        barcode_summary_rows.append(
            {
                "barcode": barcode,
                "bc_seq": bc_seq,
                "expected_genome": expected_genome,
                "allowed_genome": allowed_genome,
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

    # ------------------------------------------------------------------
    # 4. Optional: per-read decontam from assignments
    # ------------------------------------------------------------------
    reads_to_drop_path: Optional[Path] = None

    if assignments is not None:
        typer.echo(f"[decontam] Streaming assignments from {assignments}")

        # For QC: per-barcode read counts
        reads_total: Dict[str, int] = {}
        reads_dropped: Dict[str, int] = {}

        reads_to_drop_path = out_dir / f"{sample_name}_reads_to_drop.tsv.gz"
        with gzip.open(reads_to_drop_path, "wt") as fh:
            # Header
            fh.write(
                "read_id\tbarcode\tallowed_genome\twinner_genome\treason\n"
            )

            # Stream in chunks
            is_first_chunk = True
            for chunk in pd.read_csv(
                assignments,
                sep="\t",
                chunksize=chunksize,
                dtype=str,
            ):
                if is_first_chunk:
                    required_assign_cols = {"barcode", "read_id", "genome"}
                    missing_assign = required_assign_cols - set(chunk.columns)
                    if missing_assign:
                        raise ValueError(
                            f"{assignments} missing required columns: "
                            f"{', '.join(sorted(missing_assign))}"
                        )
                    is_first_chunk = False

                # Ensure we have the expected columns as strings
                chunk["barcode"] = chunk["barcode"].astype(str)
                chunk["read_id"] = chunk["read_id"].astype(str)
                chunk["genome"] = chunk["genome"].astype(str)

                # Map barcode → allowed genome
                chunk["allowed_genome"] = chunk["barcode"].map(
                    allowed_genome_map
                )

                # Drop-barcode flag: explicitly dropped in design logic OR
                # barcode has no allowed genome.
                chunk["drop_barcode"] = (
                    chunk["barcode"].isin(barcodes_to_drop_set)
                    | chunk["allowed_genome"].isna()
                )

                # Winner mismatch: barcode kept but genome != allowed_genome
                chunk["drop_winner"] = (
                    ~chunk["drop_barcode"]
                    & (chunk["genome"] != chunk["allowed_genome"])
                )

                mask_drop = chunk["drop_barcode"] | chunk["drop_winner"]

                # Reason column
                chunk["reason"] = "winner_not_allowed"
                chunk.loc[chunk["drop_barcode"], "reason"] = "drop_barcode"

                # Write dropped reads (append, no header)
                drop_df = chunk.loc[
                    mask_drop,
                    [
                        "read_id",
                        "barcode",
                        "allowed_genome",
                        "genome",
                        "reason",
                    ],
                ]
                drop_df.to_csv(
                    fh,
                    sep="\t",
                    header=False,
                    index=False,
                )

                # Update per-barcode counts
                total_counts = chunk["barcode"].value_counts()
                drop_counts = chunk.loc[mask_drop, "barcode"].value_counts()

                for bc, n in total_counts.items():
                    reads_total[bc] = reads_total.get(bc, 0) + int(n)

                for bc, n in drop_counts.items():
                    reads_dropped[bc] = reads_dropped.get(bc, 0) + int(n)

        typer.echo(f"[decontam] Wrote {reads_to_drop_path}")

        # Update barcode_summary_rows with read stats
        bs = pd.DataFrame.from_records(barcode_summary_rows)
        bs.set_index("barcode", inplace=True, drop=False)

        # Map counts
        bs["reads_total"] = bs["barcode"].map(reads_total).fillna(0).astype(int)
        bs["reads_dropped"] = (
            bs["barcode"].map(reads_dropped).fillna(0).astype(int)
        )

        # Fraction removed; 0 if no reads
        with pd.option_context("mode.use_inf_as_na", True):
            frac = bs["reads_dropped"] / bs["reads_total"].replace(0, pd.NA)
        bs["fraction_removed"] = frac

        barcode_summary = bs.reset_index(drop=True)
    else:
        # No assignments: just materialize DataFrame as-is
        barcode_summary = pd.DataFrame.from_records(barcode_summary_rows)

    # ------------------------------------------------------------------
    # 5. Save outputs
    # ------------------------------------------------------------------
    barcodes_to_drop = pd.DataFrame.from_records(barcodes_to_drop_rows)
    per_barcode_rules = pd.DataFrame.from_records(per_barcode_rules_rows)

    barcode_summary_path = out_dir / f"{sample_name}_barcode_summary.tsv.gz"
    barcodes_to_drop_path = out_dir / f"{sample_name}_barcodes_to_drop.tsv.gz"
    per_barcode_rules_path = (
        out_dir / f"{sample_name}_per_barcode_rules.tsv.gz"
    )
    params_path = out_dir / f"{sample_name}_decontam_params.json"

    barcode_summary.to_csv(barcode_summary_path, sep="\t", index=False)

    # Always write barcodes_to_drop file (even if empty, with header)
    barcodes_to_drop.to_csv(barcodes_to_drop_path, sep="\t", index=False)

    per_barcode_rules.to_csv(per_barcode_rules_path, sep="\t", index=False)

    params = {
        "cells_calls": str(cells_calls),
        "design_file": str(design_file) if design_file is not None else None,
        "layout_file": layout_file,
        "assignments": str(assignments) if assignments is not None else None,
        "chunksize": chunksize,
        "strict_design_drop_mismatch": strict_design_drop_mismatch,
        "sample_name": sample_name,
        "reads_to_drop": (
            str(reads_to_drop_path) if reads_to_drop_path is not None else None
        ),
    }
    with params_path.open("w") as fh:
        json.dump(params, fh, indent=2)

    typer.echo(f"[decontam] Wrote {barcode_summary_path}")
    typer.echo(f"[decontam] Wrote {barcodes_to_drop_path}")
    typer.echo(f"[decontam] Wrote {per_barcode_rules_path}")
    typer.echo(f"[decontam] Wrote {params_path}")
    if reads_to_drop_path is not None:
        typer.echo(f"[decontam] Wrote {reads_to_drop_path}")
