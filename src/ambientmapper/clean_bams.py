# ambientmapper/clean_bams.py

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import pysam
import typer

app = typer.Typer(help="Apply decontamination rules to BAM files.")


@app.command("clean-bams")
def clean_bams_cmd(
    bam_map: Path = typer.Option(..., exists=True, help="TSV with columns: id<tab>path_to_bam"),
    rules_tsv: Path = typer.Option(
        ..., "--drop-list", exists=True,
        help="Per-barcode rules TSV (from decontam): barcode, drop_genome_ne."
    ),
    drop_barcodes_tsv: Optional[Path] = typer.Option(
        None,
        "--drop-barcodes",
        exists=True,
        help="Optional barcodes_to_drop TSV (from decontam).",
    ),
    out_dir: Path = typer.Option(..., help="Output directory for cleaned BAMs."),
    barcode_tag: str = typer.Option(
        "BC", help="SAM tag holding the cell/barcode (e.g., BC or CB)."
    ),
):
    """
    Use per-barcode rules to create genome-cleaned BAMs.

    For each input BAM:
      - Determine its 'genome id' from the bam_map id column.
      - For each read:
          - extract barcode from barcode_tag
          - if barcode ∈ bad_barcodes → drop
          - else, if barcode has a rule in rules_tsv, keep only if
            current_bam_genome == drop_genome_ne(barcode)
    """
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load rules
    rules_df = pd.read_csv(rules_tsv, sep="\t")
    if "barcode" not in rules_df.columns or "drop_genome_ne" not in rules_df.columns:
        raise ValueError(
            f"{rules_tsv} must contain 'barcode' and 'drop_genome_ne' columns."
        )

    # dict: barcode -> required genome (the only genome to keep)
    rules = dict(zip(rules_df["barcode"], rules_df["drop_genome_ne"]))

    # 2. Load barcodes_to_drop (if provided)
    bad_barcodes = set()
    if drop_barcodes_tsv is not None:
        drop_df = pd.read_csv(drop_barcodes_tsv, sep="\t")
        if "barcode" not in drop_df.columns:
            raise ValueError(
                f"{drop_barcodes_tsv} must contain a 'barcode' column."
            )
        bad_barcodes = set(drop_df["barcode"].astype(str))

    # 3. Load bam_map
    bam_map_df = pd.read_csv(bam_map, sep="\t", header=None, names=["id", "path"])
    for row in bam_map_df.itertuples():
        bam_id = row.id
        bam_path = Path(row.path)

        if not bam_path.exists():
            typer.echo(f"[clean-bams] WARNING: BAM not found: {bam_path}", err=True)
            continue

        current_bam_genome = str(bam_id)  # or custom mapping if desired

        typer.echo(f"[clean-bams] Cleaning {bam_path} (genome_id={current_bam_genome})")

        in_bam = pysam.AlignmentFile(str(bam_path), "rb")
        out_path = out_dir / f"{bam_path.stem}.clean.bam"
        out_bam = pysam.AlignmentFile(str(out_path), "wb", template=in_bam)

        n_in = 0
        n_out = 0
        n_dropped_barcode = 0
        n_dropped_genome_mismatch = 0

        for read in in_bam:
            n_in += 1
            try:
                bc = read.get_tag(barcode_tag)
            except KeyError:
                # no barcode; keep or drop? For now, drop silently.
                continue

            bc = str(bc)

            # 1. Drop barcode-level
            if bc in bad_barcodes:
                n_dropped_barcode += 1
                continue

            # 2. Apply genome rule, if any
            req_genome = rules.get(bc)
            if req_genome is not None:
                if current_bam_genome != str(req_genome):
                    n_dropped_genome_mismatch += 1
                    continue

            out_bam.write(read)
            n_out += 1

        in_bam.close()
        out_bam.close()
        pysam.index(str(out_path))

        typer.echo(
            f"[clean-bams] {bam_path} → {out_path} | "
            f"in={n_in}, kept={n_out}, "
            f"dropped_barcode={n_dropped_barcode}, "
            f"dropped_genome_mismatch={n_dropped_genome_mismatch}"
        )
