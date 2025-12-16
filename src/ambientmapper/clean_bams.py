# ambientmapper/clean_bams.py
from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import pysam
import typer

app = typer.Typer(help="Apply AmbientMapper decontamination decisions to BAM files.")


def norm_bc(raw: str) -> str:
    # Match decontam.py convention
    return str(raw).split("-")[0]


@app.command("clean-bams")
def clean_bams_cmd(
    bam_map: Path = typer.Option(
        ...,
        exists=True,
        help="TSV with header: bam_id, bam_path, genome_id",
    ),
    reads_to_drop_tsv: Path = typer.Option(
        ...,
        "--reads-to-drop",
        exists=True,
        help="reads_to_drop.tsv.gz from decontam (must contain read_id).",
    ),
    rules_tsv: Optional[Path] = typer.Option(
        None,
        "--rules",
        exists=True,
        help="Optional per-barcode rules (barcode, drop_genome_ne).",
    ),
    drop_barcodes_tsv: Optional[Path] = typer.Option(
        None,
        "--drop-barcodes",
        exists=True,
        help="Optional barcodes_to_drop TSV (from decontam).",
    ),
    out_dir: Path = typer.Option(..., help="Output directory for cleaned BAMs."),
    barcode_tag: str = typer.Option("BC", help="SAM tag for barcode (BC or CB)."),
    combined_genome_id: str = typer.Option(
        "COMBINED",
        help="genome_id value indicating a combined-reference BAM.",
    ),
    log_every: int = typer.Option(1_000_000, help="Progress log interval (reads)."),
):
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load reads_to_drop (QNAMEs)
    rtd = pd.read_csv(reads_to_drop_tsv, sep="\t", dtype=str)
    if "read_id" not in rtd.columns:
        raise ValueError(f"{reads_to_drop_tsv} must contain a 'read_id' column.")
    bad_reads = set(rtd["read_id"].astype(str))

    # 2) Load optional barcode drop list
    bad_barcodes = set()
    if drop_barcodes_tsv is not None:
        bdf = pd.read_csv(drop_barcodes_tsv, sep="\t", dtype=str)
        if "barcode" not in bdf.columns:
            raise ValueError(f"{drop_barcodes_tsv} must contain a 'barcode' column.")
        bad_barcodes = set(bdf["barcode"].astype(str).map(norm_bc))

    # 3) Load optional per-barcode allowed genome rule
    keep_genome = {}
    if rules_tsv is not None:
        rules_df = pd.read_csv(rules_tsv, sep="\t", dtype=str)
        if not {"barcode", "drop_genome_ne"}.issubset(rules_df.columns):
            raise ValueError(f"{rules_tsv} must contain columns: barcode, drop_genome_ne")
        # normalize barcode key here too
        keep_genome = dict(
            zip(rules_df["barcode"].astype(str).map(norm_bc), rules_df["drop_genome_ne"].astype(str))
        )

    # 4) Load bam_map with explicit genome_id
    bm = pd.read_csv(bam_map, sep="\t", dtype=str)
    missing = {"bam_id", "bam_path", "genome_id"} - set(bm.columns)
    if missing:
        raise ValueError(f"{bam_map} missing columns: {', '.join(sorted(missing))}")

    for row in bm.itertuples(index=False):
        bam_id = row.bam_id
        bam_path = Path(row.bam_path)
        genome_id = str(row.genome_id)

        if not bam_path.exists():
            typer.echo(f"[clean-bams] WARNING: BAM not found: {bam_path}", err=True)
            continue

        typer.echo(f"[clean-bams] Cleaning {bam_path} (bam_id={bam_id}, genome_id={genome_id})")

        in_bam = pysam.AlignmentFile(str(bam_path), "rb")
        out_path = out_dir / f"{bam_path.stem}.clean.bam"
        out_bam = pysam.AlignmentFile(str(out_path), "wb", template=in_bam)

        n_in = n_out = 0
        n_drop_read = n_drop_bc = n_drop_genome = 0

        for i, read in enumerate(in_bam, start=1):
            n_in += 1
            if log_every and i % log_every == 0:
                typer.echo(f"[clean-bams]  processed {i/1e6:.1f}M reads ...")

            qname = read.query_name
            if qname in bad_reads:
                n_drop_read += 1
                continue

            try:
                bc_raw = read.get_tag(barcode_tag)
            except KeyError:
                # Policy choice: drop reads lacking a barcode
                continue

            bc = norm_bc(bc_raw)

            if bc in bad_barcodes:
                n_drop_bc += 1
                continue

            # Only apply genome rule when this BAM is genome-specific
            req = keep_genome.get(bc)
            if req is not None and genome_id != combined_genome_id:
                if genome_id != req:
                    n_drop_genome += 1
                    continue

            out_bam.write(read)
            n_out += 1

        in_bam.close()
        out_bam.close()
        pysam.index(str(out_path))

        typer.echo(
            f"[clean-bams] {bam_path} → {out_path} | "
            f"in={n_in}, kept={n_out}, "
            f"dropped_read={n_drop_read}, "
            f"dropped_barcode={n_drop_bc}, "
            f"dropped_genome={n_drop_genome}"
        )
