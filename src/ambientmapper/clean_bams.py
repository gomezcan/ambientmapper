# src/ambientmapper/clean_bams.py
from __future__ import annotations

import gzip
from pathlib import Path
from typing import Optional, Set

import pandas as pd
import pysam
import typer

app = typer.Typer(help="Apply AmbientMapper decontamination decisions to BAM files (read-id filtering only).")


def _load_read_ids(reads_to_drop_tsv: Path) -> Set[str]:
    """
    Load read IDs (QNAMEs) to drop.
    Expects a TSV/TSV.GZ with a 'read_id' column.
    """
    df = pd.read_csv(reads_to_drop_tsv, sep="\t", dtype=str)
    if "read_id" not in df.columns:
        raise ValueError(f"{reads_to_drop_tsv} must contain a 'read_id' column.")
    # Drop NA, enforce str
    return set(df["read_id"].dropna().astype(str).tolist())


def _iter_bams(bam: Optional[Path], bam_map: Optional[Path]):
    if bam is None and bam_map is None:
        raise typer.BadParameter("Provide either --bam or --bam-map.")
    if bam is not None and bam_map is not None:
        raise typer.BadParameter("Provide only one of --bam or --bam-map (not both).")

    if bam is not None:
        yield ("bam", Path(bam))
        return

    # bam_map path
    bm = pd.read_csv(Path(bam_map), sep="\t", dtype=str)
    if not {"bam_id", "bam_path"}.issubset(bm.columns):
        raise ValueError(f"{bam_map} must have columns: bam_id, bam_path")
    for row in bm.itertuples(index=False):
        yield (str(row.bam_id), Path(row.bam_path))


@app.command("clean-bams")
def clean_bams_cmd(
    reads_to_drop: Path = typer.Option(
        ...,
        "--reads-to-drop",
        exists=True,
        help="reads_to_drop.tsv.gz from ambientmapper decontam (must contain 'read_id').",
    ),
    bam: Optional[Path] = typer.Option(
        None,
        "--bam",
        exists=True,
        help="Single BAM to clean.",
    ),
    bam_map: Optional[Path] = typer.Option(
        None,
        "--bam-map",
        exists=True,
        help="TSV listing BAMs to clean (columns: bam_id, bam_path).",
    ),
    out_dir: Path = typer.Option(
        ...,
        "--out-dir",
        help="Output directory for cleaned BAMs.",
    ),
    out_suffix: str = typer.Option(
        ".clean.bam",
        "--out-suffix",
        help="Suffix appended to each output BAM filename.",
    ),
    index: bool = typer.Option(
        True,
        "--index/--no-index",
        help="Create .bai index for cleaned BAMs.",
    ),
    log_every: int = typer.Option(
        2_000_000,
        "--log-every",
        help="Progress log interval (reads). Use 0 to disable.",
    ),
):
    """
    Remove reads from BAM(s) based solely on read IDs (QNAMEs).

    This is reference-agnostic and guarantees the same reads are removed from any BAM derived
    from the same FASTQs (assuming QNAMEs are preserved).
    """
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    bad_reads = _load_read_ids(reads_to_drop)
    typer.echo(f"[clean-bams] Loaded {len(bad_reads):,} read_ids to drop from {reads_to_drop}")

    for bam_id, bam_path in _iter_bams(bam, bam_map):
        if not bam_path.exists():
            typer.echo(f"[clean-bams] WARNING: BAM not found: {bam_path}", err=True)
            continue

        typer.echo(f"[clean-bams] Cleaning bam_id={bam_id} path={bam_path}")

        in_bam = pysam.AlignmentFile(str(bam_path), "rb")
        out_path = out_dir / f"{bam_path.stem}{out_suffix}"
        out_bam = pysam.AlignmentFile(str(out_path), "wb", template=in_bam)

        n_in = 0
        n_out = 0
        n_drop = 0

        for i, read in enumerate(in_bam, start=1):
            n_in += 1
            if log_every and i % log_every == 0:
                typer.echo(f"[clean-bams]  processed {i/1e6:.1f}M reads ...")

            if read.query_name in bad_reads:
                n_drop += 1
                continue

            out_bam.write(read)
            n_out += 1

        in_bam.close()
        out_bam.close()

        if index:
            pysam.index(str(out_path))

        typer.echo(
            f"[clean-bams] {bam_path} → {out_path} | "
            f"in={n_in:,}, kept={n_out:,}, dropped={n_drop:,}"
        )
