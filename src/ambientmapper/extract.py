# src/ambientmapper/extract.py
from pathlib import Path
import csv, pysam
import pyarrow as pa
from .normalization import canonicalize_bc_seq_sample_force


# Locked schema for filtered/extracted QC parquet files. Single source of truth
# shared by extract (PR3 writer), filter (PR2 writer), and the migration tool
# `_convert_to_parquet` (legacy TSV path; still casts to FLOAT for historical
# back-compat, but new writers conform to this typed schema).
#
# Column naming notes:
#   - `as_` is the BAM `AS` alignment-score tag, renamed to avoid the SQL
#     reserved word in DuckDB queries.
#   - `frag_loc` is `chrom:pos:mate_pos`; empty string when the source has
#     no chrom/mate.
QC_PARQUET_SCHEMA: pa.Schema = pa.schema([
    pa.field("Read",     pa.string(),  nullable=False),
    pa.field("BC",       pa.string(),  nullable=False),
    pa.field("MAPQ",     pa.int16(),   nullable=False),
    pa.field("as_",      pa.int32(),   nullable=True),   # BAM AS tag is optional
    pa.field("NM",       pa.int32(),   nullable=True),   # BAM NM tag is optional
    pa.field("XAcount",  pa.int16(),   nullable=False),
    pa.field("frag_loc", pa.string(),  nullable=False),
])


def bam_to_qc(bam_path: Path, out_path: Path, sample_name: str | None = None):
    """
    Write per-read QC with normalized BC = '<seq>-<sample_name>'.
    """
    with pysam.AlignmentFile(str(bam_path), "rb") as bam, open(out_path, "w", newline="") as out:
        w = csv.writer(out, delimiter="\t")
        for aln in bam.fetch(until_eof=True):
            if aln.is_unmapped or aln.is_secondary or aln.is_supplementary:
                continue
            read = aln.query_name
            mapq = int(aln.mapping_quality)
            ascore = aln.get_tag("AS") if aln.has_tag("AS") else ""
            nm = aln.get_tag("NM") if aln.has_tag("NM") else ""
            bc_raw = ""
            if aln.has_tag("CB"): bc_raw = aln.get_tag("CB")
            elif aln.has_tag("BC"): bc_raw = aln.get_tag("BC")
            # NEW: normalize to seq-sample
            bc = canonicalize_bc_seq_sample_force(bc_raw, sample_name)

            xa_count = 0
            if aln.has_tag("XA"):
                xa = aln.get_tag("XA")
                xa_count = xa.count(";") if isinstance(xa, str) else 0

            # Fragment location: chr:R1_start:mate_start
            chrom = aln.reference_name or "*"
            pos = aln.reference_start
            mate_pos = (
                aln.next_reference_start
                if aln.is_paired and not aln.mate_is_unmapped
                else -1
            )
            frag_loc = f"{chrom}:{pos}:{mate_pos}"

            w.writerow([read, bc, mapq, ascore, nm, xa_count, frag_loc])
