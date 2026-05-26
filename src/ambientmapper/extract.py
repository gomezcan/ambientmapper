# src/ambientmapper/extract.py
"""
Per-genome QCMapping extraction from BAM.

Output format is dispatched by the ``out_path`` suffix:

* ``.parquet`` (default in 0.2): typed, snappy-compressed Parquet
  conforming to :data:`QC_PARQUET_SCHEMA`. Streamed in BAM-order with a
  small in-memory buffer flushed as a Parquet row group every
  ``row_group_size`` rows.
* ``.txt`` (legacy, pre-0.2): header-less tab-delimited TSV, preserved
  verbatim for back-compat with ``extract --format txt``.

The new parquet path avoids the multi-GB CSV intermediate that the
on-the-fly ``_convert_to_parquet`` step otherwise has to parse inside
``assign``. See ``CHANGELOG.md`` and ``README.md`` for the migration
story.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import pysam
import pyarrow as pa
import pyarrow.parquet as pq

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
# All fields nullable: DuckDB's `COPY TO PARQUET` (used by the new filter)
# always emits nullable columns, and pyarrow's ParquetWriter defaults also
# lean nullable. Marking `Read/BC/MAPQ/XAcount/frag_loc` as non-null at the
# schema level created cross-writer friction (DuckDB-emitted parquet failed
# `Table.schema.equals(QC_PARQUET_SCHEMA)`) with no real safety benefit —
# the writers in this package already guarantee these columns are populated.
QC_PARQUET_SCHEMA: pa.Schema = pa.schema([
    ("Read",     pa.string()),
    ("BC",       pa.string()),
    ("MAPQ",     pa.int16()),
    ("as_",      pa.int32()),    # BAM AS tag is optional
    ("NM",       pa.int32()),    # BAM NM tag is optional
    ("XAcount",  pa.int16()),
    ("frag_loc", pa.string()),
])


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def bam_to_qc(
    bam_path: Path,
    out_path: Path,
    sample_name: str | None = None,
    *,
    row_group_size: int = 100_000,
) -> int:
    """Write per-read QC with normalized BC = '<seq>-<sample_name>'.

    Output format dispatched by ``out_path.suffix``:

    - ``.parquet`` (default in 0.2): typed parquet via
      :func:`_bam_to_qc_parquet`. Conforms to :data:`QC_PARQUET_SCHEMA`.
    - ``.txt`` (legacy): header-less TSV via :func:`_bam_to_qc_tsv`.

    Returns the number of data rows written (alignments surviving the
    unmapped/secondary/supplementary filter).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".txt":
        return _bam_to_qc_tsv(bam_path, out_path, sample_name)
    return _bam_to_qc_parquet(bam_path, out_path, sample_name, row_group_size)


# ----------------------------------------------------------------------------
# Parquet writer (new in 0.2)
# ----------------------------------------------------------------------------

def _bam_to_qc_parquet(
    bam_path: Path,
    out_path: Path,
    sample_name: str | None,
    row_group_size: int,
) -> int:
    """Stream a BAM to Parquet conforming to :data:`QC_PARQUET_SCHEMA`.

    Memory: ~17-25 MB resident per process at flush time. The BAM iteration
    is one pass; the in-memory buffer holds at most ``row_group_size``
    pending rows across the 7 columns.

    Atomic write: data is staged to ``out_path.with_suffix('.parquet.tmp')``
    and ``os.replace`` is used on success. Partial writes from a crash leave
    the ``.tmp`` file behind (next run overwrites it).
    """
    bam_path = Path(bam_path)
    out_path = Path(out_path)
    tmp_out = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_out.exists():
        tmp_out.unlink()

    # Per-column accumulators (cleared after each row-group flush)
    col_reads:    list[str]          = []
    col_bcs:      list[str]          = []
    col_mapqs:    list[int]          = []
    col_as:       list[int | None]   = []
    col_nms:      list[int | None]   = []
    col_xacounts: list[int]          = []
    col_frags:    list[str]          = []

    def _build_batch() -> pa.RecordBatch:
        # Pass schema= (not names=) so the resulting RecordBatch inherits the
        # nullability flags from QC_PARQUET_SCHEMA. Without this, pa.array()
        # defaults to nullable=True for every column, which the ParquetWriter
        # (opened with QC_PARQUET_SCHEMA) refuses with:
        #   ValueError: Table schema does not match schema used to create file
        return pa.RecordBatch.from_arrays(
            [
                pa.array(col_reads,    type=pa.string()),
                pa.array(col_bcs,      type=pa.string()),
                pa.array(col_mapqs,    type=pa.int16()),
                pa.array(col_as,       type=pa.int32()),
                pa.array(col_nms,      type=pa.int32()),
                pa.array(col_xacounts, type=pa.int16()),
                pa.array(col_frags,    type=pa.string()),
            ],
            schema=QC_PARQUET_SCHEMA,
        )

    def _flush(writer: pq.ParquetWriter) -> None:
        if not col_reads:
            return
        writer.write_batch(_build_batch())
        col_reads.clear(); col_bcs.clear(); col_mapqs.clear()
        col_as.clear();    col_nms.clear(); col_xacounts.clear(); col_frags.clear()

    n_rows = 0
    with pysam.AlignmentFile(str(bam_path), "rb") as bam, \
         pq.ParquetWriter(
             tmp_out, QC_PARQUET_SCHEMA, compression="snappy",
         ) as writer:
        for aln in bam.fetch(until_eof=True):
            if aln.is_unmapped or aln.is_secondary or aln.is_supplementary:
                continue

            read = aln.query_name
            mapq = int(aln.mapping_quality)
            ascore = int(aln.get_tag("AS")) if aln.has_tag("AS") else None
            nm = int(aln.get_tag("NM")) if aln.has_tag("NM") else None

            bc_raw = ""
            if aln.has_tag("CB"):
                bc_raw = aln.get_tag("CB")
            elif aln.has_tag("BC"):
                bc_raw = aln.get_tag("BC")
            bc = canonicalize_bc_seq_sample_force(bc_raw, sample_name)

            xa_count = 0
            if aln.has_tag("XA"):
                xa = aln.get_tag("XA")
                xa_count = xa.count(";") if isinstance(xa, str) else 0

            chrom = aln.reference_name or "*"
            pos = aln.reference_start
            mate_pos = (
                aln.next_reference_start
                if aln.is_paired and not aln.mate_is_unmapped
                else -1
            )
            frag_loc = f"{chrom}:{pos}:{mate_pos}"

            col_reads.append(read)
            col_bcs.append(bc)
            col_mapqs.append(mapq)
            col_as.append(ascore)
            col_nms.append(nm)
            col_xacounts.append(xa_count)
            col_frags.append(frag_loc)
            n_rows += 1

            if len(col_reads) >= row_group_size:
                _flush(writer)

        # Final flush of any partial row group.
        _flush(writer)

    # Atomic move into place. ParquetWriter closing on an empty BAM still
    # produces a valid empty parquet with the embedded schema.
    os.replace(tmp_out, out_path)
    return n_rows


# ----------------------------------------------------------------------------
# Legacy TSV writer (preserved verbatim; called when out_path.suffix == ".txt")
# ----------------------------------------------------------------------------

def _bam_to_qc_tsv(
    bam_path: Path,
    out_path: Path,
    sample_name: str | None = None,
) -> int:
    """Header-less TSV writer matching the pre-0.2 extract output.

    Columns (positional, no header):
      Read, BC, MAPQ, AS, NM, XAcount, frag_loc

    Returns the number of data rows written.
    """
    n_rows = 0
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
            n_rows += 1
    return n_rows
