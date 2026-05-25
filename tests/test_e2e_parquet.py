"""End-to-end equivalence test for the parquet-native pipeline (PR3).

Verifies that extract → filter produces equivalent filtered Parquet output
regardless of whether extract emits parquet directly (the new default) or
emits the legacy TSV and filter reads from it. This is the load-bearing
guarantee that the format dispatch is semantically identical.

Chunks and assign are out of scope here (they're covered by
test_chunks_parquet.py and the assign codebase's existing tests; they
read filtered_QCFiles/ via a precedence rule that's been exercised by
PR2's tests).
"""
from __future__ import annotations

import pathlib

import pyarrow.parquet as pq
import pysam
import pytest

from ambientmapper.extract import bam_to_qc
from ambientmapper.filtering import filter_qc_file


# Reuse the BAM-building helper pattern from test_extract_parquet.
def _make_bam(path: pathlib.Path, reads: list[dict]) -> pathlib.Path:
    header = {"HD": {"VN": "1.6"}, "SQ": [{"LN": 1_000_000, "SN": "chr1"}]}
    with pysam.AlignmentFile(str(path), "wb", header=header) as bam:
        for spec in reads:
            a = pysam.AlignedSegment(bam.header)
            a.query_name = spec["read"]
            a.reference_id = 0
            a.reference_start = spec.get("pos", 100)
            a.mapping_quality = int(spec.get("mapq", 30))
            a.cigar = ((0, 50),)
            a.query_sequence = "A" * 50
            a.query_qualities = pysam.qualitystring_to_array("?" * 50)
            flag = 0
            if spec.get("mate_pos") is not None:
                flag |= 1
                a.next_reference_id = 0
                a.next_reference_start = int(spec["mate_pos"])
            a.flag = flag
            a.set_tag("CB", spec["bc"])
            if "as_" in spec:
                a.set_tag("AS", int(spec["as_"]))
            if "nm" in spec:
                a.set_tag("NM", int(spec["nm"]))
            bam.write(a)
    return path


def test_extract_parquet_then_filter_equals_extract_txt_then_filter(tmp_path: pathlib.Path):
    """End-to-end equivalence: parquet path == TSV path through filter.

    Both routes should produce bit-equivalent filtered Parquet output for
    the same input BAM and same filter parameters.
    """
    # Synthetic BAM with 5 reads spanning 2 BCs (one above min_freq, one below)
    reads = [
        {"read": "r1", "pos": 100, "mate_pos": 200, "mapq": 30, "bc": "AAAAAA", "as_": 100, "nm": 0},
        {"read": "r2", "pos": 110, "mate_pos": 210, "mapq": 28, "bc": "AAAAAA", "as_": 95,  "nm": 1},
        {"read": "r3", "pos": 120, "mate_pos": 220, "mapq": 30, "bc": "AAAAAA", "as_": 99,  "nm": 0},
        {"read": "r4", "pos": 130, "mate_pos": 230, "mapq": 30, "bc": "BBBBBB", "as_": 100, "nm": 0},
        # BBBBBB is a singleton; min_freq=2 should drop it.
    ]

    route_a = tmp_path / "route_a"; route_a.mkdir()
    route_b = tmp_path / "route_b"; route_b.mkdir()

    bam_a = _make_bam(route_a / "in.bam", reads)
    bam_b = _make_bam(route_b / "in.bam", reads)

    # Route A: parquet from extract → parquet from filter
    qc_a = route_a / "qc.parquet"
    out_a = route_a / "filtered.parquet"
    bam_to_qc(bam_a, qc_a, sample_name="S1")
    filter_qc_file(qc_a, out_a, min_freq=2, sample_name=None)

    # Route B: legacy TSV from extract → parquet from filter
    qc_b = route_b / "qc.txt"
    out_b = route_b / "filtered.parquet"
    bam_to_qc(bam_b, qc_b, sample_name="S1")
    filter_qc_file(qc_b, out_b, min_freq=2, sample_name=None)

    tbl_a = pq.read_table(out_a)
    tbl_b = pq.read_table(out_b)

    # Schema must be identical (the locked QC_PARQUET_SCHEMA).
    assert tbl_a.schema.equals(tbl_b.schema), (tbl_a.schema, tbl_b.schema)
    # Row count and content equality.
    assert tbl_a.num_rows == tbl_b.num_rows == 3  # 3 AAAAAA-S1 rows; BBBBBB dropped
    assert tbl_a.to_pydict() == tbl_b.to_pydict()

    # Verify the filter dropped the singleton BC.
    assert set(tbl_a.column("BC").to_pylist()) == {"AAAAAA-S1"}
