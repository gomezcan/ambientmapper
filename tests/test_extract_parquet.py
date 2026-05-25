"""Tests for parquet-native extract (PR3)."""
from __future__ import annotations

import csv
import pathlib

import pyarrow.parquet as pq
import pysam
import pytest

from ambientmapper.extract import bam_to_qc, QC_PARQUET_SCHEMA


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_bam(path: pathlib.Path, reads: list[dict]) -> pathlib.Path:
    """Write a tiny synthetic BAM at ``path``.

    ``reads`` items are dicts with optional keys:
      read (str), pos (int, default 100), mate_pos (int|None, None=unpaired),
      mapq (int), bc (str), as_ (int), nm (int), xa (str),
      flag_override (int): force a specific flag (for unmapped/secondary/supp tests).
    """
    header = {"HD": {"VN": "1.6"}, "SQ": [{"LN": 1_000_000, "SN": "chr1"}]}
    with pysam.AlignmentFile(str(path), "wb", header=header) as bam:
        for spec in reads:
            a = pysam.AlignedSegment(bam.header)
            a.query_name = spec.get("read", "r")
            a.reference_id = 0
            a.reference_start = spec.get("pos", 100)
            a.mapping_quality = int(spec.get("mapq", 30))
            a.cigar = ((0, 50),)  # 50M
            a.query_sequence = "A" * 50
            a.query_qualities = pysam.qualitystring_to_array("?" * 50)
            if "flag_override" in spec:
                a.flag = int(spec["flag_override"])
            else:
                # Default: primary, mapped. Pair flag set if mate_pos provided.
                flag = 0
                if spec.get("mate_pos") is not None:
                    flag |= 1   # paired
                    a.next_reference_id = 0
                    a.next_reference_start = int(spec["mate_pos"])
                a.flag = flag
            if "bc" in spec:
                a.set_tag("CB", spec["bc"])
            if "as_" in spec:
                a.set_tag("AS", int(spec["as_"]))
            if "nm" in spec:
                a.set_tag("NM", int(spec["nm"]))
            if "xa" in spec:
                a.set_tag("XA", spec["xa"])
            bam.write(a)
    return path


# ---------------------------------------------------------------------------
# Parquet writer
# ---------------------------------------------------------------------------

def test_extract_writes_parquet_with_schema(tmp_path: pathlib.Path):
    bam = _make_bam(tmp_path / "in.bam", [
        {"read": "r1", "pos": 100, "mate_pos": 200, "mapq": 30,
         "bc": "AAAAAA", "as_": 100, "nm": 0},
        {"read": "r2", "pos": 300, "mate_pos": 400, "mapq": 28,
         "bc": "BBBBBB", "as_": 95, "nm": 1, "xa": "alt1;alt2;"},
    ])
    out = tmp_path / "qc.parquet"
    n = bam_to_qc(bam, out, sample_name="X")
    assert n == 2

    tbl = pq.read_table(out)
    assert tbl.schema.equals(QC_PARQUET_SCHEMA), (tbl.schema, QC_PARQUET_SCHEMA)

    rows = tbl.to_pydict()
    assert rows["Read"] == ["r1", "r2"]
    # BC normalized to <seq>-<sample> by canonicalize_bc_seq_sample_force
    assert rows["BC"] == ["AAAAAA-X", "BBBBBB-X"]
    assert rows["MAPQ"] == [30, 28]
    assert rows["as_"] == [100, 95]
    assert rows["NM"] == [0, 1]
    # XA count = number of ';' in the tag string
    assert rows["XAcount"] == [0, 2]
    assert rows["frag_loc"] == ["chr1:100:200", "chr1:300:400"]


def test_extract_empty_bam_writes_valid_empty_parquet(tmp_path: pathlib.Path):
    bam = _make_bam(tmp_path / "empty.bam", [])
    out = tmp_path / "qc.parquet"
    n = bam_to_qc(bam, out, sample_name="X")
    assert n == 0
    assert out.exists()
    tbl = pq.read_table(out)
    assert tbl.num_rows == 0
    assert tbl.schema.equals(QC_PARQUET_SCHEMA)


def test_extract_missing_optional_tags_writes_nulls(tmp_path: pathlib.Path):
    """AS and NM are nullable; missing tags → None in parquet (not 0)."""
    bam = _make_bam(tmp_path / "in.bam", [
        {"read": "r1", "pos": 100, "mate_pos": 200, "mapq": 30, "bc": "AAA"},
        # No AS, NM tags; XA absent → xa_count=0; CB present.
    ])
    out = tmp_path / "qc.parquet"
    bam_to_qc(bam, out, sample_name="X")
    rows = pq.read_table(out).to_pydict()
    assert rows["as_"] == [None]
    assert rows["NM"] == [None]
    assert rows["XAcount"] == [0]


def test_extract_filters_unmapped_secondary_supplementary(tmp_path: pathlib.Path):
    """Only primary mapped alignments are emitted; unmapped/secondary/supp dropped."""
    UNMAPPED, SECONDARY, SUPPLEMENTARY = 4, 256, 2048
    bam = _make_bam(tmp_path / "in.bam", [
        {"read": "good", "mapq": 30, "bc": "AAA", "as_": 100, "nm": 0},
        {"read": "unmapped", "mapq": 0, "bc": "AAA", "as_": 0, "nm": 0,
         "flag_override": UNMAPPED},
        {"read": "secondary", "mapq": 30, "bc": "AAA", "as_": 100, "nm": 0,
         "flag_override": SECONDARY},
        {"read": "supplementary", "mapq": 30, "bc": "AAA", "as_": 100, "nm": 0,
         "flag_override": SUPPLEMENTARY},
    ])
    out = tmp_path / "qc.parquet"
    n = bam_to_qc(bam, out, sample_name="X")
    assert n == 1
    rows = pq.read_table(out).to_pydict()
    assert rows["Read"] == ["good"]


def test_extract_row_group_size_flush(tmp_path: pathlib.Path):
    """With row_group_size=2 and 5 reads, the writer should emit 3 row groups
    (2 + 2 + 1). Verifies the buffered flush logic without depending on
    pyarrow's internal default."""
    reads = [
        {"read": f"r{i}", "pos": 100 + i, "mate_pos": 200 + i, "mapq": 30,
         "bc": "AAA", "as_": 100, "nm": 0}
        for i in range(5)
    ]
    bam = _make_bam(tmp_path / "in.bam", reads)
    out = tmp_path / "qc.parquet"
    n = bam_to_qc(bam, out, sample_name="X", row_group_size=2)
    assert n == 5

    pf = pq.ParquetFile(out)
    assert pf.num_row_groups == 3
    # row group sizes: 2, 2, 1
    assert [pf.metadata.row_group(i).num_rows for i in range(3)] == [2, 2, 1]


# ---------------------------------------------------------------------------
# Legacy TSV writer (preserved verbatim; --format txt escape hatch)
# ---------------------------------------------------------------------------

def test_extract_legacy_tsv_path(tmp_path: pathlib.Path):
    """`.txt` output dispatches to the pre-0.2 header-less TSV writer."""
    bam = _make_bam(tmp_path / "in.bam", [
        {"read": "r1", "pos": 100, "mate_pos": 200, "mapq": 30,
         "bc": "AAA", "as_": 100, "nm": 0},
        {"read": "r2", "pos": 300, "mate_pos": 400, "mapq": 28,
         "bc": "BBB", "as_": 95, "nm": 1, "xa": "alt1;alt2;"},
    ])
    out = tmp_path / "qc.txt"
    n = bam_to_qc(bam, out, sample_name="X")
    assert n == 2

    with out.open() as f:
        rows = list(csv.reader(f, delimiter="\t"))
    assert len(rows) == 2  # no header (pre-0.2 format)
    assert rows[0][0] == "r1"
    assert rows[0][1] == "AAA-X"
    assert rows[0][2] == "30"
    assert rows[0][3] == "100"
    assert rows[0][4] == "0"
    assert rows[0][5] == "0"
    assert rows[0][6] == "chr1:100:200"


def test_extract_dispatch_extension_matters(tmp_path: pathlib.Path):
    """The function dispatches on out_path.suffix, regardless of caller intent."""
    bam = _make_bam(tmp_path / "in.bam", [
        {"read": "r1", "pos": 100, "mapq": 30, "bc": "AAA", "as_": 100, "nm": 0},
    ])
    # parquet suffix → parquet
    out_pq = tmp_path / "x.parquet"
    bam_to_qc(bam, out_pq, sample_name="X")
    assert pq.read_table(out_pq).num_rows == 1
    # txt suffix → text
    out_txt = tmp_path / "x.txt"
    bam_to_qc(bam, out_txt, sample_name="X")
    assert out_txt.read_text().startswith("r1\tAAA-X\t30")
