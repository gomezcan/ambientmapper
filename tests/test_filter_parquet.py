"""Tests for the parquet-native filter step (PR2).

These exercise `filter_qc_file` with both the new parquet output path
(default in 0.2) and the preserved legacy TSV path.
"""
from __future__ import annotations

import pathlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ambientmapper.extract import QC_PARQUET_SCHEMA
from ambientmapper.filtering import filter_qc_file


# ---------------------------------------------------------------------------
# Fixtures: small synthetic inputs
# ---------------------------------------------------------------------------

def _make_tsv_input(tmp_path: pathlib.Path, rows: list[tuple]) -> pathlib.Path:
    """Write a header-less TSV matching the legacy extract output schema:
    Read, BC, MAPQ, AS, NM, XAcount, frag_loc.
    """
    p = tmp_path / "in.txt"
    with p.open("w") as f:
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")
    return p


def _make_parquet_input(tmp_path: pathlib.Path, rows: list[tuple]) -> pathlib.Path:
    """Write a parquet input matching QC_PARQUET_SCHEMA. ``rows`` are
    7-tuples in column order (Read, BC, MAPQ, as_, NM, XAcount, frag_loc).
    """
    p = tmp_path / "in.parquet"
    arrays = {name: [] for name in QC_PARQUET_SCHEMA.names}
    for r in rows:
        for name, value in zip(QC_PARQUET_SCHEMA.names, r):
            arrays[name].append(value)
    tbl = pa.table(arrays, schema=QC_PARQUET_SCHEMA)
    pq.write_table(tbl, p, compression="snappy")
    return p


# ---------------------------------------------------------------------------
# Parquet-out (the new default)
# ---------------------------------------------------------------------------

def test_parquet_in_parquet_out_sorts_by_bc(tmp_path: pathlib.Path):
    # BC counts: AAA=3, BBB=2, CCC=1. min_freq=2 keeps AAA + BBB only.
    rows = [
        ("r1", "CCC", 30, 100, 0, 0, "chr1:1:2"),
        ("r2", "AAA", 30, 100, 0, 0, "chr1:3:4"),
        ("r3", "BBB", 30, 100, 0, 0, "chr1:5:6"),
        ("r4", "AAA", 28, 95, 1, 1, "chr1:7:8"),
        ("r5", "BBB", 31, 102, 0, 0, "chr1:9:10"),
        ("r6", "AAA", 30, 99, 0, 0, "chr1:11:12"),
    ]
    in_path = _make_parquet_input(tmp_path, rows)
    out_path = tmp_path / "filtered.parquet"

    n = filter_qc_file(in_path, out_path, min_freq=2, sample_name=None)
    assert n == 5  # 3 AAA + 2 BBB

    tbl = pq.read_table(out_path)
    assert tbl.schema.equals(QC_PARQUET_SCHEMA), (tbl.schema, QC_PARQUET_SCHEMA)

    bcs = tbl.column("BC").to_pylist()
    assert bcs == sorted(bcs), "rows must be sorted by BC"
    assert set(bcs) == {"AAA", "BBB"}
    assert bcs.count("AAA") == 3
    assert bcs.count("BBB") == 2


def test_txt_in_parquet_out(tmp_path: pathlib.Path):
    # Legacy header-less TSV input (post-extract format), parquet output.
    rows = [
        ("r1", "AAA", 30, 100, 0, 0, "chr1:1:2"),
        ("r2", "AAA", 30, 100, 0, 0, "chr1:3:4"),
        ("r3", "BBB", 30, 100, 0, 0, "chr1:5:6"),  # singleton
    ]
    in_path = _make_tsv_input(tmp_path, rows)
    out_path = tmp_path / "filtered.parquet"

    n = filter_qc_file(in_path, out_path, min_freq=2, sample_name=None)
    assert n == 2

    tbl = pq.read_table(out_path)
    assert tbl.schema.equals(QC_PARQUET_SCHEMA)
    assert tbl.column("BC").to_pylist() == ["AAA", "AAA"]


def test_empty_after_cutoff_writes_valid_empty_parquet(tmp_path: pathlib.Path):
    # Every BC is a singleton; min_freq=2 → no rows pass.
    rows = [
        ("r1", "AAA", 30, 100, 0, 0, "chr1:1:2"),
        ("r2", "BBB", 30, 100, 0, 0, "chr1:3:4"),
        ("r3", "CCC", 30, 100, 0, 0, "chr1:5:6"),
    ]
    in_path = _make_parquet_input(tmp_path, rows)
    out_path = tmp_path / "filtered.parquet"

    n = filter_qc_file(in_path, out_path, min_freq=2, sample_name=None)
    assert n == 0

    tbl = pq.read_table(out_path)
    assert tbl.num_rows == 0
    assert tbl.schema.equals(QC_PARQUET_SCHEMA)


def test_normalize_bc_with_sample_name(tmp_path: pathlib.Path):
    # BCs already include some sample suffix; normalization rewrites to
    # '<seq>-<sample_name>'. After normalization, all rows share BC and
    # min_freq=2 keeps them all.
    rows = [
        ("r1", "AAAAAA-other_sample", 30, 100, 0, 0, "chr1:1:2"),
        ("r2", "AAAAAA-other_sample", 30, 100, 0, 0, "chr1:3:4"),
        ("r3", "AAAAAA",               30, 100, 0, 0, "chr1:5:6"),
    ]
    in_path = _make_tsv_input(tmp_path, rows)
    out_path = tmp_path / "filtered.parquet"

    n = filter_qc_file(in_path, out_path, min_freq=2, sample_name="X")
    assert n == 3

    bcs = pq.read_table(out_path).column("BC").to_pylist()
    assert all(bc == "AAAAAA-X" for bc in bcs)


def test_input_extension_dispatch(tmp_path: pathlib.Path):
    """Parquet and TSV inputs with the same content produce equivalent output."""
    rows = [
        ("r1", "AAA", 30, 100, 0, 0, "chr1:1:2"),
        ("r2", "AAA", 28, 95, 1, 1, "chr1:3:4"),
        ("r3", "BBB", 30, 102, 0, 0, "chr1:5:6"),
        ("r4", "BBB", 30, 100, 0, 0, "chr1:7:8"),
    ]
    # Use separate subdirs so the helpers' canonical "in.parquet"/"in.txt"
    # filenames don't conflict in shared tmp_path.
    pq_sub = tmp_path / "pq"; pq_sub.mkdir()
    tsv_sub = tmp_path / "tsv"; tsv_sub.mkdir()
    pq_in = _make_parquet_input(pq_sub, rows)
    tsv_in = _make_tsv_input(tsv_sub, rows)

    out_pq = tmp_path / "from_parquet.parquet"
    out_txt = tmp_path / "from_txt.parquet"
    filter_qc_file(pq_in, out_pq, min_freq=2)
    filter_qc_file(tsv_in, out_txt, min_freq=2)

    a = pq.read_table(out_pq).to_pydict()
    b = pq.read_table(out_txt).to_pydict()
    assert a == b


# ---------------------------------------------------------------------------
# TSV-out path (legacy, --format txt escape hatch)
# ---------------------------------------------------------------------------

def test_txt_out_legacy_path_preserved(tmp_path: pathlib.Path):
    """The TSV-out dispatch must continue to write the pre-0.2 TSV format
    (header + tab-delimited rows, no parquet involved).
    """
    rows = [
        ("r1", "AAA", 30, 100, 0, 0, "chr1:1:2"),
        ("r2", "AAA", 28, 95, 1, 1, "chr1:3:4"),
        ("r3", "BBB", 30, 102, 0, 0, "chr1:5:6"),  # singleton
    ]
    in_path = _make_tsv_input(tmp_path, rows)
    out_path = tmp_path / "filtered.txt"

    n = filter_qc_file(in_path, out_path, min_freq=2, sample_name=None)
    assert n == 2

    lines = out_path.read_text().splitlines()
    assert lines[0] == "Read\tBC\tMAPQ\tAS\tNM\tXAcount\tfrag_loc"
    assert len(lines) == 3  # header + 2 data rows
    # All data rows have BC = AAA
    for ln in lines[1:]:
        parts = ln.split("\t")
        assert parts[1] == "AAA"
