"""Tests for QC_PARQUET_SCHEMA constant and _filtered_io helpers (PR1)."""
from __future__ import annotations

import pathlib

import pyarrow as pa
import pytest

from ambientmapper.extract import QC_PARQUET_SCHEMA
from ambientmapper._filtered_io import (
    genome_from_filename,
    discover_filtered_files,
)


# --- QC_PARQUET_SCHEMA -------------------------------------------------------


def test_qc_parquet_schema_is_pyarrow_schema():
    assert isinstance(QC_PARQUET_SCHEMA, pa.Schema)


def test_qc_parquet_schema_columns_in_order():
    expected = ["Read", "BC", "MAPQ", "as_", "NM", "XAcount", "frag_loc"]
    assert QC_PARQUET_SCHEMA.names == expected


def test_qc_parquet_schema_types():
    # Column types match the locked plan. Catches drift between extract,
    # filter, and the migration tool's outputs.
    expected_types = {
        "Read":     pa.string(),
        "BC":       pa.string(),
        "MAPQ":     pa.int16(),
        "as_":      pa.int32(),
        "NM":       pa.int32(),
        "XAcount":  pa.int16(),
        "frag_loc": pa.string(),
    }
    for name, want in expected_types.items():
        got = QC_PARQUET_SCHEMA.field(name).type
        assert got.equals(want), f"{name}: got {got}, want {want}"


def test_qc_parquet_schema_all_fields_nullable():
    # All fields are nullable in 0.2 — DuckDB's COPY TO PARQUET emits nullable
    # columns and pyarrow's ParquetWriter defaults also lean nullable, so
    # marking required fields as non-null at the schema level created
    # cross-writer friction with no safety benefit (writers already control
    # which columns are populated).
    for field in QC_PARQUET_SCHEMA:
        assert field.nullable, f"{field.name} should be nullable (0.2 contract)"


# --- genome_from_filename ----------------------------------------------------


@pytest.mark.parametrize(
    "filename,want",
    [
        ("filtered_B97_QCMapping.parquet", "B97"),
        ("filtered_B97_QCMapping.txt", "B97"),
        ("filtered_CML103_QCMapping.parquet", "CML103"),
        ("filtered_CML322_QCMapping.txt", "CML322"),
        # Genome names with underscores must round-trip cleanly.
        ("filtered_Zm_NAM_v5_QCMapping.parquet", "Zm_NAM_v5"),
    ],
)
def test_genome_from_filename_known_suffixes(filename, want):
    assert genome_from_filename(pathlib.Path(filename)) == want


def test_genome_from_filename_unknown_pattern_returns_basename():
    p = pathlib.Path("/some/dir/unrelated.txt")
    assert genome_from_filename(p) == "unrelated.txt"


# --- discover_filtered_files -------------------------------------------------


def _touch(p: pathlib.Path) -> pathlib.Path:
    p.write_bytes(b"")
    return p


def test_discover_raises_on_missing_dir(tmp_path: pathlib.Path):
    with pytest.raises(FileNotFoundError):
        discover_filtered_files(tmp_path / "does_not_exist")


def test_discover_raises_on_empty_dir(tmp_path: pathlib.Path):
    d = tmp_path / "filtered_QCFiles"
    d.mkdir()
    with pytest.raises(FileNotFoundError):
        discover_filtered_files(d)


def test_discover_returns_tsv_when_only_tsv(tmp_path: pathlib.Path):
    d = tmp_path / "filtered_QCFiles"; d.mkdir()
    _touch(d / "filtered_B97_QCMapping.txt")
    _touch(d / "filtered_CML103_QCMapping.txt")
    out = discover_filtered_files(d)
    assert [p.name for p in out] == [
        "filtered_B97_QCMapping.txt",
        "filtered_CML103_QCMapping.txt",
    ]


def test_discover_prefers_parquet_when_complete(tmp_path: pathlib.Path):
    # Both parquet and TSV present for the same 2 genomes; parquet wins.
    d = tmp_path / "filtered_QCFiles"; d.mkdir()
    for g in ("B97", "CML103"):
        _touch(d / f"filtered_{g}_QCMapping.txt")
        _touch(d / f"filtered_{g}_QCMapping.parquet")
    out = discover_filtered_files(d)
    assert [p.suffix for p in out] == [".parquet", ".parquet"]


def test_discover_falls_back_to_tsv_when_parquet_incomplete(tmp_path: pathlib.Path):
    # parquet covers only B97; CML103 has only TSV — fall back to all-TSV.
    d = tmp_path / "filtered_QCFiles"; d.mkdir()
    _touch(d / "filtered_B97_QCMapping.txt")
    _touch(d / "filtered_B97_QCMapping.parquet")
    _touch(d / "filtered_CML103_QCMapping.txt")
    out = discover_filtered_files(d)
    assert [p.suffix for p in out] == [".txt", ".txt"]


def test_discover_returns_parquet_when_only_parquet(tmp_path: pathlib.Path):
    # Post-PR3 world: no TSVs at all. Parquet-only directory should resolve.
    d = tmp_path / "filtered_QCFiles"; d.mkdir()
    _touch(d / "filtered_B97_QCMapping.parquet")
    _touch(d / "filtered_CML103_QCMapping.parquet")
    out = discover_filtered_files(d)
    assert [p.name for p in out] == [
        "filtered_B97_QCMapping.parquet",
        "filtered_CML103_QCMapping.parquet",
    ]
