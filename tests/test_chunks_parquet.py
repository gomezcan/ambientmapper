"""Tests for the parquet-aware chunks step (PR2)."""
from __future__ import annotations

import pathlib

import pyarrow as pa
import pyarrow.parquet as pq

from ambientmapper.chunks import _iter_barcodes_from_filtered, make_barcode_chunks
from ambientmapper.extract import QC_PARQUET_SCHEMA


def _write_filtered_parquet(d: pathlib.Path, genome: str, bcs: list[str]) -> pathlib.Path:
    arrays = {name: [] for name in QC_PARQUET_SCHEMA.names}
    for bc in bcs:
        arrays["Read"].append("r")
        arrays["BC"].append(bc)
        arrays["MAPQ"].append(30)
        arrays["as_"].append(100)
        arrays["NM"].append(0)
        arrays["XAcount"].append(0)
        arrays["frag_loc"].append("chr1:1:2")
    tbl = pa.table(arrays, schema=QC_PARQUET_SCHEMA)
    p = d / f"filtered_{genome}_QCMapping.parquet"
    pq.write_table(tbl, p, compression="snappy")
    return p


def _write_filtered_tsv(d: pathlib.Path, genome: str, bcs: list[str]) -> pathlib.Path:
    p = d / f"filtered_{genome}_QCMapping.txt"
    with p.open("w") as f:
        f.write("Read\tBC\tMAPQ\tAS\tNM\tXAcount\tfrag_loc\n")
        for bc in bcs:
            f.write(f"r\t{bc}\t30\t100\t0\t0\tchr1:1:2\n")
    return p


# ---------------------------------------------------------------------------
# _iter_barcodes_from_filtered direct
# ---------------------------------------------------------------------------

def test_iter_barcodes_from_parquet(tmp_path: pathlib.Path):
    p = _write_filtered_parquet(tmp_path, "G1", ["BC1", "BC2", "BC3"])
    got = list(_iter_barcodes_from_filtered(p))
    assert got == ["BC1", "BC2", "BC3"]


def test_iter_barcodes_from_tsv(tmp_path: pathlib.Path):
    p = _write_filtered_tsv(tmp_path, "G1", ["BC1", "BC2"])
    got = list(_iter_barcodes_from_filtered(p))
    assert got == ["BC1", "BC2"]


def test_iter_barcodes_skips_empty_bcs_in_parquet(tmp_path: pathlib.Path):
    p = _write_filtered_parquet(tmp_path, "G1", ["BC1", "", "BC3"])
    got = list(_iter_barcodes_from_filtered(p))
    assert got == ["BC1", "BC3"]


# ---------------------------------------------------------------------------
# make_barcode_chunks dispatch
# ---------------------------------------------------------------------------

def _read_chunks(chunks_dir: pathlib.Path, sample: str) -> list[str]:
    """Return concatenated barcodes across all chunk files, in chunk-order."""
    out = []
    for cf in sorted(chunks_dir.glob(f"{sample}_cell_map_ref_chunk_*.txt")):
        out.extend(cf.read_text().strip().split("\n"))
    return out


def test_chunks_reads_parquet_inputs(tmp_path: pathlib.Path):
    """Two parquet inputs → chunks step dedups + sorts the union."""
    filtered_dir = tmp_path / "filtered_QCFiles"; filtered_dir.mkdir()
    _write_filtered_parquet(filtered_dir, "G1", ["BC1", "BC2", "BC3"])
    _write_filtered_parquet(filtered_dir, "G2", ["BC2", "BC4"])

    chunks_dir = tmp_path / "chunks"
    n = make_barcode_chunks(filtered_dir, chunks_dir, "S1", chunk_size=10)
    assert n == 1
    got = _read_chunks(chunks_dir, "S1")
    assert got == ["BC1", "BC2", "BC3", "BC4"]


def test_chunks_reads_tsv_inputs(tmp_path: pathlib.Path):
    """Legacy TSV inputs still work."""
    filtered_dir = tmp_path / "filtered_QCFiles"; filtered_dir.mkdir()
    _write_filtered_tsv(filtered_dir, "G1", ["BC1", "BC2"])
    _write_filtered_tsv(filtered_dir, "G2", ["BC2", "BC3"])

    chunks_dir = tmp_path / "chunks"
    n = make_barcode_chunks(filtered_dir, chunks_dir, "S1", chunk_size=10)
    assert n == 1
    assert _read_chunks(chunks_dir, "S1") == ["BC1", "BC2", "BC3"]


def test_chunks_prefers_parquet_when_complete(tmp_path: pathlib.Path):
    """Precedence rule: parquet wins when it covers every genome."""
    filtered_dir = tmp_path / "filtered_QCFiles"; filtered_dir.mkdir()
    _write_filtered_tsv(filtered_dir, "G1", ["BC_FROM_TSV_G1"])
    _write_filtered_tsv(filtered_dir, "G2", ["BC_FROM_TSV_G2"])
    _write_filtered_parquet(filtered_dir, "G1", ["BC_FROM_PQ_G1"])
    _write_filtered_parquet(filtered_dir, "G2", ["BC_FROM_PQ_G2"])

    chunks_dir = tmp_path / "chunks"
    make_barcode_chunks(filtered_dir, chunks_dir, "S1", chunk_size=10)
    got = _read_chunks(chunks_dir, "S1")
    # Only parquet-sourced barcodes should appear; TSV ignored when parquet
    # covers every genome.
    assert set(got) == {"BC_FROM_PQ_G1", "BC_FROM_PQ_G2"}


def test_chunks_falls_back_to_tsv_when_parquet_incomplete(tmp_path: pathlib.Path):
    """If parquet doesn't cover every genome, fall back to all-TSV (avoids
    mixing per-genome formats within a single pool)."""
    filtered_dir = tmp_path / "filtered_QCFiles"; filtered_dir.mkdir()
    _write_filtered_tsv(filtered_dir, "G1", ["BC_FROM_TSV_G1"])
    _write_filtered_tsv(filtered_dir, "G2", ["BC_FROM_TSV_G2"])
    _write_filtered_parquet(filtered_dir, "G1", ["BC_FROM_PQ_G1"])
    # No parquet for G2 → should fall back to TSV for BOTH genomes.

    chunks_dir = tmp_path / "chunks"
    make_barcode_chunks(filtered_dir, chunks_dir, "S1", chunk_size=10)
    got = _read_chunks(chunks_dir, "S1")
    assert set(got) == {"BC_FROM_TSV_G1", "BC_FROM_TSV_G2"}


def test_chunks_empty_filtered_dir_returns_zero(tmp_path: pathlib.Path):
    """No filtered files at all → no error, returns 0."""
    filtered_dir = tmp_path / "filtered_QCFiles"; filtered_dir.mkdir()
    chunks_dir = tmp_path / "chunks"
    n = make_barcode_chunks(filtered_dir, chunks_dir, "S1", chunk_size=10)
    assert n == 0


def test_chunks_splits_into_multiple_files_when_over_chunk_size(tmp_path: pathlib.Path):
    """Chunk size enforced: 5 unique BCs at chunk_size=2 → 3 chunk files."""
    filtered_dir = tmp_path / "filtered_QCFiles"; filtered_dir.mkdir()
    _write_filtered_parquet(filtered_dir, "G1", ["BC1", "BC2", "BC3", "BC4", "BC5"])

    chunks_dir = tmp_path / "chunks"
    n = make_barcode_chunks(filtered_dir, chunks_dir, "S1", chunk_size=2)
    assert n == 3
