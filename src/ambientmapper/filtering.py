# src/ambientmapper/filtering.py
from __future__ import annotations

import os
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from .extract import QC_PARQUET_SCHEMA
from .normalization import canonicalize_bc_seq_sample_force

try:
    import duckdb as _duckdb
    _HAS_DUCKDB = True
except ImportError:  # pragma: no cover
    _duckdb = None
    _HAS_DUCKDB = False


# ----------------------------------------------------------------------------
# Public API
# ----------------------------------------------------------------------------

def filter_qc_file(
    in_path: Path,
    out_path: Path,
    min_freq: int,
    sample_name: Optional[str] = None,
    *,
    duckdb_threads: int = 2,
    duckdb_memory_limit: str = "16GB",
    row_group_size: int = 100_000,
) -> int:
    """Filter a QCMapping file by barcode frequency.

    Output format is dispatched by ``out_path.suffix``:

    - ``.parquet`` (default in 0.2): DuckDB-backed single-pass scan that
      counts BCs, filters by ``min_freq``, sorts by ``BC``, and writes a
      typed sorted Parquet conforming to :data:`extract.QC_PARQUET_SCHEMA`.
    - ``.txt`` (legacy, pre-0.2): pure-Python two-pass TSV scan, preserved
      verbatim for back-compat with users who pass ``--format txt`` on
      the filter command (or migrate via ``ambientmapper prepare``).

    Input format is auto-detected from ``in_path.suffix``:

    - ``.parquet``: ``read_parquet(...)`` (post-PR3 extract output)
    - ``.txt``: ``read_csv(..., header=false, ...)`` (legacy header-less
      TSV from extract; columns are positionally typed)

    Memory:
      Parquet path: ~1-2 GB peak (DuckDB external sort spills to
      ``$TMPDIR``); honors ``duckdb_memory_limit``.
      TSV path: O(unique_barcodes) for the keep-set, ~100-200 MB.

    Returns
    -------
    int
        Number of data rows written (excludes header).
    """
    in_path = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix == ".txt":
        return _filter_qc_file_tsv(in_path, out_path, min_freq, sample_name)
    return _filter_qc_file_parquet(
        in_path,
        out_path,
        min_freq,
        sample_name,
        duckdb_threads=duckdb_threads,
        duckdb_memory_limit=duckdb_memory_limit,
        row_group_size=row_group_size,
    )


# ----------------------------------------------------------------------------
# Parquet path (new in 0.2)
# ----------------------------------------------------------------------------

def _filter_qc_file_parquet(
    in_path: Path,
    out_path: Path,
    min_freq: int,
    sample_name: Optional[str],
    *,
    duckdb_threads: int,
    duckdb_memory_limit: str,
    row_group_size: int,
) -> int:
    """Single-SQL DuckDB pipeline: count → filter → sort → write parquet."""
    if not _HAS_DUCKDB:
        raise RuntimeError(
            "duckdb not installed; parquet output requires duckdb>=0.9. "
            "Either install duckdb or pass an .txt output path to use the "
            "legacy TSV writer."
        )
    if not in_path.exists():
        raise FileNotFoundError(in_path)

    min_freq = int(min_freq)

    # SQL fragment for reading the input. Naming the BAM `AS` column as
    # `as_` at read time avoids SQL reserved-word quoting downstream.
    if in_path.suffix == ".parquet":
        src_sql = f"read_parquet('{_sql_quote(str(in_path))}')"
    else:
        # Legacy header-less TSV from extract: 7 positional columns.
        # Use SMALLINT/INTEGER to match QC_PARQUET_SCHEMA types.
        src_sql = (
            f"read_csv('{_sql_quote(str(in_path))}', delim='\\t', header=false, "
            f"columns={{"
            f"'Read': 'VARCHAR', 'BC': 'VARCHAR', "
            f"'MAPQ': 'SMALLINT', 'as_': 'INTEGER', 'NM': 'INTEGER', "
            f"'XAcount': 'SMALLINT', 'frag_loc': 'VARCHAR'"
            f"}}, nullstr=[''])"
        )

    # Atomic write: stage to .tmp and rename on success.
    tmp_out = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp_out.exists():
        tmp_out.unlink()

    con = _duckdb.connect()
    try:
        con.execute(f"SET threads TO {max(1, int(duckdb_threads))}")
        con.execute(f"SET memory_limit='{duckdb_memory_limit}'")
        con.execute(f"SET temp_directory='{_sql_quote(tempfile.gettempdir())}'")
        con.execute("SET preserve_insertion_order=false")

        # BC normalization via Python UDF (if sample_name provided).
        # ~1 µs/row overhead, comparable to the legacy Python loop.
        if sample_name:
            _sample = sample_name  # capture for closure
            def _norm_bc(bc):
                return canonicalize_bc_seq_sample_force(bc or "", _sample)
            # Use string type names (portable across DuckDB 0.9+).
            # `duckdb.typing.*` constants only exist on 0.10+ and pyproject
            # allows older.
            con.create_function(
                "normalize_bc", _norm_bc,
                ["VARCHAR"], "VARCHAR",
            )
            bc_expr = "normalize_bc(BC) AS BC"
        else:
            bc_expr = "BC AS BC"

        # Count → filter → sort → write in a single COPY.
        # COALESCE on frag_loc + numeric tags so missing values write as defaults
        # consistent with the legacy TSV writer (mapq/as=0, NM=10^9, xa=0).
        copy_sql = f"""
            COPY (
                WITH src AS (
                    SELECT
                        Read,
                        {bc_expr},
                        COALESCE(MAPQ, 0) AS MAPQ,
                        as_,
                        NM,
                        COALESCE(XAcount, 0) AS XAcount,
                        COALESCE(frag_loc, '') AS frag_loc
                    FROM {src_sql}
                ),
                counted AS (
                    SELECT *, COUNT(*) OVER (PARTITION BY BC) AS bc_n
                    FROM src
                )
                SELECT
                    Read,
                    BC,
                    CAST(MAPQ AS SMALLINT)    AS MAPQ,
                    CAST(as_ AS INTEGER)       AS as_,
                    CAST(NM AS INTEGER)        AS NM,
                    CAST(XAcount AS SMALLINT)  AS XAcount,
                    frag_loc
                FROM counted
                WHERE bc_n >= {min_freq}
                ORDER BY BC
            )
            TO '{_sql_quote(str(tmp_out))}'
            (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE {int(row_group_size)})
        """
        con.execute(copy_sql)
    except Exception:
        con.close()
        if tmp_out.exists():
            tmp_out.unlink()
        raise
    finally:
        con.close()

    # Atomic rename.
    os.replace(tmp_out, out_path)

    # Handle the empty-result case: DuckDB writes a valid empty parquet, but
    # if no rows matched we still want a schema-conformant file so downstream
    # readers don't choke. Re-emit via ParquetWriter only when needed.
    n_rows = _parquet_num_rows(out_path)
    if n_rows == 0:
        _write_empty_parquet(out_path)
    return int(n_rows)


def _parquet_num_rows(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def _write_empty_parquet(out_path: Path) -> None:
    """Replace ``out_path`` with a header-only parquet matching the schema."""
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with pq.ParquetWriter(tmp, QC_PARQUET_SCHEMA, compression="snappy") as w:
        # Write a zero-row table so the file has the schema embedded.
        w.write_table(pa.table({name: [] for name in QC_PARQUET_SCHEMA.names},
                                schema=QC_PARQUET_SCHEMA))
    os.replace(tmp, out_path)


def _sql_quote(s: str) -> str:
    """Quote a string for safe SQL literal embedding (DuckDB single-quote rules)."""
    return s.replace("'", "''")


# ----------------------------------------------------------------------------
# Legacy TSV path (preserved verbatim; called when out_path.suffix == ".txt")
# ----------------------------------------------------------------------------

def _filter_qc_file_tsv(
    in_path: Path,
    out_path: Path,
    min_freq: int,
    sample_name: Optional[str],
) -> int:
    """
    Stream QCMapping file in two passes:

      Pass 1) Count barcodes (after optional normalization to '<seq>-<sample>'),
              build keep-set of barcodes with count >= min_freq.

      Pass 2) Re-stream file, emit rows whose barcode is in keep-set
              directly to output. No in-memory aggregation.

    Input format (no header), tab-delimited:
      Read  BC  MAPQ  AS  NM  XAcount  [frag_loc]

    Output format (WITH header), tab-delimited:
      Read  BC  MAPQ  AS  NM  XAcount  frag_loc

    Returns:
      total rows written (excluding header). May include duplicate
      (Read, BC) pairs from input; downstream stages (assign, chunks)
      handle dedup independently.

    Memory: O(unique_barcodes) for the keep-set, typically 100-200 MB.
    """
    in_path = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    min_freq = int(min_freq)

    # -------------------------
    # Pass 1: count (normalized) barcodes
    # -------------------------
    bc_counts: Dict[str, int] = defaultdict(int)

    # Small optimization: if sample_name is provided, normalization is needed;
    # otherwise, we assume BC is already in desired form.
    do_norm = bool(sample_name)

    with in_path.open("r") as f:
        for line in f:
            if not line:
                continue
            line = line.rstrip("\n")
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                continue

            bc = parts[1]
            if do_norm:
                # canonicalize handles empty/malformed; keep it robust
                bc = canonicalize_bc_seq_sample_force(bc or "", sample_name)  # type: ignore[arg-type]

            # intern to reduce memory if many repeats
            bc = sys.intern(bc)
            bc_counts[bc] += 1

    keep = {bc for bc, c in bc_counts.items() if c >= min_freq}

    if not keep:
        # Write empty file with header, matching previous behavior
        with out_path.open("w") as out:
            out.write("Read\tBC\tMAPQ\tAS\tNM\tXAcount\tfrag_loc\n")
        return 0

    # Allow GC to reclaim counts dict memory before heavy pass 2
    bc_counts.clear()

    # -------------------------
    # Pass 2: stream-filter and emit
    # -------------------------
    def _to_int(x: str, default: int) -> int:
        try:
            return int(x)
        except Exception:
            return default

    n_written = 0

    with in_path.open("r") as f, out_path.open("w") as out:
        out.write("Read\tBC\tMAPQ\tAS\tNM\tXAcount\tfrag_loc\n")

        for line in f:
            if not line:
                continue
            line = line.rstrip("\n")
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 6:
                continue

            bc = parts[1]
            if do_norm:
                bc = canonicalize_bc_seq_sample_force(bc or "", sample_name)  # type: ignore[arg-type]

            if bc not in keep:
                continue

            # Sanitize metrics (robustness against malformed values)
            mapq = _to_int(parts[2], 0)
            alsc = _to_int(parts[3], 0)
            nm   = _to_int(parts[4], 10**9)
            xac  = _to_int(parts[5], 0)
            frag_loc = parts[6] if len(parts) >= 7 else ""

            out.write(f"{parts[0]}\t{bc}\t{mapq}\t{alsc}\t{nm}\t{xac}\t{frag_loc}\n")
            n_written += 1

    return n_written
