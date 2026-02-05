from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional
import os
import sqlite3


def _iter_barcodes_from_filtered(path: Path) -> Iterable[str]:
    """
    Stream barcodes from a filtered QCMapping file.
    Assumes header: Read\tBC\tMAPQ\tAS\tNM\tXAcount
    """
    with path.open("r") as f:
        header = f.readline()
        # If file is headerless for some reason, treat first line as data
        if header and header.startswith("Read\t"):
            pass
        else:
            # header was actually data
            if header:
                parts = header.rstrip("\n").split("\t")
                if len(parts) >= 2 and parts[1]:
                    yield parts[1]

        for line in f:
            if not line:
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) >= 2:
                bc = parts[1]
                if bc:
                    yield bc


def _sqlite_pragmas(conn: sqlite3.Connection, *, fast: bool = True) -> None:
    """
    Pragmas tuned for bulk insert. Safe enough for scratch DB used in one run.
    """
    cur = conn.cursor()
    if fast:
        cur.execute("PRAGMA journal_mode = WAL;")
        cur.execute("PRAGMA synchronous = NORMAL;")
        cur.execute("PRAGMA temp_store = MEMORY;")
        cur.execute("PRAGMA cache_size = -200000;")  # ~200MB cache (negative = KB)
    cur.execute("PRAGMA foreign_keys = OFF;")
    conn.commit()


def make_barcode_chunks(
    filtered_dir: Path,
    chunks_dir: Path,
    sample: str,
    chunk_size: int,
    *,
    db_path: Optional[Path] = None,
    batch_insert: int = 200_000,
) -> int:
    """
    Create barcode chunk files from filtered QCMapping files.

    Memory-safe implementation:
      - Streams BCs from all filtered_*_QCMapping.txt
      - Deduplicates using a SQLite table with PRIMARY KEY
      - Iterates barcodes in sorted order and writes chunk files

    Returns number of chunk files written.
    """
    filtered_dir = Path(filtered_dir)
    chunks_dir = Path(chunks_dir)
    chunks_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(filtered_dir.glob("filtered_*_QCMapping.txt"))
    if not paths:
        return 0

    n = int(chunk_size)
    if n <= 0:
        raise ValueError("chunk_size must be > 0")

    # Put DB inside chunks_dir by default (same filesystem = faster)
    if db_path is None:
        db_path = chunks_dir / f"{sample}.barcodes.sqlite"

    # Rebuild DB each time to avoid stale state.
    if db_path.exists():
        db_path.unlink()

    conn = sqlite3.connect(str(db_path))
    try:
        _sqlite_pragmas(conn, fast=True)
        cur = conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS barcodes (bc TEXT PRIMARY KEY);")
        conn.commit()

        # Bulk insert with OR IGNORE to deduplicate
        buf = []
        for p in paths:
            for bc in _iter_barcodes_from_filtered(p):
                buf.append((bc,))
                if len(buf) >= batch_insert:
                    cur.executemany("INSERT OR IGNORE INTO barcodes(bc) VALUES (?);", buf)
                    conn.commit()
                    buf.clear()
        if buf:
            cur.executemany("INSERT OR IGNORE INTO barcodes(bc) VALUES (?);", buf)
            conn.commit()
            buf.clear()

        # Write chunks in sorted order
        k = 0
        out_lines = []

        # Iterate sorted without loading all rows
        for (bc,) in cur.execute("SELECT bc FROM barcodes ORDER BY bc;"):
            out_lines.append(bc)
            if len(out_lines) >= n:
                k += 1
                (chunks_dir / f"{sample}_cell_map_ref_chunk_{k:04d}.txt").write_text(
                    "\n".join(out_lines) + "\n"
                )
                out_lines.clear()

        if out_lines:
            k += 1
            (chunks_dir / f"{sample}_cell_map_ref_chunk_{k:04d}.txt").write_text(
                "\n".join(out_lines) + "\n"
            )
            out_lines.clear()

        # Optional: clean up DB to save space (comment out if you want to inspect)
        try:
            conn.close()
        finally:
            conn = None
        # Remove DB after success (keeps run directory smaller)
        if db_path.exists():
            db_path.unlink()

        return k

    finally:
        if conn is not None:
            conn.close()
