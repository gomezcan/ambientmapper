"""
Shared helpers for discovering and identifying per-genome filtered QC files.

These were originally private functions in ``assign_streaming.py``
(``_genome_from_filename`` and ``_filtered_files``). They were lifted here
so ``chunks.py`` can use the same precedence rule for choosing between
legacy ``.txt`` and current ``.parquet`` inputs.

Public API:
  - ``genome_from_filename(path)`` — extract the genome name from a
    ``filtered_<GENOME>_QCMapping.{txt,parquet}`` file.
  - ``discover_filtered_files(filtered_dir)`` — list per-genome files in a
    ``filtered_QCFiles/`` directory, preferring Parquet when it covers all
    genomes that have a matching TSV (all-or-nothing precedence).
"""
from __future__ import annotations

from pathlib import Path


_KNOWN_SUFFIXES: tuple[str, ...] = ("_QCMapping.parquet", "_QCMapping.txt")


def genome_from_filename(p: Path) -> str:
    """Return the genome name from a ``filtered_<GENOME>_QCMapping.{txt,parquet}`` path.

    Falls back to the bare filename if neither known suffix matches, so callers
    that pass unrelated paths still get a deterministic (if useless) string.
    """
    n = p.name
    for suffix in _KNOWN_SUFFIXES:
        if n.startswith("filtered_") and n.endswith(suffix):
            return n[len("filtered_") : -len(suffix)]
    return n


def discover_filtered_files(filtered_dir: Path) -> list[Path]:
    """Return per-genome filtered QC files from ``filtered_dir``.

    Precedence rule (matches the historical ``_filtered_files`` behavior in
    ``assign_streaming.py``): prefer ``.parquet`` when it covers all genomes
    that also have a ``.txt`` companion; otherwise return the ``.txt`` files.
    This avoids mixing per-genome formats within a single pool, which would
    risk schema-cast surprises in DuckDB SQL.

    Raises
    ------
    FileNotFoundError
        If ``filtered_dir`` does not exist, or if it contains neither
        ``.parquet`` nor ``.txt`` files.
    """
    d = Path(filtered_dir)
    if not d.exists():
        raise FileNotFoundError(f"Not found: {d}")
    pq_files = sorted(d.glob("filtered_*_QCMapping.parquet"))
    tsv_files = sorted(d.glob("filtered_*_QCMapping.txt"))
    if pq_files:
        pq_genomes = {genome_from_filename(p) for p in pq_files}
        tsv_genomes = {genome_from_filename(p) for p in tsv_files}
        # Prefer parquet only when it covers every genome that has a TSV.
        # If TSVs exist for additional genomes, fall back to all-TSV so the
        # caller sees a consistent set.
        if not tsv_genomes or pq_genomes >= tsv_genomes:
            return pq_files
    if not tsv_files:
        raise FileNotFoundError(f"No filtered_* files under {d}")
    return tsv_files
