#!/usr/bin/env python3
"""
assign_streaming.py

A memory- and IO-aware "assign" implementation for AmbientMapper-like workflows.

Design goals
------------
- Keep per-(Read,BC) accumulator lightweight (dataclass with slots)
- Avoid nested dict-of-dict-of-sets patterns
- Use stable dtypes (string/float32) and avoid repeated astype(str)
- Parallelize at a safe granularity (batch of chunk files per worker)
- Avoid nested parallelism (common source of BrokenProcessPool/OOM)
- Provide:
    Pass A: learn global AS/MAPQ decile edges from winner distributions
    Pass B: learn per-decile ΔAS / ΔMAPQ tail ECDFs (stored as hist counts)
    Pass C: score chunks, producing:
        - raw per-read summary table (winner, runner-up, deltas, p-values)
        - filtered per-read/per-genome table (keeps winner rows + ambiguous rows)

Notes / assumptions
-------------------
- Input genome-mapping files live under: {workdir}/{sample}/filtered_QCFiles/
  matching: filtered_*_QCMapping.txt
- Chunk files are barcode lists (one barcode per line), matching:
  *_cell_map_ref_chunk_*.txt

- We assume the filtered mapping files are TAB-separated with header and columns:
  Read, BC, MAPQ, AS, NM, XAcount

- "Winner" ranking is: NM asc, then AS desc, then MAPQ desc.
- "Ambiguity" is assessed using empirical tail probabilities of ΔAS and ΔMAPQ
  within deciles of winner AS/MAPQ:
    p_as = P(ΔAS >= observed | winner-AS decile)
    p_mq = P(ΔMQ >= observed | winner-MAPQ decile)
  Then a read is called:
    winner if (p_as <= alpha) OR (p_mq <= alpha) OR (runner-up absent) OR (ΔNM > 0)
    ambiguous otherwise

Performance knobs
-----------------
- --chunksize: pandas read_csv chunk size for genome files (rows per chunk)
- --batch-size: number of chunk files handled per worker in batched passes
- --workers: number of worker processes for batched passes
- --max-reads-per-genome: optional cap during model learning for speed

Outputs
-------
Models saved to:
  {workdir}/{sample}/ExplorationReadLevel/global_edges.npz
  {workdir}/{sample}/ExplorationReadLevel/global_ecdf.npz

Scored chunk outputs to:
  {workdir}/{sample}/raw_cell_map_ref_chunks/{sample}_chunk{N}_raw.tsv.gz
  {workdir}/{sample}/cell_map_ref_chunks/{sample}_chunk{N}_filtered.tsv.gz
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Iterable, Tuple

import numpy as np
import pandas as pd
import typer

app = typer.Typer(add_completion=False)

try:
    import duckdb as _duckdb
    _HAS_DUCKDB = True
except ImportError:
    _duckdb = None  # type: ignore
    _HAS_DUCKDB = False

# -----------------------------
# Logging helpers
# -----------------------------
_LOG_LOCK = Lock()


def _log(msg: str, verbose: bool) -> None:
    if not verbose:
        return
    with _LOG_LOCK:
        typer.echo(msg)


def _log_ok(msg: str, verbose: bool) -> None:
    if not verbose:
        return
    with _LOG_LOCK:
        typer.secho(msg, fg="green")


def _log_err(msg: str, verbose: bool) -> None:
    with _LOG_LOCK:
        typer.secho(msg, fg="red")


# -----------------------------
# Small utils
# -----------------------------
def _chunk_bcs(chunk_file: Path) -> set[str]:
    with open(chunk_file, "r") as f:
        return {ln.strip() for ln in f if ln.strip()}


def _genome_from_filename(p: Path) -> str:
    n = p.name
    if n.startswith("filtered_") and n.endswith("_QCMapping.txt"):
        return n[len("filtered_") : -len("_QCMapping.txt")]
    return n


def _filtered_files(workdir: Path, sample: str) -> list[Path]:
    d = Path(workdir) / sample / "filtered_QCFiles"
    if not d.exists():
        raise FileNotFoundError(f"Not found: {d}")
    files = sorted(d.glob("filtered_*_QCMapping.txt"))
    if not files:
        raise FileNotFoundError(f"No filtered_* files under {d}")
    return files


def _build_duckdb_union_sql(
    files: list,
    mapq_min: int,
    xa_max: int,
    bc_table: str = "_bcs",
) -> str:
    """
    Build a UNION ALL SQL fragment over all genome files.
    AS is a SQL reserved word — aliased to as_ throughout.
    Filters are applied inside each SELECT for DuckDB pushdown.
    xa_max < 0 means no XAcount filter.
    """
    parts = []
    for fp in files:
        genome = _genome_from_filename(fp).replace("'", "''")
        path   = str(fp).replace("'", "''")
        where  = [f"BC IN (SELECT bc FROM {bc_table})"]
        if mapq_min > 0:
            where.append(f"MAPQ >= {mapq_min}")
        if xa_max >= 0:
            where.append(f"XAcount <= {xa_max}")
        parts.append(
            f"SELECT Read, BC, CAST(\"AS\" AS FLOAT) AS as_, "
            f"CAST(MAPQ AS FLOAT) AS MAPQ, CAST(NM AS FLOAT) AS NM, "
            f"CAST(XAcount AS FLOAT) AS XAcount, '{genome}' AS Genome "
            f"FROM read_csv('{path}', delim='\\t', header=true) "
            f"WHERE {' AND '.join(where)}"
        )
    return "\n    UNION ALL\n    ".join(parts)


def _learn_edges_duckdb(
    files: list,
    sampled_bcs: set,
    mapq_min: int,
    xa_max: int,
    duckdb_threads: int,
    lo_as: float,
    hi_as: float,
    nb_as: int,
    lo_mq: float,
    hi_mq: float,
    nb_mq: int,
    verbose: bool,
) -> tuple:
    """
    DuckDB fast path for Pass A: compute winner AS/MAPQ histograms in one SQL pass.

    Scans each genome file exactly once (total = len(files) file scans).
    Returns (as_counts, mq_counts, n_winners).
    """
    import pyarrow as pa

    con = _duckdb.connect()
    con.execute(f"SET threads TO {max(1, duckdb_threads)}")

    con.register(
        "_bcs",
        pa.table({"bc": pa.array(sorted(sampled_bcs), type=pa.string())}),
    )

    union_sql = _build_duckdb_union_sql(files, mapq_min, xa_max, "_bcs")

    sql = f"""
    WITH all_reads AS (
        {union_sql}
    ),
    deduped AS (
        SELECT Read, BC, Genome, as_, MAPQ, NM
        FROM (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY Read, BC, Genome
                    ORDER BY NM ASC, as_ DESC, MAPQ DESC
                ) AS _rg
            FROM all_reads
        ) t WHERE _rg = 1
    ),
    ranked AS (
        SELECT as_, MAPQ,
            ROW_NUMBER() OVER (
                PARTITION BY Read, BC
                ORDER BY NM ASC, as_ DESC, MAPQ DESC
            ) AS _rn
        FROM deduped
    )
    SELECT as_, MAPQ FROM ranked WHERE _rn = 1
    """

    if verbose:
        _log("[assign/edges] DuckDB query running ...", True)
    winners_df = con.execute(sql).df()
    con.close()

    n_winners = len(winners_df)
    if n_winners == 0:
        return (
            np.zeros(nb_as, dtype=np.int64),
            np.zeros(nb_mq, dtype=np.int64),
            0,
        )

    as_vals = winners_df["as_"].values.astype(np.float64)
    mq_vals = winners_df["MAPQ"].values.astype(np.float64)

    as_counts = np.histogram(
        as_vals, bins=nb_as, range=(lo_as, hi_as)
    )[0].astype(np.int64)
    mq_counts = np.histogram(
        mq_vals, bins=nb_mq, range=(lo_mq, hi_mq)
    )[0].astype(np.int64)

    return as_counts, mq_counts, n_winners


def _learn_ecdfs_duckdb(
    files: list,
    sampled_bcs: set,
    mapq_min: int,
    xa_max: int,
    duckdb_threads: int,
    as_edges: np.ndarray,
    mq_edges: np.ndarray,
    k: int,
    lo_das: float,
    hi_das: float,
    nb_das: int,
    lo_dmq: float,
    hi_dmq: float,
    nb_dmq: int,
    verbose: bool,
) -> tuple:
    """
    DuckDB fast path for Pass B: compute per-decile delta histograms in one SQL pass.

    For each (Read, BC), finds best1 and best2 alignments across genomes,
    computes dAS = best1_as - best2_as and dMQ = best1_mq - best2_mq,
    then bins deltas by winner decile.

    Returns (dAS_counts, dAS_over, dMQ_counts, dMQ_over, n_pairs).
    """
    import pyarrow as pa

    con = _duckdb.connect()
    con.execute(f"SET threads TO {max(1, duckdb_threads)}")

    con.register(
        "_bcs",
        pa.table({"bc": pa.array(sorted(sampled_bcs), type=pa.string())}),
    )

    union_sql = _build_duckdb_union_sql(files, mapq_min, xa_max, "_bcs")

    sql = f"""
    WITH all_reads AS (
        {union_sql}
    ),
    deduped AS (
        SELECT Read, BC, Genome, as_, MAPQ, NM
        FROM (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY Read, BC, Genome
                    ORDER BY NM ASC, as_ DESC, MAPQ DESC
                ) AS _rg
            FROM all_reads
        ) t WHERE _rg = 1
    ),
    ranked AS (
        SELECT Read, BC, as_, MAPQ, NM,
            ROW_NUMBER() OVER (
                PARTITION BY Read, BC
                ORDER BY NM ASC, as_ DESC, MAPQ DESC
            ) AS _rn
        FROM deduped
    ),
    best12 AS (
        SELECT
            Read, BC,
            MAX(CASE WHEN _rn = 1 THEN as_ END) AS as1,
            MAX(CASE WHEN _rn = 1 THEN MAPQ END) AS mq1,
            MAX(CASE WHEN _rn = 2 THEN as_ END) AS as2,
            MAX(CASE WHEN _rn = 2 THEN MAPQ END) AS mq2
        FROM ranked WHERE _rn <= 2
        GROUP BY Read, BC
        HAVING COUNT(*) = 2
    )
    SELECT as1, mq1, as1 - as2 AS dAS, mq1 - mq2 AS dMQ
    FROM best12
    """

    if verbose:
        _log("[assign/ecdf] DuckDB query running ...", True)
    df = con.execute(sql).df()
    con.close()

    n_pairs = len(df)
    if n_pairs == 0:
        return (
            np.zeros((k, nb_das), dtype=np.int64),
            np.zeros((k,), dtype=np.int64),
            np.zeros((k, nb_dmq), dtype=np.int64),
            np.zeros((k,), dtype=np.int64),
            0,
        )

    as1 = df["as1"].values.astype(np.float64)
    mq1 = df["mq1"].values.astype(np.float64)
    dAS = df["dAS"].values.astype(np.float64)
    dMQ = df["dMQ"].values.astype(np.float64)

    # Assign winner to deciles (same logic as _ecdf_chunk_worker / score_chunk)
    if as_edges.size > 0:
        dec_as = np.clip(
            np.searchsorted(as_edges, as1, side="left") + 1, 1, k
        ).astype(np.int64)
    else:
        dec_as = np.ones(n_pairs, dtype=np.int64)

    if mq_edges.size > 0:
        dec_mq = np.clip(
            np.searchsorted(mq_edges, mq1, side="left") + 1, 1, k
        ).astype(np.int64)
    else:
        dec_mq = np.ones(n_pairs, dtype=np.int64)

    # Per-decile histogram binning
    dAS_counts = np.zeros((k, nb_das), dtype=np.int64)
    dAS_over = np.zeros((k,), dtype=np.int64)
    dMQ_counts = np.zeros((k, nb_dmq), dtype=np.int64)
    dMQ_over = np.zeros((k,), dtype=np.int64)

    for d in range(1, k + 1):
        mask_as = dec_as == d
        vals = dAS[mask_as]
        if vals.size > 0:
            dAS_counts[d - 1] = np.histogram(
                vals, bins=nb_das, range=(lo_das, hi_das)
            )[0].astype(np.int64)
            dAS_over[d - 1] = int(np.sum(vals >= hi_das))

        mask_mq = dec_mq == d
        vals = dMQ[mask_mq]
        if vals.size > 0:
            dMQ_counts[d - 1] = np.histogram(
                vals, bins=nb_dmq, range=(lo_dmq, hi_dmq)
            )[0].astype(np.int64)
            dMQ_over[d - 1] = int(np.sum(vals >= hi_dmq))

    return dAS_counts, dAS_over, dMQ_counts, dMQ_over, n_pairs


def _score_chunk_duckdb(
    files: list,
    bcs: set,
    mapq_min: int,
    xa_max: int,
    duckdb_threads: int,
):
    """
    DuckDB-backed Pass C helpers.

    Returns (top3_df, lazy_ambig) where:
      top3_df   — DataFrame with _rn=1 (best1), _rn=2 (best2), _rn_worst=1 (worst), n_genomes
      lazy_ambig — callable(ambiguous_bcs) → DataFrame of ALL genome rows for those BCs

    The DuckDB connection is kept alive inside the closure; it is released when
    lazy_ambig is garbage-collected at the end of score_chunk.
    """
    import pyarrow as pa

    con = _duckdb.connect()
    con.execute(f"SET threads TO {max(1, duckdb_threads)}")

    # Register BC set via Arrow (zero-copy)
    con.register("_bcs", pa.table({"bc": pa.array(list(bcs), type=pa.string())}))

    union_sql = _build_duckdb_union_sql(files, mapq_min, xa_max, "_bcs")

    q1 = f"""
    WITH all_reads AS (
        {union_sql}
    ),
    deduped AS (
        SELECT Read, BC, Genome, as_, MAPQ, NM, XAcount
        FROM (
            SELECT *,
                ROW_NUMBER() OVER (
                    PARTITION BY Read, BC, Genome
                    ORDER BY NM ASC, as_ DESC, MAPQ DESC
                ) AS _rg
            FROM all_reads
        ) t WHERE _rg = 1
    ),
    ranked AS (
        SELECT *,
            ROW_NUMBER() OVER (
                PARTITION BY Read, BC ORDER BY NM ASC, as_ DESC, MAPQ DESC
            ) AS _rn,
            ROW_NUMBER() OVER (
                PARTITION BY Read, BC ORDER BY NM DESC, as_ ASC, MAPQ ASC
            ) AS _rn_worst,
            COUNT(*) OVER (PARTITION BY Read, BC) AS n_genomes
        FROM deduped
    )
    SELECT Read, BC, Genome, as_, MAPQ, NM, XAcount, _rn, _rn_worst, n_genomes
    FROM ranked
    WHERE _rn <= 2 OR _rn_worst = 1
    """

    top3_df = con.execute(q1).df()

    def lazy_ambig(ambiguous_bcs: set) -> "pd.DataFrame":
        if not ambiguous_bcs:
            return pd.DataFrame(
                columns=["Read", "BC", "Genome", "as_", "MAPQ", "NM", "XAcount"]
            )
        con.register(
            "_ambiguous_bcs",
            pa.table({"bc": pa.array(list(ambiguous_bcs), type=pa.string())})
        )
        # Re-use the same union_sql but restrict to ambiguous BCs only
        ambig_union = _build_duckdb_union_sql(files, mapq_min, xa_max, "_ambiguous_bcs")
        q2 = f"""
        WITH all_reads AS (
            {ambig_union}
        ),
        deduped AS (
            SELECT Read, BC, Genome, as_, MAPQ, NM, XAcount
            FROM (
                SELECT *,
                    ROW_NUMBER() OVER (
                        PARTITION BY Read, BC, Genome
                        ORDER BY NM ASC, as_ DESC, MAPQ DESC
                    ) AS _rg
                FROM all_reads
            ) t WHERE _rg = 1
        )
        SELECT * FROM deduped
        """
        return con.execute(q2).df()

    return top3_df, lazy_ambig


# -----------------------------
# Histogram + tail-prob model
# -----------------------------
class DeltaHist:
    """
    Fixed-bin histogram with overflow bucket and tail-prob query.
    tail_p(x) = P(X >= x) under the empirical distribution.
    """

    __slots__ = ("lo", "hi", "nbins", "edges", "counts", "overflow", "_total", "_suffix", "_frozen")

    def __init__(self, lo: float, hi: float, nbins: int):
        self.lo = float(lo)
        self.hi = float(hi)
        self.nbins = int(nbins)
        self.edges = np.linspace(self.lo, self.hi, self.nbins + 1, dtype=np.float64)
        self.counts = np.zeros(self.nbins, dtype=np.int64)
        self.overflow = 0
        self._total: int = 0
        self._suffix: Optional[np.ndarray] = None
        self._frozen: bool = False

    def add(self, x: float) -> None:
        if self._frozen:
            raise RuntimeError("DeltaHist.add() called after freeze()")
        if not np.isfinite(x):
            return
        if x >= self.hi:
            self.overflow += 1
            self._total += 1
            return
        if x <= self.lo:
            idx = 0
        else:
            idx = int((x - self.lo) / (self.hi - self.lo) * self.nbins)
            if idx < 0:
                idx = 0
            elif idx >= self.nbins:
                idx = self.nbins - 1
        self.counts[idx] += 1
        self._total += 1

    def freeze(self) -> None:
        """Precompute suffix-sum for O(1) tail_p. Call once after all add() calls."""
        self._total = int(self.counts.sum()) + self.overflow
        self._suffix = np.cumsum(self.counts[::-1])[::-1].astype(np.int64)
        self._frozen = True

    def tail_p(self, x: float) -> float:
        if not np.isfinite(x):
            return np.nan
        if self._frozen:
            total = self._total
            if total == 0:
                return np.nan
            if x >= self.hi:
                return self.overflow / total
            if x <= self.lo:
                return 1.0
            idx = int((x - self.lo) / (self.hi - self.lo) * self.nbins)
            idx = max(0, min(idx, self.nbins - 1))
            return (int(self._suffix[idx]) + self.overflow) / total
        else:
            # Unfrozen path: used during Pass A/B accumulation if tail_p is ever called
            total = int(self.counts.sum() + self.overflow)
            if total == 0:
                return np.nan
            if x >= self.hi:
                return self.overflow / total
            if x <= self.lo:
                return 1.0
            idx = int((x - self.lo) / (self.hi - self.lo) * self.nbins)
            idx = max(0, min(idx, self.nbins - 1))
            return int(self.counts[idx:].sum() + self.overflow) / total


# winner hist specs
_AS_RANGE = (0.0, 200.0, 2000)  # lo, hi, nbins
_MQ_RANGE = (0.0, 60.0, 120)

# delta hist specs
_DAS_RANGE = (0.0, 100.0, 200)
_DMQ_RANGE = (0.0, 60.0, 120)


def _new_hist(spec: tuple[float, float, int]) -> DeltaHist:
    lo, hi, nb = spec
    return DeltaHist(lo, hi, nb)


def edges_from_hist_counts(counts: np.ndarray, lo: float, hi: float, k: int) -> np.ndarray:
    """
    Given histogram counts over [lo,hi] with len(counts)=nbins,
    compute k-1 quantile cutpoints (decile edges when k=10).
    """
    counts = counts.astype(np.int64, copy=False)
    if counts.sum() == 0:
        return np.array([], dtype=np.float64)

    nbins = counts.shape[0]
    edges = np.linspace(lo, hi, nbins + 1, dtype=np.float64)
    cum = np.cumsum(counts)
    tot = int(cum[-1])
    targets = (np.linspace(0, 1, k + 1)[1:-1] * tot).astype(np.int64)

    out: list[float] = []
    for t in targets:
        idx = int(np.searchsorted(cum, t, side="left"))
        if idx < 0:
            idx = 0
        elif idx >= nbins:
            idx = nbins - 1
        out.append(edges[idx])
    return np.unique(np.array(out, dtype=np.float64))


def assign_decile_scalar(v: float, edges: np.ndarray) -> int:
    """
    Return 1..k where k=len(edges)+1. If edges empty => 1.
    """
    if edges.size == 0 or not np.isfinite(v):
        return 1
    return int(np.searchsorted(edges, v, side="left") + 1)


# -----------------------------
# Lightweight per-(Read,BC) accumulator
# -----------------------------
@dataclass(slots=True)
class ReadAcc:
    best1_g: Optional[str] = None
    best1_as: float = -np.inf
    best1_mq: float = -np.inf
    best1_nm: float = np.inf
    best1_xa: float = np.nan   # XAcount for best1 alignment; np.nan for Pass A/B callers

    best2_g: Optional[str] = None
    best2_as: float = -np.inf
    best2_mq: float = -np.inf
    best2_nm: float = np.inf

    worst_g: Optional[str] = None
    worst_as: float = -np.inf
    worst_mq: float = -np.inf
    worst_nm: float = np.inf

    n_genomes: int = 0


def _better(nm1: float, as1: float, mq1: float, nm2: float, as2: float, mq2: float) -> bool:
    # NM asc, AS desc, MAPQ desc
    if nm1 != nm2:
        return nm1 < nm2
    if as1 != as2:
        return as1 > as2
    return mq1 > mq2


def update_acc(acc: ReadAcc, genome: str, AS: float, MAPQ: float, NM: float, xa: float = np.nan) -> None:
    acc.n_genomes += 1

    if acc.best1_g is None:
        acc.best1_g = genome
        acc.best1_as = AS
        acc.best1_mq = MAPQ
        acc.best1_nm = NM
        acc.best1_xa = xa

        acc.worst_g = genome
        acc.worst_as = AS
        acc.worst_mq = MAPQ
        acc.worst_nm = NM
        return

    # worst (keep "worst" as the *least good* alignment by the same comparator, i.e., inverse of _better)
    # If current is not better than stored worst, update worst.
    if not _better(NM, AS, MAPQ, acc.worst_nm, acc.worst_as, acc.worst_mq):
        acc.worst_g = genome
        acc.worst_as = AS
        acc.worst_mq = MAPQ
        acc.worst_nm = NM

    # best1 / best2
    if _better(NM, AS, MAPQ, acc.best1_nm, acc.best1_as, acc.best1_mq):
        acc.best2_g = acc.best1_g
        acc.best2_as = acc.best1_as
        acc.best2_mq = acc.best1_mq
        acc.best2_nm = acc.best1_nm

        acc.best1_g = genome
        acc.best1_as = AS
        acc.best1_mq = MAPQ
        acc.best1_nm = NM
        acc.best1_xa = xa
    else:
        if acc.best2_g is None or _better(NM, AS, MAPQ, acc.best2_nm, acc.best2_as, acc.best2_mq):
            acc.best2_g = genome
            acc.best2_as = AS
            acc.best2_mq = MAPQ
            acc.best2_nm = NM


# -----------------------------
# Streaming row iterator
# -----------------------------
def _iter_rows(
    fp: Path,
    bcs: set[str],
    chunksize: int,
    mapq_min: int,
    xa_max: int,
    max_rows_for_genome: Optional[int] = None,
) -> Iterable[Tuple[str, str, float, float, float]]:
    """
    Yield (Read, BC, AS, MAPQ, NM) for rows passing filters.
    Uses stable dtypes (string/float32) to reduce memory.
    """
    usecols = ["Read", "BC", "MAPQ", "AS", "NM", "XAcount"]
    seen = 0

    for df in pd.read_csv(
        fp,
        sep="\t",
        usecols=usecols,
        chunksize=chunksize,
        dtype={
            "Read": "string",
            "BC": "string",
            "AS": "float32",
            "MAPQ": "float32",
            "NM": "float32",
            "XAcount": "float32",
        },
        engine="c",
        low_memory=True,
        memory_map=False,
    ):
        if max_rows_for_genome is not None and seen >= max_rows_for_genome:
            break

        df = df[df["BC"].isin(bcs)]
        if df.empty:
            continue

        if mapq_min > 0:
            df = df[df["MAPQ"] >= mapq_min]
        if xa_max >= 0:
            df = df[df["XAcount"] <= xa_max]
        if df.empty:
            continue

        # dedup within chunk
        df = df.drop_duplicates(subset=["Read", "BC"], keep="first")

        if max_rows_for_genome is not None:
            remaining = max_rows_for_genome - seen
            if remaining <= 0:
                break
            if len(df) > remaining:
                df = df.iloc[:remaining]

        seen += len(df)

        # itertuples gives python scalars; string dtype yields python str
        for r in df.itertuples(index=False):
            # r.Read, r.BC are pandas String scalars => convert to str once
            yield (str(r.Read), str(r.BC), float(r.AS), float(r.MAPQ), float(r.NM))


def _iter_rows_with_genome_xa(
    fp: Path,
    genome: str,
    bcs: set[str],
    chunksize: int,
    mapq_min: int,
    xa_max: int,
) -> Iterable[Tuple[str, str, float, float, float, float, str]]:
    """
    Yield (Read, BC, AS, MAPQ, NM, XAcount, genome) for rows passing filters.
    Identical filtering logic to _iter_rows but also yields XAcount and genome.
    Used exclusively in score_chunk (Pass C) to support single-pass output.
    """
    usecols = ["Read", "BC", "MAPQ", "AS", "NM", "XAcount"]

    for df in pd.read_csv(
        fp,
        sep="\t",
        usecols=usecols,
        chunksize=chunksize,
        dtype={
            "Read": "string",
            "BC": "string",
            "AS": "float32",
            "MAPQ": "float32",
            "NM": "float32",
            "XAcount": "float32",
        },
        engine="c",
        low_memory=True,
        memory_map=False,
    ):
        df = df[df["BC"].isin(bcs)]
        if df.empty:
            continue

        if mapq_min > 0:
            df = df[df["MAPQ"] >= mapq_min]
        if xa_max >= 0:
            df = df[df["XAcount"] <= xa_max]
        if df.empty:
            continue

        df = df.drop_duplicates(subset=["Read", "BC"], keep="first")

        for r in df.itertuples(index=False):
            yield (str(r.Read), str(r.BC), float(r.AS), float(r.MAPQ), float(r.NM), float(r.XAcount), genome)


# -----------------------------
# Worker: learn winner hist counts for a batch of chunk files
# -----------------------------
def _edges_batch_worker(
    bi: int,
    chunk_files: list[str],
    genome_files: list[str],
    mapq_min: int,
    xa_max: int,
    chunksize: int,
    lo_as: float,
    hi_as: float,
    nb_as: int,
    lo_mq: float,
    hi_mq: float,
    nb_mq: int,
    max_reads_per_genome: Optional[int],
    verbose: bool,
) -> tuple[int, np.ndarray, np.ndarray, int]:
    chunk_paths = [Path(x) for x in chunk_files]
    files = [Path(x) for x in genome_files]

    # Build BC -> chunk routing
    bc_to_chunk: dict[str, int] = {}
    per_chunk_acc: list[dict[tuple[str, str], ReadAcc]] = [dict() for _ in chunk_paths]

    for i, ch in enumerate(chunk_paths):
        for bc in _chunk_bcs(ch):
            bc_to_chunk[bc] = i

    batch_bcs = set(bc_to_chunk.keys())
    if verbose:
        _log(f"[assign/edges] ▶ batch {bi} chunks={len(chunk_paths)} BCs={len(batch_bcs):,}", True)

    # Stream each genome file once
    for fp in files:
        genome = _genome_from_filename(fp)
        kept = 0
        for read, bc, AS, MQ, NM in _iter_rows(
            fp, batch_bcs, chunksize, mapq_min, xa_max, max_rows_for_genome=max_reads_per_genome
        ):
            idx = bc_to_chunk.get(bc)
            if idx is None:
                continue
            rb = per_chunk_acc[idx]
            key = (read, bc)
            acc = rb.get(key)
            if acc is None:
                acc = ReadAcc()
                rb[key] = acc
            update_acc(acc, genome, AS, MQ, NM)
            kept += 1
        if verbose:
            _log(f"[assign/edges]    ✓ genome {genome} kept≈{kept:,}", True)

    # Reduce winners -> hist counts
    as_counts = np.zeros(nb_as, dtype=np.int64)
    mq_counts = np.zeros(nb_mq, dtype=np.int64)

    scale_as = hi_as - lo_as
    scale_mq = hi_mq - lo_mq

    winners = 0
    for rb in per_chunk_acc:
        for acc in rb.values():
            if acc.best1_g is None:
                continue
            asv = acc.best1_as
            mqv = acc.best1_mq

            if np.isfinite(asv):
                if asv <= lo_as:
                    idx_as = 0
                elif asv >= hi_as:
                    idx_as = nb_as - 1
                else:
                    idx_as = int((asv - lo_as) / scale_as * nb_as)
                    if idx_as < 0:
                        idx_as = 0
                    elif idx_as >= nb_as:
                        idx_as = nb_as - 1
                as_counts[idx_as] += 1

            if np.isfinite(mqv):
                if mqv <= lo_mq:
                    idx_mq = 0
                elif mqv >= hi_mq:
                    idx_mq = nb_mq - 1
                else:
                    idx_mq = int((mqv - lo_mq) / scale_mq * nb_mq)
                    if idx_mq < 0:
                        idx_mq = 0
                    elif idx_mq >= nb_mq:
                        idx_mq = nb_mq - 1
                mq_counts[idx_mq] += 1

            winners += 1

    return bi, as_counts, mq_counts, winners


# -----------------------------
# Worker: learn Δ hist counts for a single chunk (safe parallel unit)
# -----------------------------
def _ecdf_chunk_worker(
    chunk_file: str,
    genome_files: list[str],
    mapq_min: int,
    xa_max: int,
    chunksize: int,
    as_edges: np.ndarray,
    mq_edges: np.ndarray,
    k: int,
    verbose: bool,
) -> tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    ch = Path(chunk_file)
    files = [Path(x) for x in genome_files]
    bcs = _chunk_bcs(ch)

    H_dAS = [_new_hist(_DAS_RANGE) for _ in range(k)]
    H_dMQ = [_new_hist(_DMQ_RANGE) for _ in range(k)]

    rb_map: dict[tuple[str, str], ReadAcc] = {}

    if verbose:
        _log(f"[assign/ecdf] ▶ map {ch.name} BCs={len(bcs):,}", True)

    for fp in files:
        genome = _genome_from_filename(fp)
        for read, bc, AS, MQ, NM in _iter_rows(fp, bcs, chunksize, mapq_min, xa_max):
            key = (read, bc)
            acc = rb_map.get(key)
            if acc is None:
                acc = ReadAcc()
                rb_map[key] = acc
            update_acc(acc, genome, AS, MQ, NM)

    pairs = 0
    for acc in rb_map.values():
        if acc.best1_g is None or acc.best2_g is None:
            continue

        dAS = acc.best1_as - acc.best2_as
        dMQ = acc.best1_mq - acc.best2_mq

        dec_as = assign_decile_scalar(acc.best1_as, as_edges)
        dec_mq = assign_decile_scalar(acc.best1_mq, mq_edges)

        if 1 <= dec_as <= k:
            H_dAS[dec_as - 1].add(dAS)
        if 1 <= dec_mq <= k:
            H_dMQ[dec_mq - 1].add(dMQ)
        pairs += 1

    dAS_counts = np.stack([h.counts for h in H_dAS], axis=0)
    dAS_over = np.array([h.overflow for h in H_dAS], dtype=np.int64)

    dMQ_counts = np.stack([h.counts for h in H_dMQ], axis=0)
    dMQ_over = np.array([h.overflow for h in H_dMQ], dtype=np.int64)

    return ch.name, dAS_counts, dAS_over, dMQ_counts, dMQ_over, pairs


# -----------------------------
# Pass A: learn edges (batched parallel)
# -----------------------------
@app.command()
def learn_edges(
    workdir: Path = typer.Option(..., "--workdir"),
    sample: str = typer.Option(..., "--sample"),
    chunks_dir: Path = typer.Option(..., "--chunks-dir"),
    out_model: Optional[Path] = typer.Option(None, "--out-model"),
    mapq_min: int = typer.Option(20, "--mapq-min"),
    xa_max: int = typer.Option(2, "--xa-max"),
    chunksize: int = typer.Option(500_000, "--chunksize"),
    k: int = typer.Option(10, "--k"),
    batch_size: int = typer.Option(16, "--batch-size"),
    workers: int = typer.Option(4, "--workers"),
    max_reads_per_genome: Optional[int] = typer.Option(None, "--max-reads-per-genome"),
    edges_subsample: int = typer.Option(
        50_000,
        "--edges-subsample",
        help="Subsample N BCs for decile learning (0 = use all). Default: 50000.",
    ),
    edges_duckdb: bool = typer.Option(
        True,
        "--edges-duckdb/--edges-no-duckdb",
        help="Use DuckDB for Pass A (faster). Falls back to Python if duckdb unavailable.",
    ),
    duckdb_threads: int = typer.Option(
        4,
        "--edges-duckdb-threads",
        help="DuckDB threads for Pass A edge learning (default: 4).",
    ),
    verbose: bool = typer.Option(True, "--verbose/--quiet"),
):
    """
    Learn global AS/MAPQ decile edges from winner distributions.

    By default, subsamples 50 000 barcodes and uses DuckDB to scan each genome
    file exactly once.  Set --edges-subsample 0 to use all barcodes (legacy).
    """
    files = _filtered_files(workdir, sample)
    chunk_files = sorted(Path(chunks_dir).glob("*_cell_map_ref_chunk_*.txt"))
    if not chunk_files:
        raise typer.BadParameter(f"No chunk files under {chunks_dir}")

    lo_as, hi_as, nb_as = _AS_RANGE
    lo_mq, hi_mq, nb_mq = _MQ_RANGE

    t0 = time.time()

    # ---- Collect all BCs and optionally subsample ----
    all_bcs: set = set()
    for cf in chunk_files:
        all_bcs.update(_chunk_bcs(cf))

    if edges_subsample > 0 and len(all_bcs) > edges_subsample:
        import random
        rng = random.Random(42)
        sampled_bcs = set(rng.sample(sorted(all_bcs), edges_subsample))
        _log(
            f"[assign/edges] subsampled {len(sampled_bcs):,} / {len(all_bcs):,} BCs (seed=42)",
            verbose,
        )
    else:
        sampled_bcs = all_bcs
        _log(
            f"[assign/edges] using all {len(all_bcs):,} BCs (no subsampling needed)",
            verbose,
        )

    # ---- DuckDB fast path ----
    if edges_duckdb and _HAS_DUCKDB:
        _log(
            f"[assign/edges] DuckDB fast path: {len(files)} genome files, "
            f"{duckdb_threads} threads, {len(sampled_bcs):,} BCs",
            verbose,
        )
        as_counts, mq_counts, winners = _learn_edges_duckdb(
            files, sampled_bcs, mapq_min, xa_max, duckdb_threads,
            lo_as, hi_as, nb_as, lo_mq, hi_mq, nb_mq, verbose,
        )
        _log_ok(
            f"[assign/edges] DuckDB done: winners={winners:,} ({time.time()-t0:.1f}s)",
            verbose,
        )

    # ---- Python fallback (single batch with sampled BCs) ----
    else:
        if edges_duckdb and not _HAS_DUCKDB:
            _log_err(
                "[assign/edges] duckdb not installed — falling back to Python path",
                True,
            )

        import tempfile
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="edges_bcs_",
        )
        tmp.write("\n".join(sorted(sampled_bcs)))
        tmp.close()
        tf_path = tmp.name

        _log(
            f"[assign/edges] Python path: 1 batch, {len(files)} genome files, "
            f"{len(sampled_bcs):,} BCs",
            verbose,
        )

        try:
            _, as_counts, mq_counts, winners = _edges_batch_worker(
                1,
                [tf_path],
                [str(p) for p in files],
                mapq_min,
                xa_max,
                chunksize,
                lo_as, hi_as, nb_as,
                lo_mq, hi_mq, nb_mq,
                max_reads_per_genome,
                verbose,
            )
        finally:
            Path(tf_path).unlink(missing_ok=True)

        _log_ok(
            f"[assign/edges] Python done: winners={winners:,} ({time.time()-t0:.1f}s)",
            verbose,
        )

    # ---- Compute edges (unchanged) ----
    as_edges = edges_from_hist_counts(as_counts, lo_as, hi_as, k)
    mq_edges = edges_from_hist_counts(mq_counts, lo_mq, hi_mq, k)

    exp_dir = Path(workdir) / sample / "ExplorationReadLevel"
    exp_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_model or (exp_dir / "global_edges.npz")

    np.savez_compressed(model_path, as_edges=as_edges, mq_edges=mq_edges, k=np.array(k))
    _log_ok(f"[assign/edges] done → {model_path} ({time.time()-t0:0.1f}s)", verbose)
    return model_path


# -----------------------------
# Pass B: learn ECDFs (per-chunk parallel map-reduce)
# -----------------------------
@app.command()
def learn_ecdfs(
    workdir: Path = typer.Option(..., "--workdir"),
    sample: str = typer.Option(..., "--sample"),
    chunks_dir: Path = typer.Option(..., "--chunks-dir"),
    edges_model: Path = typer.Option(..., "--edges"),
    out_model: Optional[Path] = typer.Option(None, "--out-model"),
    mapq_min: int = typer.Option(20, "--mapq-min"),
    xa_max: int = typer.Option(2, "--xa-max"),
    chunksize: int = typer.Option(500_000, "--chunksize"),
    workers: int = typer.Option(4, "--workers"),
    ecdf_subsample: int = typer.Option(
        50_000,
        "--ecdf-subsample",
        help="Subsample N BCs for ECDF learning (0 = use all). Default: 50000.",
    ),
    ecdf_duckdb: bool = typer.Option(
        True,
        "--ecdf-duckdb/--ecdf-no-duckdb",
        help="Use DuckDB for Pass B (faster). Falls back to Python if duckdb unavailable.",
    ),
    ecdf_duckdb_threads: int = typer.Option(
        4,
        "--ecdf-duckdb-threads",
        help="DuckDB threads for Pass B ECDF learning (default: 4).",
    ),
    verbose: bool = typer.Option(True, "--verbose/--quiet"),
):
    """
    Learn per-decile ΔAS/ΔMAPQ hist counts + overflow.

    By default, subsamples 50 000 barcodes and uses DuckDB to scan each genome
    file exactly once.  Set --ecdf-subsample 0 to use all barcodes (legacy).
    """
    files = _filtered_files(workdir, sample)
    chunk_files = sorted(Path(chunks_dir).glob("*_cell_map_ref_chunk_*.txt"))
    if not chunk_files:
        raise typer.BadParameter(f"No chunk files under {chunks_dir}")

    dat = np.load(edges_model)
    as_edges = dat["as_edges"]
    mq_edges = dat["mq_edges"]
    k = int(dat["k"])

    lo_das, hi_das, nb_das = _DAS_RANGE
    lo_dmq, hi_dmq, nb_dmq = _DMQ_RANGE

    t0 = time.time()

    # ---- Collect all BCs and optionally subsample ----
    all_bcs: set = set()
    for cf in chunk_files:
        all_bcs.update(_chunk_bcs(cf))

    if ecdf_subsample > 0 and len(all_bcs) > ecdf_subsample:
        import random
        rng = random.Random(42)
        sampled_bcs = set(rng.sample(sorted(all_bcs), ecdf_subsample))
        _log(
            f"[assign/ecdf] subsampled {len(sampled_bcs):,} / {len(all_bcs):,} BCs (seed=42)",
            verbose,
        )
    else:
        sampled_bcs = all_bcs
        _log(
            f"[assign/ecdf] using all {len(all_bcs):,} BCs (no subsampling needed)",
            verbose,
        )

    # ---- DuckDB fast path ----
    if ecdf_duckdb and _HAS_DUCKDB:
        _log(
            f"[assign/ecdf] DuckDB fast path: {len(files)} genome files, "
            f"{ecdf_duckdb_threads} threads, {len(sampled_bcs):,} BCs",
            verbose,
        )
        dAS_counts, dAS_over, dMQ_counts, dMQ_over, n_pairs = _learn_ecdfs_duckdb(
            files, sampled_bcs, mapq_min, xa_max, ecdf_duckdb_threads,
            as_edges, mq_edges, k,
            lo_das, hi_das, nb_das, lo_dmq, hi_dmq, nb_dmq,
            verbose,
        )
        _log_ok(
            f"[assign/ecdf] DuckDB done: pairs={n_pairs:,} ({time.time()-t0:.1f}s)",
            verbose,
        )

    # ---- Python fallback (single batch with sampled BCs) ----
    else:
        if ecdf_duckdb and not _HAS_DUCKDB:
            _log_err(
                "[assign/ecdf] duckdb not installed — falling back to Python path",
                True,
            )

        import tempfile
        tmp = tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, prefix="ecdf_bcs_",
        )
        tmp.write("\n".join(sorted(sampled_bcs)))
        tmp.close()
        tf_path = tmp.name

        _log(
            f"[assign/ecdf] Python path: 1 chunk, {len(files)} genome files, "
            f"{len(sampled_bcs):,} BCs",
            verbose,
        )

        try:
            _, dAS_counts, dAS_over, dMQ_counts, dMQ_over, n_pairs = _ecdf_chunk_worker(
                tf_path,
                [str(p) for p in files],
                mapq_min,
                xa_max,
                chunksize,
                as_edges,
                mq_edges,
                k,
                verbose,
            )
        finally:
            Path(tf_path).unlink(missing_ok=True)

        _log_ok(
            f"[assign/ecdf] Python done: pairs={n_pairs:,} ({time.time()-t0:.1f}s)",
            verbose,
        )

    # ---- Save model (same format as before) ----
    exp_dir = Path(workdir) / sample / "ExplorationReadLevel"
    exp_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_model or (exp_dir / "global_ecdf.npz")

    np.savez_compressed(
        model_path,
        k=np.array(k),
        as_edges=as_edges,
        mq_edges=mq_edges,
        dAS_lo=np.array([lo_das]),
        dAS_hi=np.array([hi_das]),
        dAS_nbins=np.array([nb_das]),
        dMQ_lo=np.array([lo_dmq]),
        dMQ_hi=np.array([hi_dmq]),
        dMQ_nbins=np.array([nb_dmq]),
        dAS_counts=dAS_counts,
        dAS_overflow=dAS_over,
        dMQ_counts=dMQ_counts,
        dMQ_overflow=dMQ_over,
    )

    _log_ok(f"[assign/ecdf] done → {model_path} ({time.time()-t0:0.1f}s)", verbose)
    return model_path


def _load_ecdf_model(path: Path):
    dat = np.load(path)
    k = int(dat["k"])
    as_edges = dat["as_edges"]
    mq_edges = dat["mq_edges"]

    dAS = []
    dMQ = []

    for i in range(k):
        H = DeltaHist(float(dat["dAS_lo"][0]), float(dat["dAS_hi"][0]), int(dat["dAS_nbins"][0]))
        H.counts = dat["dAS_counts"][i].astype(np.int64)
        H.overflow = int(dat["dAS_overflow"][i])
        H.freeze()
        dAS.append(H)

    for i in range(k):
        H = DeltaHist(float(dat["dMQ_lo"][0]), float(dat["dMQ_hi"][0]), int(dat["dMQ_nbins"][0]))
        H.counts = dat["dMQ_counts"][i].astype(np.int64)
        H.overflow = int(dat["dMQ_overflow"][i])
        H.freeze()
        dMQ.append(H)

    return k, as_edges, mq_edges, dAS, dMQ


# -----------------------------
# Pass C: score one chunk (single-pass + streaming output)
# -----------------------------
@app.command()
def score_chunk(
    workdir: Path = typer.Option(..., "--workdir"),
    sample: str = typer.Option(..., "--sample"),
    chunk_file: Path = typer.Option(..., "--chunk-file"),
    ecdf_model: Path = typer.Option(..., "--model"),
    out_raw_dir: Optional[Path] = typer.Option(None, "--out-raw"),
    out_filtered_dir: Optional[Path] = typer.Option(None, "--out-filtered"),
    mapq_min: int = typer.Option(20, "--mapq-min"),
    xa_max: int = typer.Option(2, "--xa-max"),
    chunksize: int = typer.Option(500_000, "--chunksize"),
    alpha: float = typer.Option(0.05, "--alpha"),
    use_duckdb: bool = typer.Option(True, "--score-duckdb/--score-no-duckdb"),
    duckdb_threads: int = typer.Option(2, "--duckdb-threads"),
    verbose: bool = typer.Option(True, "--verbose/--quiet"),
):
    """
    Score a single chunk:
      - build per-(Read,BC) best1/best2/worst across genomes
      - compute p-values for ΔAS/ΔMAPQ using global ECDF model
      - write raw per-read summary and filtered per-read/per-genome table

    Filtering rule (compatible with your current behavior):
      - Always keep winner genome rows
      - Keep all genome rows for ambiguous reads (so downstream can see alternatives)
    """
    files = _filtered_files(workdir, sample)
    k, as_edges, mq_edges, H_dAS, H_dMQ = _load_ecdf_model(ecdf_model)

    out_raw_dir = out_raw_dir or (Path(workdir) / sample / "raw_cell_map_ref_chunks")
    out_filtered_dir = out_filtered_dir or (Path(workdir) / sample / "cell_map_ref_chunks")
    out_raw_dir.mkdir(parents=True, exist_ok=True)
    out_filtered_dir.mkdir(parents=True, exist_ok=True)

    tag = chunk_file.stem
    suff = tag.split("_cell_map_ref_chunk_")[-1] if "_cell_map_ref_chunk_" in tag else tag
    base = f"{sample}_chunk{suff}"
    raw_out = out_raw_dir / f"{base}_raw.tsv.gz"
    filt_out = out_filtered_dir / f"{base}_filtered.tsv.gz"

    bcs = _chunk_bcs(chunk_file)
    rb_map: dict[tuple[str, str], ReadAcc] = {}

    _log(f"[assign/score] ▶ {chunk_file.name} BCs={len(bcs):,}", verbose)
    t0 = time.time()

    # ----------------------------------------------------------------
    # DuckDB fast path (Pass C)
    # ----------------------------------------------------------------
    if use_duckdb and _HAS_DUCKDB:
        top3_df, lazy_ambig = _score_chunk_duckdb(
            files, bcs, mapq_min, xa_max, duckdb_threads
        )
        if top3_df.empty:
            pd.DataFrame().to_csv(raw_out, sep="\t", index=False, compression="gzip")
            pd.DataFrame().to_csv(filt_out, sep="\t", index=False, compression="gzip")
            _log_ok(f"[assign/score] ■ empty chunk → wrote empty outputs", verbose)
            return

        best1 = (top3_df[top3_df["_rn"] == 1]
                 .drop_duplicates(["Read", "BC"])
                 .set_index(["Read", "BC"]))
        best2 = (top3_df[top3_df["_rn"] == 2]
                 .drop_duplicates(["Read", "BC"])
                 .set_index(["Read", "BC"]))
        worst = (top3_df[top3_df["_rn_worst"] == 1]
                 .drop_duplicates(["Read", "BC"])
                 .set_index(["Read", "BC"]))

        b1 = best1.join(
            best2[["Genome", "as_", "MAPQ", "NM"]].rename(
                columns={"Genome": "Genome_2", "as_": "as_2", "MAPQ": "MAPQ_2", "NM": "NM_2"}
            ),
            how="left",
        ).join(
            worst[["Genome", "as_", "MAPQ", "NM"]].rename(
                columns={"Genome": "Genome_last", "as_": "as_last", "MAPQ": "MAPQ_last", "NM": "NM_last"}
            ),
            how="left",
        )

        as1_arr = b1["as_"].values.astype(np.float64)
        mq1_arr = b1["MAPQ"].values.astype(np.float64)
        nm1_arr = b1["NM"].values.astype(np.float64)
        as2_arr = b1["as_2"].values.astype(np.float64)
        mq2_arr = b1["MAPQ_2"].values.astype(np.float64)
        nm2_arr = b1["NM_2"].values.astype(np.float64)

        dAS_arr = as1_arr - as2_arr
        dMQ_arr = mq1_arr - mq2_arr
        dNM_arr = nm2_arr - nm1_arr

        if as_edges.size > 0:
            dec_as = np.clip(
                np.searchsorted(as_edges, as1_arr, side="left") + 1, 1, k
            ).astype(np.int64)
        else:
            dec_as = np.ones(len(b1), dtype=np.int64)

        if mq_edges.size > 0:
            dec_mq = np.clip(
                np.searchsorted(mq_edges, mq1_arr, side="left") + 1, 1, k
            ).astype(np.int64)
        else:
            dec_mq = np.ones(len(b1), dtype=np.int64)

        # p-value loop — O(N_reads), not O(N_reads × G_genomes)
        p_as_arr = np.array(
            [H_dAS[int(d) - 1].tail_p(float(v)) if np.isfinite(v) else np.nan
             for v, d in zip(dAS_arr, dec_as)],
            dtype=np.float64,
        )
        p_mq_arr = np.array(
            [H_dMQ[int(d) - 1].tail_p(float(v)) if np.isfinite(v) else np.nan
             for v, d in zip(dMQ_arr, dec_mq)],
            dtype=np.float64,
        )

        has_best2 = ~np.isnan(as2_arr)
        klass_arr = np.where(
            (np.isfinite(p_as_arr) & (p_as_arr <= alpha))
            | (np.isfinite(p_mq_arr) & (p_mq_arr <= alpha))
            | (np.isfinite(dNM_arr) & (dNM_arr > 0))
            | ~has_best2,
            "winner",
            "ambiguous",
        )

        read_idx = b1.index.get_level_values("Read")
        bc_idx   = b1.index.get_level_values("BC")

        raw_df = pd.DataFrame({
            "Read":                 read_idx,
            "BC":                   bc_idx,
            "Genome_winner":        b1["Genome"].values,
            "AS_winner":            as1_arr.astype(np.float32),
            "MAPQ_winner":          mq1_arr.astype(np.float32),
            "NM_winner":            nm1_arr.astype(np.float32),
            "Genome_2":             b1["Genome_2"].fillna("").values,
            "AS_2":                 as2_arr.astype(np.float32),
            "MAPQ_2":               mq2_arr.astype(np.float32),
            "NM_2":                 nm2_arr.astype(np.float32),
            "Genome_last":          b1["Genome_last"].fillna("").values,
            "AS_last":              b1["as_last"].values.astype(np.float32),
            "MAPQ_last":            b1["MAPQ_last"].values.astype(np.float32),
            "NM_last":              b1["NM_last"].values.astype(np.float32),
            "delta_AS_1_2":         dAS_arr.astype(np.float32),
            "delta_MAPQ_1_2":       dMQ_arr.astype(np.float32),
            "delta_NM_1_2":         dNM_arr.astype(np.float32),
            "n_genomes_considered": b1["n_genomes"].values.astype(np.int32),
            "decile_AS":            dec_as.astype(np.int16),
            "decile_MAPQ":          dec_mq.astype(np.int16),
            "p_as":                 p_as_arr.astype(np.float32),
            "p_mq":                 p_mq_arr.astype(np.float32),
            "assigned_class":       klass_arr,
        })
        raw_df.to_csv(raw_out, sep="\t", index=False, compression="gzip")

        # Filtered output: winners from b1; ambiguous via lazy Query 2
        win_mask = klass_arr == "winner"
        wb1 = b1[win_mask]
        p_as_s = pd.Series(p_as_arr, index=b1.index)
        p_mq_s = pd.Series(p_mq_arr, index=b1.index)

        winner_df = pd.DataFrame({
            "Read":           wb1.index.get_level_values("Read"),
            "BC":             wb1.index.get_level_values("BC"),
            "Genome":         wb1["Genome"].values,
            "AS":             wb1["as_"].values.astype(np.float32),
            "MAPQ":           wb1["MAPQ"].values.astype(np.float32),
            "NM":             wb1["NM"].values.astype(np.float32),
            "XAcount":        wb1["XAcount"].values.astype(np.float32),
            "assigned_class": "winner",
            "p_as":           p_as_s[win_mask].values.astype(np.float32),
            "p_mq":           p_mq_s[win_mask].values.astype(np.float32),
        })

        ambiguous_bcs_d = set(bc_idx[klass_arr == "ambiguous"].tolist())
        ambig_raw = lazy_ambig(ambiguous_bcs_d)
        if not ambig_raw.empty:
            ambig_idx = pd.MultiIndex.from_arrays(
                [ambig_raw["Read"].values, ambig_raw["BC"].values]
            )
            ambig_df = ambig_raw.rename(columns={"as_": "AS"}).copy()
            ambig_df["assigned_class"] = "ambiguous"
            ambig_df["p_as"] = p_as_s.reindex(ambig_idx).values.astype(np.float32)
            ambig_df["p_mq"] = p_mq_s.reindex(ambig_idx).values.astype(np.float32)
            ambig_df = ambig_df[["Read", "BC", "Genome", "AS", "MAPQ", "NM",
                                  "XAcount", "assigned_class", "p_as", "p_mq"]]
        else:
            ambig_df = pd.DataFrame(columns=winner_df.columns)

        out_df = pd.concat([winner_df, ambig_df], ignore_index=True)
        out_df.to_csv(filt_out, sep="\t", index=False, compression="gzip")

        _log_ok(
            f"[assign/score] ■ done {chunk_file.name} ({time.time()-t0:0.1f}s) "
            f"[duckdb] raw={raw_out.name} filt={filt_out.name}",
            verbose,
        )
        return

    if use_duckdb and not _HAS_DUCKDB:
        _log_err("[assign/score] duckdb not installed — falling back to Python path", True)

    # Pass 1: accumulate best1/best2/worst across all genome files; best1_xa tracks XAcount
    # of the winner alignment. per_read_genome is NOT built here — winners are reconstructed
    # directly from rb_map; ambiguous reads get a targeted second scan below.
    for fp in files:
        genome = _genome_from_filename(fp)
        for read, bc, AS, MQ, NM, XA, gname in _iter_rows_with_genome_xa(
            fp, genome, bcs, chunksize, mapq_min, xa_max
        ):
            key = (read, bc)
            acc = rb_map.get(key)
            if acc is None:
                acc = ReadAcc()
                rb_map[key] = acc
            update_acc(acc, gname, AS, MQ, NM, xa=XA)

    if not rb_map:
        # Write empty outputs
        pd.DataFrame().to_csv(raw_out, sep="\t", index=False, compression="gzip")
        pd.DataFrame().to_csv(filt_out, sep="\t", index=False, compression="gzip")
        _log_ok(f"[assign/score] ■ empty chunk → wrote empty outputs", verbose)
        return

    # Build raw summary table using per-column lists (avoids dict-per-row overhead)
    col_read: list[str] = []
    col_bc: list[str] = []
    col_gwin: list[str] = []
    col_as1: list[float] = []
    col_mq1: list[float] = []
    col_nm1: list[float] = []
    col_g2: list[str] = []
    col_as2: list[float] = []
    col_mq2: list[float] = []
    col_nm2: list[float] = []
    col_glast: list[str] = []
    col_aslast: list[float] = []
    col_mqlast: list[float] = []
    col_nmlast: list[float] = []
    col_das: list[float] = []
    col_dmq: list[float] = []
    col_dnm: list[float] = []
    col_ngen: list[int] = []
    col_decas: list[int] = []
    col_decmq: list[int] = []
    col_pas: list[float] = []
    col_pmq: list[float] = []
    col_klass: list[str] = []

    class_by: dict[tuple[str, str], str] = {}
    p_lookup: dict[tuple[str, str], tuple[float, float]] = {}
    gwin_by: dict[tuple[str, str], str] = {}

    for (read, bc), acc in rb_map.items():
        if acc.best1_g is None:
            continue

        as1, mq1, nm1 = acc.best1_as, acc.best1_mq, acc.best1_nm

        if acc.best2_g is not None:
            as2, mq2, nm2 = acc.best2_as, acc.best2_mq, acc.best2_nm
            dAS = as1 - as2
            dMQ = mq1 - mq2
            dNM = nm2 - nm1
        else:
            as2 = mq2 = nm2 = np.nan
            dAS = dMQ = dNM = np.nan

        # Inline searchsorted (faster than wrapping scalar in list for np.digitize)
        dec_as = (int(np.searchsorted(as_edges, as1, side="left") + 1)
                  if as_edges.size > 0 and np.isfinite(as1) else 1)
        dec_mq = (int(np.searchsorted(mq_edges, mq1, side="left") + 1)
                  if mq_edges.size > 0 and np.isfinite(mq1) else 1)

        p_as_val = H_dAS[dec_as - 1].tail_p(dAS) if np.isfinite(dAS) else np.nan
        p_mq_val = H_dMQ[dec_mq - 1].tail_p(dMQ) if np.isfinite(dMQ) else np.nan

        clear_by_p = (np.isfinite(p_as_val) and p_as_val <= alpha) or \
                     (np.isfinite(p_mq_val) and p_mq_val <= alpha)
        klass = "winner" if (clear_by_p or (np.isfinite(dNM) and dNM > 0) or acc.best2_g is None) \
                else "ambiguous"

        key = (read, bc)
        class_by[key] = klass
        p_lookup[key] = (float(p_as_val) if np.isfinite(p_as_val) else np.nan,
                         float(p_mq_val) if np.isfinite(p_mq_val) else np.nan)
        gwin_by[key] = acc.best1_g

        col_read.append(read)
        col_bc.append(bc)
        col_gwin.append(acc.best1_g)
        col_as1.append(float(as1))
        col_mq1.append(float(mq1))
        col_nm1.append(float(nm1))
        col_g2.append(acc.best2_g or "")
        col_as2.append(float(as2) if np.isfinite(as2) else np.nan)
        col_mq2.append(float(mq2) if np.isfinite(mq2) else np.nan)
        col_nm2.append(float(nm2) if np.isfinite(nm2) else np.nan)
        col_glast.append(acc.worst_g or "")
        col_aslast.append(float(acc.worst_as) if np.isfinite(acc.worst_as) else np.nan)
        col_mqlast.append(float(acc.worst_mq) if np.isfinite(acc.worst_mq) else np.nan)
        col_nmlast.append(float(acc.worst_nm) if np.isfinite(acc.worst_nm) else np.nan)
        col_das.append(float(dAS) if np.isfinite(dAS) else np.nan)
        col_dmq.append(float(dMQ) if np.isfinite(dMQ) else np.nan)
        col_dnm.append(float(dNM) if np.isfinite(dNM) else np.nan)
        col_ngen.append(acc.n_genomes)
        col_decas.append(dec_as)
        col_decmq.append(dec_mq)
        col_pas.append(float(p_as_val) if np.isfinite(p_as_val) else np.nan)
        col_pmq.append(float(p_mq_val) if np.isfinite(p_mq_val) else np.nan)
        col_klass.append(klass)

    raw_df = pd.DataFrame({
        "Read": col_read, "BC": col_bc,
        "Genome_winner": col_gwin, "AS_winner": col_as1, "MAPQ_winner": col_mq1, "NM_winner": col_nm1,
        "Genome_2": col_g2, "AS_2": col_as2, "MAPQ_2": col_mq2, "NM_2": col_nm2,
        "Genome_last": col_glast, "AS_last": col_aslast, "MAPQ_last": col_mqlast, "NM_last": col_nmlast,
        "delta_AS_1_2": col_das, "delta_MAPQ_1_2": col_dmq, "delta_NM_1_2": col_dnm,
        "n_genomes_considered": col_ngen, "decile_AS": col_decas, "decile_MAPQ": col_decmq,
        "p_as": col_pas, "p_mq": col_pmq, "assigned_class": col_klass,
    })
    raw_df.to_csv(raw_out, sep="\t", index=False, compression="gzip")

    # Filtered per-read/per-genome table.
    # Winners: reconstructed directly from rb_map (no second file scan).
    #   best1_xa captured best1's XAcount during Pass 1.
    # Ambiguous: requires all genome rows per read → Pass 2, restricted to ambiguous BCs only
    #   (typically 10–30% of BCs), so the second scan is fast.
    f_read: list[str] = []
    f_bc: list[str] = []
    f_genome: list[str] = []
    f_as: list[float] = []
    f_mq: list[float] = []
    f_nm: list[float] = []
    f_xa: list[float] = []
    f_klass: list[str] = []
    f_pas: list[float] = []
    f_pmq: list[float] = []

    # --- Winners: reconstruct from rb_map ---
    for (read, bc), acc in rb_map.items():
        key = (read, bc)
        if class_by.get(key) != "winner":
            continue
        pa, pm = p_lookup[key]
        f_read.append(read)
        f_bc.append(bc)
        f_genome.append(acc.best1_g)
        f_as.append(float(acc.best1_as))
        f_mq.append(float(acc.best1_mq))
        f_nm.append(float(acc.best1_nm))
        f_xa.append(float(acc.best1_xa))
        f_klass.append("winner")
        f_pas.append(pa)
        f_pmq.append(pm)

    # --- Ambiguous: Pass 2 over genome files, BC-filtered to ambiguous reads only ---
    ambiguous_bcs = {bc for (_, bc), klass in class_by.items() if klass == "ambiguous"}
    if ambiguous_bcs:
        per_read_genome: dict[tuple[str, str], list[tuple[str, float, float, float, float]]] = {}
        for fp in files:
            genome = _genome_from_filename(fp)
            for read, bc, AS, MQ, NM, XA, gname in _iter_rows_with_genome_xa(
                fp, genome, ambiguous_bcs, chunksize, mapq_min, xa_max
            ):
                key = (read, bc)
                if class_by.get(key) != "ambiguous":
                    # guard: same BC shared between an ambiguous and a winner read
                    continue
                grecs = per_read_genome.get(key)
                if grecs is None:
                    grecs = []
                    per_read_genome[key] = grecs
                grecs.append((gname, AS, MQ, NM, XA))

        for key, grecs in per_read_genome.items():
            pa, pm = p_lookup[key]
            for gname, AS, MQ, NM, XA in grecs:
                f_read.append(key[0])
                f_bc.append(key[1])
                f_genome.append(gname)
                f_as.append(AS)
                f_mq.append(MQ)
                f_nm.append(NM)
                f_xa.append(XA)
                f_klass.append("ambiguous")
                f_pas.append(pa)
                f_pmq.append(pm)

    out_df = pd.DataFrame({
        "Read": f_read, "BC": f_bc, "Genome": f_genome,
        "AS": f_as, "MAPQ": f_mq, "NM": f_nm, "XAcount": f_xa,
        "assigned_class": f_klass, "p_as": f_pas, "p_mq": f_pmq,
    })
    out_df.to_csv(filt_out, sep="\t", index=False, compression="gzip")

    _log_ok(f"[assign/score] ■ done {chunk_file.name} ({time.time()-t0:0.1f}s) raw={raw_out.name} filt={filt_out.name}", verbose)


# -----------------------------
# Convenience pipeline: edges -> ecdfs -> score all chunks
# -----------------------------
@app.command()
def assign_streaming_pipeline(
    workdir: Path = typer.Option(..., "--workdir"),
    sample: str = typer.Option(..., "--sample"),
    chunks_dir: Path = typer.Option(..., "--chunks-dir"),
    alpha: float = typer.Option(0.05, "--alpha"),
    mapq_min: int = typer.Option(20, "--mapq-min"),
    xa_max: int = typer.Option(2, "--xa-max"),
    chunksize: int = typer.Option(500_000, "--chunksize"),
    k: int = typer.Option(10, "--k"),
    batch_size: int = typer.Option(16, "--batch-size"),
    workers: int = typer.Option(4, "--workers"),
    max_reads_per_genome: Optional[int] = typer.Option(None, "--max-reads-per-genome"),
    edges_subsample: int = typer.Option(
        50_000,
        "--edges-subsample",
        help="Subsample N BCs for decile learning (0 = use all). Default: 50000.",
    ),
    edges_duckdb: bool = typer.Option(
        True,
        "--edges-duckdb/--edges-no-duckdb",
        help="Use DuckDB for Pass A (faster). Falls back to Python if duckdb unavailable.",
    ),
    edges_duckdb_threads: int = typer.Option(
        4,
        "--edges-duckdb-threads",
        help="DuckDB threads for Pass A edge learning (default: 4).",
    ),
    ecdf_subsample: int = typer.Option(
        50_000,
        "--ecdf-subsample",
        help="Subsample N BCs for ECDF learning (0 = use all). Default: 50000.",
    ),
    ecdf_duckdb: bool = typer.Option(
        True,
        "--ecdf-duckdb/--ecdf-no-duckdb",
        help="Use DuckDB for Pass B (faster). Falls back to Python if duckdb unavailable.",
    ),
    ecdf_duckdb_threads: int = typer.Option(
        4,
        "--ecdf-duckdb-threads",
        help="DuckDB threads for Pass B ECDF learning (default: 4).",
    ),
    verbose: bool = typer.Option(True, "--verbose/--quiet"),
):
    """
    Convenience:
      - learn edges if missing
      - learn ecdfs if missing
      - score all chunks serially (safe) OR you can parallelize externally via SLURM array
    """
    exp_dir = Path(workdir) / sample / "ExplorationReadLevel"
    exp_dir.mkdir(parents=True, exist_ok=True)
    edges_path = exp_dir / "global_edges.npz"
    ecdf_path = exp_dir / "global_ecdf.npz"

    if not edges_path.exists():
        _log(f"[assign] learn edges → {edges_path}", verbose)
        learn_edges(  # type: ignore
            workdir=workdir,
            sample=sample,
            chunks_dir=chunks_dir,
            out_model=edges_path,
            mapq_min=mapq_min,
            xa_max=xa_max,
            chunksize=chunksize,
            k=k,
            batch_size=batch_size,
            workers=workers,
            max_reads_per_genome=max_reads_per_genome,
            edges_subsample=edges_subsample,
            edges_duckdb=edges_duckdb,
            duckdb_threads=edges_duckdb_threads,
            verbose=verbose,
        )
    else:
        _log(f"[assign] reuse edges model: {edges_path}", verbose)

    if not ecdf_path.exists():
        _log(f"[assign] learn ECDFs → {ecdf_path}", verbose)
        learn_ecdfs(  # type: ignore
            workdir=workdir,
            sample=sample,
            chunks_dir=chunks_dir,
            edges_model=edges_path,
            out_model=ecdf_path,
            mapq_min=mapq_min,
            xa_max=xa_max,
            chunksize=chunksize,
            workers=workers,
            ecdf_subsample=ecdf_subsample,
            ecdf_duckdb=ecdf_duckdb,
            ecdf_duckdb_threads=ecdf_duckdb_threads,
            verbose=verbose,
        )
    else:
        _log(f"[assign] reuse ECDF model: {ecdf_path}", verbose)

    chunk_files = sorted(Path(chunks_dir).glob("*_cell_map_ref_chunk_*.txt"))
    if not chunk_files:
        raise typer.BadParameter(f"No chunk files under {chunks_dir}")

    _log(f"[assign/score] scoring {len(chunk_files)} chunks (serial). For parallel, run score-chunk-v2 in SLURM array.", verbose)

    for ch in chunk_files:
        score_chunk(  # type: ignore
            workdir=workdir,
            sample=sample,
            chunk_file=ch,
            ecdf_model=ecdf_path,
            out_raw_dir=None,
            out_filtered_dir=None,
            mapq_min=mapq_min,
            xa_max=xa_max,
            chunksize=chunksize,
            alpha=alpha,
            verbose=verbose,
        )


if __name__ == "__main__":
    app()
