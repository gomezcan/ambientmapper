#!/usr/bin/env python3
"""
assign_streaming_v2.py

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


# -----------------------------
# Histogram + tail-prob model
# -----------------------------
class DeltaHist:
    """
    Fixed-bin histogram with overflow bucket and tail-prob query.
    tail_p(x) = P(X >= x) under the empirical distribution.
    """

    __slots__ = ("lo", "hi", "nbins", "edges", "counts", "overflow")

    def __init__(self, lo: float, hi: float, nbins: int):
        self.lo = float(lo)
        self.hi = float(hi)
        self.nbins = int(nbins)
        self.edges = np.linspace(self.lo, self.hi, self.nbins + 1, dtype=np.float64)
        self.counts = np.zeros(self.nbins, dtype=np.int64)
        self.overflow = 0

    def add(self, x: float) -> None:
        if not np.isfinite(x):
            return
        if x >= self.hi:
            self.overflow += 1
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

    def tail_p(self, x: float) -> float:
        if not np.isfinite(x):
            return np.nan
        total = int(self.counts.sum() + self.overflow)
        if total == 0:
            return np.nan
        if x >= self.hi:
            return self.overflow / total
        if x <= self.lo:
            return 1.0
        idx = int((x - self.lo) / (self.hi - self.lo) * self.nbins)
        if idx < 0:
            idx = 0
        elif idx >= self.nbins:
            idx = self.nbins - 1
        tail = int(self.counts[idx:].sum() + self.overflow)
        return tail / total


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
    # np.digitize returns 0..len(edges)
    return int(np.digitize([v], bins=edges, right=True)[0] + 1)


# -----------------------------
# Lightweight per-(Read,BC) accumulator
# -----------------------------
@dataclass(slots=True)
class ReadAcc:
    best1_g: Optional[str] = None
    best1_as: float = -np.inf
    best1_mq: float = -np.inf
    best1_nm: float = np.inf

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


def update_acc(acc: ReadAcc, genome: str, AS: float, MAPQ: float, NM: float) -> None:
    acc.n_genomes += 1

    if acc.best1_g is None:
        acc.best1_g = genome
        acc.best1_as = AS
        acc.best1_mq = MAPQ
        acc.best1_nm = NM

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
        if df.duplicated(subset=["Read", "BC"]).any():
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
def learn_edges_v2(
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
    verbose: bool = typer.Option(True, "--verbose/--quiet"),
):
    """
    Learn global AS/MAPQ decile edges from winner distributions.
    Parallelization: chunk batches across workers; each worker streams each genome file once.
    """
    files = _filtered_files(workdir, sample)
    chunk_files = sorted(Path(chunks_dir).glob("*_cell_map_ref_chunk_*.txt"))
    if not chunk_files:
        raise typer.BadParameter(f"No chunk files under {chunks_dir}")

    lo_as, hi_as, nb_as = _AS_RANGE
    lo_mq, hi_mq, nb_mq = _MQ_RANGE

    batches: list[tuple[int, list[Path]]] = []
    for bi, ofs in enumerate(range(0, len(chunk_files), batch_size), start=1):
        batches.append((bi, chunk_files[ofs : ofs + batch_size]))

    as_counts = np.zeros(nb_as, dtype=np.int64)
    mq_counts = np.zeros(nb_mq, dtype=np.int64)

    _log(f"[assign/edges] start: chunks={len(chunk_files)} batches={len(batches)} workers={workers}", verbose)

    max_workers = max(1, min(int(workers), len(batches)))
    t0 = time.time()

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = []
        for bi, chs in batches:
            futs.append(
                ex.submit(
                    _edges_batch_worker,
                    bi,
                    [str(p) for p in chs],
                    [str(p) for p in files],
                    mapq_min,
                    xa_max,
                    chunksize,
                    lo_as,
                    hi_as,
                    nb_as,
                    lo_mq,
                    hi_mq,
                    nb_mq,
                    max_reads_per_genome,
                    verbose,
                )
            )

        done = 0
        for fut in as_completed(futs):
            try:
                bi, as_part, mq_part, winners = fut.result()
            except Exception as e:
                _log_err(f"[assign/edges][ERROR] worker failed: {e}", True)
                raise

            as_counts += as_part
            mq_counts += mq_part
            done += 1
            _log_ok(f"[assign/edges] ■ batch {bi} reduced winners={winners:,} ({done}/{len(futs)})", verbose)

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
def learn_ecdfs_v2(
    workdir: Path = typer.Option(..., "--workdir"),
    sample: str = typer.Option(..., "--sample"),
    chunks_dir: Path = typer.Option(..., "--chunks-dir"),
    edges_model: Path = typer.Option(..., "--edges"),
    out_model: Optional[Path] = typer.Option(None, "--out-model"),
    mapq_min: int = typer.Option(20, "--mapq-min"),
    xa_max: int = typer.Option(2, "--xa-max"),
    chunksize: int = typer.Option(500_000, "--chunksize"),
    workers: int = typer.Option(4, "--workers"),
    verbose: bool = typer.Option(True, "--verbose/--quiet"),
):
    """
    Learn per-decile ΔAS/ΔMAPQ hist counts + overflow, in parallel over chunks.
    Each worker loads all genomes for *one* chunk => stable memory profile.
    """
    files = _filtered_files(workdir, sample)
    chunk_files = sorted(Path(chunks_dir).glob("*_cell_map_ref_chunk_*.txt"))
    if not chunk_files:
        raise typer.BadParameter(f"No chunk files under {chunks_dir}")

    dat = np.load(edges_model)
    as_edges = dat["as_edges"]
    mq_edges = dat["mq_edges"]
    k = int(dat["k"])

    # global accumulators
    Htmp_AS = _new_hist(_DAS_RANGE)
    Htmp_MQ = _new_hist(_DMQ_RANGE)

    dAS_counts = np.zeros((k, Htmp_AS.nbins), dtype=np.int64)
    dAS_over = np.zeros((k,), dtype=np.int64)

    dMQ_counts = np.zeros((k, Htmp_MQ.nbins), dtype=np.int64)
    dMQ_over = np.zeros((k,), dtype=np.int64)

    max_workers = max(1, min(int(workers), len(chunk_files)))
    _log(f"[assign/ecdf] start: chunks={len(chunk_files)} workers={max_workers}", verbose)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {
            ex.submit(
                _ecdf_chunk_worker,
                str(ch),
                [str(p) for p in files],
                mapq_min,
                xa_max,
                chunksize,
                as_edges,
                mq_edges,
                k,
                verbose,
            ): ch
            for ch in chunk_files
        }

        done = 0
        for fut in as_completed(futs):
            ch = futs[fut]
            try:
                _, das, daso, dmq, dmqo, pairs = fut.result()
            except Exception as e:
                _log_err(f"[assign/ecdf][ERROR] {ch.name}: {e}", True)
                raise

            dAS_counts += das
            dAS_over += daso
            dMQ_counts += dmq
            dMQ_over += dmqo

            done += 1
            if verbose and (done % 5 == 0 or done == len(chunk_files)):
                _log(f"[assign/ecdf] {done}/{len(chunk_files)} chunks reduced (last pairs={pairs:,})", True)

    exp_dir = Path(workdir) / sample / "ExplorationReadLevel"
    exp_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_model or (exp_dir / "global_ecdf.npz")

    np.savez_compressed(
        model_path,
        k=np.array(k),
        as_edges=as_edges,
        mq_edges=mq_edges,
        dAS_lo=np.array([Htmp_AS.lo]),
        dAS_hi=np.array([Htmp_AS.hi]),
        dAS_nbins=np.array([Htmp_AS.nbins]),
        dMQ_lo=np.array([Htmp_MQ.lo]),
        dMQ_hi=np.array([Htmp_MQ.hi]),
        dMQ_nbins=np.array([Htmp_MQ.nbins]),
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
        dAS.append(H)

    for i in range(k):
        H = DeltaHist(float(dat["dMQ_lo"][0]), float(dat["dMQ_hi"][0]), int(dat["dMQ_nbins"][0]))
        H.counts = dat["dMQ_counts"][i].astype(np.int64)
        H.overflow = int(dat["dMQ_overflow"][i])
        dMQ.append(H)

    return k, as_edges, mq_edges, dAS, dMQ


# -----------------------------
# Pass C: score one chunk (single-pass + streaming output)
# -----------------------------
@app.command()
def score_chunk_v2(
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

    # Accumulate best1/best2/worst across all genomes
    for fp in files:
        genome = _genome_from_filename(fp)
        for read, bc, AS, MQ, NM in _iter_rows(fp, bcs, chunksize, mapq_min, xa_max):
            key = (read, bc)
            acc = rb_map.get(key)
            if acc is None:
                acc = ReadAcc()
                rb_map[key] = acc
            update_acc(acc, genome, AS, MQ, NM)

    if not rb_map:
        # Write empty outputs
        pd.DataFrame().to_csv(raw_out, sep="\t", index=False, compression="gzip")
        pd.DataFrame().to_csv(filt_out, sep="\t", index=False, compression="gzip")
        _log_ok(f"[assign/score] ■ empty chunk → wrote empty outputs", verbose)
        return

    # Build raw summary table (vectorized-ish via list accumulation)
    raw_rows = []
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

        dec_as = assign_decile_scalar(as1, as_edges)
        dec_mq = assign_decile_scalar(mq1, mq_edges)

        p_as = H_dAS[dec_as - 1].tail_p(dAS) if np.isfinite(dAS) else np.nan
        p_mq = H_dMQ[dec_mq - 1].tail_p(dMQ) if np.isfinite(dMQ) else np.nan

        clear_by_p = (np.isfinite(p_as) and p_as <= alpha) or (np.isfinite(p_mq) and p_mq <= alpha)
        clear_by_nm = np.isfinite(dNM) and dNM > 0
        single_hit = acc.best2_g is None

        klass = "winner" if (clear_by_p or clear_by_nm or single_hit) else "ambiguous"

        key = (read, bc)
        class_by[key] = klass
        p_lookup[key] = (float(p_as) if np.isfinite(p_as) else np.nan, float(p_mq) if np.isfinite(p_mq) else np.nan)
        gwin_by[key] = acc.best1_g

        raw_rows.append(
            {
                "Read": read,
                "BC": bc,
                "Genome_winner": acc.best1_g,
                "AS_winner": as1,
                "MAPQ_winner": mq1,
                "NM_winner": nm1,
                "Genome_2": acc.best2_g or "",
                "AS_2": as2,
                "MAPQ_2": mq2,
                "NM_2": nm2,
                "Genome_last": acc.worst_g or "",
                "AS_last": acc.worst_as if np.isfinite(acc.worst_as) else np.nan,
                "MAPQ_last": acc.worst_mq if np.isfinite(acc.worst_mq) else np.nan,
                "NM_last": acc.worst_nm if np.isfinite(acc.worst_nm) else np.nan,
                "delta_AS_1_2": dAS,
                "delta_MAPQ_1_2": dMQ,
                "delta_NM_1_2": dNM,
                "n_genomes_considered": acc.n_genomes,
                "decile_AS": dec_as,
                "decile_MAPQ": dec_mq,
                "p_as": p_as,
                "p_mq": p_mq,
                "assigned_class": klass,
            }
        )

    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(raw_out, sep="\t", index=False, compression="gzip")

    # Filtered per-read/per-genome table:
    # Keep all genome rows for ambiguous reads; for winner reads keep only winner genome.
    keep_pairs = set(class_by.keys())
    bc_keep = {bc for (_, bc) in keep_pairs}

    out_rows = []
    usecols = ["Read", "BC", "MAPQ", "AS", "NM", "XAcount"]

    for fp in files:
        genome = _genome_from_filename(fp)

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
            df = df[df["BC"].isin(bc_keep)]
            if df.empty:
                continue

            if mapq_min > 0:
                df = df[df["MAPQ"] >= mapq_min]
            if xa_max >= 0:
                df = df[df["XAcount"] <= xa_max]
            if df.empty:
                continue

            # We only care about (Read,BC) pairs in rb_map
            # Build pairs vector; avoid Python list membership per-row by using merge keys
            reads = df["Read"].astype(str).to_numpy()
            bcs2 = df["BC"].astype(str).to_numpy()
            mask = np.fromiter(((r, b) in keep_pairs for r, b in zip(reads, bcs2)), dtype=bool, count=len(df))
            if not mask.any():
                continue
            df = df.loc[mask]

            for row in df.itertuples(index=False):
                read = str(row.Read)
                bc = str(row.BC)
                key = (read, bc)

                klass = class_by.get(key, "ambiguous")
                gwin = gwin_by.get(key, "")
                pa, pm = p_lookup.get(key, (np.nan, np.nan))

                # if winner-class, keep only the winner genome row
                if klass == "winner" and genome != gwin:
                    continue

                out_rows.append(
                    {
                        "Read": read,
                        "BC": bc,
                        "Genome": genome,
                        "AS": float(row.AS),
                        "MAPQ": float(row.MAPQ),
                        "NM": float(row.NM),
                        "XAcount": float(row.XAcount),
                        "assigned_class": klass,
                        "p_as": pa,
                        "p_mq": pm,
                    }
                )

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(filt_out, sep="\t", index=False, compression="gzip")

    _log_ok(f"[assign/score] ■ done {chunk_file.name} ({time.time()-t0:0.1f}s) raw={raw_out.name} filt={filt_out.name}", verbose)


# -----------------------------
# Convenience pipeline: edges -> ecdfs -> score all chunks
# -----------------------------
@app.command()
def assign_streaming_pipeline_v2(
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
        learn_edges_v2(  # type: ignore
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
            verbose=verbose,
        )
    else:
        _log(f"[assign] reuse edges model: {edges_path}", verbose)

    if not ecdf_path.exists():
        _log(f"[assign] learn ECDFs → {ecdf_path}", verbose)
        learn_ecdfs_v2(  # type: ignore
            workdir=workdir,
            sample=sample,
            chunks_dir=chunks_dir,
            edges_model=edges_path,
            out_model=ecdf_path,
            mapq_min=mapq_min,
            xa_max=xa_max,
            chunksize=chunksize,
            workers=workers,
            verbose=verbose,
        )
    else:
        _log(f"[assign] reuse ECDF model: {ecdf_path}", verbose)

    chunk_files = sorted(Path(chunks_dir).glob("*_cell_map_ref_chunk_*.txt"))
    if not chunk_files:
        raise typer.BadParameter(f"No chunk files under {chunks_dir}")

    _log(f"[assign/score] scoring {len(chunk_files)} chunks (serial). For parallel, run score-chunk-v2 in SLURM array.", verbose)

    for ch in chunk_files:
        score_chunk_v2(  # type: ignore
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
