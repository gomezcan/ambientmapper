# src/ambientmapper/assign_streaming.py
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, Tuple, Optional
from threading import Lock
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed


import numpy as np
import pandas as pd
import typer

app = typer.Typer(add_completion=False)

__all__ = [
    "learn_edges", "learn_ecdfs", "score_chunk", "assign_streaming_pipeline",
    "learn_edges_parallel", "learn_ecdfs_parallel", "learn_ecdfs_batched",
]

# ---------- logging helpers ----------
_LOG_LOCK = Lock()

def _log(msg: str, verbose: bool):
    if not verbose:
        return
    with _LOG_LOCK:
        typer.echo(msg)

def _log_ok(msg: str, verbose: bool):
    if not verbose:
        return
    with _LOG_LOCK:
        typer.secho(msg, fg="green")

def _log_err(msg: str, verbose: bool):
    with _LOG_LOCK:
        typer.secho(msg, fg="red")


# ---------- small utils ----------
def _chunk_bcs(chunk_file: Path) -> set[str]:
    with open(chunk_file, "r") as f:
        return {ln.strip() for ln in f if ln.strip()}

def _genome_from_filename(p: Path) -> str:
    n = p.name
    if n.startswith("filtered_") and n.endswith("_QCMapping.txt"):
        return n[len("filtered_"):-len("_QCMapping.txt")]
    return n

def _key_tuple(genome: str, AS: float, MAPQ: float, NM: float) -> tuple:
    # comparator key: lower is better (AS↓ negative to sort desc)
    return (-AS, -MAPQ, NM, genome)

def _pick_best_two_and_worst(acc: dict, genome: str, AS: float, MAPQ: float, NM: float, XA: float):
    cand = {"g": genome, "AS": AS, "MAPQ": MAPQ, "NM": NM, "XA": XA}
    cand["key"] = _key_tuple(genome, AS, MAPQ, NM)
    if acc.get("best1") is None:
        acc["best1"] = cand
        acc["worst"] = cand
        acc.setdefault("seen_g", set()).add(genome)
        return
    acc.setdefault("seen_g", set()).add(genome)
    if cand["key"] > acc["worst"]["key"]:
        acc["worst"] = cand
    if cand["key"] < acc["best1"]["key"]:
        acc["best2"] = acc["best1"]
        acc["best1"] = cand
    else:
        b2 = acc.get("best2")
        if b2 is None or cand["key"] < b2["key"]:
            acc["best2"] = cand


# ---------- fixed histogram specs (shared across workers) ----------
_AS_RANGE = (0.0, 200.0, 2000)   # lo, hi, nbins
_MQ_RANGE = (0.0,  60.0,  120)

class DeltaHist:
    def __init__(self, lo=0.0, hi=100.0, nbins=200):
        self.lo, self.hi, self.nbins = float(lo), float(hi), int(nbins)
        self.edges = np.linspace(self.lo, self.hi, self.nbins + 1)
        self.counts = np.zeros(self.nbins, dtype=np.int64)
        self.overflow = 0

    def add(self, x: float):
        if not np.isfinite(x):
            return
        if x >= self.hi:
            self.overflow += 1
            return
        if x <= self.lo:
            idx = 0
        else:
            idx = int((x - self.lo) / (self.hi - self.lo) * self.nbins)
            idx = min(max(idx, 0), self.nbins - 1)
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
        idx = min(max(idx, 0), self.nbins - 1)
        tail = int(self.counts[idx:].sum() + self.overflow)
        return tail / total

def _new_hist_from_spec(spec: tuple[float, float, int]) -> DeltaHist:
    lo, hi, nb = spec
    return DeltaHist(lo=lo, hi=hi, nbins=nb)

# ---------- IO helpers ----------
def _filtered_files(workdir: Path, sample: str) -> list[Path]:
    d = Path(workdir) / sample / "filtered_QCFiles"
    if not d.exists():
        raise FileNotFoundError(f"Not found: {d}")
    files = sorted(d.glob("filtered_*_QCMapping.txt"))
    if not files:
        raise FileNotFoundError(f"No filtered_* files under {d}")
    return files

def _iter_read_bc_rows(fp: Path, bcs: set[str], chunksize: Optional[int], mapq_min: int, xa_max: int):
    usecols = ["Read","BC","MAPQ","AS","NM","XAcount"]
    for c in pd.read_csv(
        fp, sep="\t", header=0, usecols=usecols, chunksize=chunksize,
        dtype={"Read":"string","BC":"string"}, engine="c", low_memory=True, memory_map=False
    ):
        c = c[c["BC"].astype(str).isin(bcs)]
        if c.empty:
            continue
        for col in ("AS","MAPQ","NM","XAcount"):
            c[col] = pd.to_numeric(c[col], errors="coerce")
        if mapq_min > 0:
            c = c[c["MAPQ"].fillna(0) >= mapq_min]
        if xa_max >= 0:
            c = c[c["XAcount"].fillna(0) <= xa_max]
        if c.empty:
            continue
        c = c.drop_duplicates(subset=["Read","BC"], keep="first")
        yield c.itertuples(index=False)

# ---------- decile helpers ----------
def compute_decile_edges(values: np.ndarray, k: int = 10) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.array([])
    qs = np.linspace(0, 1, k + 1)
    edges = np.quantile(vals, qs[1:-1])
    return np.unique(edges)

def assign_decile(v: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if edges.size == 0:
        return np.full_like(v, np.nan, dtype=float)
    return (np.digitize(v, bins=edges, right=True) + 1).astype(float)

def edges_from_hist(H: DeltaHist, k: int) -> np.ndarray:
    counts = H.counts.astype(np.int64)
    if counts.sum() == 0:
        return np.array([])
    cum = np.cumsum(counts)
    tot = int(cum[-1] + H.overflow)
    targets = (np.linspace(0, 1, k + 1)[1:-1] * tot).astype(int)
    edges = []
    for t in targets:
        idx = int(np.searchsorted(cum, t, side="left"))
        idx = min(max(idx, 0), H.nbins - 1)
        edges.append(H.edges[idx])
    return np.unique(np.array(edges, dtype=float))

def _edges_from_hist_counts(as_counts: np.ndarray, mq_counts: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    H_AS = _new_hist_from_spec(_AS_RANGE); H_AS.counts = as_counts.copy()
    H_MQ = _new_hist_from_spec(_MQ_RANGE); H_MQ.counts = mq_counts.copy()
    return edges_from_hist(H_AS, k), edges_from_hist(H_MQ, k)

# ---------- map workers ----------
def _map_winner_hist_for_chunk(
    workdir: Path, sample: str, chunk_file: Path,
    mapq_min: int, xa_max: int, chunksize: Optional[int], verbose: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Return (AS_winner_counts, MAPQ_winner_counts) hist counts for this chunk."""
    files = _filtered_files(workdir, sample)
    bcs = _chunk_bcs(chunk_file)
    H_AS = _new_hist_from_spec(_AS_RANGE)
    H_MQ = _new_hist_from_spec(_MQ_RANGE)

    rb_map: Dict[tuple[str,str], dict] = {}
    t0 = time.time()
    last_log = t0
    genomes_done = 0
    reads_kept = 0

    _log(f"[edges-map] ▶ start {chunk_file.name}  BCs={len(bcs)}", verbose)
    for fp in files:
        genome = _genome_from_filename(fp)
        per_file_reads = 0
        per_file_kept  = 0
        for i, rows in enumerate(_iter_read_bc_rows(fp, bcs, chunksize, mapq_min, xa_max), start=1):
            for row in rows:
                key = (str(row.Read), str(row.BC))
                acc = rb_map.get(key) or {"best1": None, "best2": None, "worst": None, "seen_g": set()}
                _pick_best_two_and_worst(
                    acc, genome,
                    AS=float(row.AS) if pd.notna(row.AS) else float("-inf"),
                    MAPQ=float(row.MAPQ) if pd.notna(row.MAPQ) else float("-inf"),
                    NM=float(row.NM) if pd.notna(row.NM) else float("inf"),
                    XA=float(row.XAcount) if pd.notna(row.XAcount) else float("inf"),
                )
                rb_map[key] = acc
                per_file_reads += 1
                per_file_kept  += 1

            now = time.time()
            if verbose and (now - last_log > 30 or i % 50 == 0):
                _log(f"[edges-map] … {chunk_file.name} + {fp.name}  chunks={i}  kept≈{per_file_kept:,}", verbose)
                last_log = now

        genomes_done += 1
        reads_kept   += per_file_kept
        _log(f"[edges-map] ✓ {chunk_file.name} + {fp.name}  kept={per_file_kept:,}", verbose)

    # finalize hist fill
    for acc in rb_map.values():
        b1 = acc["best1"]
        if b1 is None:
            continue
        H_AS.add(b1["AS"])
        H_MQ.add(b1["MAPQ"])

    dt = time.time() - t0
    _log_ok(f"[edges-map] ■ done {chunk_file.name}  genomes={genomes_done}  winners={len(rb_map):,}  kept≈{reads_kept:,}  {dt:0.1f}s", verbose)
    return H_AS.counts, H_MQ.counts

def _map_delta_hists_for_chunk(
    workdir: Path, sample: str, chunk_file: Path,
    as_edges: np.ndarray, mq_edges: np.ndarray, k: int,
    mapq_min: int, xa_max: int, chunksize: int, verbose: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return stacked counts for ΔAS and ΔMAPQ per decile.
      dAS_counts: (k, nbins_AS), dAS_over: (k,)
      dMQ_counts: (k, nbins_MQ), dMQ_over: (k,)
    """
    files = _filtered_files(workdir, sample)
    bcs = _chunk_bcs(chunk_file)
    H_dAS = [_new_hist_from_spec((0.0, 100.0, 200)) for _ in range(k)]
    H_dMQ = [_new_hist_from_spec((0.0,  60.0, 120)) for _ in range(k)]
    rb_map: Dict[tuple[str,str], dict] = {}

    t0 = time.time(); last_log = t0
    _log(f"[ecdf-map] ▶ start {chunk_file.name}  BCs={len(bcs)}", verbose)

    for fp in files:
        genome = _genome_from_filename(fp)
        per_file_kept = 0
        for i, rows in enumerate(_iter_read_bc_rows(fp, bcs, chunksize, mapq_min, xa_max), start=1):
            for row in rows:
                key = (str(row.Read), str(row.BC))
                acc = rb_map.get(key) or {"best1": None, "best2": None, "worst": None, "seen_g": set()}
                _pick_best_two_and_worst(
                    acc, genome,
                    AS=float(row.AS) if pd.notna(row.AS) else float("-inf"),
                    MAPQ=float(row.MAPQ) if pd.notna(row.MAPQ) else float("-inf"),
                    NM=float(row.NM) if pd.notna(row.NM) else float("inf"),
                    XA=float(row.XAcount) if pd.notna(row.XAcount) else float("inf"),
                )
                rb_map[key] = acc
                per_file_kept += 1

            now = time.time()
            if verbose and (now - last_log > 30 or i % 50 == 0):
                _log(f"[ecdf-map] … {chunk_file.name} + {fp.name}  chunks={i}  kept≈{per_file_kept:,}", verbose)
                last_log = now

        _log(f"[ecdf-map] ✓ {chunk_file.name} + {fp.name}  kept={per_file_kept:,}", verbose)

    # fill histograms
    pairs = 0
    for acc in rb_map.values():
        b1, b2 = acc["best1"], acc.get("best2")
        if b1 is None or b2 is None:
            continue
        dAS = b1["AS"] - b2["AS"]
        dMQ = b1["MAPQ"] - b2["MAPQ"]
        dec_as = int(assign_decile(np.array([b1["AS"]]), as_edges)[0]) if as_edges.size else 1
        dec_mq = int(assign_decile(np.array([b1["MAPQ"]]), mq_edges)[0]) if mq_edges.size else 1
        if 1 <= dec_as <= k: H_dAS[dec_as-1].add(dAS)
        if 1 <= dec_mq <= k: H_dMQ[dec_mq-1].add(dMQ)
        pairs += 1

    dt = time.time() - t0
    _log_ok(f"[ecdf-map] ■ done {chunk_file.name}  pairs={pairs:,}  {dt:0.1f}s", verbose)

    dAS_counts = np.stack([H.counts for H in H_dAS], axis=0)
    dAS_over   = np.array([H.overflow for H in H_dAS], dtype=np.int64)
    dMQ_counts = np.stack([H.counts for H in H_dMQ], axis=0)
    dMQ_over   = np.array([H.overflow for H in H_dMQ], dtype=np.int64)
    return dAS_counts, dAS_over, dMQ_counts, dMQ_over

####
def _edges_batch_worker(
    bi: int,
    ch_paths: list[Path],
    files: list[Path],
    mapq_min: int,
    xa_max: int,
    chunksize: int,
    k: int,
    verbose: bool,
    as_range: tuple[float, float, int],
    mq_range: tuple[float, float, int],
    max_reads_per_genome: int | None = None,
):
    import numpy as np
    import pandas as pd

    lo_as, hi_as, nb_as = as_range
    lo_mq, hi_mq, nb_mq = mq_range
    scale_as = (hi_as - lo_as)
    scale_mq = (hi_mq - lo_mq)

    # per-batch hist buffers
    as_counts = np.zeros(nb_as, dtype=np.int64)
    mq_counts = np.zeros(nb_mq, dtype=np.int64)

    # Build BC -> chunk map + per-chunk accumulators
    bc_to_chunk: dict[str, Path] = {}
    per_chunk_acc: dict[Path, dict[tuple[str, str], dict]] = {}
    for ch in ch_paths:
        per_chunk_acc[ch] = {}
        for bc in _chunk_bcs(ch):
            bc_to_chunk[bc] = ch
    batch_bcs = set(bc_to_chunk.keys())
    if verbose:
        _log(f"[assign/edges] ▶ batch {bi}  chunks={len(ch_paths)}  BCs={len(batch_bcs):,}", verbose)

    # Stream each genome once
    for fp in files:
        kept_rows = 0
        genome = _genome_from_filename(fp)
        rows_seen_for_genome = 0
        for c in pd.read_csv(
            fp, sep="\t",
            usecols=["Read","BC","MAPQ","AS","NM","XAcount"],
            chunksize=chunksize,
            dtype={"Read":"string","BC":"string"},
            engine="c", low_memory=True, memory_map=False,
        ):
            # Optional cap to speed up model learning
            if max_reads_per_genome is not None and rows_seen_for_genome >= max_reads_per_genome:
                break

            c = c[c["BC"].astype(str).isin(batch_bcs)]
            if c.empty:
                continue

            for col in ("AS","MAPQ","NM","XAcount"):
                c[col] = pd.to_numeric(c[col], errors="coerce")

            if mapq_min > 0:
                c = c[c["MAPQ"].fillna(0) >= mapq_min]
            if xa_max >= 0:
                c = c[c["XAcount"].fillna(0) <= xa_max]
            if c.empty:
                continue

            c = c.drop_duplicates(subset=["Read","BC"], keep="first")

            # If capped, trim this chunk so total per genome doesn't exceed cap
            if max_reads_per_genome is not None:
                remaining = max_reads_per_genome - rows_seen_for_genome
                if remaining <= 0:
                    break
                if len(c) > remaining:
                    c = c.iloc[:remaining]

            for row in c.itertuples(index=False):
                key_bc = str(row.BC)
                ch = bc_to_chunk.get(key_bc)
                if ch is None:
                    continue
                rb = per_chunk_acc[ch]
                key = (str(row.Read), key_bc)
                acc = rb.get(key)
                if acc is None:
                    acc = {"best1": None, "best2": None, "worst": None, "seen_g": set()}
                    rb[key] = acc
                _pick_best_two_and_worst(
                    acc, genome,
                    AS=float(row.AS) if pd.notna(row.AS) else float("-inf"),
                    MAPQ=float(row.MAPQ) if pd.notna(row.MAPQ) else float("-inf"),
                    NM=float(row.NM) if pd.notna(row.NM) else float("inf"),
                    XA=float(row.XAcount) if pd.notna(row.XAcount) else float("inf"),
                )
                kept_rows += 1
                rows_seen_for_genome += 1

        if verbose:
            _log(f"[assign/edges]    ✓ genome {genome}  kept≈{kept_rows:,}", verbose)

    # Reduce winners -> per-batch hist
    batch_winners = 0
    for rb in per_chunk_acc.values():
        for acc in rb.values():
            b1 = acc["best1"]
            if b1 is None:
                continue
            asv = b1["AS"]
            if np.isfinite(asv):
                if asv >= hi_as: idx_as = nb_as - 1
                elif asv <= lo_as: idx_as = 0
                else:
                    idx_as = int((asv - lo_as) / scale_as * nb_as)
                    idx_as = min(max(idx_as, 0), nb_as - 1)
                as_counts[idx_as] += 1

            mqv = b1["MAPQ"]
            if np.isfinite(mqv):
                if mqv >= hi_mq: idx_mq = nb_mq - 1
                elif mqv <= lo_mq: idx_mq = 0
                else:
                    idx_mq = int((mqv - lo_mq) / scale_mq * nb_mq)
                    idx_mq = min(max(idx_mq, 0), nb_mq - 1)
                mq_counts[idx_mq] += 1

            batch_winners += 1

    # Return the partial histograms and counts
    return bi, as_counts, mq_counts, batch_winners


# ---------- Pass A (CLI): learn global decile edges (serial) ----------
@app.command()
def learn_edges(
    workdir: Path = typer.Option(..., "--workdir"),
    sample: str = typer.Option(..., "--sample"),
    chunks_dir: Path = typer.Option(..., "--chunks-dir"),
    out_model: Optional[Path] = typer.Option(None, "--out-model"),
    mapq_min: int = 20, xa_max: int = 2, chunksize: int = 1_000_000, k: int = 10,
):
    files = _filtered_files(workdir, sample)
    chunk_list = sorted(Path(chunks_dir).glob("*_cell_map_ref_chunk_*.txt"))
    if not chunk_list:
        raise typer.BadParameter(f"No chunk files under {chunks_dir}")

    H_AS = DeltaHist(lo=0, hi=200, nbins=2000)
    H_MQ = DeltaHist(lo=0, hi=60,  nbins=120)

    for chunk_file in chunk_list:
        bcs = _chunk_bcs(chunk_file)
        rb_map: Dict[Tuple[str,str], dict] = {}
        for fp in files:
            genome = _genome_from_filename(fp)
            for rows in _iter_read_bc_rows(fp, bcs, chunksize, mapq_min, xa_max):
                for row in rows:
                    key = (str(row.Read), str(row.BC))
                    acc = rb_map.get(key) or {"best1": None, "best2": None, "worst": None, "seen_g": set()}
                    _pick_best_two_and_worst(
                        acc, genome,
                        AS=float(row.AS) if pd.notna(row.AS) else float("-inf"),
                        MAPQ=float(row.MAPQ) if pd.notna(row.MAPQ) else float("-inf"),
                        NM=float(row.NM) if pd.notna(row.NM) else float("inf"),
                        XA=float(row.XAcount) if pd.notna(row.XAcount) else float("inf"),
                    )
                    rb_map[key] = acc
        for acc in rb_map.values():
            b1 = acc["best1"]
            if b1 is None:
                continue
            H_AS.add(b1["AS"])
            H_MQ.add(b1["MAPQ"])

    as_edges = edges_from_hist(H_AS, k)
    mq_edges = edges_from_hist(H_MQ, k)
    model_path = out_model or (Path(workdir)/sample/"ExplorationReadLevel"/"global_edges.npz")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(model_path, as_edges=as_edges, mq_edges=mq_edges, k=k)
    typer.echo(f"[assign/edges] done → {model_path}")
    return model_path


# ---------- Pass B (CLI): learn per-decile Δ ECDFs (serial) ----------
@app.command()
def learn_ecdfs(
    workdir: Path = typer.Option(..., "--workdir"),
    sample: str = typer.Option(..., "--sample"),
    chunks_dir: Path = typer.Option(..., "--chunks-dir"),
    edges_model: Path = typer.Option(..., "--edges"),
    out_model: Optional[Path] = typer.Option(None, "--out-model"),
    mapq_min: int = 20, xa_max: int = 2, chunksize: int = 1_000_000,
):
    files = _filtered_files(workdir, sample)
    chunk_list = sorted(Path(chunks_dir).glob("*_cell_map_ref_chunk_*.txt"))
    if not chunk_list:
        raise typer.BadParameter(f"No chunk files under {chunks_dir}")

    dat = np.load(edges_model)
    as_edges, mq_edges, k = dat["as_edges"], dat["mq_edges"], int(dat["k"])
    H_dAS = {d: DeltaHist(lo=0, hi=100, nbins=200) for d in range(1, k+1)}
    H_dMQ = {d: DeltaHist(lo=0, hi=60,  nbins=120) for d in range(1, k+1)}

    for chunk_file in chunk_list:
        bcs = _chunk_bcs(chunk_file)
        rb_map: Dict[Tuple[str,str], dict] = {}
        for fp in files:
            genome = _genome_from_filename(fp)
            for rows in _iter_read_bc_rows(fp, bcs, chunksize, mapq_min, xa_max):
                for row in rows:
                    key = (str(row.Read), str(row.BC))
                    acc = rb_map.get(key) or {"best1": None, "best2": None, "worst": None, "seen_g": set()}
                    _pick_best_two_and_worst(
                        acc, genome,
                        AS=float(row.AS) if pd.notna(row.AS) else float("-inf"),
                        MAPQ=float(row.MAPQ) if pd.notna(row.MAPQ) else float("-inf"),
                        NM=float(row.NM) if pd.notna(row.NM) else float("inf"),
                        XA=float(row.XAcount) if pd.notna(row.XAcount) else float("inf"),
                    )
                    rb_map[key] = acc
        for acc in rb_map.values():
            b1, b2 = acc["best1"], acc.get("best2")
            if b1 is None or b2 is None:
                continue
            dAS = b1["AS"] - b2["AS"]
            dMQ = b1["MAPQ"] - b2["MAPQ"]
            dec_as = int(assign_decile(np.array([b1["AS"]]), as_edges)[0]) if as_edges.size else 1
            dec_mq = int(assign_decile(np.array([b1["MAPQ"]]), mq_edges)[0]) if mq_edges.size else 1
            if 1 <= dec_as <= k: H_dAS[dec_as].add(dAS)
            if 1 <= dec_mq <= k: H_dMQ[dec_mq].add(dMQ)

    payload = {
        "k": np.array(k),
        "as_edges": as_edges, "mq_edges": mq_edges,
        "dAS_lo": np.array([H_dAS[1].lo]), "dAS_hi": np.array([H_dAS[1].hi]), "dAS_nbins": np.array([H_dAS[1].nbins]),
        "dMQ_lo": np.array([H_dMQ[1].lo]), "dMQ_hi": np.array([H_dMQ[1].hi]), "dMQ_nbins": np.array([H_dMQ[1].nbins]),
        "dAS_counts": np.stack([H_dAS[d].counts for d in range(1,k+1)], axis=0),
        "dAS_overflow": np.array([H_dAS[d].overflow for d in range(1,k+1)], dtype=np.int64),
        "dMQ_counts": np.stack([H_dMQ[d].counts for d in range(1,k+1)], axis=0),
        "dMQ_overflow": np.array([H_dMQ[d].overflow for d in range(1,k+1)], dtype=np.int64),
    }
    model_path = out_model or (Path(workdir)/sample/"ExplorationReadLevel"/"global_ecdf.npz")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(model_path, **payload)
    typer.echo(f"[assign/ecdf] saved {model_path}")
    return model_path


# ---------- Parallel map-reduce learners (library-level helpers) ----------
def learn_edges_parallel(
    workdir: Path, 
    sample: str, 
    chunks_dir: Path,
    out_model: Optional[Path], 
    mapq_min: int, 
    xa_max: int, 
    chunksize: int, 
    k: int,
    batch_size: int = 32, 
    threads: int = 8, 
    verbose: bool = False,
    edges_workers: Optional[int] = None,
    edges_max_reads: Optional[int] = None, 
) -> Path:
    """
    Parallel *batched* map-reduce with concurrent batches:
    read each genome once per batch, route rows by BC, then reduce winners to hist counts.
    """
    import numpy as np

    chunk_files = sorted(Path(chunks_dir).glob(f"{sample}_cell_map_ref_chunk_*.txt"))
    if not chunk_files:
        raise typer.BadParameter(f"No chunk files under {chunks_dir}")
    files = _filtered_files(workdir, sample)

    # global hist buffers
    as_counts = np.zeros(_AS_RANGE[2], dtype=np.int64)
    mq_counts = np.zeros(_MQ_RANGE[2], dtype=np.int64)

    total_chunks = len(chunk_files)
    batches = [(i, chunk_files[i:i+batch_size]) for i in range(0, total_chunks, batch_size)]
    _log(f"[assign/edges] start: {total_chunks} chunks in {len(batches)} batches (batch_size={batch_size})", verbose)

    max_workers = min(edges_workers or threads, len(batches))
    futures = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for bi, ch_paths in batches:
            fut = ex.submit(
                _edges_batch_worker,
                bi, ch_paths, files,
                mapq_min, xa_max, chunksize, k, verbose,
                _AS_RANGE, _MQ_RANGE,
                edges_max_reads
            )
            futures.append(fut)

        for f in as_completed(futures):
            bi, as_part, mq_part, won = f.result()
            # reduce partials
            as_counts += as_part
            mq_counts += mq_part
            _log_ok(f"[assign/edges] ■ batch {bi} reduced  winners={won:,}", verbose)

    # finalize
    as_edges, mq_edges = _edges_from_hist_counts(as_counts, mq_counts, k)
    model_path = out_model or (Path(workdir) / sample / "ExplorationReadLevel" / "global_edges.npz")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(model_path, as_edges=as_edges, mq_edges=mq_edges, k=k)
    _log_ok(f"[assign/edges] done → {model_path}", verbose)
    return model_path
    
def learn_ecdfs_parallel(
    workdir: Path,
    sample: str,
    chunks_dir: Path,
    edges_model: Path,
    out_model: Optional[Path],
    mapq_min: int,
    xa_max: int,
    chunksize: int,
    threads: int = 8,
    verbose: bool = False,
    edges_workers: Optional[int] = None,
) -> Path:
    """Parallel map-reduce: per-chunk per-decile Δ histograms → global ECDF model."""
    
    # choose pool size based on chunks available
    max_workers = min(ecdf_workers or threads,  max(1, len(list(Path(chunks_dir).glob("*_cell_map_ref_chunk_*.txt")))))
    
    dat = np.load(edges_model)
    as_edges, mq_edges, k = dat["as_edges"], dat["mq_edges"], int(dat["k"])

    chunk_files = sorted(Path(chunks_dir).glob(f"{sample}_cell_map_ref_chunk_*.txt"))
    if not chunk_files:
        raise typer.BadParameter(f"No chunk files under {chunks_dir}")

    Htmp_AS = _new_hist_from_spec((0.0, 100.0, 200))
    Htmp_MQ = _new_hist_from_spec((0.0,  60.0, 120))
    dAS_counts = np.zeros((k, Htmp_AS.nbins), dtype=np.int64)
    dAS_over   = np.zeros((k,), dtype=np.int64)
    dMQ_counts = np.zeros((k, Htmp_MQ.nbins), dtype=np.int64)
    dMQ_over   = np.zeros((k,), dtype=np.int64)

    def _worker(chf: Path):
        return _map_delta_hists_for_chunk(
            workdir, sample, chf, as_edges, mq_edges, k, mapq_min, xa_max, chunksize, verbose=verbose
        )

    total = len(chunk_files)
    _log(f"[assign/ecdf] start: {total} chunks, threads={min(threads, total)}", verbose)
    with ThreadPoolExecutor(max_workers=min(max_workers, total)) as ex:
        fut = {ex.submit(_worker, ch): ch for ch in chunk_files}
        done = 0
        for f in as_completed(fut):
            ch = fut[f]
            try:
                das, daso, dmq, dmqo = f.result()
                dAS_counts += das; dAS_over += daso
                dMQ_counts += dmq; dMQ_over += dmqo
            except Exception as e:
                _log_err(f"[assign/ecdf][ERROR] {ch.name}: {e}", verbose=True)
                raise
            done += 1
            if verbose and (done % 5 == 0 or done == total):
                _log(f"[assign/ecdf] {done}/{total} chunks", verbose)

    payload = {
        "k": np.array(k),
        "as_edges": as_edges, "mq_edges": mq_edges,
        "dAS_lo": np.array([0.0]), "dAS_hi": np.array([100.0]), "dAS_nbins": np.array([Htmp_AS.nbins]),
        "dMQ_lo": np.array([0.0]), "dMQ_hi": np.array([ 60.0]), "dMQ_nbins": np.array([Htmp_MQ.nbins]),
        "dAS_counts": dAS_counts, "dAS_overflow": dAS_over,
        "dMQ_counts": dMQ_counts, "dMQ_overflow": dMQ_over,
    }
    model_path = out_model or (Path(workdir)/sample/"ExplorationReadLevel"/"global_ecdf.npz")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(model_path, **payload)
    _log_ok(f"[assign/ecdf] done → {model_path}", verbose)
    return model_path

def learn_ecdfs_batched(
    workdir: Path, sample: str, chunks_dir: Path,
    edges_model: Path, out_model: Optional[Path],
    mapq_min: int, xa_max: int, chunksize: Optional[int],
    batch_size: int = 32, verbose: bool = True
) -> Path:
    """
    Batched map-reduce: read each genome once per batch of chunk files, route rows
    by BC to per-chunk accumulators, then reduce to global per-decile Δ histograms.
    Much lower I/O than per-chunk workers when you have hundreds/thousands of chunks.
    """
    # 0) load decile edges (for assigning deciles of the winner AS/MAPQ)
    dat = np.load(edges_model)
    as_edges, mq_edges, k = dat["as_edges"], dat["mq_edges"], int(dat["k"])

    # 1) global Δ histogram buffers (same binning as your current parallel version)
    Htmp_AS = _new_hist_from_spec((0.0, 100.0, 200))
    Htmp_MQ = _new_hist_from_spec((0.0,  60.0, 120))
    dAS_counts = np.zeros((k, Htmp_AS.nbins), dtype=np.int64)
    dAS_over   = np.zeros((k,), dtype=np.int64)
    dMQ_counts = np.zeros((k, Htmp_MQ.nbins), dtype=np.int64)
    dMQ_over   = np.zeros((k,), dtype=np.int64)

    chunk_files = sorted(Path(chunks_dir).glob("*_cell_map_ref_chunk_*.txt"))

    files = _filtered_files(workdir, sample)
    if not chunk_files:
        raise typer.BadParameter(f"No chunk files under {chunks_dir}")

    _log(f"[assign/ecdf-batched] start: {len(chunk_files)} chunks, batch_size={batch_size}", verbose)
    t0_all = time.time()

    for ofs in range(0, len(chunk_files), batch_size):
        batch = chunk_files[ofs:ofs + batch_size]
        t0 = time.time()
        _log(f"[assign/ecdf-batched] ▶ batch {ofs//batch_size+1}: {len(batch)} chunks", verbose)

        # 2) routing
        bc_to_chunk: dict[str, Path] = {}
        per_chunk_acc: dict[Path, dict[tuple[str, str], dict]] = {}
        for ch in batch:
            per_chunk_acc[ch] = {}
            for bc in _chunk_bcs(ch):
                bc_to_chunk[bc] = ch

        # 3) stream each genome once; route rows to the right batch-chunk accumulator
        for fp in files:
            genome = _genome_from_filename(fp)
            per_file_kept = 0
            for c in pd.read_csv(
                fp, sep="\t",
                usecols=["Read", "BC", "MAPQ", "AS", "NM", "XAcount"],
                chunksize=chunksize, 
                dtype={
                    "Read": "string",
                    "BC": "string",
                    "AS": "float32",
                    "MAPQ": "float32",
                    "NM": "float32",
                    "XAcount": "float32",
                },
                engine="c", low_memory=True, memory_map=False
            ):
                # keep only BCs present in THIS batch
                bc_view = set(bc_to_chunk.keys()
                c = c[c["BC"].astype(str).isin(bc_view)]
                if c.empty:
                    continue
                
                if mapq_min > 0:
                    c = c[c["MAPQ"].fillna(0) >= mapq_min]
                if xa_max >= 0:
                    c = c[c["XAcount"].fillna(0) <= xa_max]
                if c.empty:
                    continue
                
                # at most one (Read,BC) per genome file                
                if c.duplicated(subset=["Read","BC"]).any():
                    c = c.drop_duplicates(subset=["Read","BC"], keep="first")

                for row in c.itertuples(index=False):
                    ch = bc_to_chunk.get(str(row.BC))
                    if ch is None:
                        continue
                    rb = per_chunk_acc[ch]
                    key = (str(row.Read), str(row.BC))
                    acc = rb.get(key)
                    if acc is None:
                        acc = {"best1": None, "best2": None, "worst": None, "seen_g": set()}
                        rb[key] = acc
                    _pick_best_two_and_worst(
                        acc, genome,
                        AS=float(row.AS) if pd.notna(row.AS) else float("-inf"),
                        MAPQ=float(row.MAPQ) if pd.notna(row.MAPQ) else float("-inf"),
                        NM=float(row.NM) if pd.notna(row.NM) else float("inf"),
                        XA=float(row.XAcount) if pd.notna(row.XAcount) else float("inf"),
                    )
                    per_file_kept += 1
            _log(f"[assign/ecdf-batched]   ✓ {fp.name} kept≈{per_file_kept:,}", verbose)

        # 4) reduce THIS batch → add to global Δ histograms
        pairs = 0
        for rb in per_chunk_acc.values():
            for acc in rb.values():
                b1, b2 = acc["best1"], acc.get("best2")
                if b1 is None or b2 is None:
                    continue
                dAS = b1["AS"] - b2["AS"]
                dMQ = b1["MAPQ"] - b2["MAPQ"]
                # assign winner deciles
                dec_as = int(assign_decile(np.array([b1["AS"]]), as_edges)[0]) if as_edges.size else 1
                dec_mq = int(assign_decile(np.array([b1["MAPQ"]]), mq_edges)[0]) if mq_edges.size else 1
                # add ΔAS
                if 1 <= dec_as <= k:
                    H = Htmp_AS
                    if not np.isfinite(dAS):
                        pass
                    elif dAS >= H.hi:
                        dAS_over[dec_as - 1] += 1
                    else:
                        idx = int(min(max((dAS - H.lo) / (H.hi - H.lo) * H.nbins, 0), H.nbins - 1))
                        dAS_counts[dec_as - 1, idx] += 1
                # add ΔMAPQ
                if 1 <= dec_mq <= k:
                    H = Htmp_MQ
                    if not np.isfinite(dMQ):
                        pass
                    elif dMQ >= H.hi:
                        dMQ_over[dec_mq - 1] += 1
                    else:
                        idx = int(min(max((dMQ - H.lo) / (H.hi - H.lo) * H.nbins, 0), H.nbins - 1))
                        dMQ_counts[dec_mq - 1, idx] += 1
                pairs += 1

        dt = time.time() - t0
        _log_ok(f"[assign/ecdf-batched] ■ batch {ofs//batch_size+1} done  pairs={pairs:,}  {dt:0.1f}s", verbose)

    # 5) save model
    model_path = out_model or (Path(workdir) / sample / "ExplorationReadLevel" / "global_ecdf.npz")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        model_path,
        k=np.array(k),
        as_edges=as_edges, mq_edges=mq_edges,
        dAS_lo=np.array([Htmp_AS.lo]), dAS_hi=np.array([Htmp_AS.hi]), dAS_nbins=np.array([Htmp_AS.nbins]),
        dMQ_lo=np.array([Htmp_MQ.lo]), dMQ_hi=np.array([Htmp_MQ.hi]), dMQ_nbins=np.array([Htmp_MQ.nbins]),
        dAS_counts=dAS_counts, dAS_overflow=dAS_over,
        dMQ_counts=dMQ_counts, dMQ_overflow=dMQ_over,
    )
    _log_ok(f"[assign/ecdf-batched] done → {model_path}  total {time.time()-t0_all:0.1f}s", verbose)
    return model_path


# ---------- model loader ----------
def _load_ecdf_model(path: Path):
    dat = np.load(path)
    k = int(dat["k"]); as_edges, mq_edges = dat["as_edges"], dat["mq_edges"]
    dAS, dMQ = [], []
    for i in range(k):
        H = DeltaHist(float(dat["dAS_lo"][0]), float(dat["dAS_hi"][0]), int(dat["dAS_nbins"][0]))
        H.counts = dat["dAS_counts"][i].astype(np.int64); H.overflow = int(dat["dAS_overflow"][i]); dAS.append(H)
    for i in range(k):
        H = DeltaHist(float(dat["dMQ_lo"][0]), float(dat["dMQ_hi"][0]), int(dat["dMQ_nbins"][0]))
        H.counts = dat["dMQ_counts"][i].astype(np.int64); H.overflow = int(dat["dMQ_overflow"][i]); dMQ.append(H)
    return k, as_edges, mq_edges, dAS, dMQ


# ---------- Pass C (CLI): score one chunk ----------
@app.command()
def score_chunk(
    workdir: Path = typer.Option(..., "--workdir"),
    sample: str = typer.Option(..., "--sample"),
    chunk_file: Path = typer.Option(..., "--chunk-file"),
    ecdf_model: Path = typer.Option(..., "--model"),
    out_raw_dir: Optional[Path] = typer.Option(None, "--out-raw"),
    out_filtered_dir: Optional[Path] = typer.Option(None, "--out-filtered"),
    mapq_min: int = 20, xa_max: int = 2, chunksize: int = 1_000_000, alpha: float = 0.05,
):
    k, as_edges, mq_edges, H_dAS, H_dMQ = _load_ecdf_model(ecdf_model)
    files = _filtered_files(workdir, sample)

    out_raw_dir = out_raw_dir or (Path(workdir)/sample/"raw_cell_map_ref_chunks")
    out_filtered_dir = out_filtered_dir or (Path(workdir)/sample/"cell_map_ref_chunks")
    out_raw_dir.mkdir(parents=True, exist_ok=True)
    out_filtered_dir.mkdir(parents=True, exist_ok=True)

    tag = chunk_file.stem
    suff = tag.split("_cell_map_ref_chunk_")[-1] if "_cell_map_ref_chunk_" in tag else tag
    base = f"{sample}_chunk{suff}"
    raw_out = out_raw_dir / f"{base}_raw.tsv.gz"
    filt_out = out_filtered_dir / f"{base}_filtered.tsv.gz"

    bcs = _chunk_bcs(chunk_file)
    rb_map: Dict[Tuple[str,str], dict] = {}

    for fp in files:
        genome = _genome_from_filename(fp)
        for rows in _iter_read_bc_rows(fp, bcs, chunksize, mapq_min, xa_max):
            for row in rows:
                key = (str(row.Read), str(row.BC))
                acc = rb_map.get(key) or {"best1": None, "best2": None, "worst": None, "seen_g": set()}
                _pick_best_two_and_worst(
                    acc, genome,
                    AS=float(row.AS) if pd.notna(row.AS) else float("-inf"),
                    MAPQ=float(row.MAPQ) if pd.notna(row.MAPQ) else float("-inf"),
                    NM=float(row.NM) if pd.notna(row.NM) else float("inf"),
                    XA=float(row.XAcount) if pd.notna(row.XAcount) else float("inf"),
                )
                rb_map[key] = acc

    # RAW table
    raw_rows = []
    for (read, bc), acc in rb_map.items():
        b1, b2, bw = acc["best1"], acc.get("best2"), acc.get("worst")
        if b1 is None:
            continue
        as1, mq1, nm1 = b1["AS"], b1["MAPQ"], b1["NM"]
        if b2 is not None:
            as2, mq2, nm2 = b2["AS"], b2["MAPQ"], b2["NM"]
            dAS, dMQ, dNM = as1 - as2, mq1 - mq2, nm2 - nm1
        else:
            as2 = mq2 = nm2 = np.nan
            dAS = dMQ = dNM = np.nan
        dec_as = int(assign_decile(np.array([as1]), as_edges)[0]) if as_edges.size else 1
        dec_mq = int(assign_decile(np.array([mq1]), mq_edges)[0]) if mq_edges.size else 1
        p_as = H_dAS[dec_as-1].tail_p(dAS) if np.isfinite(dAS) else np.nan
        p_mq = H_dMQ[dec_mq-1].tail_p(dMQ) if np.isfinite(dMQ) else np.nan
        clear_by_p  = (np.isfinite(p_as) and p_as <= alpha) or (np.isfinite(p_mq) and p_mq <= alpha)
        clear_by_nm = (np.isfinite(dNM) and dNM > 0)
        single_hit  = b2 is None
        klass = "winner" if (clear_by_p or clear_by_nm or single_hit) else "ambiguous"
        raw_rows.append({
            "Read": read, "BC": bc,
            "Genome_winner": b1["g"], "AS_winner": as1, "MAPQ_winner": mq1, "NM_winner": nm1,
            "Genome_2": b2["g"] if b2 else "", "AS_2": as2, "MAPQ_2": mq2, "NM_2": nm2,
            "Genome_last": bw["g"] if bw else "", "AS_last": (bw["AS"] if bw else np.nan),
            "MAPQ_last": (bw["MAPQ"] if bw else np.nan), "NM_last": (bw["NM"] if bw else np.nan),
            "delta_AS_1_2": dAS, "delta_MAPQ_1_2": dMQ, "delta_NM_1_2": dNM,
            "delta_AS_1_last": (as1 - (bw["AS"] if bw else np.nan)),
            "delta_MAPQ_1_last": (mq1 - (bw["MAPQ"] if bw else np.nan)),
            "delta_NM_1_last": ((bw["NM"] if bw else np.nan) - nm1),
            "n_genomes_considered": len(acc.get("seen_g", [])),
            "decile_AS": dec_as, "decile_MAPQ": dec_mq, "p_as": p_as, "p_mq": p_mq,
            "assigned_class": klass,
        })
    raw_df = pd.DataFrame(raw_rows)
    p_lookup = {(r, b): (pa, pm) for r, b, pa, pm in zip(
        raw_df["Read"].astype(str),
        raw_df["BC"].astype(str),
        raw_df["p_as"].astype(float),
        raw_df["p_mq"].astype(float),
    )}
    
    raw_df.to_csv(raw_out, sep="\t", index=False, compression="gzip")

    # FILTERED alignment-level rows
    keep_pairs = set(zip(raw_df["Read"].astype(str), raw_df["BC"].astype(str)))
    class_by = {(r, b): k for r, b, k in zip(raw_df["Read"].astype(str), raw_df["BC"].astype(str), raw_df["assigned_class"])}
    gwin_by  = {(r, b): g for r, b, g in zip(raw_df["Read"].astype(str), raw_df["BC"].astype(str), raw_df["Genome_winner"])}


    out_rows = []
    usecols2 = ["Read","BC","AS","MAPQ","NM","XAcount"]  # Genome supplied externally
    for fp in files:
        genome = _genome_from_filename(fp)
        for c in pd.read_csv(fp, sep="\t", header=0, usecols=usecols2, chunksize=chunksize):
            if c.empty:
                continue
            c = c[c["BC"].astype(str).isin({bc for _, bc in rb_map.keys()})]
            if c.empty:
                continue
            c = c[list(zip(c["Read"].astype(str), c["BC"].astype(str))).__iter__()]
+           c = c[[(str(r), str(b)) in keep_pairs for r, b in zip(c["Read"], c["BC"])]]
            if c.empty:
                continue
            for col in ("AS","MAPQ","NM","XAcount"):
                c[col] = pd.to_numeric(c[col], errors="coerce")
            for row in c.itertuples(index=False):
                r, b = str(row.Read), str(row.BC)
                klass = class_by.get((r, b), "ambiguous")                
                pa, pm = p_lookup.get((r, b), (np.nan, np.nan))
                base = row._asdict()
                base.update({
                    "Genome": genome,
                    "delta_AS": np.nan,            # (you can drop or fill later)
                    "assigned_class": klass,
                    "p_as": pa,
                    "p_mq": pm,
                })
                if klass == "winner":
                    if genome == gwin_by.get((r, b)):
                        out_rows.append(base)
                    else:
                        out_rows.append(base)
    pd.DataFrame(out_rows).to_csv(filt_out, sep="\t", index=False, compression="gzip")


# ---------- convenience: run all three passes serially ----------
@app.command()
def assign_streaming_pipeline(
    workdir: Path = typer.Option(..., "--workdir"),
    sample: str = typer.Option(..., "--sample"),
    chunks_dir: Path = typer.Option(..., "--chunks-dir"),
    alpha: float = typer.Option(0.05, "--alpha"),
    mapq_min: int = 20, xa_max: int = 2, chunksize: int = 1_000_000, k: int = 10,
):
    edges_path = Path(workdir)/sample/"ExplorationReadLevel"/"global_edges.npz"
    ecdf_path  = Path(workdir)/sample/"ExplorationReadLevel"/"global_ecdf.npz"
    learn_edges(workdir, sample, chunks_dir, edges_path, mapq_min, xa_max, chunksize, k)  # type: ignore
    learn_ecdfs(workdir, sample, chunks_dir, edges_path, ecdf_path, mapq_min, xa_max, chunksize)   # type: ignore
    for chunk_file in sorted(Path(chunks_dir).glob("*_cell_map_ref_chunk_*.txt")):
        score_chunk(workdir, sample, chunk_file, ecdf_path, None, None, mapq_min, xa_max, chunksize, alpha)  # type: ignore

if __name__ == "__main__":
    app()
