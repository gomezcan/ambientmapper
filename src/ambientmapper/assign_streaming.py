# src/ambientmapper/assign_streaming.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, Tuple, Optional, Iterable
import numpy as np
import pandas as pd
import typer
from concurrent.futures import ThreadPoolExecutor, as_completed


app = typer.Typer(add_completion=False)

# ---------- reuse-friendly helpers ----------
def _chunk_bcs(chunk_file: Path) -> set[str]:
    with open(chunk_file, "r") as f:
        return {ln.strip() for ln in f if ln.strip()}

def _genome_from_filename(p: Path) -> str:
    n = p.name
    if n.startswith("filtered_") and n.endswith("_QCMapping.txt"):
        return n[len("filtered_"):-len("_QCMapping.txt")]
    return n

def _key_tuple(genome: str, AS: float, MAPQ: float, NM: float) -> tuple:
    return (-AS, -MAPQ, NM, genome)

def _pick_best_two_and_worst(acc: dict, genome: str, AS: float, MAPQ: float, NM: float, XA: float):
    cand = {"g": genome, "AS": AS, "MAPQ": MAPQ, "NM": NM, "XA": XA}
    cand["key"] = _key_tuple(genome, AS, MAPQ, NM)
    if acc.get("best1") is None:
        acc["best1"] = cand; acc["worst"] = cand; acc.setdefault("seen_g", set()).add(genome); return
    acc.setdefault("seen_g", set()).add(genome)
    if cand["key"] > acc["worst"]["key"]: acc["worst"] = cand
    if cand["key"] < acc["best1"]["key"]:
        acc["best2"] = acc["best1"]; acc["best1"] = cand
    else:
        b2 = acc.get("best2")
        if b2 is None or cand["key"] < b2["key"]: acc["best2"] = cand

# --- NEW: fixed bin specs so all workers align ---
_AS_RANGE = (0.0, 200.0, 2000)   # lo, hi, nbins (tune if needed)
_MQ_RANGE = (0.0,  60.0,  120)

def _new_hist_from_spec(spec):
    lo, hi, nb = spec
    return DeltaHist(lo=lo, hi=hi, nbins=nb)

def _sum_counts_inplace(dst: DeltaHist, src: DeltaHist):
    dst.counts += src.counts
    dst.overflow += src.overflow

# --- MAP workers ---

def _map_winner_hist_for_chunk(workdir: Path, sample: str, chunk_file: Path,
                               mapq_min: int, xa_max: int, chunksize: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (AS_winner_counts, MAPQ_winner_counts) for this chunk."""
    files = _filtered_files(workdir, sample)
    bcs = _chunk_bcs(chunk_file)
    H_AS = _new_hist_from_spec(_AS_RANGE)
    H_MQ = _new_hist_from_spec(_MQ_RANGE)

    rb_map: Dict[tuple[str,str], dict] = {}
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
        if b1 is None: continue
        H_AS.add(b1["AS"]); H_MQ.add(b1["MAPQ"])

    return H_AS.counts, H_MQ.counts  # fixed shapes

def _map_delta_hists_for_chunk(workdir: Path, sample: str, chunk_file: Path,
                               as_edges: np.ndarray, mq_edges: np.ndarray, k: int,
                               mapq_min: int, xa_max: int, chunksize: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return stacked counts for ΔAS and ΔMAPQ per decile.
    Shapes:
      dAS_counts: (k, nbins_AS)
      dAS_overflow: (k,)
      dMQ_counts: (k, nbins_MQ)
      dMQ_overflow: (k,)
    """
    files = _filtered_files(workdir, sample)
    bcs = _chunk_bcs(chunk_file)

    H_dAS = [ _new_hist_from_spec((0.0, 100.0, 200)) for _ in range(k) ]
    H_dMQ = [ _new_hist_from_spec((0.0,  60.0, 120)) for _ in range(k) ]

    rb_map: Dict[tuple[str,str], dict] = {}
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
        if b1 is None or b2 is None: continue
        dAS = b1["AS"] - b2["AS"]
        dMQ = b1["MAPQ"] - b2["MAPQ"]
        dec_as = int(assign_decile(np.array([b1["AS"]]), as_edges)[0]) if as_edges.size else 1
        dec_mq = int(assign_decile(np.array([b1["MAPQ"]]), mq_edges)[0]) if mq_edges.size else 1
        if 1 <= dec_as <= k: H_dAS[dec_as-1].add(dAS)
        if 1 <= dec_mq <= k: H_dMQ[dec_mq-1].add(dMQ)

    dAS_counts = np.stack([H.counts for H in H_dAS], axis=0)
    dAS_over   = np.array([H.overflow for H in H_dAS], dtype=np.int64)
    dMQ_counts = np.stack([H.counts for H in H_dMQ], axis=0)
    dMQ_over   = np.array([H.overflow for H in H_dMQ], dtype=np.int64)
    return dAS_counts, dAS_over, dMQ_counts, dMQ_over

# --- REDUCE helpers ---

def _edges_from_hist_counts(as_counts: np.ndarray, mq_counts: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    # rebuild temporary hists to get edges
    H_AS = _new_hist_from_spec(_AS_RANGE); H_AS.counts = as_counts.copy()
    H_MQ = _new_hist_from_spec(_MQ_RANGE); H_MQ.counts = mq_counts.copy()

    def edges_from_hist(H: DeltaHist, k: int) -> np.ndarray:
        counts = H.counts.astype(np.int64)
        if counts.sum() == 0: return np.array([])
        cum = np.cumsum(counts)
        tot = int(cum[-1] + H.overflow)
        targets = (np.linspace(0, 1, k + 1)[1:-1] * tot).astype(int)
        edges = []
        for t in targets:
            idx = int(np.searchsorted(cum, t, side="left"))
            idx = min(max(idx, 0), H.nbins - 1)
            edges.append(H.edges[idx])
        return np.unique(np.array(edges, dtype=float))

    return edges_from_hist(H_AS, k), edges_from_hist(H_MQ, k)

def compute_decile_edges(values: np.ndarray, k: int = 10) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0: return np.array([])
    qs = np.linspace(0, 1, k + 1)
    edges = np.quantile(vals, qs[1:-1])
    return np.unique(edges)

def assign_decile(v: np.ndarray, edges: np.ndarray) -> np.ndarray:
    if edges.size == 0: return np.full_like(v, np.nan, dtype=float)
    return (np.digitize(v, bins=edges, right=True) + 1).astype(float)

class DeltaHist:
    def __init__(self, lo=0.0, hi=100.0, nbins=200):
        self.lo, self.hi, self.nbins = float(lo), float(hi), int(nbins)
        self.edges = np.linspace(self.lo, self.hi, self.nbins + 1)
        self.counts = np.zeros(self.nbins, dtype=np.int64)
        self.overflow = 0
    def add(self, x):
        if not np.isfinite(x): return
        if x >= self.hi: self.overflow += 1; return
        if x <= self.lo: idx = 0
        else:
            idx = int((x - self.lo) / (self.hi - self.lo) * self.nbins)
            idx = min(max(idx, 0), self.nbins - 1)
        self.counts[idx] += 1
    def tail_p(self, x):
        if not np.isfinite(x): return np.nan
        total = int(self.counts.sum() + self.overflow)
        if total == 0: return np.nan
        if x >= self.hi: return self.overflow / total
        if x <= self.lo: return 1.0
        idx = int((x - self.lo) / (self.hi - self.lo) * self.nbins)
        idx = min(max(idx, 0), self.nbins - 1)
        tail = int(self.counts[idx:].sum() + self.overflow)
        return tail / total

def _filtered_files(workdir: Path, sample: str) -> list[Path]:
    d = Path(workdir) / sample / "filtered_QCFiles"
    if not d.exists(): raise FileNotFoundError(f"Not found: {d}")
    files = sorted(d.glob("filtered_*_QCMapping.txt"))
    if not files: raise FileNotFoundError(f"No filtered_* files under {d}")
    return files

def _iter_read_bc_rows(fp: Path, bcs: set[str], chunksize: int, mapq_min: int, xa_max: int):
    usecols = ["Read","BC","MAPQ","AS","NM","XAcount"]
    for c in pd.read_csv(fp, sep="\t", header=0, usecols=usecols, chunksize=chunksize,
                        dtype={"Read":"string","BC":"string"}, engine="c", low_memory=True, memory_map=False):
        c = c[c["BC"].astype(str).isin(bcs)]
        if c.empty: continue
        for col in ("AS","MAPQ","NM","XAcount"): c[col] = pd.to_numeric(c[col], errors="coerce")
        if mapq_min > 0: c = c[c["MAPQ"].fillna(0) >= mapq_min]
        if xa_max >= 0: c = c[c["XAcount"].fillna(0) <= xa_max]
        if c.empty: continue
        c = c.drop_duplicates(subset=["Read","BC"], keep="first")
        yield c.itertuples(index=False)

# ---------- Pass A: learn global decile edges ----------
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
    if not chunk_list: raise typer.BadParameter(f"No chunk files under {chunks_dir}")

    # coarse histograms for winners
    H_AS, H_MQ = DeltaHist(lo=0, hi=200, nbins=2000), DeltaHist(lo=0, hi=60, nbins=120)

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
            if b1 is None: continue
            H_AS.add(b1["AS"]); H_MQ.add(b1["MAPQ"])

    def learn_edges_parallel(workdir: Path, sample: str, chunks_dir: Path,
                         out_model: Path | None, mapq_min: int, xa_max: int, chunksize: int, k: int,
                         threads: int = 8) -> Path:
        """Parallel map-reduce: per-chunk winner histograms → global decile edges."""
        chunk_files = sorted(Path(chunks_dir).glob(f"{sample}_cell_map_ref_chunk_*.txt"))
        if not chunk_files: raise typer.BadParameter(f"No chunk files under {chunks_dir}")

        # reduce buffers
        as_counts = np.zeros(_AS_RANGE[2], dtype=np.int64)
        mq_counts = np.zeros(_MQ_RANGE[2], dtype=np.int64)

    def _worker(chf: Path):
        return _map_winner_hist_for_chunk(workdir, sample, chf, mapq_min, xa_max, chunksize)
        
    total = len(chunk_files)
    typer.echo(f"[assign/edges] start: {total} chunks, threads={min(threads, total)}")
    with ThreadPoolExecutor(max_workers=min(threads, total)) as ex:
        fut = {ex.submit(_worker, ch): ch for ch in chunk_files}
        done = 0
        for f in as_completed(fut):
            ch = fut[f]
            try:
                as_c, mq_c = f.result()
                as_counts += as_c; mq_counts += mq_c
            except Exception as e:
                typer.echo(f"[assign/edges][ERROR] {ch.name}: {e}")
                raise
                done += 1
                if done % 5 == 0 or done == total:
                    typer.echo(f"[assign/edges] {done}/{total} chunks")
        
    as_edges, mq_edges = _edges_from_hist_counts(as_counts, mq_counts, k)
    model_path = out_model or (Path(workdir)/sample/"ExplorationReadLevel"/"global_edges.npz")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(model_path, as_edges=as_edges, mq_edges=mq_edges, k=k)
    typer.echo(f"[assign/edges] done → {model_path}")
    return model_path


# ---------- Pass B: learn per-decile Δ ECDFs (histograms) ----------
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
    if not chunk_list: raise typer.BadParameter(f"No chunk files under {chunks_dir}")
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
            if b1 is None or b2 is None: continue
            dAS = b1["AS"] - b2["AS"]; dMQ = b1["MAPQ"] - b2["MAPQ"]
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

def learn_ecdfs_parallel(workdir: Path, sample: str, chunks_dir: Path,
                         edges_model: Path, out_model: Path | None,
                         mapq_min: int, xa_max: int, chunksize: int, threads: int = 8) -> Path:
    """Parallel map-reduce: per-chunk per-decile Δ hist → global Δ ECDF hist model."""
    dat = np.load(edges_model)
    as_edges, mq_edges, k = dat["as_edges"], dat["mq_edges"], int(dat["k"])

    chunk_files = sorted(Path(chunks_dir).glob(f"{sample}_cell_map_ref_chunk_*.txt"))
    if not chunk_files: raise typer.BadParameter(f"No chunk files under {chunks_dir}")

    # reduce buffers
    Htmp_AS = _new_hist_from_spec((0.0, 100.0, 200))
    Htmp_MQ = _new_hist_from_spec((0.0, 60.0, 120))
    dAS_counts = np.zeros((k, Htmp_AS.nbins), dtype=np.int64)
    dAS_over   = np.zeros((k,), dtype=np.int64)
    dMQ_counts = np.zeros((k, Htmp_MQ.nbins), dtype=np.int64)
    dMQ_over   = np.zeros((k,), dtype=np.int64)

    def _worker(chf: Path):
        return _map_delta_hists_for_chunk(workdir, sample, chf, as_edges, mq_edges, k,
                                          mapq_min, xa_max, chunksize)
        total = len(chunk_files)
        typer.echo(f"[assign/ecdf] start: {total} chunks, threads={min(threads, total)}")
        with ThreadPoolExecutor(max_workers=min(threads, total)) as ex:
            fut = {ex.submit(_worker, ch): ch for ch in chunk_files}
            done = 0
            for f in as_completed(fut):
                ch = fut[f]
                try:
                    das, daso, dmq, dmqo = f.result()
                    dAS_counts += das; dAS_over += daso
                    dMQ_counts += dmq; dMQ_over += dmqo
                except Exception as e:
                    typer.echo(f"[assign/ecdf][ERROR] {ch.name}: {e}")
                    raise
                done += 1
                if done % 5 == 0 or done == total:
                    typer.echo(f"[assign/ecdf] {done}/{total} chunks")

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
        return model_path
                             
def edges_from_hist(H: DeltaHist, k: int) -> np.ndarray:
    counts = H.counts.astype(np.int64)
    if counts.sum() == 0: return np.array([])
    cum = np.cumsum(counts)
    tot = int(cum[-1] + H.overflow)
    targets = (np.linspace(0, 1, k + 1)[1:-1] * tot).astype(int)
    edges = []
    for t in targets:
        idx = int(np.searchsorted(cum, t, side="left"))
        idx = min(max(idx, 0), H.nbins - 1)
        edges.append(H.edges[idx])
    return np.unique(np.array(edges, dtype=float))

    as_edges, mq_edges = edges_from_hist(H_AS, k), edges_from_hist(H_MQ, k)
    model_path = out_model or (Path(workdir)/sample/"ExplorationReadLevel"/"global_edges.npz")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(model_path, as_edges=as_edges, mq_edges=mq_edges, k=k)
    typer.echo(f"[assign/edges] saved {model_path}")


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

    def _worker(chf: Path):
        return _map_delta_hists_for_chunk(workdir, sample, chf, as_edges, mq_edges, k,
                                          mapq_min, xa_max, chunksize)

    with ThreadPoolExecutor(max_workers=min(threads, len(chunk_files))) as ex:
        for das, daso, dmq, dmqo in ex.map(_worker, chunk_files):
            dAS_counts += das; dAS_over += daso
            dMQ_counts += dmq; dMQ_over += dmqo

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
    return model_path
    
# ---------- Pass C: score one chunk -> write *raw.tsv.gz and *filtered.tsv.gz ----------
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
    out_raw_dir.mkdir(parents=True, exist_ok=True); out_filtered_dir.mkdir(parents=True, exist_ok=True)

    # naming compatible with previous pipeline
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

    # RAW
    raw_rows = []
    for (read, bc), acc in rb_map.items():
        b1, b2, bw = acc["best1"], acc.get("best2"), acc.get("worst")
        if b1 is None: continue
        as1, mq1, nm1 = b1["AS"], b1["MAPQ"], b1["NM"]
        if b2 is not None:
            as2, mq2, nm2 = b2["AS"], b2["MAPQ"], b2["NM"]
            dAS, dMQ, dNM = as1-as2, mq1-mq2, nm2-nm1
        else:
            as2=mq2=nm2=np.nan; dAS=dMQ=dNM=np.nan
        dec_as = int(assign_decile(np.array([as1]), as_edges)[0]) if as_edges.size else 1
        dec_mq = int(assign_decile(np.array([mq1]), mq_edges)[0]) if mq_edges.size else 1
        p_as = H_dAS[dec_as-1].tail_p(dAS) if (np.isfinite(dAS)) else np.nan
        p_mq = H_dMQ[dec_mq-1].tail_p(dMQ) if (np.isfinite(dMQ)) else np.nan
        clear_by_p  = (np.isfinite(p_as) and p_as <= alpha) or (np.isfinite(p_mq) and p_mq <= alpha)
        clear_by_nm = (np.isfinite(dNM) and dNM > 0)
        single_hit  = b2 is None
        klass = "winner" if (clear_by_p or clear_by_nm or single_hit) else "ambiguous"
        raw_rows.append({
            "Read": read, "BC": bc,
            "Genome_winner": b1["g"], "AS_winner": as1, "MAPQ_winner": mq1, "NM_winner": nm1,
            "Genome_2": b2["g"] if b2 else "", "AS_2": as2, "MAPQ_2": mq2, "NM_2": nm2,
            "Genome_last": bw["g"] if bw else "", "AS_last": bw["AS"] if bw else np.nan,
            "MAPQ_last": bw["MAPQ"] if bw else np.nan, "NM_last": bw["NM"] if bw else np.nan,
            "delta_AS_1_2": dAS, "delta_MAPQ_1_2": dMQ, "delta_NM_1_2": dNM,
            "delta_AS_1_last": (as1 - (bw["AS"] if bw else np.nan)),
            "delta_MAPQ_1_last": (mq1 - (bw["MAPQ"] if bw else np.nan)),
            "delta_NM_1_last": ((bw["NM"] if bw else np.nan) - nm1),
            "n_genomes_considered": len(acc.get("seen_g", [])),
            "decile_AS": dec_as, "decile_MAPQ": dec_mq, "p_as_decile": p_as, "p_mq_decile": p_mq,
            "assigned_class": klass,
        })
    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(raw_out, sep="\t", index=False, compression="gzip")

    # FILTERED (winner-only if class==winner; else all)
    keep_reads = set(raw_df["Read"].tolist())
    class_by = dict(zip(raw_df["Read"], raw_df["assigned_class"]))
    gwin_by  = dict(zip(raw_df["Read"], raw_df["Genome_winner"]))
    out_rows = []
    usecols2 = ["Read","BC","Genome","AS","MAPQ","NM","XAcount"]
    for fp in files:
        genome = _genome_from_filename(fp)
        for c in pd.read_csv(fp, sep="\t", header=0, usecols=usecols2, chunksize=chunksize):
            c = c[c["BC"].astype(str).isin({bc for _, bc in rb_map.keys()})]
            if c.empty: continue
            c = c[c["Read"].astype(str).isin(keep_reads)]
            if c.empty: continue
            for col in ("AS","MAPQ","NM","XAcount"): c[col] = pd.to_numeric(c[col], errors="coerce")
            for row in c.itertuples(index=False):
                r = str(row.Read); klass = class_by.get(r, "ambiguous")
                if klass == "winner":
                    if genome == gwin_by.get(r):
                        out_rows.append(row._asdict() | {"Genome": genome, "delta_AS": np.nan, "assigned_class": "winner"})
                else:
                    out_rows.append(row._asdict() | {"Genome": genome, "delta_AS": np.nan, "assigned_class": "ambiguous"})
    pd.DataFrame(out_rows).to_csv(filt_out, sep="\t", index=False, compression="gzip")

# ---------- Convenience: run all passes for a library ----------
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
