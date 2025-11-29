from pathlib import Path
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Callable, Sequence, Any
import json, time, hashlib
import typer

# --------------------------------------------------------------------
# Public entrypoint
# --------------------------------------------------------------------
def run_pipeline(cfg: dict, params: dict, version: str,
                 resume: bool, force: List[str], skip_to: str, only: List[str]) -> Dict[str, List[str]]:
    """
    Execute pipeline with resumable sentinels.

    Steps: extract → filter → chunks → assign → genotyping 

    Args:
      cfg: single-sample config (sample, workdir, genomes, …)
      params: runtime knobs (threads, verbose, assign={}, sentinel_dir=?, steps=?)
      version: string tag recorded in sentinels
      resume: keep completed steps if True
      force: list of step names to recompute
      skip_to: optional step name to start from
      only: optional list of steps to run exclusively

    Returns:
      {"executed": [...], "skipped": [...]}
    """
    dirs = _dirs(cfg)
    sentinel_dir = Path(params.get("sentinel_dir", dirs["root"] / "_sentinels"))
    steps_all = params.get("steps", STEP_ORDER[:])

    plan = _plan(steps_all, skip_to=skip_to, only=only)
    _apply_resume_policy(sentinel_dir, plan, resume=resume, force=force)

    ctx = Ctx(cfg=cfg, params=params, dirs=dirs, version=version, sentinel_dir=sentinel_dir)

    executed, skipped = [], []
    for step in plan:
        sp = _sentinel_path(sentinel_dir, step)
        if sp.exists():
            skipped.append(step)
            continue

        _ensure_step_prereqs(ctx, step)
        RUNNERS[step](ctx)

        _mark_done(sentinel_dir, step, meta={
            "sample": cfg.get("sample"),
            "version": version,
            "time": _now(),
            "hash": _cfg_hash(cfg, params),
        })
        executed.append(step)

    return {"executed": executed, "skipped": skipped}

# --------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------
STEP_ORDER: List[str] = ["extract", "filter", "chunks", "assign", "genotyping"]

# --------------------------------------------------------------------
# Context and dirs
# --------------------------------------------------------------------
@dataclass
class Ctx:
    cfg: dict
    params: dict
    dirs: Dict[str, Path]
    version: str
    sentinel_dir: Path

def _dirs(cfg: dict) -> Dict[str, Path]:
    root = Path(cfg["workdir"]).expanduser().resolve() / cfg["sample"]
    return {
        "root": root,
        "qc": root / "qc",
        "filtered": root / "filtered_QCFiles",
        "chunks": root / "cell_map_ref_chunks",
        "final": root / "final",
    }

def _ensure_dirs(d: Dict[str, Path]) -> None:
    for p in d.values():
        if isinstance(p, Path):
            p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------
# Planning and sentinels
# --------------------------------------------------------------------
def _plan(all_steps: Sequence[str], skip_to: str, only: List[str]) -> List[str]:
    valid = set(STEP_ORDER)
    if only:
        bad = [s for s in only if s not in valid]
        if bad:
            raise ValueError(f"Unknown steps in --only: {', '.join(bad)}")
        return [s for s in STEP_ORDER if s in set(only)]
    if skip_to:
        if skip_to not in valid:
            raise ValueError(f"--skip-to must be one of: {', '.join(STEP_ORDER)}")
        idx = STEP_ORDER.index(skip_to)
        return STEP_ORDER[idx:]
    allowed = set(all_steps) & valid
    return [s for s in STEP_ORDER if s in allowed]

def _sentinel_path(sentinel_dir: Path, step: str) -> Path:
    return sentinel_dir / f"{step}.ok"

def _apply_resume_policy(sentinel_dir: Path, run_steps: List[str], resume: bool, force: List[str]) -> None:
    sentinel_dir.mkdir(parents=True, exist_ok=True)
    if not resume:
        for s in run_steps:
            sp = _sentinel_path(sentinel_dir, s)
            if sp.exists():
                sp.unlink()
    for s in force or []:
        sp = _sentinel_path(sentinel_dir, s)
        if sp.exists():
            sp.unlink()

def _mark_done(sentinel_dir: Path, step: str, meta: dict) -> None:
    p = _sentinel_path(sentinel_dir, step)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"step": step, **meta}, indent=2))

def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())

def _cfg_hash(cfg: dict, params: dict) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(cfg, sort_keys=True, default=str).encode())
    h.update(b"|")
    h.update(json.dumps(params, sort_keys=True, default=str).encode())
    return h.hexdigest()[:12]

# --------------------------------------------------------------------
# Prereqs
# --------------------------------------------------------------------
def _ensure_minimal_chunk(dirs: Dict[str, Path], sample: str) -> None:
    chunks = dirs["chunks"]
    chunks.mkdir(parents=True, exist_ok=True)
    has_manifest = (chunks / "manifest.json").exists()
    has_txt = any(chunks.glob("*_cell_map_ref_chunk_*.txt"))
    if has_manifest or has_txt:
        return
    dummy = chunks / f"{sample}_cell_map_ref_chunk_0001.txt"
    dummy.touch(exist_ok=True)
    (chunks / "manifest.json").write_text(json.dumps({
        "version": 1, "n_chunks": 1,
        "chunks": [{"id": dummy.stem, "path": dummy.name}],
        "derived_from": {"filtered_inputs_sha256": "bootstrap", "upstream_generation": 0},
    }, indent=2))

def _ensure_step_prereqs(ctx: Ctx, step: str) -> None:
    _ensure_dirs(ctx.dirs)
    if step in ("assign", "genotyping", "summarize", "interpool", "chunks"):
        _ensure_minimal_chunk(ctx.dirs, ctx.cfg["sample"])

# --------------------------------------------------------------------
# Step runners
# --------------------------------------------------------------------
def _run_extract(ctx: Ctx) -> None:
    from .extract import bam_to_qc
    cfg = ctx.cfg; d = ctx.dirs
    genomes = sorted(cfg["genomes"].items())
    threads = int(ctx.params.get("threads", 4))
    nprocs = max(1, min(threads, len(genomes)))
    with ProcessPoolExecutor(max_workers=nprocs) as ex:
        futs = [ex.submit(bam_to_qc, Path(bam), d["qc"] / f"{g}_QCMapping.txt", cfg["sample"])
                for g, bam in genomes]
        for f in as_completed(futs):
            f.result()

def _run_filter(ctx: Ctx) -> None:
    from .filtering import filter_qc_file
    cfg = ctx.cfg; d = ctx.dirs
    genomes = sorted(cfg["genomes"].keys())
    minf = int(cfg.get("min_barcode_freq", 10))
    threads = int(ctx.params.get("threads", 4))
    nprocs = max(1, min(threads, len(genomes)))
    with ProcessPoolExecutor(max_workers=nprocs) as ex:
        futs = []
        for g in genomes:
            ip = d["qc"] / f"{g}_QCMapping.txt"
            op = d["filtered"] / f"filtered_{g}_QCMapping.txt"
            futs.append(ex.submit(filter_qc_file, ip, op, minf, cfg["sample"]))
        for f in as_completed(futs):
            f.result()

def _run_chunks(ctx: Ctx) -> None:
    from .chunks import make_barcode_chunks
    cfg = ctx.cfg; d = ctx.dirs
    _ensure_minimal_chunk(d, cfg["sample"])
    _ = make_barcode_chunks(d["filtered"], d["chunks"], cfg["sample"], int(cfg.get("chunk_size_cells", 5000)))

def _run_assign(ctx: Ctx) -> None:
    from .assign_streaming import learn_edges_parallel, learn_ecdfs_parallel, score_chunk
    cfg = ctx.cfg; d = ctx.dirs
    workdir = Path(cfg["workdir"]); sample = cfg["sample"]
    chunks_dir = d["chunks"]

    aconf = cfg.get("assign", {}) if isinstance(cfg.get("assign"), dict) else {}
    alpha    = float(aconf.get("alpha", 0.05))
    k        = int(aconf.get("k", 10))
    mapq_min = int(aconf.get("mapq_min", 20))
    xa_max   = int(aconf.get("xa_max", 2))
    chunksize_val  = int(aconf.get("chunksize", 1_000_000))
    batch_size_val = int(aconf.get("batch_size", 32))
    if "chunks_dir" in aconf:
        chunks_dir = workdir / sample / str(aconf["chunks_dir"])

    chunk_files = sorted(chunks_dir.glob("*_cell_map_ref_chunk_*.txt"))
    exp_dir = workdir / sample / "ExplorationReadLevel"; exp_dir.mkdir(parents=True, exist_ok=True)
    edges_npz = exp_dir / "global_edges.npz"
    ecdf_npz  = exp_dir / "global_ecdf.npz"

    threads = int(ctx.params.get("threads", 16))
    verbose = bool(ctx.params.get("verbose", True))
    edges_workers = aconf.get("edges_workers", None)
    edges_max_reads = aconf.get("edges_max_reads", None)
    ecdf_workers = aconf.get("ecdf_workers", None)
    if edges_workers is not None:
        edges_workers = max(1, min(int(edges_workers), threads))
    if ecdf_workers is None:
        ecdf_workers = threads
    ecdf_workers = max(1, min(int(ecdf_workers), threads))

    learn_edges_parallel(
        workdir=workdir, sample=sample, chunks_dir=chunks_dir, out_model=edges_npz,
        mapq_min=parse_int(mapq_min), xa_max=parse_int(xa_max),
        chunksize=parse_int(chunksize_val), k=parse_int(k), batch_size=parse_int(batch_size_val),
        threads=parse_int(threads), verbose=verbose,
        edges_workers=edges_workers, edges_max_reads=edges_max_reads
    )
    learn_ecdfs_parallel(
        workdir=workdir, sample=sample, chunks_dir=chunks_dir, edges_model=edges_npz, out_model=ecdf_npz,
        mapq_min=parse_int(mapq_min), xa_max=parse_int(xa_max),
        chunksize=parse_int(chunksize_val), verbose=verbose, workers=parse_int(ecdf_workers)
    )

    pool_n = max(1, min(threads, len(chunk_files)))
    with ProcessPoolExecutor(max_workers=pool_n) as ex:
        fut = {
            ex.submit(
                score_chunk,
                workdir=workdir, sample=sample, chunk_file=chf, ecdf_model=ecdf_npz,
                out_raw_dir=None, out_filtered_dir=None,
                mapq_min=mapq_min, xa_max=xa_max, chunksize=chunksize_val, alpha=alpha,
            ): chf
            for chf in chunk_files
        }
        for f in as_completed(fut):
            f.result()

def _run_genotyping(ctx: Ctx) -> None:
    from .genotyping import genotyping as _run
    import typer

    cfg = ctx.cfg
    d = ctx.dirs

    # Only use assign outputs from the genotyping step (filtered TSVs)
    assign_glob = str(d["chunks"] / "**" / "*filtered.tsv.gz")

    # Global pipeline threads (fallback for genotyping-specific threads)
    threads_global = int(ctx.params.get("threads", 1))

    # Genotyping-specific overrides from params (set by CLI `run`)
    # Example:
    # "genotyping": {
    #   "min_reads": 100,
    #   "beta": 0.5,
    #   "winner_only": true,
    #   ...
    # }
    gconf = ctx.params.get("genotyping", {}) or {}

    min_reads = int(gconf.get("min_reads", 100))
    beta = float(gconf.get("beta", 1))
    w_as = float(gconf.get("w_as", 0.5))
    w_mapq = float(gconf.get("w_mapq", 1.0))
    w_nm = float(gconf.get("w_nm", 1.0))
    ambient_const = float(gconf.get("ambient_const", 1e-3))
    tau_drop = float(gconf.get("tau_drop", 8.0))
    topk_genomes = int(gconf.get("topk_genomes", 3))
    shards = int(gconf.get("shards", 32))
    chunk_rows = int(gconf.get("chunk_rows", 1_000_000))
    winner_only = bool(gconf.get("winner_only", False))
    
    # New: winner-only flag
    winner_only = bool(gconf.get("winner_only", False))
    single_mass_min = float(gconf.get("single_mass_min", 0.7))
    ratio_top1_top2_min = float(gconf.get("ratio_top1_top2_min", 2.0))
    bic_margin = float(gconf.get("bic_margin", 6.0))
    doublet_minor_min = float(gconf.get("doublet_minor_min", 0.2))


    threads = int(gconf.get("threads", threads_global))
    if threads < 1:
        threads = 1

    pass1_workers = gconf.get("pass1_workers", None)
    if pass1_workers is not None:
        pass1_workers = max(1, int(pass1_workers))

    # Log the effective genotyping parameters
    # Log the effective genotyping parameters (single positional argument!)
    msg = (
        "[genotyping] effective parameters: "
        f"min_reads={min_reads}, beta={beta}, w_as={w_as}, "
        f"w_mapq={w_mapq}, w_nm={w_nm}, ambient_const={ambient_const}, "
        f"single_mass_min={single_mass_min}, bic_margin={bic_margin}, "
        f"ratio_top1_top2_min={ratio_top1_top2_min}, winner_only={winner_only}, doublet_minor_min={doublet_minor_min}"
    )
    typer.echo(msg)

    _run(
        assign=assign_glob,
        outdir=d["final"],
        sample=cfg["sample"],
        min_reads=min_reads,
        beta=beta,
        w_as=w_as,
        w_mapq=w_mapq,
        w_nm=w_nm,
        ambient_const=ambient_const,
        tau_drop=tau_drop,
        bic_margin=bic_margin,
        topk_genomes=topk_genomes,
        make_report=True,
        threads=threads,
        shards=shards,
        chunk_rows=chunk_rows,
        pass1_workers=pass1_workers,
        winner_only=winner_only,   
        ratio_top1_top2_min = ratio_top1_top2_min,
        single_mass_min=single_mass_min,
        doublet_minor_min=doublet_minor_min,
    )

def _run_interpool(ctx: Ctx) -> None:
    # Cross-sample driver. No-op here; call CLI `interpool` separately.
    return

RUNNERS: Dict[str, Callable[[Ctx], None]] = {
    "extract": _run_extract,
    "filter": _run_filter,
    "chunks": _run_chunks,
    "assign": _run_assign,
    "genotyping": _run_genotyping,
}

# --------------------------------------------------------------------
# Utils
# --------------------------------------------------------------------
def parse_int(x) -> int:
    return int(x)
