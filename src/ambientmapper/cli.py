#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import json, csv, hashlib, time
import typer
from typing import Dict, List, Optional
import json as _json

# pipeline runner
from .pipeline import run_pipeline

app = typer.Typer(help="ambientmapper: local-first ambient cleaning pipeline")

# ----------------
# Helpers
# ----------------
def _clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))

def _parse_csv_list(s: str | None) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

def _cfg_dirs(cfg: Dict[str, object]) -> Dict[str, Path]:
    root = Path(cfg["workdir"]) / cfg["sample"]
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

def _cfg_from_inline(sample: str, genomes_csv: str, bams_csv: str,
                     workdir: str, min_barcode_freq: int, chunk_size_cells: int) -> Dict[str, object]:
    genomes = _parse_csv_list(genomes_csv)
    bam_paths = [Path(p).expanduser().resolve() for p in _parse_csv_list(bams_csv)]
    if len(genomes) != len(bam_paths):
        raise typer.BadParameter(f"#genome names ({len(genomes)}) must equal #bams ({len(bam_paths)}).")
    for p in bam_paths:
        if not p.exists():
            raise typer.BadParameter(f"BAM not found: {p}")
    if len(set(genomes)) != len(genomes):
        raise typer.BadParameter("Genome names must be unique.")
    return {
        "sample": sample,
        "genomes": {g: str(p) for g, p in zip(genomes, bam_paths)},
        "min_barcode_freq": int(min_barcode_freq),
        "chunk_size_cells": int(chunk_size_cells),
        "workdir": str(Path(workdir).expanduser().resolve()),
    }

def _load_config(config: Path) -> Dict[str, object]:
    """Load a single-sample JSON config."""
    cfg = json.loads(Path(config).read_text())
    if "sample" not in cfg or "workdir" not in cfg or "genomes" not in cfg:
        raise typer.BadParameter("Config must include: sample, workdir, genomes")
    return cfg

def _cfgs_from_tsv(tsv: Path, min_barcode_freq: int, chunk_size_cells: int) -> List[Dict[str, object]]:
    groups: Dict[str, Dict[str, str]] = {}
    workdirs: Dict[str, str] = {}
    with open(tsv, "r", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        need = {"sample","genome","bam","workdir"}
        if need - set(r.fieldnames or []):
            raise typer.BadParameter(f"TSV must include columns: {sorted(need)}")
        for row in r:
            s = row["sample"].strip()
            g = row["genome"].strip()
            b = str(Path(row["bam"]).expanduser().resolve())
            w = str(Path(row["workdir"]).expanduser().resolve())
            if s not in groups:
                groups[s] = {}
                workdirs[s] = w
            if g in groups[s]:
                raise typer.BadParameter(f"Duplicate genome '{g}' for sample '{s}' in TSV.")
            if not Path(b).exists():
                raise typer.BadParameter(f"BAM not found: {b}")
            groups[s][g] = b
    cfgs: List[Dict[str, object]] = []
    for s, gmap in groups.items():
        if not gmap:
            continue
        cfgs.append({
            "sample": s,
            "genomes": gmap,
            "min_barcode_freq": int(min_barcode_freq),
            "chunk_size_cells": int(chunk_size_cells),
            "workdir": workdirs[s],
        })
    return cfgs

def _apply_assign_overrides(
    cfg: Dict[str, object],
    alpha: Optional[float] = None,
    k: Optional[int] = None,
    mapq_min: Optional[int] = None,
    xa_max: Optional[int] = None,
    chunksize: Optional[int] = None,
    batch_size: Optional[int] = None,
    edges_workers: Optional[int] = None,
    edges_max_reads: Optional[int] = None,
    ecdf_workers: Optional[int] = None,
) -> None:
    """Ensure cfg['assign'] exists and apply CLI overrides if provided."""
    assign = cfg.get("assign")
    if not isinstance(assign, dict):
        assign = {}
        cfg["assign"] = assign

    if alpha is not None:          assign["alpha"] = float(alpha)
    if k is not None:              assign["k"] = int(k)
    if mapq_min is not None:       assign["mapq_min"] = int(mapq_min)
    if xa_max is not None:         assign["xa_max"] = int(xa_max)
    if chunksize is not None:      assign["chunksize"] = int(chunksize)
    if batch_size is not None:     assign["batch_size"] = int(batch_size)
    if edges_workers is not None:  assign["edges_workers"] = int(edges_workers)
    if edges_max_reads is not None:
        assign["edges_max_reads"] = int(edges_max_reads)
        typer.echo(f"[assign] edges_max_reads cap in effect: {int(edges_max_reads):,} rows/genome")
    if ecdf_workers is not None:   assign["ecdf_workers"] = int(ecdf_workers)

def _ensure_minimal_chunk(workdir: Path, sample: str) -> None:
    """
    If the sample's chunks dir has no manifest and no *_cell_map_ref_chunk_*.txt,
    synthesize a single dummy chunk and a manifest so the DAG can start.
    """
    chunks = (Path(workdir) / sample / "cell_map_ref_chunks")
    chunks.mkdir(parents=True, exist_ok=True)

    has_manifest = (chunks / "manifest.json").exists()
    has_txt = any(chunks.glob("*_cell_map_ref_chunk_*.txt"))
    if has_manifest or has_txt:
        return

    dummy = chunks / f"{sample}_cell_map_ref_chunk_0001.txt"
    dummy.touch(exist_ok=True)
    man = {
        "version": 1,
        "n_chunks": 1,
        "chunks": [{"id": dummy.stem, "path": dummy.name}],
        "derived_from": {"filtered_inputs_sha256": "bootstrap", "upstream_generation": 0},
    }
    (chunks / "manifest.json").write_text(json.dumps(man, indent=2))

STEP_ORDER = ["extract", "filter", "chunks", "assign", "genotyping", "summarize", "interpool"]

def _norm_steps(xs: list[str]) -> list[str]:
    s = set(STEP_ORDER)
    out = []
    for x in xs:
        if x not in s:
            raise typer.BadParameter(f"Unknown step '{x}'. Valid: {', '.join(STEP_ORDER)}")
        out.append(x)
    return out

def _sentinel_root(cfg: dict) -> Path:
    return Path(cfg["workdir"]) / cfg["sample"] / "_sentinels"

def _sentinel_path(cfg: dict, step: str) -> Path:
    return _sentinel_root(cfg) / f"{step}.ok"

def _is_done(cfg: dict, step: str) -> bool:
    return _sentinel_path(cfg, step).exists()

def _mark_done(cfg: dict, step: str, meta: dict | None = None) -> None:
    p = _sentinel_path(cfg, step)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "sample": cfg.get("sample"),
        "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
        "meta": meta or {},
    }
    p.write_text(_json.dumps(payload, indent=2))

def _clear_step(cfg: dict, step: str) -> None:
    p = _sentinel_path(cfg, step)
    if p.exists():
        p.unlink()

def _cfg_hash(cfg: dict, params: dict) -> str:
    h = hashlib.sha256()
    h.update(_json.dumps(cfg, sort_keys=True, default=str).encode())
    h.update(b"|")
    h.update(_json.dumps(params, sort_keys=True, default=str).encode())
    return h.hexdigest()[:12]

def _plan_steps(resume: bool, force: list[str], skip_to: str, only: list[str]) -> list[str]:
    """Return ordered steps to run given policy flags."""
    base = STEP_ORDER[:]
    if only:
        return _norm_steps(only)
    if skip_to:
        if skip_to not in STEP_ORDER:
            raise typer.BadParameter(f"--skip-to must be one of: {', '.join(STEP_ORDER)}")
        i = STEP_ORDER.index(skip_to)
        base = STEP_ORDER[i:]
    _ = _norm_steps(force) if force else []
    return base

def _apply_resume_policy(cfg: dict, resume: bool, force: list[str], run_steps: list[str]) -> None:
    """Enforce resume/force by pruning sentinels."""
    root = _sentinel_root(cfg); root.mkdir(parents=True, exist_ok=True)
    if not resume:
        for s in run_steps:
            _clear_step(cfg, s)
    for s in force:
        _clear_step(cfg, s)
# ---- end helpers --------------------------------------------------------------

# --------------------------------------------------------------------------------
# Existing stepwise commands
# --------------------------------------------------------------------------------

@app.command()
def extract(config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
            threads: int = typer.Option(4, "--threads", "-t", min=1)):
    from .extract import bam_to_qc
    cfg = json.loads(Path(config).read_text()); d = _cfg_dirs(cfg); _ensure_dirs(d)
    genomes = sorted(cfg["genomes"].items())
    with ProcessPoolExecutor(max_workers=_clamp(threads, 1, len(genomes))) as ex:
        futs = [ex.submit(bam_to_qc, Path(bam), d["qc"] / f"{g}_QCMapping.txt", cfg["sample"])
                for g, bam in genomes]
        for f in as_completed(futs):
            f.result()
    typer.echo("[extract] done")

@app.command()
def filter(config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
           threads: int = typer.Option(4, "--threads", "-t", min=1)):
    from .filtering import filter_qc_file
    cfg = json.loads(Path(config).read_text()); d = _cfg_dirs(cfg); _ensure_dirs(d)
    genomes = sorted(cfg["genomes"].keys()); minf = int(cfg["min_barcode_freq"])
    with ProcessPoolExecutor(max_workers=_clamp(threads, 1, len(genomes))) as ex:
        futs = []
        for g in genomes:
            ip = d["qc"] / f"{g}_QCMapping.txt"
            op = d["filtered"] / f"filtered_{g}_QCMapping.txt"
            futs.append(ex.submit(filter_qc_file, ip, op, minf, cfg["sample"]))
        for f in as_completed(futs):
            f.result()
    typer.echo("[filter] done")

@app.command()
def chunks(config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True)):
    from .chunks import make_barcode_chunks
    cfg = json.loads(Path(config).read_text()); d = _cfg_dirs(cfg); _ensure_dirs(d)
    n = make_barcode_chunks(d["filtered"], d["chunks"], cfg["sample"], int(cfg["chunk_size_cells"]))
    typer.echo(f"[chunks] wrote {n} chunk files")

@app.command()
def assign(
    config: Path = typer.Option(..., "-c", "--config", exists=True),
    threads: int = typer.Option(16, "-t", "--threads"),
    edges_workers: Optional[int] = typer.Option(None, "--edges-workers",
        help="Max process workers for learn_edges_parallel (default: = --threads)"),
    edges_max_reads: Optional[int] = typer.Option(None, "--edges-max-reads",
        help="Optional cap of reads per genome when learning edges"),
    ecdf_workers: Optional[int] = typer.Option(None, "--ecdf-workers",
        help="Max workers for learn_ecdfs (default: = --threads)"),
    chunksize: Optional[int] = typer.Option(
        None, "--chunksize",
        help="Override pandas read_csv chunk size (rows per chunk). CLI > assign{} > top-level > default(1000000)."),
    batch_size: Optional[int] = typer.Option(
        None, "--batch-size",
        help="Override batch size for batched winner-edge learning. CLI > assign{} > top-level > default(32)."),
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="Print per-chunk progress")
) -> None:
    """Learn edges/ECDFs and score each chunk (parallel)."""
    from .assign_streaming import learn_edges_parallel, learn_ecdfs_parallel, score_chunk
    from concurrent.futures import ProcessPoolExecutor, as_completed

    cfg = _load_config(config)
    d = _cfg_dirs(cfg); _ensure_dirs(d)

    _ensure_minimal_chunk(Path(cfg["workdir"]), cfg["sample"])

    workdir = Path(cfg["workdir"])
    sample = str(cfg["sample"])
    chunks_dir = d["chunks"]

    aconf = cfg.get("assign", {}) if isinstance(cfg.get("assign"), dict) else {}
    alpha    = float(aconf.get("alpha", 0.05))
    k        = int(aconf.get("k", 10))
    mapq_min = int(aconf.get("mapq_min", 20))
    xa_max   = int(aconf.get("xa_max", 2))

    chunksize_val  = int(chunksize if chunksize is not None else aconf.get("chunksize", cfg.get("chunksize", 1_000_000)))
    batch_size_val = int(batch_size if batch_size is not None else aconf.get("batch_size",  cfg.get("batch_size", 32)))

    if "chunks_dir" in aconf:
        chunks_dir = Path(workdir) / sample / str(aconf["chunks_dir"])

    chunk_files = sorted(chunks_dir.glob("*_cell_map_ref_chunk_*.txt"))
    if not chunk_files:
        dummy = chunks_dir / f"{sample}_cell_map_ref_chunk_0001.txt"
        dummy.parent.mkdir(parents=True, exist_ok=True)
        dummy.touch(exist_ok=True)
        (chunks_dir / "manifest.json").write_text(json.dumps({
            "version": 1,
            "n_chunks": 1,
            "chunks": [{"id": dummy.stem, "path": dummy.name}],
            "derived_from": {"filtered_inputs_sha256": "bootstrap", "upstream_generation": 0},
        }, indent=2))
        typer.echo(f"[assign] No chunk files found; created {dummy.name} to proceed.")
        chunk_files = [dummy]

    threads_eff = max(1, int(threads))
    if edges_workers is not None:
        edges_workers = 1 if edges_workers <= 0 else min(edges_workers, threads_eff)
    if edges_max_reads is not None and edges_max_reads <= 0:
        edges_max_reads = None
    if k <= 0:
        typer.echo("[assign] warn: k<=0 deciles → forcing k=10"); k = 10
    if chunksize_val <= 0:
        typer.echo("[assign] warn: chunksize<=0 → forcing 1_000_000"); chunksize_val = 1_000_000
    if batch_size_val <= 0:
        typer.echo("[assign] warn: batch_size<=0 → forcing 32"); batch_size_val = 32

    exp_dir   = Path(workdir) / sample / "ExplorationReadLevel"
    exp_dir.mkdir(parents=True, exist_ok=True)
    edges_npz = exp_dir / "global_edges.npz"
    ecdf_npz  = exp_dir / "global_ecdf.npz"

    if verbose:
        extra = []
        if edges_workers is not None: extra.append(f"edges_workers={edges_workers}")
        if edges_max_reads is not None: extra.append(f"edges_max_reads={edges_max_reads:,}")
        if ecdf_workers is not None:   extra.append(f"ecdf_workers={min(max(ecdf_workers or 0,1), threads_eff)}")
        typer.echo(f"[assign] effective chunksize={chunksize_val:,}  batch_size={batch_size_val}  "
                   f"threads={threads_eff}" + ("  " + "  ".join(extra) if extra else ""))

    learn_edges_parallel(
        workdir=workdir, sample=sample, chunks_dir=chunks_dir, out_model=edges_npz,
        mapq_min=mapq_min, xa_max=xa_max, chunksize=chunksize_val, k=k, batch_size=batch_size_val,
        threads=threads_eff, verbose=verbose, edges_workers=edges_workers, edges_max_reads=edges_max_reads
    )
    ecdf_workers_eff = (ecdf_workers if (ecdf_workers and ecdf_workers > 0) else threads_eff)
    ecdf_workers_eff = min(ecdf_workers_eff, threads_eff)
    learn_ecdfs_parallel(
        workdir=workdir, sample=sample, chunks_dir=chunks_dir, edges_model=edges_npz, out_model=ecdf_npz,
        mapq_min=mapq_min, xa_max=xa_max, chunksize=chunksize_val, verbose=verbose, workers=ecdf_workers_eff
    )

    pool_n = min(threads_eff, len(chunk_files))
    typer.echo(f"[assign/score] start: {len(chunk_files)} chunks, procs={pool_n}")
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
        done = 0; total = len(fut)
        for f in as_completed(fut):
            f.result()
            done += 1
            if done % 5 == 0 or done == total:
                typer.echo(f"[assign/score] {done}/{total} chunks")
    typer.echo("[assign/score] done")

@app.command()
def genotyping(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
    assign_glob: Optional[str] = typer.Option(None, "--assign", help="Glob to assign outputs (parquet/csv/tsv)"),
    outdir: Optional[Path] = typer.Option(None, "--outdir"),
    sample: Optional[str] = typer.Option(None, "--sample"),
    make_report: bool = typer.Option(True, "--report/--no-report"),
    threads: int = typer.Option(1, "--threads", help="Parallel workers for per-cell model selection."),
):
    """Posterior-aware genotyping (merge + summarize + optional post-steps)."""
    from .genotyping import genotyping as _run_genotyping
    import glob as _glob

    cfg = _json.loads(Path(config).read_text())
    d = _cfg_dirs(cfg); _ensure_dirs(d)

    sample = sample or cfg["sample"]
    outdir = outdir or d["final"]

    if not assign_glob:
        chunks_dir = d["chunks"]
        candidate_patterns = [
            str(chunks_dir / "**" / "*filtered.tsv.gz"),
            str(chunks_dir / "**" / "*.tsv.gz"),
            str(chunks_dir / "**" / "*.csv.gz"),
        ]
        matched = []
        for pat in candidate_patterns:
            matched.extend(_glob.glob(pat, recursive=True))
        if not matched:
            raise typer.BadParameter(
                f"No assign outputs found under {chunks_dir}.\n"
                f"Expected files like 'Sample_chunkNNN_filtered.tsv.gz'.\n"
                f"Provide --assign or run the 'assign' step first."
            )
        assign_glob = str(chunks_dir / "**" / "*filtered.tsv.gz")

    if not _glob.glob(assign_glob, recursive=True):
        raise typer.BadParameter(f"No assign files matched: {assign_glob}")

    if "sample" in cfg and cfg["sample"] != sample:
        typer.echo(f"[genotyping] warn: config sample={cfg['sample']} but CLI --sample={sample}")


        # I/O hygiene
    # Bound how many files genotyping tries to open in parallel downstream.
    # Your genotyper already has a threads parameter; keep it smaller by default.
    threads = max(1, int(threads))
    if threads > 16:
        typer.echo(f"[genotyping] capping threads from {threads} to 8 for merge memory safety")
        threads = 16

    typer.echo(f"[genotyping] sample={sample}  outdir={outdir}")
    typer.echo(f"[genotyping] assign_glob={assign_glob}")
    typer.echo(f"[genotyping] threads={threads}  report={'on' if make_report else 'off'}")
    typer.echo(f"[genotyping] sample={sample}  outdir={outdir}")
    typer.echo(f"[genotyping] assign_glob={assign_glob}")
    typer.echo(f"[genotyping] threads={threads}  report={'on' if make_report else 'off'}")

    _run_genotyping(
        assign=assign_glob,
        outdir=outdir,
        sample=sample,
        make_report=make_report,
        threads=int(threads),
        # explicit numeric defaults to avoid OptionInfo leaking into pydantic:
        beta=0.5,
        w_as=1.0,
        w_mapq=0.5,
        w_nm=0.25,
        ambient_const=1e-3,
        min_reads=100,
        tau_drop=8.0,
        topk_genomes=3,
)

@app.command()
def summarize(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
    assign_glob: Optional[str] = typer.Option(None, "--assign", help="Glob to assign outputs (parquet/tsv/csv)"),
):
    """[Deprecated] Back-compat wrapper to new merge.summarize."""
    from .merge import run as posterior_merge_run
    cfg = json.loads(Path(config).read_text()); d = _cfg_dirs(cfg); _ensure_dirs(d)
    if not assign_glob:
        assign_glob = str(d["chunks"] / "**" / "*")
    posterior_merge_run(assign=assign_glob, outdir=d["final"], sample=cfg["sample"], make_report=True)
    typer.echo(f"[summarize] outputs in {d['final']}")

@app.command()
def interpool(
    configs: Path = typer.Option(..., "--configs", help="TSV with: sample, genome, bam, workdir"),
    outdir: Path = typer.Option(None, "--outdir", help="Output dir for inter-pool summary (default: <first workdir>/interpool)")
):
    """Compare multiple pools (samples) after they have been run."""
    from .interpool import interpool_summary
    res = interpool_summary(configs_tsv=configs, outdir=outdir)
    typer.echo(f"[interpool] BC counts: {res['bc_counts']}")
    typer.echo(f"[interpool] Read composition: {res['read_comp']}")
    typer.echo(f"[interpool] PDF: {res['pdf']}")

@app.command()
def plate(
    workdir: Path = typer.Option(..., "--workdir", help="Output root where pool folders live"),
    plate_map: Path = typer.Option(..., "--plate-map", help="TSV mapping Sample<TAB>Wells (e.g., 'A1-12,B1-12')"),
    outdir: Path = typer.Option(None, "--outdir", help="Where to write plate outputs (default: WORKDIR/PlateSummary)"),
    xa_max: int = typer.Option(2, "--xa-max", help="Keep winner rows with XAcount <= xa_max; set -1 to disable"),
):
    """Plate-level summary: contamination distributions + 96-well heatmap."""
    from .plate import plate_summary_cli
    out = plate_summary_cli(workdir=workdir, plate_map=plate_map, outdir=outdir, xa_max=xa_max)
    typer.echo(f"[plate] per-BC: {out['per_bc']}")
    typer.echo(f"[plate] per-sample: {out['per_sample']}")
    typer.echo(f"[plate] density: {out['density_pdf']}")
    typer.echo(f"[plate] heatmap: {out['heatmap_pdf']}")

# --------------------------------------------------------------------------------
# End-to-end: resumable DAG + sentinels
# --------------------------------------------------------------------------------
@app.command()
def run(
    # mode A: single JSON
    config: Path = typer.Option(None, "--config", "-c", exists=True, readable=True, help="Single JSON config"),
    # mode B: inline one-sample
    sample: str = typer.Option(None, "--sample", help="Sample name"),
    genome: str = typer.Option(None, "--genome", help="Comma-separated genome names (e.g. B73,Mo17)"),
    bam: str = typer.Option(None, "--bam", help="Comma-separated BAM paths, same order as --genome"),
    workdir: Path = typer.Option(None, "--workdir", help="Output root directory"),
    # mode C: many samples from TSV
    configs: Path = typer.Option(None, "--configs", help="TSV with: sample, genome, bam, workdir"),
    # common knobs
    verbose: bool = typer.Option(True, "--verbose", help="Print per-chunk progress (step tools may honor this)"),
    min_barcode_freq: int = typer.Option(10, "--min-barcode-freq"),
    chunk_size_cells: int = typer.Option(5000, "--chunk-size-cells"),
    threads: int = typer.Option(8, "--threads", "-t", min=1),
    # DAG resume controls
    resume: bool = typer.Option(True, '--resume/--no-resume'),
    force_steps: str = typer.Option('', '--force-steps', help='Comma-separated step names to force recompute'),
    skip_to: str = typer.Option('', '--skip-to', help='Start at a specific step (validate prereqs).'),
    only_steps: str = typer.Option('', '--only-steps', help='Run only these steps (comma-separated).'),
    # assign overrides (forwarded into cfg["assign"])
    assign_alpha: Optional[float] = typer.Option(None, "--assign-alpha"),
    assign_k: Optional[int] = typer.Option(None, "--assign-k"),
    assign_mapq_min: Optional[int] = typer.Option(None, "--assign-mapq-min"),
    assign_xa_max: Optional[int] = typer.Option(None, "--assign-xa-max"),
    assign_chunksize: Optional[int] = typer.Option(None, "--assign-chunksize"),
    assign_batch_size: Optional[int] = typer.Option(None, "--assign-batch-size"),
    assign_edges_workers: Optional[int] = typer.Option(None, "--assign-edges-workers"),
    assign_edges_max_reads: Optional[int] = typer.Option(None, "--assign-edges-max-reads"),
    ecdf_workers: Optional[int] = typer.Option(None, "--ecdf-workers"),
):
    """
    Run the full pipeline with pipeline-wide resume via sentinels.

    Modes:
      (A) --config path.json
      (B) --sample + --genome "A,B,..." + --bam "/p/a.bam,/p/b.bam" + --workdir /out
      (C) --configs samples.tsv   (columns: sample, genome, bam, workdir)
    """
    inline_ready = all([sample, genome, bam, workdir])
    inline_partial = any([sample, genome, bam, workdir]) and not inline_ready
    modes_used = sum([1 if config else 0, 1 if inline_ready else 0, 1 if configs else 0])

    if inline_partial:
        raise typer.BadParameter("Inline mode requires --sample, --genome, --bam, and --workdir together.")
    if modes_used != 1:
        raise typer.BadParameter("Choose exactly one mode: --config OR (--sample/--genome/--bam/--workdir) OR --configs")

    def _do_one(cfg: dict):
        _apply_assign_overrides(
            cfg,
            alpha=assign_alpha, k=assign_k, mapq_min=assign_mapq_min, xa_max=assign_xa_max,
            chunksize=assign_chunksize, batch_size=assign_batch_size,
            edges_workers=assign_edges_workers, edges_max_reads=assign_edges_max_reads, ecdf_workers=ecdf_workers,
        )
        params = {
            "threads": int(threads),
            "verbose": bool(verbose),
            "assign": cfg.get("assign", {}),
        }
        force = [s.strip() for s in force_steps.split(",") if s.strip()]
        only  = [s.strip() for s in only_steps.split(",") if s.strip()]

        # sentinel policy is handled inside run_pipeline in your design;
        # if you want to enforce here per-step, call _apply_resume_policy.
        result = run_pipeline(cfg, params, version="0.4.0", resume=resume, force=force, skip_to=skip_to, only=only)
        typer.echo(f"[run] Executed: {result['executed']}")
        typer.echo(f"[run] Skipped:  {result['skipped']}")
        typer.echo(f"[run] {cfg['sample']} pipeline complete")

    if config:
        cfg = _load_config(config)
        _ensure_minimal_chunk(Path(cfg["workdir"]), cfg["sample"])
        _do_one(cfg)
        return
    elif inline_ready:
        cfg = _cfg_from_inline(sample, genome, bam, str(workdir), min_barcode_freq, chunk_size_cells)
        _ensure_minimal_chunk(Path(cfg["workdir"]), cfg["sample"])
        _do_one(cfg)
        return
    else:
        batch = _cfgs_from_tsv(configs, min_barcode_freq, chunk_size_cells)
        if not batch:
            typer.echo("[run] no configs found in TSV")
            return
        for cfg in batch:
            _apply_assign_overrides(
                cfg,
                alpha=assign_alpha, k=assign_k, mapq_min=assign_mapq_min, xa_max=assign_xa_max,
                chunksize=assign_chunksize, batch_size=assign_batch_size,
                edges_workers=assign_edges_workers, edges_max_reads=assign_edges_max_reads, ecdf_workers=ecdf_workers,
            )
            typer.echo(f"[run] starting {cfg['sample']}")
            _ensure_minimal_chunk(Path(cfg["workdir"]), cfg["sample"])
            _do_one(cfg)
        return

if __name__ == "__main__":
    app()
