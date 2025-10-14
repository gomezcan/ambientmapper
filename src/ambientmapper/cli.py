from __future__ import annotations
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import json, csv
import typer
from typing import Dict, List, Optional

# NEW: import the resumable pipeline engine
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

# --------------------------------------------------------------------------------
# Existing stepwise commands (unchanged behavior)
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
        for f in as_completed(futs): f.result()
    typer.echo("[extract] done")

@app.command()
def filter(config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
           threads: int = typer.Option(4, "--threads", "-t", min=1)):
    from .filtering import filter_qc_file
    cfg = json.loads(Path(config).read_text()); d = _cfg_dirs(cfg); _ensure_dirs(d)
    genomes = sorted(cfg["genomes"].keys()); minf = int(cfg["min_barcode_freq"])
    with ProcessPoolExecutor(max_workers=_clamp(threads, 1, len(genomes))) as ex:
        futs=[]
        for g in genomes:
            ip = d["qc"]/f"{g}_QCMapping.txt"
            op = d["filtered"]/f"filtered_{g}_QCMapping.txt"
            futs.append(ex.submit(filter_qc_file, ip, op, minf, cfg["sample"]))
        for f in as_completed(futs): f.result()
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
        help="Optional cap of reads per genome when learning edges (speeds up on huge files)"),
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
    """
    Modular assign: learn edges (global), learn ECDFs (global), then score each chunk (parallel)
    """
    # keeps your current assign implementation intact (no DAG here)
    from .assign_streaming import learn_edges_parallel, learn_ecdfs_parallel, score_chunk
    from concurrent.futures import ProcessPoolExecutor, as_completed

    cfg = _load_config(config)
    d = _cfg_dirs(cfg); _ensure_dirs(d)
    workdir = Path(cfg["workdir"])
    sample = str(cfg["sample"])
    chunks_dir = d["chunks"]

    aconf       = cfg.get("assign", {}) if isinstance(cfg.get("assign"), dict) else {}
    alpha       = float(aconf.get("alpha", 0.05))
    k           = int(aconf.get("k", 10))
    mapq_min    = int(aconf.get("mapq_min", 20))
    xa_max      = int(aconf.get("xa_max", 2))

    chunksize_val = int(chunksize if chunksize is not None else aconf.get("chunksize", cfg.get("chunksize", 1_000_000)))
    batch_size_val = int(batch_size if batch_size is not None else aconf.get("batch_size", cfg.get("batch_size", 32)))

    if "chunks_dir" in aconf:
        chunks_dir = Path(workdir) / sample / str(aconf["chunks_dir"])

    threads_eff = max(1, int(threads))
    if edges_workers is not None:
        if edges_workers <= 0:
            typer.echo("[assign] warn: edges_workers <= 0 → using 1")
            edges_workers = 1
        edges_workers = min(edges_workers, threads_eff)
    if edges_max_reads is not None and edges_max_reads <= 0:
        typer.echo("[assign] warn: edges_max_reads <= 0 → ignoring cap")
        edges_max_reads = None
    if k <= 0:
        typer.echo("[assign] warn: k<=0 deciles → forcing k=10"); k = 10
    if chunksize_val <= 0:
        typer.echo("[assign] warn: chunksize<=0 → forcing 1_000_000"); chunksize_val = 1_000_000
    if batch_size_val <= 0:
        typer.echo("[assign] warn: batch_size<=0 → forcing 32"); batch_size_val = 32

    exp_dir   = Path(workdir) / sample / "ExplorationReadLevel"
    edges_npz = exp_dir / "global_edges.npz"
    ecdf_npz  = exp_dir / "global_ecdf.npz"
    exp_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        extra = []
        if edges_workers is not None: extra.append(f"edges_workers={edges_workers}")
        if edges_max_reads is not None: extra.append(f"edges_max_reads={edges_max_reads:,}")
        if ecdf_workers is not None: extra.append(f"ecdf_workers={min(max(ecdf_workers,1), threads_eff)}")
        typer.echo(f"[assign] effective chunksize={chunksize_val:,}  batch_size={batch_size_val}  "
                   f"threads={threads_eff}" + ("  " + "  ".join(extra) if extra else ""))

    # 1) global models
    learn_edges_parallel(
        workdir=workdir, sample=sample, chunks_dir=chunks_dir, out_model=edges_npz,
        mapq_min=mapq_min, xa_max=xa_max, chunksize=chunksize_val, k=k, batch_size=batch_size_val,
        threads=threads_eff, verbose=verbose, edges_workers=edges_workers, edges_max_reads=edges_max_reads
    )
    ecdf_workers_eff = ecdf_workers if (ecdf_workers and ecdf_workers > 0) else threads_eff
    ecdf_workers_eff = min(ecdf_workers_eff, threads_eff)
    learn_ecdfs_parallel(
        workdir=workdir, sample=sample, chunks_dir=chunks_dir, edges_model=edges_npz, out_model=ecdf_npz,
        mapq_min=mapq_min, xa_max=xa_max, chunksize=chunksize_val, verbose=verbose, workers=ecdf_workers_eff
    )

    # 2) per-chunk scoring in parallel
    chunk_files = sorted(chunks_dir.glob("*_cell_map_ref_chunk_*.txt"))
    if not chunk_files:
        typer.echo(f"[assign] No chunk files in {chunks_dir}")
        raise typer.Exit(code=2)
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
):
    """Posterior-aware genotyping (merge + summarize + optional post-steps)."""
    from .genotyping import genotyping as _run_genotyping
    cfg = json.loads(Path(config).read_text())
    d = _cfg_dirs(cfg); _ensure_dirs(d)
    assign_glob = assign_glob or str(d["chunks"] / "**" / "*")
    outdir = outdir or d["final"]
    sample = sample or cfg["sample"]
    _run_genotyping.callback(assign=assign_glob, outdir=outdir, sample=sample, make_report=make_report)

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
# End-to-end: now powered by the resumable DAG + sentinels
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
        # apply assign overrides (stored in cfg['assign'])
        _apply_assign_overrides(
            cfg,
            alpha=assign_alpha, k=assign_k, mapq_min=assign_mapq_min, xa_max=assign_xa_max,
            chunksize=assign_chunksize, batch_size=assign_batch_size,
            edges_workers=assign_edges_workers, edges_max_reads=assign_edges_max_reads, ecdf_workers=ecdf_workers,
        )
        # build params that the DAG can use (extend as you please)
        params = {
            "threads": int(threads),
            "verbose": bool(verbose),
            # mirror assign overrides for downstream helpers if needed:
            "assign": cfg.get("assign", {}),
        }
        force = [s.strip() for s in force_steps.split(",") if s.strip()]
        only  = [s.strip() for s in only_steps.split(",") if s.strip()]

        # hand-off to the resumable DAG
        result = run_pipeline(cfg, params, version="0.4.0", resume=resume, force=force, skip_to=skip_to, only=only)
        typer.echo(f"[run] Executed: {result['executed']}")
        typer.echo(f"[run] Skipped:  {result['skipped']}")
        typer.echo(f"[run] {cfg['sample']} pipeline complete")

    if config:
        cfg = _load_config(config)
        _do_one(cfg)
        return
    elif inline_ready:
        cfg = _cfg_from_inline(sample, genome, bam, str(workdir), min_barcode_freq, chunk_size_cells)
        _do_one(cfg)
        return
    else:  # configs TSV
        batch = _cfgs_from_tsv(configs, min_barcode_freq, chunk_size_cells)
        if not batch:
            typer.echo("[run] no configs found in TSV"); return
        for cfg in batch:
            typer.echo(f"[run] starting {cfg['sample']}")
            _do_one(cfg)
        return
