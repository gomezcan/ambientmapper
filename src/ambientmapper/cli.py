from __future__ import annotations

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import json, csv
import concurrent.futures as cf
from typing import Dict, List, Optional
import typer


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

def _apply_assign_overrides(cfg: Dict[str, object],
                            alpha: Optional[float] = None,
                            k: Optional[int] = None,
                            mapq_min: Optional[int] = None,
                            xa_max: Optional[int] = None,
                            chunksize: Optional[int] = None,
                            batch_size: Optional[int] = None,
                            edges_workers: Optional[int] = None,
                            edges_max_reads: Optional[int] = None) -> None:
    """Ensure cfg['assign'] exists and apply CLI overrides if provided."""
    assign = cfg.get("assign")
    if not isinstance(assign, dict):
        assign = {}
        cfg["assign"] = assign
    if alpha      is not None: assign["alpha"] = float(alpha)
    if k          is not None: assign["k"] = int(k)
    if mapq_min   is not None: assign["mapq_min"] = int(mapq_min)
    if xa_max     is not None: assign["xa_max"] = int(xa_max)
    if chunksize  is not None: assign["chunksize"] = int(chunksize)
    if batch_size is not None: assign["batch_size"] = int(batch_size)
    if edges_workers   is not None: assign["edges_workers"] = int(edges_workers)
    if edges_max_reads is not None: assign["edges_max_reads"] = int(edges_max_reads)
    if edges_max_reads is not None:
        typer.echo(f"[assign] edges_max_reads cap in effect: {edges_max_reads:,} rows/genome")


def _run_pipeline(cfg: Dict[str, object], threads: int) -> None:
    # Lazy imports
    from .extract import bam_to_qc
    from .filtering import filter_qc_file
    from .chunks import make_barcode_chunks    
    from .assign_streaming import learn_edges_parallel, learn_ecdfs_batched, score_chunk
    from .genotyping import genotyping as _run_genotyping
    
    d = _cfg_dirs(cfg); _ensure_dirs(d)
    genomes = sorted(cfg["genomes"].items())
    if not genomes:
        typer.echo("[run] no genomes in config; nothing to do")
        return

    # 10) extract (parallel over genomes)
    with ProcessPoolExecutor(max_workers=_clamp(threads, 1, len(genomes))) as ex:
        futs = [ex.submit(bam_to_qc, Path(bam), d["qc"] / f"{g}_QCMapping.txt", cfg["sample"])
                for g, bam in genomes]
        for f in as_completed(futs):
            f.result()

    # 20) filter (parallel over genomes)
    minf = int(cfg["min_barcode_freq"])
    with ProcessPoolExecutor(max_workers=_clamp(threads, 1, len(genomes))) as ex:
        futs = []
        for g, _ in genomes:
            ip = d["qc"] / f"{g}_QCMapping.txt"
            op = d["filtered"] / f"filtered_{g}_QCMapping.txt"
            futs.append(ex.submit(filter_qc_file, ip, op, minf, cfg["sample"]))
        for f in as_completed(futs):
            f.result()

    # 25) chunks
    typer.echo("[run] chunks: creating per-BC subsets…")
    make_barcode_chunks(d["filtered"], d["chunks"], cfg["sample"], int(cfg["chunk_size_cells"]))

    # 30) assign (learn global models once, then score chunks in parallel)
    typer.echo("[run] assign: learning edges/ECDFs & scoring chunks…")
    aconf      = cfg.get("assign", {}) if isinstance(cfg.get("assign"), dict) else {}
    alpha      = float(aconf.get("alpha", 0.05))
    k          = int(aconf.get("k", 10))    
    mapq_min   = int(aconf.get("mapq_min", 20))
    xa_max     = int(aconf.get("xa_max", 2))    
    
    chunksize_val  = int(aconf.get("chunksize", cfg.get("chunksize", 1_000_000)))  # default 1M
    batch_size     = int(aconf.get("batch_size", cfg.get("batch_size", 32)))

    ew  = aconf.get("edges_workers")
    emr = aconf.get("edges_max_reads")
    edges_workers  = int(ew)  if ew  is not None else None
    edges_max_reads= int(emr) if emr is not None else None
    
    # ---- validation & clamping ----
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
        typer.echo("[assign] warn: k<=0 deciles → forcing k=10")
        k = 10
    if chunksize_val <= 0:
        typer.echo("[assign] warn: chunksize<=0 → forcing 1000000")
        chunksize_val = 1_000_000
    if batch_size <= 0:
        typer.echo("[assign] warn: batch_size<=0 → forcing 32")
        batch_size = 32
    
    chunks_dir = d["chunks"]
    
    # Model paths
    exp_dir   = d["root"] / "ExplorationReadLevel"
    edges_npz = exp_dir / "global_edges.npz"
    ecdf_npz  = exp_dir / "global_ecdf.npz"
    exp_dir.mkdir(parents=True, exist_ok=True)
     
    # IMPORTANT: the streaming functions expect a workdir that *contains* <sample>/
    pool_workdir = d["root"].parent
    
    # Verbose summary of effective assign settings
    if True:  # keep: aligns with your --verbose default
        extras = []
        if edges_workers is not None:  extras.append(f"edges_workers={edges_workers}")
        if edges_max_reads is not None:extras.append(f"edges_max_reads={edges_max_reads:,}")
        extras_s = ("  " + "  ".join(extras)) if extras else ""
        typer.echo(
            "[assign] "
            f"alpha={alpha} k={k} mapq_min={mapq_min} xa_max={xa_max}  "
            f"chunksize={chunksize_val:,} batch_size={batch_size} threads={threads_eff}{extras_s}"
        )

    # 1) global models (map-reduce)
    
    learn_edges_parallel(
        workdir=pool_workdir,
        sample=cfg["sample"],
        chunks_dir=chunks_dir,
        out_model=edges_npz, 
        mapq_min=mapq_min,
        xa_max=xa_max,
        chunksize=chunksize_val,
        k=k, batch_size=batch_size,
        threads=threads_eff, verbose=True,
        edges_workers=edges_workers, 
        edges_max_reads=edges_max_reads
    )

    learn_ecdfs_parallel(
        workdir=pool_workdir,
        sample=cfg["sample"],
        chunks_dir=chunks_dir,
        edges_model=edges_npz,
        out_model=ecdf_npz,
        mapq_min=mapq_min,
        xa_max=xa_max,
        chunksize=chunksize_val,
        batch_size=batch_size,
        ecdf_workers: edges_workers,
        verbose=True
    )

    # 2) score each chunk (parallel) 
    chunk_files = sorted(chunks_dir.glob("*_cell_map_ref_chunk_*.txt"))

    if not chunk_files:
        typer.echo(f"[assign] No chunk files in {chunks_dir}")
        return

    def _score_one(chf: Path):
        return score_chunk(
            workdir=pool_workdir, sample=cfg["sample"], chunk_file=chf, ecdf_model=ecdf_npz,
            out_raw_dir=None, out_filtered_dir=None,
            mapq_min=mapq_min, xa_max=xa_max, chunksize=chunksize_val, alpha=alpha
        )

    
    with cf.ThreadPoolExecutor(max_workers=min(threads_eff, len(chunk_files))) as ex:
        for _ in ex.map(_score_one, chunk_files):
            pass
        
    # 40) posterior merge + summarize (replaces old winner-merge + summarize)
    typer.echo("[run] merge.summarize: posterior-aware merge + QC…")
    
    # robust default: look for assign outputs in the chunks folder (parquet/tsv/csv)
    assign_glob = str(_cfg_dirs(cfg)["chunks"] / "**" / "*")
    _run_genotyping.callback(assign=assign_glob,
                             outdir=_cfg_dirs(cfg)["final"],
                             sample=cfg["sample"],
                             make_report=True)
    typer.echo(f"[run] merge.summarize → {d['final']}")

# ----------------
# Stepwise commands (optional)
# ----------------
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
    chunksize: Optional[int] = typer.Option(
        None, "--chunksize",
        help="Override pandas read_csv chunk size (rows per chunk). CLI > assign{} > top-level > default(1000000)."
    ),
    batch_size: Optional[int] = typer.Option(
        None, "--batch-size",
        help="Override batch size for batched winner-edge learning. CLI > assign{} > top-level > default(32)."
    ),
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="Print per-chunk progress")
) -> None:
    """    
    Modular assign: learn edges (global), learn ECDFs (global), then score each chunk (parallel)
    """
    
    from .assign_streaming import learn_edges_parallel, learn_ecdfs_batched, score_chunk
    
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
    
    # precedence: CLI > assign{} > top-level > default
    chunksize_val = int(
        chunksize
        if chunksize is not None
        else aconf.get("chunksize", cfg.get("chunksize", 1_000_000))
    )
    batch_size_val = int(
        batch_size
        if batch_size is not None
        else aconf.get("batch_size", cfg.get("batch_size", 32))
    )
    
    # Optional per-config override of chunks directory
    if "chunks_dir" in aconf:
        chunks_dir = Path(workdir) / sample / str(aconf["chunks_dir"])

    # ----- validation & clamping (mirror _run_pipeline) -----
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

    # model paths
    exp_dir   = Path(workdir) / sample / "ExplorationReadLevel"
    edges_npz = exp_dir / "global_edges.npz"
    ecdf_npz  = exp_dir / "global_ecdf.npz"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # small log so users can see the effective settings
    if verbose:
        extra = []
        if edges_workers is not None:   extra.append(f"edges_workers={edges_workers}")
        if edges_max_reads is not None: extra.append(f"edges_max_reads={edges_max_reads:,}")
        extra_s = ("  " + "  ".join(extra)) if extra else ""
        typer.echo(
            "[assign] "
            f"alpha={alpha} k={k} mapq_min={mapq_min} xa_max={xa_max}  "
            f"chunksize={chunksize_val:,} batch_size={batch_size_val} threads={threads_eff}{extra_s}"
        )

    # 1) global models
    learn_edges_parallel(
        workdir=workdir,
        sample=sample,
        chunks_dir=chunks_dir,
        out_model=edges_npz,
        mapq_min=mapq_min,
        xa_max=xa_max,
        chunksize=chunksize_val,
        k=k,
        batch_size=batch_size_val,
        threads=threads_eff,
        verbose=verbose,
        edges_workers=edges_workers,
        edges_max_reads=edges_max_reads
    )
    
    learn_ecdfs_batched(
        workdir=workdir, sample=sample, chunks_dir=chunks_dir,
        edges_model=edges_npz, out_model=ecdf_npz,
        mapq_min=mapq_min, xa_max=xa_max, chunksize=chunksize_val,
        batch_size=batch_size_val, verbose=verbose,
    )
    
    # 2) per-chunk scoring in parallel
    chunk_files = sorted(chunks_dir.glob("*_cell_map_ref_chunk_*.txt"))

    if not chunk_files:
        typer.echo(f"[assign] No chunk files in {chunks_dir}")
        raise typer.Exit(code=2)
    
    typer.echo(f"[assign/score] start: {len(chunk_files)} chunks, threads={min(threads, len(chunk_files))}")
    
    with cf.ThreadPoolExecutor(max_workers=min(threads, len(chunk_files))) as ex:
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
        for f in cf.as_completed(fut):
            ch = fut[f]
            try:
                f.result()
            except Exception as e:
                typer.echo(f"[assign/score][ERROR] {ch.name}: {e}")
                raise
            done += 1
            if done % 5 == 0 or done == total:
                typer.echo(f"[assign/score] {done}/{total} chunks")
        typer.echo("[assign/score] done")

    typer.echo(f"[assign] Done. Models in {exp_dir}. Outputs in raw_cell_map_ref_chunks/ and cell_map_ref_chunks/")


@app.command()
def genotyping(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
    assign_glob: Optional[str] = typer.Option(None, "--assign", help="Glob to assign outputs (parquet/csv/tsv)"),
    outdir: Optional[Path] = typer.Option(None, "--outdir"),
    sample: Optional[str] = typer.Option(None, "--sample"),
    make_report: bool = typer.Option(True, "--report/--no-report"),
):
    """
    Posterior-aware genotyping (merge + summarize + optional decontam).
    """
    from .genotyping import genotyping as _run_genotyping
    cfg = json.loads(Path(config).read_text())
    d = _cfg_dirs(cfg); _ensure_dirs(d)
    assign_glob = assign_glob or str(d["chunks"] / "**" / "*")
    outdir = outdir or d["final"]
    sample = sample or cfg["sample"]
    _run_genotyping.callback(  # call the Typer command function directly
        assign=assign_glob, outdir=outdir, sample=sample, make_report=make_report
    )

@app.command()
def summarize(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
    assign_glob: Optional[str] = typer.Option(None, "--assign", help="Glob to assign outputs (parquet/tsv/csv)"),
):
    """
    [Deprecated] Kept for backward compatibility.
    Calls the new posterior-aware merge.summarize.
    """
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
    """
    Compare multiple pools (samples) after they have been run.
    Produces:
      - interpool_bc_counts.tsv
      - interpool_read_composition.tsv
      - interpool_summary.pdf
    """
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
    """
    Plate-level summary: contamination distributions by pool and a 96-well heatmap.
    """
    from .plate import plate_summary_cli
    out = plate_summary_cli(workdir=workdir, plate_map=plate_map, outdir=outdir, xa_max=xa_max)
    typer.echo(f"[plate] per-BC: {out['per_bc']}")
    typer.echo(f"[plate] per-sample: {out['per_sample']}")
    typer.echo(f"[plate] density: {out['density_pdf']}")
    typer.echo(f"[plate] heatmap: {out['heatmap_pdf']}")



# ----------------
# End-to-end (3 modes)
# ----------------
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
    verbose: bool = typer.Option(True, "--verbose", help="Print per-chunk progress"),
    min_barcode_freq: int = typer.Option(10, "--min-barcode-freq"),
    chunk_size_cells: int = typer.Option(5000, "--chunk-size-cells"),
    threads: int = typer.Option(8, "--threads", "-t", min=1),
    
    #with_summary: bool = typer.Option(False, "--with-summary", help="Produce summary PDF + PASS BCs + Reads_to_discard"),
    #pool_design: Path = typer.Option(None, "--pool-design", exists=True, readable=True, help="TSV with columns: Genome, Pool [optional: Plate]"),
    #xa_max: int = typer.Option(2, "--xa-max", help="Keep winners with XAcount <= xa_max in summary; set -1 to disable"),
    
    # NEW: posterior-merge options (optional knobs if you want to expose later)    
    # (Using defaults inside merge.run for now)
    
    assign_alpha: Optional[float] = typer.Option(None, "--assign-alpha", help="Override assign.alpha (e.g., 0.05)"),
    assign_k: Optional[int] = typer.Option(None, "--assign-k", help="Override assign.k deciles (default 10)"),
    assign_mapq_min: Optional[int] = typer.Option(None, "--assign-mapq-min", help="Override assign.mapq_min (default 20)"),
    assign_xa_max: Optional[int] = typer.Option(None, "--assign-xa-max", help="Override assign.xa_max (default 2)"),
    assign_chunksize: Optional[int] = typer.Option(None, "--assign-chunksize",
                                         help="Override assign.chunksize rows per read_csv (default 1000000)"),
    assign_batch_size: Optional[int] = typer.Option(None, "--assign-batch-size",
                                                    help="Override assign.batch_size for learn_edges/ecdf (default 32)"),
    assign_edges_workers: Optional[int] = typer.Option(None, "--assign-edges-workers",
                                                       help="Process workers for learn_edges_parallel (default: = --threads)"),
    assign_edges_max_reads: Optional[int] = typer.Option(None, "--assign-edges-max-reads",
                                                         help="Cap rows per genome during edges learning (speeds up huge files)"),
):
    """
    Run the pipeline.

    Modes:
      (A) --config path.json
      (B) --sample + --genome "A,B,..." + --bam "/p/a.bam,/p/b.bam" + --workdir /out
      (C) --configs samples.tsv   (columns: sample, genome, bam, workdir)
    """
    inline_ready = all([sample, genome, bam, workdir])
    inline_partial = any([sample, genome, bam, workdir]) and not inline_ready

    modes_used = sum([
        1 if config else 0,
        1 if inline_ready else 0,
        1 if configs else 0,
    ])
    if inline_partial:
        raise typer.BadParameter("Inline mode requires --sample, --genome, --bam, and --workdir together.")
    if modes_used != 1:
        raise typer.BadParameter("Choose exactly one mode: --config OR (--sample/--genome/--bam/--workdir) OR --configs")
    
    typer.echo(f"[run] mode=inline={inline_ready} config={bool(config)} tsv={bool(configs)}")

    def _do_one(cfg: dict):
        # run the pipeline
        _run_pipeline(cfg, threads)
        typer.echo(f"[run] {cfg['sample']} pipeline complete")
        
    if config:
        cfg = json.loads(Path(config).read_text())
        _apply_assign_overrides(cfg,
                                alpha=assign_alpha,
                                k=assign_k,
                                mapq_min=assign_mapq_min,
                                xa_max=assign_xa_max, 
                                chunksize=assign_chunksize,
                                batch_size=assign_batch_size,
                                edges_workers=assign_edges_workers,
                                edges_max_reads=assign_edges_max_reads)                     
        _do_one(cfg)
        return
        
    elif inline_ready:
        cfg = _cfg_from_inline(sample, genome, bam, str(workdir), min_barcode_freq, chunk_size_cells)
        _apply_assign_overrides(cfg,
                                alpha=assign_alpha, 
                                k=assign_k, 
                                mapq_min=assign_mapq_min,
                                xa_max=assign_xa_max, 
                                chunksize=assign_chunksize,
                                edges_workers=assign_edges_workers,                                
                                edges_max_reads=assign_edges_max_reads,                                
                                batch_size=assign_batch_size)
        _do_one(cfg)
        return
    elif configs:
        batch = _cfgs_from_tsv(configs, min_barcode_freq, chunk_size_cells)
        if not batch:
            typer.echo("[run] no configs found in TSV"); return
        for cfg in batch:
            _apply_assign_overrides(cfg,
                                    alpha=assign_alpha,
                                    k=assign_k,
                                    mapq_min=assign_mapq_min,
                                    xa_max=assign_xa_max,
                                    chunksize=assign_chunksize,
                                    batch_size=assign_batch_size,
                                    edges_workers=assign_edges_workers,
                                    edges_max_reads=assign_edges_max_reads
                                   )

            typer.echo(f"[run] starting {cfg['sample']}")
            _do_one(cfg)
        return
