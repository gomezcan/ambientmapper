# src/ambientmapper/cli.py
#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional

import typer

from .pipeline import run_pipeline

# -----------------------------------------------------------------------------
# Root CLI
# -----------------------------------------------------------------------------
app = typer.Typer(
    help="ambientmapper: local-first ambient cleaning pipeline",
    add_completion=False,
)

# -----------------------------------------------------------------------------
# Sub-apps / commands
# -----------------------------------------------------------------------------
from .decontam import app as decontam_app  # noqa: E402
app.add_typer(decontam_app, name="decontam")

from .clean_bams import clean_bams_cmd  # noqa: E402
app.command("clean-bams")(clean_bams_cmd)

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))


def _parse_csv_list(s: str | None) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _cfg_dirs(cfg: Dict[str, object]) -> Dict[str, Path]:
    root = Path(str(cfg["workdir"])) / str(cfg["sample"])
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


def _cfg_from_inline(
    sample: str,
    genomes_csv: str,
    bams_csv: str,
    workdir: str,
    min_barcode_freq: int,
    chunk_size_cells: int,
) -> Dict[str, object]:
    genomes = _parse_csv_list(genomes_csv)
    bam_paths = [Path(p).expanduser().resolve() for p in _parse_csv_list(bams_csv)]

    if len(genomes) != len(bam_paths):
        raise typer.BadParameter(
            f"#genome names ({len(genomes)}) must equal #bams ({len(bam_paths)})."
        )
    if len(set(genomes)) != len(genomes):
        raise typer.BadParameter("Genome names must be unique.")
    for p in bam_paths:
        if not p.exists():
            raise typer.BadParameter(f"BAM not found: {p}")

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


def _cfgs_from_tsv(
    tsv: Path, min_barcode_freq: int, chunk_size_cells: int
) -> List[Dict[str, object]]:
    groups: Dict[str, Dict[str, str]] = {}
    workdirs: Dict[str, str] = {}

    with open(tsv, "r", newline="") as f:
        r = csv.DictReader(f, delimiter="\t")
        need = {"sample", "genome", "bam", "workdir"}
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
        cfgs.append(
            {
                "sample": s,
                "genomes": gmap,
                "min_barcode_freq": int(min_barcode_freq),
                "chunk_size_cells": int(chunk_size_cells),
                "workdir": workdirs[s],
            }
        )
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

    if alpha is not None:
        assign["alpha"] = float(alpha)
    if k is not None:
        assign["k"] = int(k)
    if mapq_min is not None:
        assign["mapq_min"] = int(mapq_min)
    if xa_max is not None:
        assign["xa_max"] = int(xa_max)
    if chunksize is not None:
        assign["chunksize"] = int(chunksize)
    if batch_size is not None:
        assign["batch_size"] = int(batch_size)
    if edges_workers is not None:
        assign["edges_workers"] = int(edges_workers)
    if edges_max_reads is not None:
        assign["edges_max_reads"] = int(edges_max_reads)
        typer.echo(f"[assign] edges_max_reads cap in effect: {int(edges_max_reads):,} rows/genome")
    if ecdf_workers is not None:
        assign["ecdf_workers"] = int(ecdf_workers)


def _ensure_minimal_chunk(workdir: Path, sample: str) -> None:
    """
    If the sample's chunks dir has no manifest and no *_cell_map_ref_chunk_*.txt,
    synthesize a single dummy chunk and a manifest so the DAG can start.
    """
    chunks = Path(workdir) / sample / "cell_map_ref_chunks"
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


# -----------------------------------------------------------------------------
# Stepwise commands
# -----------------------------------------------------------------------------
@app.command()
def extract(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
    threads: int = typer.Option(4, "--threads", "-t", min=1),
) -> None:
    from .extract import bam_to_qc

    cfg = _load_config(config)
    d = _cfg_dirs(cfg)
    _ensure_dirs(d)

    genomes = sorted(dict(cfg["genomes"]).items())
    with ProcessPoolExecutor(max_workers=_clamp(int(threads), 1, len(genomes))) as ex:
        futs = [
            ex.submit(bam_to_qc, Path(bam), d["qc"] / f"{g}_QCMapping.txt", str(cfg["sample"]))
            for g, bam in genomes
        ]
        for f in as_completed(futs):
            f.result()

    typer.echo("[extract] done")


@app.command()
def filter(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
    threads: int = typer.Option(4, "--threads", "-t", min=1),
) -> None:
    from .filtering import filter_qc_file

    cfg = _load_config(config)
    d = _cfg_dirs(cfg)
    _ensure_dirs(d)

    genomes = sorted(dict(cfg["genomes"]).keys())
    minf = int(cfg["min_barcode_freq"])

    with ProcessPoolExecutor(max_workers=_clamp(int(threads), 1, len(genomes))) as ex:
        futs = []
        for g in genomes:
            ip = d["qc"] / f"{g}_QCMapping.txt"
            op = d["filtered"] / f"filtered_{g}_QCMapping.txt"
            futs.append(ex.submit(filter_qc_file, ip, op, minf, str(cfg["sample"])))
        for f in as_completed(futs):
            f.result()

    typer.echo("[filter] done")


@app.command()
def chunks(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
) -> None:
    from .chunks import make_barcode_chunks

    cfg = _load_config(config)
    d = _cfg_dirs(cfg)
    _ensure_dirs(d)

    n = make_barcode_chunks(d["filtered"], d["chunks"], str(cfg["sample"]), int(cfg["chunk_size_cells"]))
    typer.echo(f"[chunks] wrote {n} chunk files")


@app.command()
def assign(
    config: Path = typer.Option(..., "-c", "--config", exists=True),
    threads: int = typer.Option(16, "-t", "--threads"),
    edges_workers: Optional[int] = typer.Option(
        None, "--edges-workers", help="Max process workers for learn_edges_parallel (default: = --threads)"
    ),
    edges_max_reads: Optional[int] = typer.Option(
        None, "--edges-max-reads", help="Optional cap of reads per genome when learning edges"
    ),
    ecdf_workers: Optional[int] = typer.Option(
        None, "--ecdf-workers", help="Max workers for learn_ecdfs (default: = --threads)"
    ),
    chunksize: Optional[int] = typer.Option(
        None,
        "--chunksize",
        help="Override pandas read_csv chunk size (rows per chunk). CLI > cfg.assign > cfg.top-level > default(1000000).",
    ),
    batch_size: Optional[int] = typer.Option(
        None,
        "--batch-size",
        help="Override batch size for batched winner-edge learning. CLI > cfg.assign > cfg.top-level > default(32).",
    ),
    verbose: bool = typer.Option(True, "--verbose/--quiet", help="Print per-chunk progress"),
) -> None:
    """Learn edges/ECDFs and score each chunk (parallel)."""
    from .assign_streaming import learn_edges_parallel, learn_ecdfs_parallel, score_chunk

    cfg = _load_config(config)
    d = _cfg_dirs(cfg)
    _ensure_dirs(d)

    _ensure_minimal_chunk(Path(str(cfg["workdir"])), str(cfg["sample"]))

    workdir = Path(str(cfg["workdir"]))
    sample = str(cfg["sample"])
    chunks_dir = d["chunks"]

    aconf = cfg.get("assign", {}) if isinstance(cfg.get("assign"), dict) else {}
    alpha = float(aconf.get("alpha", 0.05))
    k = int(aconf.get("k", 10))
    mapq_min = int(aconf.get("mapq_min", 20))
    xa_max = int(aconf.get("xa_max", 2))

    chunksize_val = int(
        chunksize if chunksize is not None else aconf.get("chunksize", cfg.get("chunksize", 1_000_000))
    )
    batch_size_val = int(
        batch_size if batch_size is not None else aconf.get("batch_size", cfg.get("batch_size", 32))
    )

    if "chunks_dir" in aconf:
        chunks_dir = workdir / sample / str(aconf["chunks_dir"])

    chunk_files = sorted(chunks_dir.glob("*_cell_map_ref_chunk_*.txt"))
    if not chunk_files:
        dummy = chunks_dir / f"{sample}_cell_map_ref_chunk_0001.txt"
        dummy.parent.mkdir(parents=True, exist_ok=True)
        dummy.touch(exist_ok=True)
        (chunks_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "version": 1,
                    "n_chunks": 1,
                    "chunks": [{"id": dummy.stem, "path": dummy.name}],
                    "derived_from": {"filtered_inputs_sha256": "bootstrap", "upstream_generation": 0},
                },
                indent=2,
            )
        )
        typer.echo(f"[assign] No chunk files found; created {dummy.name} to proceed.")
        chunk_files = [dummy]

    threads_eff = max(1, int(threads))

    if edges_workers is not None:
        edges_workers = 1 if edges_workers <= 0 else min(int(edges_workers), threads_eff)

    if edges_max_reads is not None and int(edges_max_reads) <= 0:
        edges_max_reads = None

    if ecdf_workers is not None and int(ecdf_workers) <= 0:
        ecdf_workers = None

    if k <= 0:
        typer.echo("[assign] warn: k<=0 deciles → forcing k=10")
        k = 10
    if chunksize_val <= 0:
        typer.echo("[assign] warn: chunksize<=0 → forcing 1_000_000")
        chunksize_val = 1_000_000
    if batch_size_val <= 0:
        typer.echo("[assign] warn: batch_size<=0 → forcing 32")
        batch_size_val = 32

    exp_dir = workdir / sample / "ExplorationReadLevel"
    exp_dir.mkdir(parents=True, exist_ok=True)
    edges_npz = exp_dir / "global_edges.npz"
    ecdf_npz = exp_dir / "global_ecdf.npz"

    if verbose:
        extra = []
        if edges_workers is not None:
            extra.append(f"edges_workers={edges_workers}")
        if edges_max_reads is not None:
            extra.append(f"edges_max_reads={int(edges_max_reads):,}")
        if ecdf_workers is not None:
            extra.append(f"ecdf_workers={min(max(int(ecdf_workers), 1), threads_eff)}")
        typer.echo(
            f"[assign] effective chunksize={chunksize_val:,}  batch_size={batch_size_val}  threads={threads_eff}"
            + ("  " + "  ".join(extra) if extra else "")
        )

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
        edges_max_reads=edges_max_reads,
    )

    ecdf_workers_eff = (
        int(ecdf_workers) if (ecdf_workers is not None and int(ecdf_workers) > 0) else threads_eff
    )
    ecdf_workers_eff = min(ecdf_workers_eff, threads_eff)

    learn_ecdfs_parallel(
        workdir=workdir,
        sample=sample,
        chunks_dir=chunks_dir,
        edges_model=edges_npz,
        out_model=ecdf_npz,
        mapq_min=mapq_min,
        xa_max=xa_max,
        chunksize=chunksize_val,
        verbose=verbose,
        workers=ecdf_workers_eff,
    )

    pool_n = min(threads_eff, len(chunk_files))
    typer.echo(f"[assign/score] start: {len(chunk_files)} chunks, procs={pool_n}")

    with ProcessPoolExecutor(max_workers=pool_n) as ex:
        fut = {
            ex.submit(
                score_chunk,
                workdir=workdir,
                sample=sample,
                chunk_file=chf,
                ecdf_model=ecdf_npz,
                out_raw_dir=None,
                out_filtered_dir=None,
                mapq_min=mapq_min,
                xa_max=xa_max,
                chunksize=chunksize_val,
                alpha=alpha,
            ): chf
            for chf in chunk_files
        }
        done = 0
        total = len(fut)
        for f in as_completed(fut):
            f.result()
            done += 1
            if done % 5 == 0 or done == total:
                typer.echo(f"[assign/score] {done}/{total} chunks")

    typer.echo("[assign/score] done")


@app.command()
def genotyping(
    # Core I/O and runtime
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True, help="Single-sample JSON config."),
    assign_glob: Optional[str] = typer.Option(None, "--assign", help="Glob to assign outputs. If omitted, inferred."),
    outdir: Optional[Path] = typer.Option(None, "--outdir", help="Override output directory (default: <workdir>/<sample>/final)."),
    sample: Optional[str] = typer.Option(None, "--sample", help="Override sample name from config."),

    # Performance
    threads: Optional[int] = typer.Option(None, "--threads", help="Pass-2 workers (model selection)."),
    shards: Optional[int] = typer.Option(None, "--shards", help="Shard count for pass-1 spill (default 32)."),
    chunk_rows: Optional[int] = typer.Option(None, "--chunk-rows", help="Pass-1 input chunk size (rows)."),
    pass2_chunksize: Optional[int] = typer.Option(None, "--pass2-chunksize", help="Pass-2 shard chunk size (rows)."),

    # Mode
    winner_only: Optional[bool] = typer.Option(None, "--winner-only/--no-winner-only", help="Winner-only mode (default true)."),

    # Single/Doublet modeling
    min_reads: Optional[int] = typer.Option(None, "--min-reads", help="Min reads to fit single/doublet."),
    beta: Optional[float] = typer.Option(None, "--beta", help="Softmax temperature (probabilistic mode)."),
    w_as: Optional[float] = typer.Option(None, "--w-as", help="AS weight (probabilistic mode)."),
    w_mapq: Optional[float] = typer.Option(None, "--w-mapq", help="MAPQ weight (probabilistic mode)."),
    w_nm: Optional[float] = typer.Option(None, "--w-nm", help="NM weight (probabilistic mode)."),
    ambient_const: Optional[float] = typer.Option(None, "--ambient-const", help="Per-read ambient mass."),
    bic_margin: Optional[float] = typer.Option(None, "--bic-margin", help="ΔBIC required to accept doublet over single."),
    doublet_minor_min: Optional[float] = typer.Option(None, "--doublet-minor-min", help="Min minor fraction for doublet."),
    topk_genomes: Optional[int] = typer.Option(None, "--topk-genomes", help="Top-K candidate genomes per barcode."),
    single_mass_min: Optional[float] = typer.Option(None, "--single-mass-min", help="Purity threshold for single calls."),

    # Empty / Ambient learning
    eta_iters: Optional[int] = typer.Option(None, "--eta-iters", help="Ambient refinement iterations."),
    eta_seed_quantile: Optional[float] = typer.Option(None, "--eta-seed-quantile", help="Quantile to seed eta from low-depth tail."),
    empty_bic_margin: Optional[float] = typer.Option(None, "--empty-bic-margin", help="Min ΔBIC (non-empty - empty) to call empty."),
    empty_top1_max: Optional[float] = typer.Option(None, "--empty-top1-max", help="Max top1 mass to allow empty."),
    empty_ratio12_max: Optional[float] = typer.Option(None, "--empty-ratio12-max", help="Max top1/top2 ratio to allow empty."),
    empty_entropy_min: Optional[float] = typer.Option(None, "--empty-entropy-min", help="Min entropy to allow empty."),
    empty_reads_max: Optional[int] = typer.Option(None, "--empty-reads-max", help="Optional ceiling on reads for empty calls."),
) -> None:
    """Posterior-aware genotyping (merge → per-cell genotype calls)."""
    from .genotyping import genotyping as _run_genotyping
    import glob as _glob

    cfg = _load_config(config)
    d = _cfg_dirs(cfg)
    _ensure_dirs(d)

    sample_eff = str(sample or cfg["sample"])
    outdir_eff = Path(outdir or d["final"])

    # Light validation of a few “easy to mess up” params
    if eta_seed_quantile is not None and not (0.0 < float(eta_seed_quantile) < 1.0):
        raise typer.BadParameter("--eta-seed-quantile must be in (0,1)")

    if assign_glob is None:
        chunks_dir = d["chunks"]
        patterns = [
            str(chunks_dir / "**" / "*assign*.parquet"),
            str(chunks_dir / "**" / "*filtered*.parquet"),
            str(chunks_dir / "**" / "*.parquet"),
            str(chunks_dir / "**" / "*filtered.tsv.gz"),
            str(chunks_dir / "**" / "*.tsv.gz"),
            str(chunks_dir / "**" / "*.csv.gz"),
        ]
        matched: List[str] = []
        for pat in patterns:
            matched = _glob.glob(pat, recursive=True)
            if matched:
                assign_glob = pat
                break
        if not matched:
            raise typer.BadParameter(
                f"No assign outputs found under {chunks_dir}. Provide --assign or run the 'assign' step first."
            )

    if not _glob.glob(assign_glob, recursive=True):
        raise typer.BadParameter(f"No assign files matched: {assign_glob}")

    # Build kwargs: only pass overrides if user provided them
    kwargs: Dict[str, object] = {
        "assign": assign_glob,
        "outdir": outdir_eff,
        "sample": sample_eff,
    }

    if threads is not None:
        kwargs["threads"] = max(1, int(threads))
    if shards is not None:
        kwargs["shards"] = max(1, int(shards))
    if chunk_rows is not None:
        kwargs["chunk_rows"] = max(1, int(chunk_rows))
    if pass2_chunksize is not None:
        kwargs["pass2_chunksize"] = max(1, int(pass2_chunksize))
    if winner_only is not None:
        kwargs["winner_only"] = bool(winner_only)

    if min_reads is not None:
        kwargs["min_reads"] = max(1, int(min_reads))
    if beta is not None:
        kwargs["beta"] = float(beta)
    if w_as is not None:
        kwargs["w_as"] = float(w_as)
    if w_mapq is not None:
        kwargs["w_mapq"] = float(w_mapq)
    if w_nm is not None:
        kwargs["w_nm"] = float(w_nm)
    if ambient_const is not None:
        kwargs["ambient_const"] = float(ambient_const)
    if bic_margin is not None:
        kwargs["bic_margin"] = float(bic_margin)
    if doublet_minor_min is not None:
        kwargs["doublet_minor_min"] = float(doublet_minor_min)
    if topk_genomes is not None:
        kwargs["topk_genomes"] = max(1, int(topk_genomes))
    if single_mass_min is not None:
        kwargs["single_mass_min"] = float(single_mass_min)

    if eta_iters is not None:
        kwargs["eta_iters"] = max(0, int(eta_iters))
    if eta_seed_quantile is not None:
        kwargs["eta_seed_quantile"] = float(eta_seed_quantile)
    if empty_bic_margin is not None:
        kwargs["empty_bic_margin"] = float(empty_bic_margin)
    if empty_top1_max is not None:
        kwargs["empty_top1_max"] = float(empty_top1_max)
    if empty_ratio12_max is not None:
        kwargs["empty_ratio12_max"] = float(empty_ratio12_max)
    if empty_entropy_min is not None:
        kwargs["empty_entropy_min"] = float(empty_entropy_min)
    if empty_reads_max is not None:
        kwargs["empty_reads_max"] = int(empty_reads_max)

    _run_genotyping(**kwargs)


@app.command()
def interpool(
    configs: Path = typer.Option(..., "--configs", help="TSV with: sample, genome, bam, workdir"),
    outdir: Path = typer.Option(None, "--outdir", help="Output dir for inter-pool summary (default: <first workdir>/interpool)"),
) -> None:
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
) -> None:
    """Plate-level summary: contamination distributions + 96-well heatmap."""
    from .plate import plate_summary_cli

    out = plate_summary_cli(workdir=workdir, plate_map=plate_map, outdir=outdir, xa_max=xa_max)
    typer.echo(f"[plate] per-BC: {out['per_bc']}")
    typer.echo(f"[plate] per-sample: {out['per_sample']}")
    typer.echo(f"[plate] density: {out['density_pdf']}")
    typer.echo(f"[plate] heatmap: {out['heatmap_pdf']}")


# -----------------------------------------------------------------------------
# End-to-end run
# -----------------------------------------------------------------------------
@app.command()
def run(
    # mode A: single JSON
    config: Optional[Path] = typer.Option(None, "--config", "-c", exists=True, readable=True, help="Single JSON config"),
    # mode B: inline one-sample
    sample: Optional[str] = typer.Option(None, "--sample", help="Sample name"),
    genome: Optional[str] = typer.Option(None, "--genome", help="Comma-separated genome names (e.g. B73,Mo17)"),
    bam: Optional[str] = typer.Option(None, "--bam", help="Comma-separated BAM paths, same order as --genome"),
    workdir: Optional[Path] = typer.Option(None, "--workdir", help="Output root directory"),
    # mode C: many samples from TSV
    configs: Optional[Path] = typer.Option(None, "--configs", help="TSV with: sample, genome, bam, workdir"),
    # common knobs
    verbose: bool = typer.Option(True, "--verbose", help="Print per-chunk progress (step tools may honor this)"),
    min_barcode_freq: int = typer.Option(10, "--min-barcode-freq"),
    chunk_size_cells: int = typer.Option(5000, "--chunk-size-cells"),
    threads: int = typer.Option(8, "--threads", "-t", min=1),
    # pipeline resume controls (run_pipeline owns the policy)
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    force_steps: str = typer.Option("", "--force-steps", help="Comma-separated step names to force recompute"),
    skip_to: str = typer.Option("", "--skip-to", help="Start at a specific step (validate prereqs)."),
    only_steps: str = typer.Option("", "--only-steps", help="Run only these steps (comma-separated)."),
    # assign overrides (forwarded into cfg['assign'])
    assign_alpha: Optional[float] = typer.Option(None, "--assign-alpha"),
    assign_k: Optional[int] = typer.Option(None, "--assign-k"),
    assign_mapq_min: Optional[int] = typer.Option(None, "--assign-mapq-min"),
    assign_xa_max: Optional[int] = typer.Option(None, "--assign-xa-max"),
    assign_chunksize: Optional[int] = typer.Option(None, "--assign-chunksize"),
    assign_batch_size: Optional[int] = typer.Option(None, "--assign-batch-size"),
    assign_edges_workers: Optional[int] = typer.Option(None, "--assign-edges-workers"),
    assign_edges_max_reads: Optional[int] = typer.Option(None, "--assign-edges-max-reads"),
    ecdf_workers: Optional[int] = typer.Option(None, "--ecdf-workers"),

    # genotyping overrides (forwarded into params['genotyping'])
    genotyping_min_reads: Optional[int] = typer.Option(None, "--genotyping-min-reads"),
    genotyping_beta: Optional[float] = typer.Option(None, "--genotyping-beta"),
    genotyping_w_as: Optional[float] = typer.Option(None, "--genotyping-w-as"),
    genotyping_w_mapq: Optional[float] = typer.Option(None, "--genotyping-w-mapq"),
    genotyping_w_nm: Optional[float] = typer.Option(None, "--genotyping-w-nm"),
    genotyping_ambient_const: Optional[float] = typer.Option(None, "--genotyping-ambient-const"),
    genotyping_bic_margin: Optional[float] = typer.Option(None, "--genotyping-bic-margin"),
    genotyping_topk_genomes: Optional[int] = typer.Option(None, "--genotyping-topk-genomes"),
    genotyping_threads: Optional[int] = typer.Option(None, "--genotyping-threads"),
    genotyping_shards: Optional[int] = typer.Option(None, "--genotyping-shards"),
    genotyping_chunk_rows: Optional[int] = typer.Option(None, "--genotyping-chunk-rows"),
    genotyping_pass2_chunksize: Optional[int] = typer.Option(None, "--genotyping-pass2-chunksize"),
    genotyping_winner_only: Optional[bool] = typer.Option(None, "--genotyping-winner-only/--no-genotyping-winner-only"),
    genotyping_doublet_minor_min: Optional[float] = typer.Option(None, "--genotyping-doublet-minor-min"),
    genotyping_single_mass_min: Optional[float] = typer.Option(None, "--genotyping-single-mass-min"),

    # empty / ambient overrides
    genotyping_eta_iters: Optional[int] = typer.Option(None, "--genotyping-eta-iters"),
    genotyping_eta_seed_quantile: Optional[float] = typer.Option(None, "--genotyping-eta-seed-quantile"),
    genotyping_empty_bic_margin: Optional[float] = typer.Option(None, "--genotyping-empty-bic-margin"),
    genotyping_empty_top1_max: Optional[float] = typer.Option(None, "--genotyping-empty-top1-max"),
    genotyping_empty_ratio12_max: Optional[float] = typer.Option(None, "--genotyping-empty-ratio12-max"),
    genotyping_empty_entropy_min: Optional[float] = typer.Option(None, "--genotyping-empty-entropy-min"),
    genotyping_empty_reads_max: Optional[int] = typer.Option(None, "--genotyping-empty-reads-max"),
) -> None:
    """Run the full pipeline."""
    inline_ready = all([sample, genome, bam, workdir])
    inline_partial = any([sample, genome, bam, workdir]) and not inline_ready
    modes_used = sum([1 if config else 0, 1 if inline_ready else 0, 1 if configs else 0])

    if inline_partial:
        raise typer.BadParameter("Inline mode requires --sample, --genome, --bam, and --workdir together.")
    if modes_used != 1:
        raise typer.BadParameter(
            "Choose exactly one mode: --config OR (--sample/--genome/--bam/--workdir) OR --configs"
        )

    # Build genotyping overrides (only include user-specified)
    genotyping_conf: Dict[str, object] = {}
    if genotyping_min_reads is not None:
        genotyping_conf["min_reads"] = int(genotyping_min_reads)
    if genotyping_beta is not None:
        genotyping_conf["beta"] = float(genotyping_beta)
    if genotyping_w_as is not None:
        genotyping_conf["w_as"] = float(genotyping_w_as)
    if genotyping_w_mapq is not None:
        genotyping_conf["w_mapq"] = float(genotyping_w_mapq)
    if genotyping_w_nm is not None:
        genotyping_conf["w_nm"] = float(genotyping_w_nm)
    if genotyping_ambient_const is not None:
        genotyping_conf["ambient_const"] = float(genotyping_ambient_const)
    if genotyping_bic_margin is not None:
        genotyping_conf["bic_margin"] = float(genotyping_bic_margin)
    if genotyping_topk_genomes is not None:
        genotyping_conf["topk_genomes"] = int(genotyping_topk_genomes)
    if genotyping_threads is not None:
        genotyping_conf["threads"] = int(genotyping_threads)
    if genotyping_shards is not None:
        genotyping_conf["shards"] = int(genotyping_shards)
    if genotyping_chunk_rows is not None:
        genotyping_conf["chunk_rows"] = int(genotyping_chunk_rows)
    if genotyping_pass2_chunksize is not None:
        genotyping_conf["pass2_chunksize"] = int(genotyping_pass2_chunksize)
    if genotyping_winner_only is not None:
        genotyping_conf["winner_only"] = bool(genotyping_winner_only)
    if genotyping_doublet_minor_min is not None:
        genotyping_conf["doublet_minor_min"] = float(genotyping_doublet_minor_min)
    if genotyping_single_mass_min is not None:
        genotyping_conf["single_mass_min"] = float(genotyping_single_mass_min)

    if genotyping_eta_iters is not None:
        genotyping_conf["eta_iters"] = int(genotyping_eta_iters)
    if genotyping_eta_seed_quantile is not None:
        genotyping_conf["eta_seed_quantile"] = float(genotyping_eta_seed_quantile)
    if genotyping_empty_bic_margin is not None:
        genotyping_conf["empty_bic_margin"] = float(genotyping_empty_bic_margin)
    if genotyping_empty_top1_max is not None:
        genotyping_conf["empty_top1_max"] = float(genotyping_empty_top1_max)
    if genotyping_empty_ratio12_max is not None:
        genotyping_conf["empty_ratio12_max"] = float(genotyping_empty_ratio12_max)
    if genotyping_empty_entropy_min is not None:
        genotyping_conf["empty_entropy_min"] = float(genotyping_empty_entropy_min)
    if genotyping_empty_reads_max is not None:
        genotyping_conf["empty_reads_max"] = int(genotyping_empty_reads_max)

    if genotyping_conf:
        typer.echo(
            "[run] genotyping overrides: " + ", ".join(f"{k}={v}" for k, v in genotyping_conf.items())
        )

    force = [s.strip() for s in force_steps.split(",") if s.strip()]
    only = [s.strip() for s in only_steps.split(",") if s.strip()]

    def _do_one(cfg: Dict[str, object]) -> None:
        _apply_assign_overrides(
            cfg,
            alpha=assign_alpha,
            k=assign_k,
            mapq_min=assign_mapq_min,
            xa_max=assign_xa_max,
            chunksize=assign_chunksize,
            batch_size=assign_batch_size,
            edges_workers=assign_edges_workers,
            edges_max_reads=assign_edges_max_reads,
            ecdf_workers=ecdf_workers,
        )

        params = {
            "threads": int(threads),
            "verbose": bool(verbose),
            "assign": cfg.get("assign", {}),
            "genotyping": genotyping_conf,
        }

        result = run_pipeline(
            cfg,
            params,
            version="0.4.0",
            resume=resume,
            force=force,
            skip_to=skip_to,
            only=only,
        )
        typer.echo(f"[run] Executed: {result['executed']}")
        typer.echo(f"[run] Skipped:  {result['skipped']}")
        typer.echo(f"[run] {cfg['sample']} pipeline complete")

    if config is not None:
        cfg = _load_config(config)
        _ensure_minimal_chunk(Path(str(cfg["workdir"])), str(cfg["sample"]))
        _do_one(cfg)
        return

    if inline_ready:
        assert sample is not None and genome is not None and bam is not None and workdir is not None
        cfg = _cfg_from_inline(sample, genome, bam, str(workdir), min_barcode_freq, chunk_size_cells)
        _ensure_minimal_chunk(Path(str(cfg["workdir"])), str(cfg["sample"]))
        _do_one(cfg)
        return

    # TSV multi-sample mode
    assert configs is not None
    batch = _cfgs_from_tsv(configs, min_barcode_freq, chunk_size_cells)
    if not batch:
        typer.echo("[run] no configs found in TSV")
        return

    for cfg in batch:
        typer.echo(f"[run] starting {cfg['sample']}")
        _ensure_minimal_chunk(Path(str(cfg["workdir"])), str(cfg["sample"]))
        _do_one(cfg)


if __name__ == "__main__":
    app()
