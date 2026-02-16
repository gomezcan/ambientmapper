#!/usr/bin/env python3
# src/ambientmapper/cli.py
from __future__ import annotations

import csv
import glob
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
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
# Wire sub-apps / commands
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
    if not isinstance(cfg["genomes"], dict) or not cfg["genomes"]:
        raise typer.BadParameter("Config field 'genomes' must be a non-empty mapping.")
    return cfg


def _cfgs_from_tsv(tsv: Path, min_barcode_freq: int, chunk_size_cells: int) -> List[Dict[str, object]]:
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

            if not s:
                raise typer.BadParameter("TSV row missing 'sample'")
            if not g:
                raise typer.BadParameter(f"TSV row missing 'genome' for sample='{s}'")

            if s not in groups:
                groups[s] = {}
                workdirs[s] = w
            else:
                if workdirs[s] != w:
                    raise typer.BadParameter(
                        f"Sample '{s}' has multiple workdirs in TSV: '{workdirs[s]}' vs '{w}'"
                    )

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


def _load_one_or_many_configs(
    *,
    config: Optional[Path],
    configs: Optional[Path],
    min_barcode_freq: Optional[int] = None,
    chunk_size_cells: Optional[int] = None,
) -> List[Dict[str, object]]:
    if (config is None) == (configs is None):
        raise typer.BadParameter("Provide exactly one of: --config (JSON) OR --configs (TSV).")

    if config is not None:
        return [_load_config(config)]

    mbf = int(min_barcode_freq) if min_barcode_freq is not None else 10
    csc = int(chunk_size_cells) if chunk_size_cells is not None else 5000
    assert configs is not None
    return _cfgs_from_tsv(configs, mbf, csc)


def _apply_assign_overrides(
    cfg: Dict[str, object],
    *,
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
    chunks = workdir / sample / "cell_map_ref_chunks"
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
# Sentinels (for stepwise commands)
# -----------------------------------------------------------------------------
def _sentinel_dir(d: Dict[str, Path]) -> Path:
    p = d["root"] / "_sentinels"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _stable_json(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def _hash_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def _sentinel_path(d: Dict[str, Path], step: str, signature: str) -> Path:
    return _sentinel_dir(d) / f"{step}.{signature}.done.json"


def _write_sentinel(
    d: Dict[str, Path],
    *,
    step: str,
    cfg: Dict[str, object],
    params: Dict[str, object],
    outputs: Dict[str, object],
) -> Path:
    payload = {
        "step": step,
        "sample": str(cfg.get("sample")),
        "workdir": str(cfg.get("workdir")),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "params": params,
        "outputs": outputs,
    }
    sig = _hash_str(_stable_json({"step": step, "sample": cfg.get("sample"), "params": params}))
    sp = _sentinel_path(d, step, sig)
    sp.write_text(json.dumps(payload, indent=2))
    return sp


def _has_sentinel(d: Dict[str, Path], step: str, cfg: Dict[str, object], params: Dict[str, object]) -> bool:
    sig = _hash_str(_stable_json({"step": step, "sample": cfg.get("sample"), "params": params}))
    return _sentinel_path(d, step, sig).exists()


# -----------------------------------------------------------------------------
# Stepwise commands
# -----------------------------------------------------------------------------
@app.command()
def extract(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
    threads: int = typer.Option(4, "--threads", "-t", min=1),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
) -> None:
    """Extract per-genome QCMapping files from BAMs."""
    from .extract import bam_to_qc

    cfg = _load_config(config)
    d = _cfg_dirs(cfg)
    _ensure_dirs(d)

    genomes = sorted(dict(cfg["genomes"]).items())
    params = {"threads": int(threads), "genomes": [g for g, _ in genomes]}

    if resume and _has_sentinel(d, "extract", cfg, params):
        typer.echo(f"[extract] skip (sentinel): {cfg['sample']}")
        return

    with ProcessPoolExecutor(max_workers=_clamp(int(threads), 1, len(genomes))) as ex:
        futs = [
            ex.submit(bam_to_qc, Path(bam), d["qc"] / f"{g}_QCMapping.txt", str(cfg["sample"]))
            for g, bam in genomes
        ]
        for f in as_completed(futs):
            f.result()

    outputs = {
        "qc_dir": str(d["qc"]),
        "qc_files": [str(d["qc"] / f"{g}_QCMapping.txt") for g, _ in genomes],
    }
    sp = _write_sentinel(d, step="extract", cfg=cfg, params=params, outputs=outputs)
    typer.echo(f"[extract] done ({cfg['sample']}) sentinel={sp.name}")


@app.command()
def filter(
    config: Optional[Path] = typer.Option(None, "--config", "-c", exists=True, readable=True),
    configs: Optional[Path] = typer.Option(None, "--configs", exists=True, readable=True),
    threads: int = typer.Option(4, "--threads", "-t", min=1),
    min_barcode_freq: Optional[int] = typer.Option(None, "--min-barcode-freq", min=1),
    normalize_bc: bool = typer.Option(False, "--normalize-bc/--no-normalize-bc"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
) -> None:
    """Filter QCMapping files by barcode frequency."""
    from .filtering import filter_qc_file

    cfgs = _load_one_or_many_configs(config=config, configs=configs, min_barcode_freq=min_barcode_freq)
    for cfg in cfgs:
        d = _cfg_dirs(cfg)
        _ensure_dirs(d)

        genomes = sorted(dict(cfg["genomes"]).keys())
        minf = int(min_barcode_freq if min_barcode_freq is not None else cfg.get("min_barcode_freq", 10))
        sample_for_norm = str(cfg["sample"]) if normalize_bc else None
        pool_n = _clamp(int(threads), 1, len(genomes))

        params = {
            "threads": int(threads),
            "min_barcode_freq": minf,
            "normalize_bc": bool(normalize_bc),
            "genomes": genomes,
        }

        if resume and _has_sentinel(d, "filter", cfg, params):
            typer.echo(f"[filter] skip (sentinel): {cfg['sample']}")
            continue

        with ProcessPoolExecutor(max_workers=pool_n) as ex:
            futs = []
            for g in genomes:
                ip = d["qc"] / f"{g}_QCMapping.txt"
                op = d["filtered"] / f"filtered_{g}_QCMapping.txt"
                futs.append(ex.submit(filter_qc_file, ip, op, minf, sample_for_norm))
            rows = [int(f.result()) for f in as_completed(futs)]

        outputs = {"filtered_dir": str(d["filtered"]), "n_genomes": len(genomes), "rows_per_genome": rows}
        sp = _write_sentinel(d, step="filter", cfg=cfg, params=params, outputs=outputs)
        typer.echo(f"[filter] done ({cfg['sample']}) sentinel={sp.name}")


@app.command()
def chunks(
    config: Optional[Path] = typer.Option(None, "--config", "-c", exists=True, readable=True),
    configs: Optional[Path] = typer.Option(None, "--configs", exists=True, readable=True),
    chunk_size_cells: Optional[int] = typer.Option(None, "--chunk-size-cells", min=1),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
) -> None:
    """Create barcode chunk files."""
    from .chunks import make_barcode_chunks

    cfgs = _load_one_or_many_configs(config=config, configs=configs, chunk_size_cells=chunk_size_cells)
    for cfg in cfgs:
        d = _cfg_dirs(cfg)
        _ensure_dirs(d)

        n_cells = int(chunk_size_cells if chunk_size_cells is not None else cfg.get("chunk_size_cells", 5000))
        params = {"chunk_size_cells": n_cells}

        if resume and _has_sentinel(d, "chunks", cfg, params):
            typer.echo(f"[chunks] skip (sentinel): {cfg['sample']}")
            continue

        n = make_barcode_chunks(d["filtered"], d["chunks"], str(cfg["sample"]), n_cells)
        outputs = {"chunks_dir": str(d["chunks"]), "n_chunks": int(n)}
        sp = _write_sentinel(d, step="chunks", cfg=cfg, params=params, outputs=outputs)
        typer.echo(f"[chunks] {cfg['sample']}: wrote {n} chunk files sentinel={sp.name}")


@app.command()
def assign(
    config: Optional[Path] = typer.Option(None, "-c", "--config", exists=True, readable=True),
    configs: Optional[Path] = typer.Option(None, "--configs", exists=True, readable=True),
    threads: int = typer.Option(16, "-t", "--threads", min=1),
    alpha: Optional[float] = typer.Option(None, "--alpha"),
    k: Optional[int] = typer.Option(None, "--k"),
    mapq_min: Optional[int] = typer.Option(None, "--mapq-min"),
    xa_max: Optional[int] = typer.Option(None, "--xa-max"),
    edges_workers: Optional[int] = typer.Option(None, "--edges-workers"),
    edges_max_reads: Optional[int] = typer.Option(None, "--edges-max-reads"),
    ecdf_workers: Optional[int] = typer.Option(None, "--ecdf-workers"),
    chunksize: Optional[int] = typer.Option(None, "--chunksize"),
    batch_size: Optional[int] = typer.Option(None, "--batch-size"),
    # ----------------------------
    # NEW: step control flags
    # ----------------------------
    skip_edges: bool = typer.Option(
        False,
        "--skip-edges",
        help="Skip learning global edges; requires ExplorationReadLevel/global_edges.npz to exist.",
    ),
    skip_ecdf: bool = typer.Option(
        False,
        "--skip-ecdf",
        help="Skip learning global ECDFs; requires ExplorationReadLevel/global_ecdf.npz to exist.",
    ),
    only_score: bool = typer.Option(
        False,
        "--only-score",
        help="Only score chunks; implies --skip-edges and --skip-ecdf (models must exist).",
    ),
    score_workers: Optional[int] = typer.Option(
        None,
        "--score-workers",
        help="Number of parallel workers for scoring chunks. Default: 2 (independent of --threads).",
    ),
    verbose: bool = typer.Option(True, "--verbose/--quiet"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
) -> None:
    """Learn edges/ECDFs and score each chunk (parallel)."""
    from .assign_streaming import learn_edges, learn_ecdfs, score_chunk

    if only_score:
        skip_edges = True
        skip_ecdf = True

    cfgs = _load_one_or_many_configs(config=config, configs=configs)
    for cfg in cfgs:
        d = _cfg_dirs(cfg)
        _ensure_dirs(d)

        _apply_assign_overrides(
            cfg,
            alpha=alpha,
            k=k,
            mapq_min=mapq_min,
            xa_max=xa_max,
            chunksize=chunksize,
            batch_size=batch_size,
            edges_workers=edges_workers,
            edges_max_reads=edges_max_reads,
            ecdf_workers=ecdf_workers,
        )

        workdir = Path(str(cfg["workdir"]))
        sample = str(cfg["sample"])
        _ensure_minimal_chunk(workdir, sample)

        chunks_dir = d["chunks"]
        aconf = cfg.get("assign", {}) if isinstance(cfg.get("assign"), dict) else {}

        alpha_eff = float(aconf.get("alpha", 0.05))
        k_eff = int(aconf.get("k", 10))
        mapq_min_eff = int(aconf.get("mapq_min", 20))
        xa_max_eff = int(aconf.get("xa_max", 2))
        chunksize_val = int(aconf.get("chunksize", 1_000_000))
        batch_size_val = int(aconf.get("batch_size", 32))

        threads_eff = max(1, int(threads))
        edges_workers_eff = max(1, min(int(aconf.get("edges_workers", threads_eff)), threads_eff))
        ecdf_workers_eff = max(1, min(int(aconf.get("ecdf_workers", threads_eff)), threads_eff))

        edges_max_reads_eff = aconf.get("edges_max_reads", None)
        if edges_max_reads_eff is not None and int(edges_max_reads_eff) <= 0:
            edges_max_reads_eff = None
        if edges_max_reads_eff is not None:
            edges_max_reads_eff = int(edges_max_reads_eff)

        # NOTE: sentinel should not gate partial runs; only gate full runs.
        params = {
            "threads": threads_eff,
            "alpha": alpha_eff,
            "k": k_eff,
            "mapq_min": mapq_min_eff,
            "xa_max": xa_max_eff,
            "chunksize": chunksize_val,
            "batch_size": batch_size_val,
            "edges_workers": edges_workers_eff,
            "ecdf_workers": ecdf_workers_eff,
            "edges_max_reads": edges_max_reads_eff,
            "skip_edges": bool(skip_edges),
            "skip_ecdf": bool(skip_ecdf),
            "only_score": bool(only_score),
        }

        # If doing a partial run, DO NOT skip via sentinel (because outputs are partial).
        if (not skip_edges) and (not skip_ecdf) and resume and _has_sentinel(d, "assign", cfg, params):
            typer.echo(f"[assign] skip (sentinel): {sample}")
            continue

        chunk_files = sorted(chunks_dir.glob("*_cell_map_ref_chunk_*.txt"))
        if not chunk_files:
            _ensure_minimal_chunk(workdir, sample)
            chunk_files = sorted(chunks_dir.glob("*_cell_map_ref_chunk_*.txt"))

        exp_dir = workdir / sample / "ExplorationReadLevel"
        exp_dir.mkdir(parents=True, exist_ok=True)
        edges_npz = exp_dir / "global_edges.npz"
        ecdf_npz = exp_dir / "global_ecdf.npz"

        if skip_edges and not edges_npz.exists():
            raise typer.BadParameter(
                f"--skip-edges requested but missing edges model: {edges_npz}. "
                f"Run without --skip-edges at least once."
            )
        if skip_ecdf and not ecdf_npz.exists():
            raise typer.BadParameter(
                f"--skip-ecdf requested but missing ECDF model: {ecdf_npz}. "
                f"Run without --skip-ecdf at least once."
            )

        # scoring workers is independent of --threads; default 2
        score_workers_eff = 2 if score_workers is None else max(1, int(score_workers))
        pool_n = min(score_workers_eff, len(chunk_files))

        if verbose:
            typer.echo(
                f"[assign] {sample} chunksize={chunksize_val:,} batch_size={batch_size_val} threads={threads_eff} "
                f"edges_workers={edges_workers_eff} ecdf_workers={ecdf_workers_eff}"
                + (f" edges_max_reads={int(edges_max_reads_eff):,}" if edges_max_reads_eff is not None else "")
                + (" [only-score]" if only_score else "")
                + (" [skip-edges]" if (skip_edges and not only_score) else "")
                + (" [skip-ecdf]" if (skip_ecdf and not only_score) else "")
                + f" score_workers={score_workers_eff}"
            )

        # (1) edges
        if not skip_edges:
            learn_edges(
                workdir=workdir,
                sample=sample,
                chunks_dir=chunks_dir,
                out_model=edges_npz,
                mapq_min=mapq_min_eff,
                xa_max=xa_max_eff,
                chunksize=chunksize_val,
                k=k_eff,
                batch_size=batch_size_val,
                workers=edges_workers_eff,
                max_reads_per_genome=edges_max_reads_eff,
                verbose=verbose,
            )
        else:
            if verbose:
                typer.echo(f"[assign/edges] skip: reuse {edges_npz}")

        # (2) ecdf
        if not skip_ecdf:
            learn_ecdfs(
                workdir=workdir,
                sample=sample,
                chunks_dir=chunks_dir,
                edges_model=edges_npz,
                out_model=ecdf_npz,
                mapq_min=mapq_min_eff,
                xa_max=xa_max_eff,
                chunksize=chunksize_val,
                workers=ecdf_workers_eff,
                verbose=verbose,
            )
        else:
            if verbose:
                typer.echo(f"[assign/ecdf] skip: reuse {ecdf_npz}")

        # (3) score chunks
        typer.echo(f"[assign/score] {sample}: start {len(chunk_files)} chunks, procs={pool_n}")
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
                    mapq_min=mapq_min_eff,
                    xa_max=xa_max_eff,
                    chunksize=chunksize_val,
                    alpha=alpha_eff,
                    verbose=verbose,
                ): chf
                for chf in chunk_files
            }
            for f in as_completed(fut):
                f.result()

        # Only write an assign sentinel for FULL runs (so it remains meaningful).
        if (not skip_edges) and (not skip_ecdf):
            outputs = {
                "edges_model": str(edges_npz),
                "ecdf_model": str(ecdf_npz),
                "chunks_dir": str(chunks_dir),
                "n_chunks": len(chunk_files),
            }
            sp = _write_sentinel(d, step="assign", cfg=cfg, params=params, outputs=outputs)
            typer.echo(f"[assign] done: {sample} sentinel={sp.name}")
        else:
            typer.echo(f"[assign] done: {sample} (partial run; no sentinel)")



@app.command()
def genotyping(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
    assign_glob: Optional[str] = typer.Option(None, "--assign", help="Glob to assign outputs. If omitted, inferred."),
    outdir: Optional[Path] = typer.Option(None, "--outdir", help="Override output dir (default: <workdir>/<sample>/final)."),
    sample: Optional[str] = typer.Option(None, "--sample", help="Override sample name from config."),
    # core
    min_reads: Optional[int] = typer.Option(None, "--min-reads"),
    single_mass_min: Optional[float] = typer.Option(None, "--single-mass-min"),
    ratio_top1_top2_min: Optional[float] = typer.Option(None, "--ratio-top1-top2-min"),
    threads: Optional[int] = typer.Option(None, "--threads"),
    pass1_workers: Optional[int] = typer.Option(None, "--pass1-workers"),
    shards: Optional[int] = typer.Option(None, "--shards"),
    chunk_rows: Optional[int] = typer.Option(None, "--chunk-rows"),
    pass2_chunksize: Optional[int] = typer.Option(None, "--pass2-chunksize"),
    winner_only: Optional[bool] = typer.Option(None, "--winner-only/--no-winner-only"),
    # optional read-filter (NEW)
    max_hits: Optional[int] = typer.Option(
        None,
        "--max-hits",
        help="(Optional) Drop promiscuous reads among assigned_class=ambiguous only. "
        "Enabled only if BOTH --max-hits and --hits-delta-as are provided. "
        "A read is dropped if >max-hits genomes fall within (best_AS - hits-delta-as).",
    ),
    hits_delta_mapq: Optional[int] = typer.Option(
        None,
        "--hits-delta-mapq",
        help="(Optional) MAPQ window for counting near-top hits for --max-hits filtering. "
        "Enabled only if BOTH --max-hits and --hits-delta-as are provided.",
    ),
    # fusion
    beta: Optional[float] = typer.Option(None, "--beta"),
    w_as: Optional[float] = typer.Option(None, "--w-as"),
    w_mapq: Optional[float] = typer.Option(None, "--w-mapq"),
    w_nm: Optional[float] = typer.Option(None, "--w-nm"),
    ambient_const: Optional[float] = typer.Option(None, "--ambient-const"),
    # empty gates
    empty_bic_margin: Optional[float] = typer.Option(None, "--empty-bic-margin"),
    empty_top1_max: Optional[float] = typer.Option(None, "--empty-top1-max"),
    empty_ratio12_max: Optional[float] = typer.Option(None, "--empty-ratio12-max"),
    empty_reads_max: Optional[int] = typer.Option(None, "--empty-reads-max"),
    empty_seed_bic_min: Optional[float] = typer.Option(None, "--empty-seed-bic-min"),
    empty_tau_quantile: Optional[float] = typer.Option(None, "--empty-tau-quantile"),
    empty_jsd_max: Optional[float] = typer.Option(None, "--empty-jsd-max"),
    jsd_normalize: Optional[bool] = typer.Option(None, "--jsd-normalize/--no-jsd-normalize"),
    # doublet gates
    bic_margin: Optional[float] = typer.Option(None, "--bic-margin"),
    doublet_minor_min: Optional[float] = typer.Option(None, "--doublet-minor-min"),
    # ambient iteration
    eta_iters: Optional[int] = typer.Option(None, "--eta-iters"),
    eta_seed_quantile: Optional[float] = typer.Option(None, "--eta-seed-quantile"),
    topk_genomes: Optional[int] = typer.Option(None, "--topk-genomes"),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
) -> None:
    """Posterior-aware genotyping (merge → per-cell genotype calls)."""    
    from .genotyping import _run_genotyping


    cfg = _load_config(config)
    d = _cfg_dirs(cfg)
    _ensure_dirs(d)

    sample_eff = str(sample or cfg["sample"])
    outdir_eff = Path(outdir or d["final"])
    outdir_eff.mkdir(parents=True, exist_ok=True)

    if eta_seed_quantile is not None and not (0.0 < float(eta_seed_quantile) < 1.0):
        raise typer.BadParameter("--eta-seed-quantile must be in (0,1)")

    if assign_glob is None:
        chunks_dir = d["chunks"]
        patterns = [
            str(chunks_dir / "**" / "*_filtered.tsv.gz"),
            str(chunks_dir / "**" / "*filtered.tsv.gz"),
            str(chunks_dir / "**" / "*.tsv.gz"),
            str(chunks_dir / "**" / "*.tsv"),
        ]
        chosen = None
        for pat in patterns:
            if glob.glob(pat, recursive=True):
                chosen = pat
                break
        if chosen is None:
            raise typer.BadParameter(
                f"No assign outputs found under {chunks_dir}. Provide --assign or run the 'assign' step first."
            )
        assign_glob = chosen

    if not glob.glob(assign_glob, recursive=True):
        raise typer.BadParameter(f"No assign files matched: {assign_glob}")

    kwargs: Dict[str, object] = {"assign": assign_glob, "outdir": outdir_eff, "sample": sample_eff}

    # core
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
    if single_mass_min is not None:
        kwargs["single_mass_min"] = float(single_mass_min)
    if ratio_top1_top2_min is not None:
        kwargs["ratio_top1_top2_min"] = float(ratio_top1_top2_min)
    if pass1_workers is not None:
        kwargs["pass1_workers"] = max(1, int(pass1_workers))

    # NEW: optional read-filter passthrough
    if max_hits is not None:
        kwargs["max_hits"] = int(max_hits)
    if hits_delta_mapq is not None:
        kwargs["hits_delta_mapq"] = int(hits_delta_mapq)

    # fusion / gates
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

    if empty_bic_margin is not None:
        kwargs["empty_bic_margin"] = float(empty_bic_margin)
    if empty_top1_max is not None:
        kwargs["empty_top1_max"] = float(empty_top1_max)
    if empty_ratio12_max is not None:
        kwargs["empty_ratio12_max"] = float(empty_ratio12_max)
    if empty_reads_max is not None:
        kwargs["empty_reads_max"] = int(empty_reads_max)
    if empty_seed_bic_min is not None:
        kwargs["empty_seed_bic_min"] = float(empty_seed_bic_min)
    if empty_tau_quantile is not None:
        kwargs["empty_tau_quantile"] = float(empty_tau_quantile)
    if empty_jsd_max is not None:
        kwargs["empty_jsd_max"] = float(empty_jsd_max)
    if jsd_normalize is not None:
        kwargs["jsd_normalize"] = bool(jsd_normalize)

    if bic_margin is not None:
        kwargs["bic_margin"] = float(bic_margin)
    if doublet_minor_min is not None:
        kwargs["doublet_minor_min"] = float(doublet_minor_min)

    if eta_iters is not None:
        kwargs["eta_iters"] = max(0, int(eta_iters))
    if eta_seed_quantile is not None:
        kwargs["eta_seed_quantile"] = float(eta_seed_quantile)
    if topk_genomes is not None:
        kwargs["topk_genomes"] = max(1, int(topk_genomes))

    # Sentinel for genotyping keyed by actual kwargs (Paths stringified)
    params = {k: (str(v) if isinstance(v, Path) else v) for k, v in kwargs.items()}
    if resume and _has_sentinel(d, "genotyping", cfg, params):
        typer.echo(f"[genotyping] skip (sentinel): {sample_eff}")
        return

    _run_genotyping(**kwargs)

    outputs = {"outdir": str(outdir_eff), "assign": str(assign_glob)}
    sp = _write_sentinel(d, step="genotyping", cfg=cfg, params=params, outputs=outputs)
    typer.echo(f"[genotyping] done ({sample_eff}) sentinel={sp.name}")


# -----------------------------------------------------------------------------
# Optional extra commands
# -----------------------------------------------------------------------------
@app.command()
def interpool(
    configs: Path = typer.Option(..., "--configs", help="TSV with: sample, genome, bam, workdir"),
    outdir: Optional[Path] = typer.Option(None, "--outdir"),
) -> None:
    from .interpool import interpool_summary

    res = interpool_summary(configs_tsv=configs, outdir=outdir)
    typer.echo(f"[interpool] BC counts: {res['bc_counts']}")
    typer.echo(f"[interpool] Read composition: {res['read_comp']}")
    typer.echo(f"[interpool] PDF: {res['pdf']}")


@app.command()
def plate(
    workdir: Path = typer.Option(..., "--workdir"),
    plate_map: Path = typer.Option(..., "--plate-map"),
    outdir: Optional[Path] = typer.Option(None, "--outdir"),
    xa_max: int = typer.Option(2, "--xa-max"),
) -> None:
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
    config: Optional[Path] = typer.Option(None, "--config", "-c", exists=True, readable=True),
    sample: Optional[str] = typer.Option(None, "--sample"),
    genome: Optional[str] = typer.Option(None, "--genome", help="Comma-separated genome names"),
    bam: Optional[str] = typer.Option(None, "--bam", help="Comma-separated BAM paths"),
    workdir: Optional[Path] = typer.Option(None, "--workdir"),
    configs: Optional[Path] = typer.Option(None, "--configs", help="TSV with: sample, genome, bam, workdir"),
    verbose: bool = typer.Option(True, "--verbose/--quiet"),
    min_barcode_freq: int = typer.Option(10, "--min-barcode-freq"),
    chunk_size_cells: int = typer.Option(5000, "--chunk-size-cells"),
    threads: int = typer.Option(8, "--threads", "-t", min=1),
    resume: bool = typer.Option(True, "--resume/--no-resume"),
    force_steps: str = typer.Option("", "--force-steps"),
    skip_to: str = typer.Option("", "--skip-to"),
    only_steps: str = typer.Option("", "--only-steps"),
    # assign overrides
    assign_alpha: Optional[float] = typer.Option(None, "--assign-alpha"),
    assign_k: Optional[int] = typer.Option(None, "--assign-k"),
    assign_mapq_min: Optional[int] = typer.Option(None, "--assign-mapq-min"),
    assign_xa_max: Optional[int] = typer.Option(None, "--assign-xa-max"),
    assign_chunksize: Optional[int] = typer.Option(None, "--assign-chunksize"),
    assign_batch_size: Optional[int] = typer.Option(None, "--assign-batch-size"),
    assign_edges_workers: Optional[int] = typer.Option(None, "--assign-edges-workers"),
    assign_edges_max_reads: Optional[int] = typer.Option(None, "--assign-edges-max-reads"),
    ecdf_workers: Optional[int] = typer.Option(None, "--ecdf-workers"),
    # genotyping overrides
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
    genotyping_pass1_workers: Optional[int] = typer.Option(None, "--genotyping-pass1-workers"),
    genotyping_pass2_chunksize: Optional[int] = typer.Option(None, "--genotyping-pass2-chunksize"),
    genotyping_winner_only: Optional[bool] = typer.Option(None, "--genotyping-winner-only/--no-genotyping-winner-only"),
    genotyping_doublet_minor_min: Optional[float] = typer.Option(None, "--genotyping-doublet-minor-min"),
    genotyping_single_mass_min: Optional[float] = typer.Option(None, "--genotyping-single-mass-min"),
    genotyping_ratio_top1_top2_min: Optional[float] = typer.Option(None, "--genotyping-ratio-top1-top2-min"),
    genotyping_eta_iters: Optional[int] = typer.Option(None, "--genotyping-eta-iters"),
    genotyping_eta_seed_quantile: Optional[float] = typer.Option(None, "--genotyping-eta-seed-quantile"),
    genotyping_empty_bic_margin: Optional[float] = typer.Option(None, "--genotyping-empty-bic-margin"),
    genotyping_empty_top1_max: Optional[float] = typer.Option(None, "--genotyping-empty-top1-max"),
    genotyping_empty_ratio12_max: Optional[float] = typer.Option(None, "--genotyping-empty-ratio12-max"),
    genotyping_empty_reads_max: Optional[int] = typer.Option(None, "--genotyping-empty-reads-max"),
    genotyping_empty_seed_bic_min: Optional[float] = typer.Option(None, "--genotyping-empty-seed-bic-min"),
    genotyping_empty_tau_quantile: Optional[float] = typer.Option(None, "--genotyping-empty-tau-quantile"),
    genotyping_empty_jsd_max: Optional[float] = typer.Option(None, "--genotyping-empty-jsd-max"),
    genotyping_jsd_normalize: Optional[bool] = typer.Option(None, "--genotyping-jsd-normalize/--no-genotyping-jsd-normalize"),
    # NEW: pass through max-hits filter to genotyping.py
    genotyping_max_hits: Optional[int] = typer.Option(None, "--genotyping-max-hits"),
    genotyping_hits_delta_mapq: Optional[int] = typer.Option(None, "--genotyping-hits-delta-mapq"),
) -> None:
    """Run the full pipeline."""
    inline_ready = all([sample, genome, bam, workdir])
    inline_partial = any([sample, genome, bam, workdir]) and not inline_ready
    modes_used = sum([1 if config else 0, 1 if inline_ready else 0, 1 if configs else 0])

    if inline_partial:
        raise typer.BadParameter("Inline mode requires --sample, --genome, --bam, and --workdir together.")
    if modes_used != 1:
        raise typer.BadParameter("Choose exactly one mode: --config OR inline OR --configs")

    force = [s.strip() for s in force_steps.split(",") if s.strip()]
    only = [s.strip() for s in only_steps.split(",") if s.strip()]

    genotyping_conf: Dict[str, object] = {}
    for k2, v2 in [
        ("min_reads", genotyping_min_reads),
        ("beta", genotyping_beta),
        ("w_as", genotyping_w_as),
        ("w_mapq", genotyping_w_mapq),
        ("w_nm", genotyping_w_nm),
        ("ambient_const", genotyping_ambient_const),
        ("bic_margin", genotyping_bic_margin),
        ("topk_genomes", genotyping_topk_genomes),
        ("threads", genotyping_threads),
        ("shards", genotyping_shards),
        ("chunk_rows", genotyping_chunk_rows),
        ("pass1_workers", genotyping_pass1_workers),
        ("pass2_chunksize", genotyping_pass2_chunksize),
        ("winner_only", genotyping_winner_only),
        ("doublet_minor_min", genotyping_doublet_minor_min),
        ("single_mass_min", genotyping_single_mass_min),
        ("ratio_top1_top2_min", genotyping_ratio_top1_top2_min),
        ("eta_iters", genotyping_eta_iters),
        ("eta_seed_quantile", genotyping_eta_seed_quantile),
        ("empty_bic_margin", genotyping_empty_bic_margin),
        ("empty_top1_max", genotyping_empty_top1_max),
        ("empty_ratio12_max", genotyping_empty_ratio12_max),
        ("empty_reads_max", genotyping_empty_reads_max),
        ("empty_seed_bic_min", genotyping_empty_seed_bic_min),
        ("empty_tau_quantile", genotyping_empty_tau_quantile),
        ("empty_jsd_max", genotyping_empty_jsd_max),
        ("jsd_normalize", genotyping_jsd_normalize),
        # NEW
        ("max_hits", genotyping_max_hits),
        ("hits_delta_mapq", genotyping_hits_delta_mapq),
    ]:
        if v2 is not None:
            genotyping_conf[k2] = v2

    if "eta_seed_quantile" in genotyping_conf:
        q = float(genotyping_conf["eta_seed_quantile"])
        if not (0.0 < q < 1.0):
            raise typer.BadParameter("--genotyping-eta-seed-quantile must be in (0,1)")

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
        typer.echo(f"[run] Executed: {result.get('executed')}")
        typer.echo(f"[run] Skipped:  {result.get('skipped')}")
        typer.echo(f"[run] {cfg['sample']} pipeline complete")

    if config is not None:
        cfg = _load_config(config)
        _ensure_minimal_chunk(Path(str(cfg["workdir"])), str(cfg["sample"]))
        _do_one(cfg)
        return

    if inline_ready:
        assert sample and genome and bam and workdir
        cfg = _cfg_from_inline(sample, genome, bam, str(workdir), min_barcode_freq, chunk_size_cells)
        _ensure_minimal_chunk(Path(str(cfg["workdir"])), str(cfg["sample"]))
        _do_one(cfg)
        return

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

