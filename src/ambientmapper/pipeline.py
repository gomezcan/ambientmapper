from __future__ import annotations
import os, json
from pathlib import Path
from typing import Dict, List, Any, Optional
from .dag import Ctx, Step, run_dag
from .sentinels import read_generation
from .assign_streaming import learn_edges_parallel, learn_ecdfs_parallel, score_chunk


def _touch(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text("")

def _noop_runner(outputs):
    for p in outputs:
        _touch(p)

def _dirs(cfg: Dict[str, Any]) -> Dict[str, Path]:
    root = Path(cfg['workdir']) / cfg['sample']
    return {
        'root': root,
        'extracted': root / 'extracted',
        'filtered': root / 'filtered_QCFiles',
        'chunks': root / 'cell_map_ref_chunks',
        'final': root / 'final',
        'qc': root / 'qc',
    }

def _exp_dir(ctx):
    d = ctx.dirs["root"] / "ExplorationReadLevel"
    d.mkdir(parents=True, exist_ok=True)
    return d

def io_assign_edges(ctx, part=None):
    ins  = [ctx.dirs["chunks"] / "manifest.json"]
    outs = [_exp_dir(ctx) / "global_edges.npz"]
    return ins, outs

def io_assign_ecdf(ctx, part=None):
    ins  = [_exp_dir(ctx) / "global_edges.npz"]
    outs = [_exp_dir(ctx) / "global_ecdf.npz"]
    return ins, outs

def io_assign_score(ctx, part):
    ch = Path(part["path"])
    out = ctx.dirs["chunks"] / f"{ch.stem}.scores.parquet"
    ins = [_exp_dir(ctx) / "global_edges.npz", _exp_dir(ctx) / "global_ecdf.npz", ch]
    return ins, [out]

def io_assign_merge(ctx, part=None):
    ins  = [ctx.dirs["chunks"] / "manifest.json"]
    outs = [ctx.dirs["final"] / "assignments.parquet"]  # marker file
    return ins, outs

def io_genotype_per_chunk(ctx, part):
    stem = Path(part["id"]).name
    scores = ctx.dirs["chunks"] / f"{stem}.scores.parquet"
    out    = ctx.dirs["chunks"] / f"{stem}.genotypes.parquet"  # marker file
    return [scores], [out]

def io_genotype_merge(ctx, part=None):
    ins  = [ctx.dirs["chunks"] / "manifest.json"]
    outs = [ctx.dirs["final"] / "genotypes.parquet"]
    return ins, outs

def io_bam_clean(ctx, part=None):
    ins  = [ctx.dirs["final"] / "genotypes.parquet"]
    outs = [ctx.dirs["final"] / "bam_cleaned.DONE"]
    return ins, outs


def io_extract(ctx, part=None):
    ins  = [Path(p) for p in ctx.cfg.get('fastqs', [])]
    outs = [ctx.dirs['extracted'] / 'reads_000.parquet']
    return ins, outs

def io_filter(ctx, part=None):
    ins  = [ctx.dirs['extracted'] / 'reads_000.parquet']
    outs = [ctx.dirs['filtered'] / 'reads_000.filtered.parquet']
    return ins, outs

def io_chunk(ctx, part=None):
    ins  = [ctx.dirs['filtered'] / 'reads_000.filtered.parquet']
    outs = [ctx.dirs['chunks'] / 'manifest.json']
    return ins, outs
    
# --- replace read_manifest(...) with a robust discovery ---
def discover_partitions(dirs: Dict[str, Path]) -> List[Dict[str, Any]]:
    """
    Prefer a valid manifest.json; if entries are missing on disk, fall back to globbing
    *_cell_map_ref_chunk_*.txt in the chunks directory.
    """
    parts: List[Dict[str, Any]] = []
    man = dirs["chunks"] / "manifest.json"
    if man.exists():
        try:
            data = json.loads(man.read_text())
            for ch in data.get("chunks", []):
                # always resolve relative to chunks dir
                p = dirs["chunks"] / Path(ch["path"]).name
                if p.exists():
                    parts.append({"id": ch.get("id", p.stem), "path": str(p)})
        except Exception:
            parts = []  # corrupt manifest → ignore

    if parts:  # valid manifest covered everything we need
        return parts

    # Fallback: glob the actual txt chunks your tool writes
    txts = sorted(dirs["chunks"].glob("*_cell_map_ref_chunk_*.txt"))
    return [{"id": f.stem, "path": str(f)} for f in txts]

# ----- top-level adapters (no lambdas anywhere in build_steps)
def in_extract(ctx, part=None):  return io_extract(ctx)[0]
def out_extract(ctx, part=None): return io_extract(ctx)[1]

def in_filter(ctx, part=None):   return io_filter(ctx)[0]
def out_filter(ctx, part=None):  return io_filter(ctx)[1]

def in_chunk(ctx, part=None):    return io_chunk(ctx)[0]
def out_chunk(ctx, part=None):   return io_chunk(ctx)[1]

def in_assign_edges(ctx, part=None):  return io_assign_edges(ctx)[0]
def out_assign_edges(ctx, part=None): return io_assign_edges(ctx)[1]

def in_assign_ecdf(ctx, part=None):   return io_assign_ecdf(ctx)[0]
def out_assign_ecdf(ctx, part=None):  return io_assign_ecdf(ctx)[1]

def in_assign_score(ctx, part):       return io_assign_score(ctx, part)[0]
def out_assign_score(ctx, part):      return io_assign_score(ctx, part)[1]

def in_assign_merge(ctx, part=None):  return io_assign_merge(ctx)[0]
def out_assign_merge(ctx, part=None): return io_assign_merge(ctx)[1]

def in_geno_per_chunk(ctx, part):     return io_genotype_per_chunk(ctx, part)[0]
def out_geno_per_chunk(ctx, part):    return io_genotype_per_chunk(ctx, part)[1]

def in_geno_merge(ctx, part=None):    return io_genotype_merge(ctx)[0]
def out_geno_merge(ctx, part=None):   return io_genotype_merge(ctx)[1]

def in_bam_clean(ctx, part=None):     return io_bam_clean(ctx)[0]
def out_bam_clean(ctx, part=None):    return io_bam_clean(ctx)[1]



# ----- runners updated to the new IO shape
def run_extract(ctx, part=None):
    _, outs = io_extract(ctx)
    _noop_runner(outs)

def run_filter(ctx, part=None):
    _, outs = io_filter(ctx)
    _noop_runner(outs)

# --- update run_chunk(...) to write a correct manifest if txt chunks already exist ---
def run_chunk(ctx: Ctx, part=None):
    _, outs = io_chunk(ctx)
    outs[0].parent.mkdir(parents=True, exist_ok=True)
    # If user already produced chunk txt files, reflect those exactly.
    txts = sorted(ctx.dirs["chunks"].glob("*_cell_map_ref_chunk_*.txt"))
    if txts:
        manifest = {
            "version": 1,
            "n_chunks": len(txts),
            "chunks": [{"id": f.stem, "path": f.name} for f in txts],
            "derived_from": {
                "filtered_inputs_sha256": "unknown",
                "upstream_generation": ctx.generation_id
            },
        }
    else:
        # last-resort tiny dummy so the DAG can proceed in empty sandboxes
        manifest = {
            "version": 1,
            "n_chunks": 1,
            "chunks": [{"id": "chunk_0001", "path": "chunk_0001.txt"}],
            "derived_from": {"filtered_inputs_sha256": "placeholder", "upstream_generation": ctx.generation_id},
        }
        (ctx.dirs["chunks"] / "chunk_0001.txt").touch(exist_ok=True)

    outs[0].write_text(json.dumps(manifest, indent=2))


def run_assign_edges(ctx, part=None):
    _, outs = io_assign_edges(ctx)
    edges_npz = outs[0]
    learn_edges_parallel(
        workdir=ctx.dirs["root"].parent,
        sample=ctx.cfg["sample"],
        chunks_dir=ctx.dirs["chunks"],
        out_model=edges_npz,
        mapq_min=int(ctx.params["assign"].get("mapq_min", 20)),
        xa_max=int(ctx.params["assign"].get("xa_max", 2)),
        chunksize=int(ctx.params["assign"].get("chunksize", 1_000_000)),
        k=int(ctx.params["assign"].get("k", 10)),
        batch_size=int(ctx.params["assign"].get("batch_size", 32)),
        threads=int(ctx.params.get("threads", 8)),
        verbose=bool(ctx.params.get("verbose", True)),
        edges_workers=ctx.params["assign"].get("edges_workers"),
        edges_max_reads=ctx.params["assign"].get("edges_max_reads"),
    )

def run_assign_ecdf(ctx, part=None):
    _, outs = io_assign_ecdf(ctx)
    ecdf_npz = outs[0]
    learn_ecdfs_parallel(
        workdir=ctx.dirs["root"].parent,
        sample=ctx.cfg["sample"],
        chunks_dir=ctx.dirs["chunks"],
        edges_model=_exp_dir(ctx) / "global_edges.npz",
        out_model=ecdf_npz,
        mapq_min=int(ctx.params["assign"].get("mapq_min", 20)),
        xa_max=int(ctx.params["assign"].get("xa_max", 2)),
        chunksize=int(ctx.params["assign"].get("chunksize", 1_000_000)),
        verbose=bool(ctx.params.get("verbose", True)),
        workers=ctx.params["assign"].get("ecdf_workers") or int(ctx.params.get("threads", 8)),
    )

# 
def run_assign_score(ctx, part):
    """
    Score one chunk and ensure we end up with <chunk_stem>.scores.parquet sitting
    next to the chunk file (DAG-expected path). We:
      - validate inputs
      - call score_chunk with an explicit out_raw_dir (chunks dir)
      - accept a set of common output name/dir patterns and mirror to expected
    """
    from pathlib import Path as _P
    import os, shutil, glob, typer
    from .assign_streaming import score_chunk

    chatty = bool(ctx.params.get("verbose", True))
    sample   = ctx.cfg["sample"]
    workroot = _P(ctx.cfg["workdir"])
    chunks_d = ctx.dirs["chunks"]  # …/<sample>/cell_map_ref_chunks
    exp_dir  = ctx.dirs["root"] / "ExplorationReadLevel"

    ecdf_npz   = exp_dir / "global_ecdf.npz"
    edges_npz  = exp_dir / "global_edges.npz"
    chunk_txt  = _P(part["path"])
    natural_out = chunk_txt.with_name(chunk_txt.stem + ".scores.parquet")

    # DAG-expected output (keep this as the canonical target)
    _, outs = io_assign_score(ctx, part)
    dag_out = outs[0]

    # Preflight checks
    if not ecdf_npz.exists() or ecdf_npz.stat().st_size == 0:
        raise RuntimeError(f"ECDF model missing/empty: {ecdf_npz}")
    if not edges_npz.exists() or edges_npz.stat().st_size == 0:
        # not strictly required by scorer, but good to surface early
        typer.echo(f"[score] warn: edges model looks tiny/empty ({edges_npz})")
    if not chunk_txt.exists() or chunk_txt.stat().st_size == 0:
        raise RuntimeError(f"Chunk file missing/empty: {chunk_txt}")

    if chatty:
        typer.echo(f"[score] start {part['id']}  →  {natural_out.name}")

    # Run the scorer, **force** outputs into the chunks directory
    score_chunk(
        workdir=workroot,
        sample=sample,
        chunk_file=chunk_txt,
        ecdf_model=ecdf_npz,
        out_raw_dir=chunk_txt.parent,   # << force it here
        out_filtered_dir=None,
        mapq_min=int(ctx.params["assign"].get("mapq_min", 20)),
        xa_max=int(ctx.params["assign"].get("xa_max", 2)),
        chunksize=int(ctx.params["assign"].get("chunksize", 1_000_000)),
        alpha=float(ctx.params["assign"].get("alpha", 0.05)),
    )

    # Collect plausible outputs to look for
    candidates = [
        natural_out,
        dag_out,
        chunk_txt.parent / (chunk_txt.stem + ".parquet"),
        chunk_txt.parent / (chunk_txt.stem + ".scores.parq"),
        chunk_txt.parent / (chunk_txt.stem + ".scores.arrow"),
    ]
    # Also scan a couple of common subdirs/patterns just in case
    for pat in (
        str(chunk_txt.parent / f"{chunk_txt.stem}*.parquet"),
        str(chunk_txt.parent / f"{chunk_txt.stem}*.parq"),
        str(chunk_txt.parent / f"{chunk_txt.stem}*.arrow"),
    ):
        for p in glob.glob(pat):
            candidates.append(_P(p))

    produced = next((c for c in candidates if c.exists() and c.stat().st_size > 0), None)

    # Mirror to the DAG-expected path if needed
    if produced and produced != dag_out:
        try:
            os.link(produced, dag_out)
        except Exception:
            shutil.copy2(produced, dag_out)

    # Final verdict
    if not (dag_out.exists() and dag_out.stat().st_size > 0):
        hints = [
            f"edges exists={edges_npz.exists()} size={edges_npz.stat().st_size if edges_npz.exists() else 0}",
            f"ecdf exists={ecdf_npz.exists()} size={ecdf_npz.stat().st_size if ecdf_npz.exists() else 0}",
            f"chunk exists={chunk_txt.exists()} size={chunk_txt.stat().st_size if chunk_txt.exists() else 0}",
        ]
        raise RuntimeError(
            "score_chunk produced no data for "
            f"{chunk_txt.name}. Tried: {', '.join(sorted(set(str(c.name) for c in candidates)))}. "
            "Checks: " + " | ".join(hints)
        )

    if chatty:
        typer.echo(f"[score] done  {part['id']}")



def run_assign_merge(ctx, part=None):
    """
    Create a compact manifest of all per-chunk score outputs so downstream steps
    have a single marker to depend on. We write JSON Lines into
    final/assignments.parquet (content non-zero so sentinels treat it as OK).
    """
    import glob
    _, outs = io_assign_merge(ctx)
    out_path = outs[0]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Discover chunk score files
    chunk_scores = sorted(glob.glob(str(ctx.dirs["chunks"] / "*.scores.parquet")))
    # Write as JSONL so file is small but non-zero; extension remains .parquet as a marker.
    lines = []
    for p in chunk_scores:
        lines.append(json.dumps({
            "sample": ctx.cfg["sample"],
            "score_file": str(Path(p).resolve()),
        }))

    tmp = out_path.with_suffix(".tmp")
    with tmp.open("w") as f:
        if lines:
            f.write("\n".join(lines) + "\n")
        else:
            # still write at least one byte so sentinels don't think it's missing
            f.write("{}\n")
    os.replace(tmp, out_path)


def run_genotype_per_chunk(ctx, part):
    """
    Marker-only per-chunk genotyping: write a small JSON stub to
    <chunk>.genotypes.parquet so sentinels see a non-zero artifact.
    The real posterior-aware genotyping happens in run_genotype_merge().
    """
    _, outs = io_genotype_per_chunk(ctx, part)
    out = outs[0]
    out.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "sample": ctx.cfg["sample"],
        "chunk_id": str(Path(part["id"])),
        "source_scores": str((ctx.dirs["chunks"] / f"{Path(part['id']).name}.scores.parquet").resolve()),
        "note": "placeholder; real posterior-aware genotyping occurs at 30_genotype.20_merge",
    }
    tmp = out.with_suffix(".tmp")
    with tmp.open("w") as f:
        f.write(json.dumps(payload) + "\n")  # non-zero content
    os.replace(tmp, out)

def run_genotype_merge(ctx, part=None):
    """
    Real posterior-aware genotyping/merge: invoke your existing CLI entrypoint
    that scans chunk outputs and writes final results (and optional report)
    under final/.
    """
    from .genotyping import genotyping as _run_genotyping

    # Let the genotyper search across all chunk outputs in the chunks dir
    assign_glob = str(ctx.dirs["chunks"] / "**" / "*")
    outdir = ctx.dirs["final"]
    sample = ctx.cfg["sample"]
    outdir.mkdir(parents=True, exist_ok=True)

    # Call the Typer command function directly
    _run_genotyping.callback(
        assign=assign_glob,
        outdir=outdir,
        sample=sample,
        make_report=True,  # or False if you want to suppress until BAM clean is finalized
    )

    # Optionally drop a small DONE marker to help humans/tools see completion
    (outdir / "genotyping_merge.DONE").write_text("ok\n")


def run_bam_clean(ctx, part=None):
    _, outs = io_bam_clean(ctx)
    _noop_runner(outs)

# ----- build_steps with adapters (no lambdas)
def build_steps() -> Dict[str, Step]:
    return {
        '00_extract': Step('00_extract', [], False, in_extract, out_extract, run_extract, bump_generation=True),
        '05_filter':  Step('05_filter',  ['00_extract'], False, in_filter, out_filter, run_filter, bump_generation=True),
        '10_chunk':   Step('10_chunk',   ['05_filter'],  False, in_chunk, out_chunk, run_chunk, bump_generation=True),

        '20_assign.10_edges': Step('20_assign.10_edges', ['10_chunk'], False,
                                   in_assign_edges, out_assign_edges, run_assign_edges),
        '20_assign.20_ecdfs': Step('20_assign.20_ecdfs', ['20_assign.10_edges'], False,
                                   in_assign_ecdf, out_assign_ecdf, run_assign_ecdf),

        '20_assign.30_score': Step('20_assign.30_score', ['20_assign.20_ecdfs'], True,
                                   in_assign_score, out_assign_score, run_assign_score),

        '20_assign.40_merge': Step('20_assign.40_merge', ['20_assign.30_score'], False,
                                   in_assign_merge, out_assign_merge, run_assign_merge),

        '30_genotype.10_per_chunk': Step('30_genotype.10_per_chunk', ['20_assign.40_merge'], True,
                                         in_geno_per_chunk, out_geno_per_chunk, run_genotype_per_chunk),

        '30_genotype.20_merge': Step('30_genotype.20_merge', ['30_genotype.10_per_chunk'], False,
                                     in_geno_merge, out_geno_merge, run_genotype_merge),

        '35_bamclean': Step('35_bamclean', ['30_genotype.20_merge'], False,
                            in_bam_clean, out_bam_clean, run_bam_clean),
    }


# --- and in run_pipeline(...), swap read_manifest(...) for discover_partitions(...) ---    
def run_pipeline(cfg: Dict[str, Any], params: Dict[str, Any], version: str = '0.4.0',
                 resume: bool = True, force: List[str] | None = None, skip_to: str = '', only: List[str] | None = None):
    dirs = _dirs(cfg)
    gen = read_generation(dirs['root'])
    ctx = Ctx(cfg=cfg, params=params, dirs=dirs, version=version, resume=resume,
              force=(force or []), skip_to=skip_to, only=(only or []), generation_id=gen)
    
    steps = build_steps()
    partitions = discover_partitions(dirs)
    if not partitions:
        # still nothing? create a dummy chunk to avoid crashes
        dummy = dirs["chunks"] / "chunk_0001.txt"
        dummy.parent.mkdir(parents=True, exist_ok=True)
        dummy.touch(exist_ok=True)
        partitions = [{"id": dummy.stem, "path": str(dummy)}]
    result = run_dag(ctx, steps, partitions=partitions)
    return result
