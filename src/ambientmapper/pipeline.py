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

def io_extract(ctx, part=None):
    ins = [Path(p) for p in ctx.cfg.get('fastqs', [])]
    outs = [ctx.dirs['extracted'] / 'reads_000.parquet']
    return ins, outs

def io_filter(ctx, part=None):
    ins = [ctx.dirs['extracted'] / 'reads_000.parquet']
    outs = [ctx.dirs['filtered'] / 'reads_000.filtered.parquet']
    return ins, outs

def io_chunk(ctx, part=None):
    ins = [ctx.dirs['filtered'] / 'reads_000.filtered.parquet']
    outs = [ctx.dirs['chunks'] / 'manifest.json']
    return ins, outs

def _exp_dir(ctx):  # ExplorationReadLevel
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
    ch_path = Path(part["path"])
    out = ctx.dirs["chunks"] / f"{Path(part['id']).name}.scores.parquet"
    ins = [_exp_dir(ctx) / "global_edges.npz", _exp_dir(ctx) / "global_ecdf.npz", ch_path]
    return ins, [out]

def io_assign_merge(ctx, part=None):
    ins = [ctx.dirs['chunks'] / 'manifest.json']
    outs = [ctx.dirs['final'] / 'assignments.parquet']
    return ins, outs

def io_genotype_per_chunk(ctx, part):
    from pathlib import Path as _P
    scores = ctx.dirs['chunks'] / f"{_P(part['id']).name}.scores.parquet"
    out = ctx.dirs['chunks'] / f"{_P(part['id']).name}.genotypes.parquet"
    return [scores], [out]

def io_genotype_merge(ctx, part=None):
    ins = [ctx.dirs['chunks'] / 'manifest.json']
    outs = [ctx.dirs['final'] / 'genotypes.parquet']
    return ins, outs

def io_bam_clean(ctx, part=None):
    ins = [ctx.dirs['final'] / 'genotypes.parquet']
    outs = [ctx.dirs['final'] / 'bam_cleaned.DONE']
    return ins, outs

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

def run_assign_score(ctx, part):
    _, outs = io_assign_score(ctx, part)
    out_parquet = outs[0]
    # call your scorer; it writes the parquet out automatically
    score_chunk(
        workdir=ctx.dirs["root"].parent,
        sample=ctx.cfg["sample"],
        chunk_file=Path(part["path"]),
        ecdf_model=_exp_dir(ctx) / "global_ecdf.npz",
        out_raw_dir=None,
        out_filtered_dir=None,
        mapq_min=int(ctx.params["assign"].get("mapq_min", 20)),
        xa_max=int(ctx.params["assign"].get("xa_max", 2)),
        chunksize=int(ctx.params["assign"].get("chunksize", 1_000_000)),
        alpha=float(ctx.params["assign"].get("alpha", 0.05)),
    )
    # sanity-check: fail fast if nothing was written
    if not out_parquet.exists() or out_parquet.stat().st_size == 0:
        raise RuntimeError(f"score_chunk produced no data: {out_parquet}")

def run_assign_merge(ctx, part=None):
    _, outs = io_assign_merge(ctx)
    _noop_runner(outs)

def run_genotype_per_chunk(ctx, part):
    _, outs = io_genotype_per_chunk(ctx, part)
    _noop_runner(outs)

def run_genotype_merge(ctx, part=None):
    _, outs = io_genotype_merge(ctx)
    _noop_runner(outs)

def run_bam_clean(ctx, part=None):
    _, outs = io_bam_clean(ctx)
    _noop_runner(outs)

def build_steps() -> Dict[str, Step]:
    return {
        '00_extract': Step(
            name='00_extract',
            requires=[],
            is_partitioned=False,
            inputs_fn=lambda ctx, _: io_extract(ctx)[0],
            outputs_fn=lambda ctx, _: io_extract(ctx)[1],
            runner_fn=run_extract,
            bump_generation=True,
        ),
        '05_filter': Step(
            name='05_filter',
            requires=['00_extract'],
            is_partitioned=False,
            inputs_fn=lambda ctx, _: io_filter(ctx)[0],
            outputs_fn=lambda ctx, _: io_filter(ctx)[1],
            runner_fn=run_filter,
            bump_generation=True,
        ),
        '10_chunk': Step(
            name='10_chunk',
            requires=['05_filter'],
            is_partitioned=False,
            inputs_fn=lambda ctx, _: io_chunk(ctx)[0],
            outputs_fn=lambda ctx, _: io_chunk(ctx)[1],
            runner_fn=run_chunk,
            bump_generation=True,
        ),
        '20_assign.10_edges': Step(
            name='20_assign.10_edges',
            requires=['10_chunk'],
            is_partitioned=False,
            inputs_fn=lambda ctx, _: io_assign_edges(ctx)[0],
            outputs_fn=lambda ctx, _: io_assign_edges(ctx)[1],
            runner_fn=run_assign_edges,
        ),
        '20_assign.20_ecdfs': Step(
            name='20_assign.20_ecdfs',
            requires=['20_assign.10_edges'],
            is_partitioned=False,
            inputs_fn=lambda ctx, _: io_assign_ecdf(ctx)[0],
            outputs_fn=lambda ctx, _: io_assign_ecdf(ctx)[1],
            runner_fn=run_assign_ecdf,
        ),
        '20_assign.30_score': Step(
            name='20_assign.30_score',
            requires=['20_assign.20_ecdfs'],
            is_partitioned=True,
            inputs_fn=lambda ctx, part: io_assign_score(ctx, part)[0],
            outputs_fn=lambda ctx, part: io_assign_score(ctx, part)[1],
            runner_fn=run_assign_score,
        ),
        '20_assign.40_merge': Step(
            name='20_assign.40_merge',
            requires=['20_assign.30_score'],
            is_partitioned=False,
            inputs_fn=lambda ctx, _: io_assign_merge(ctx)[0],
            outputs_fn=lambda ctx, _: io_assign_merge(ctx)[1],
            runner_fn=run_assign_merge,
        ),
        '30_genotype.10_per_chunk': Step(
            name='30_genotype.10_per_chunk',
            requires=['20_assign.40_merge'],
            is_partitioned=True,
            inputs_fn=lambda ctx, part: io_genotype_per_chunk(ctx, part)[0],
            outputs_fn=lambda ctx, part: io_genotype_per_chunk(ctx, part)[1],
            runner_fn=run_genotype_per_chunk,
        ),
        '30_genotype.20_merge': Step(
            name='30_genotype.20_merge',
            requires=['30_genotype.10_per_chunk'],
            is_partitioned=False,
            inputs_fn=lambda ctx, _: io_genotype_merge(ctx)[0],
            outputs_fn=lambda ctx, _: io_genotype_merge(ctx)[1],
            runner_fn=run_genotype_merge,
        ),
        '35_bamclean': Step(
            name='35_bamclean',
            requires=['30_genotype.20_merge'],
            is_partitioned=False,
            inputs_fn=lambda ctx, _: io_bam_clean(ctx)[0],
            outputs_fn=lambda ctx, _: io_bam_clean(ctx)[1],
            runner_fn=run_bam_clean,
        ),
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
