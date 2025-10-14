
from __future__ import annotations
import os, json
from pathlib import Path
from typing import Dict, List, Any, Optional
from .dag import Ctx, Step, run_dag
from .sentinels import read_generation

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

def read_manifest(dirs: Dict[str, Path]):
    m = dirs['chunks'] / 'manifest.json'
    if not m.exists():
        return []
    data = json.loads(m.read_text())
    parts = []
    for ch in data.get('chunks', []):
        parts.append({'id': ch['id'], 'path': str((dirs['chunks'] / Path(ch['path']).name).resolve())})
    return parts

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

def io_assign_edges(ctx, part=None):
    ins = [ctx.dirs['chunks'] / 'manifest.json']
    outs = [ctx.dirs['final'] / 'global_edges.npz']
    return ins, outs

def io_assign_ecdf(ctx, part=None):
    ins = [ctx.dirs['final'] / 'global_edges.npz']
    outs = [ctx.dirs['final'] / 'global_ecdf.npz']
    return ins, outs

def io_assign_score(ctx, part):
    from pathlib import Path as _P
    ch_path = _P(part['path'])
    ins = [ctx.dirs['final'] / 'global_edges.npz', ctx.dirs['final'] / 'global_ecdf.npz', ch_path]
    out = ctx.dirs['chunks'] / f"{_P(part['id']).name}.scores.parquet"
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

def run_chunk(ctx, part=None):
    _, outs = io_chunk(ctx)
    outs[0].parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        'version': 1,
        'n_chunks': 3,
        'chunks': [{'id': f'chunk_{i:04d}', 'path': f'chunk_{i:04d}.parquet'} for i in range(1,4)],
        'derived_from': {'filtered_inputs_sha256': 'placeholder', 'upstream_generation': ctx.generation_id},
    }
    outs[0].write_text(json.dumps(manifest, indent=2))

def run_assign_edges(ctx, part=None):
    _, outs = io_assign_edges(ctx)
    _noop_runner(outs)

def run_assign_ecdf(ctx, part=None):
    _, outs = io_assign_ecdf(ctx)
    _noop_runner(outs)

def run_assign_score(ctx, part):
    _, outs = io_assign_score(ctx, part)
    _noop_runner(outs)

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

def run_pipeline(cfg: Dict[str, Any], params: Dict[str, Any], version: str = '0.4.0',
                 resume: bool = True, force: List[str] | None = None, skip_to: str = '', only: List[str] | None = None):
    dirs = _dirs(cfg)
    gen = read_generation(dirs['root'])
    ctx = Ctx(cfg=cfg, params=params, dirs=dirs, version=version, resume=resume,
              force=(force or []), skip_to=skip_to, only=(only or []), generation_id=gen)
    steps = build_steps()
    partitions = read_manifest(dirs)
    if not partitions:
        # fallback: dummy single chunk so pipeline can run
        from pathlib import Path as _P
        dummy = dirs['chunks'] / 'chunk_0001.parquet'
        dummy.parent.mkdir(parents=True, exist_ok=True)
        dummy.touch(exist_ok=True)
        partitions = [{'id': 'chunk_0001', 'path': str(dummy)}]
    result = run_dag(ctx, steps, partitions=partitions)
    return result
