from __future__ import annotations
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import typer, math

from .config import SampleConfig
from .extract import bam_to_qc
from .filtering import filter_qc_file
from .chunks import make_barcode_chunks
from .assign import assign_winners_for_chunk
from .merge import merge_chunk_outputs

app = typer.Typer(help="ambientmap: local-first ambient cleaning pipeline")

def _clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))

@app.command()
def extract(config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True, help="Sample JSON"),
            threads: int = typer.Option(4, "--threads", "-t", min=1)):
    """Step 10: BAM -> QC (parallel across genomes)."""
    cfg = SampleConfig.load(config)
    cfg.ensure_dirs()
    jobs = []
    with ProcessPoolExecutor(max_workers=_clamp(threads, 1, len(cfg.genomes))) as ex:
        for g, bam in sorted(cfg.genomes.items()):
            out = cfg.dir_qc() / f"{g}_QCMapping.txt"
            jobs.append(ex.submit(bam_to_qc, bam, out))
        for fut in as_completed(jobs):
            fut.result()
    typer.echo("[extract] done")

@app.command()
def filter(config: Path = typer.Option(..., "--config", "-c"),
           threads: int = typer.Option(4, "--threads", "-t", min=1)):
    """Step 20: per-genome filter/collapse (parallel across genomes)."""
    cfg = SampleConfig.load(config)
    cfg.ensure_dirs()
    jobs = []
    with ProcessPoolExecutor(max_workers=_clamp(threads, 1, len(cfg.genomes))) as ex:
        for g in sorted(cfg.genomes):
            ip = cfg.dir_qc() / f"{g}_QCMapping.txt"
            op = cfg.dir_filtered() / f"filtered_{g}_QCMapping.txt"
            jobs.append(ex.submit(filter_qc_file, ip, op, cfg.min_barcode_freq))
        for fut in as_completed(jobs):
            fut.result()
    typer.echo("[filter] done")

@app.command()
def chunks(config: Path = typer.Option(..., "--config", "-c")):
    """Step 25: build barcode chunk lists."""
    cfg = SampleConfig.load(config)
    cfg.ensure_dirs()
    n = make_barcode_chunks(cfg.dir_filtered(), cfg.dir_chunks(), cfg.sample, cfg.chunk_size_cells)
    typer.echo(f"[chunks] wrote {n} chunk files")

@app.command()
def assign(config: Path = typer.Option(..., "--config", "-c"),
           threads: int = typer.Option(4, "--threads", "-t", min=1)):
    """Step 30: winner selection per chunk (parallel across chunks)."""
    cfg = SampleConfig.load(config)
    cfg.ensure_dirs()
    chunks = sorted((cfg.dir_chunks()).glob(f"{cfg.sample}_cell_map_ref_chunk_*.txt"))
    if not chunks:
        typer.echo("[assign] no chunk files found")
        raise typer.Exit(0)
    max_workers = _clamp(threads, 1, len(chunks))
    jobs = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for ch in chunks:
            out = cfg.dir_chunks() / ch.name.replace("_cell_map_ref_chunk_", "_cell_genotype_reads_chunk_")
            jobs.append(ex.submit(assign_winners_for_chunk, cfg.dir_filtered(), ch, out))
        for fut in as_completed(jobs):
            fut.result()
    typer.echo("[assign] done")

@app.command()
def merge(config: Path = typer.Option(..., "--config", "-c")):
    """Step 40: merge chunk outputs into final gz."""
    cfg = SampleConfig.load(config)
    cfg.ensure_dirs()
    out = cfg.dir_final() / f"{cfg.sample}_per_read_winner.tsv.gz"
    n = merge_chunk_outputs(cfg.dir_chunks(), cfg.sample, out)
    typer.echo(f"[merge] wrote {out} rows={n}")

@app.command()
def run(config: Path = typer.Option(..., "--config", "-c"),
        threads: int = typer.Option(8, "--threads", "-t", min=1)):
    """
    End-to-end: extract -> filter -> chunks -> assign -> merge.

    Concurrency policy:
      - per-genome steps: up to min(threads, #genomes)
      - per-chunk steps:  up to min(threads, #chunks)
    """
    extract.callback(config=config, threads=threads)   # 10
    filter.callback(config=config, threads=threads)    # 20
    chunks.callback(config=config)                     # 25
    assign.callback(config=config, threads=threads)    # 30
    merge.callback(config=config)                      # 40
    typer.echo("[run] complete")
