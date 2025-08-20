from __future__ import annotations

from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import json, csv
from typing import Dict, List
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

def _run_pipeline(cfg: Dict[str, object], threads: int) -> None:
    # Lazy import step functions so `ambientmapper --help` stays fast
    from .extract import bam_to_qc
    from .filtering import filter_qc_file
    from .chunks import make_barcode_chunks
    from .assign import assign_winners_for_chunk
    from .merge import merge_chunk_outputs

    d = _cfg_dirs(cfg); _ensure_dirs(d)
    genomes = sorted(cfg["genomes"].items())
    if not genomes:
        typer.echo("[run] no genomes in config; nothing to do")
        return

    # 10) extract (parallel over genomes)
    with ProcessPoolExecutor(max_workers=_clamp(threads, 1, len(genomes))) as ex:
        futs = [ex.submit(bam_to_qc, Path(bam), d["qc"] / f"{g}_QCMapping.txt", cfg["sample"])
                for g, bam in genomes]
        for f in as_completed(futs): f.result()

    # 20) filter (parallel over genomes)
    minf = int(cfg["min_barcode_freq"])
    with ProcessPoolExecutor(max_workers=_clamp(threads, 1, len(genomes))) as ex:
        futs = []
        for g, _ in genomes:
            ip = d["qc"] / f"{g}_QCMapping.txt"
            op = d["filtered"] / f"filtered_{g}_QCMapping.txt"
            futs.append(ex.submit(filter_qc_file, ip, op, minf, cfg["sample"]))
        for f in as_completed(futs): f.result()

    # 25) chunks
    make_barcode_chunks(d["filtered"], d["chunks"], cfg["sample"], int(cfg["chunk_size_cells"]))

    # 30) assign (parallel over chunks)
    chunk_files = sorted(d["chunks"].glob(f"{cfg['sample']}_cell_map_ref_chunk_*.txt"))
    with ProcessPoolExecutor(max_workers=_clamp(threads, 1, len(chunk_files))) as ex:
        futs = []
        for ch in chunk_files:
            out = d["chunks"] / ch.name.replace("_cell_map_ref_chunk_", "_cell_genotype_reads_chunk_")
            futs.append(ex.submit(assign_winners_for_chunk, d["filtered"], ch, out))
        for f in as_completed(futs): f.result()

    # 40) merge
    out = d["final"] / f"{cfg['sample']}_per_read_winner.tsv.gz"
    merge_chunk_outputs(d["chunks"], cfg["sample"], out)

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
def assign(config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
           threads: int = typer.Option(4, "--threads", "-t", min=1)):
    from .assign import assign_winners_for_chunk
    cfg = json.loads(Path(config).read_text()); d = _cfg_dirs(cfg); _ensure_dirs(d)
    chunks = sorted(d["chunks"].glob(f"{cfg['sample']}_cell_map_ref_chunk_*.txt"))
    if not chunks:
        typer.echo("[assign] no chunk files"); return
    with ProcessPoolExecutor(max_workers=_clamp(threads, 1, len(chunks))) as ex:
        futs=[]
        for ch in chunks:
            out = d["chunks"]/ch.name.replace("_cell_map_ref_chunk_","_cell_genotype_reads_chunk_")
            futs.append(ex.submit(assign_winners_for_chunk, d["filtered"], ch, out))
        for f in as_completed(futs): f.result()
    typer.echo("[assign] done")

@app.command()
def merge(config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True)):
    from .merge import merge_chunk_outputs
    cfg = json.loads(Path(config).read_text()); d = _cfg_dirs(cfg); _ensure_dirs(d)
    out = d["final"]/f"{cfg['sample']}_per_read_winner.tsv.gz"
    n = merge_chunk_outputs(d["chunks"], cfg["sample"], out)
    typer.echo(f"[merge] wrote {out} rows={n}")

@app.command()
def summarize(
    config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True, help="Sample JSON"),
    pool_design: Path = typer.Option(None, "--pool-design", exists=True, readable=True,
                                     help="TSV with columns: Genome, Pool [optional: Plate]"),
    xa_max: int = typer.Option(2, "--xa-max", help="Keep winners with XAcount <= xa_max; -1 disables"),
):
    """
    Summarize winner tables, make QC PDF, and produce Reads_to_discard CSV.
    """
    from .summary import summarize_cli
    out = summarize_cli(config_json=config, pool_design=pool_design, xa_max=xa_max)
    typer.echo(f"[summarize] PDF: {out['pdf']}")
    typer.echo(f"[summarize] HQ_BC: {out['hq_barcodes']}")
    typer.echo(f"[summarize] Reads_to_discard: {out['reads_to_discard']}")
    typer.echo(f"[summarize] winners={out['n_winners']} hq_bc={out['n_hq_bc']} discard_reads={out['n_discard_reads']}")

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
    min_barcode_freq: int = typer.Option(10, "--min-barcode-freq"),
    chunk_size_cells: int = typer.Option(5000, "--chunk-size-cells"),
    threads: int = typer.Option(8, "--threads", "-t", min=1),
    # NEW: summary options
    with_summary: bool = typer.Option(False, "--with-summary", 
                                      help="Produce summary PDF + PASS BCs + Reads_to_discard"),
    pool_design: Path = typer.Option(None, "--pool-design", exists=True, readable=True,
                                     help="TSV with columns: Genome, Pool [optional: Plate]"),
    xa_max: int = typer.Option(2, "--xa-max", help="Keep winners with XAcount <= xa_max in summary; set -1 to disable"),
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

        if with_summary:
            from .summary import summarize_and_mark, _load_pool_design as load_pool_design

            # Decide the in-pool genome set:
            # - If a pool design TSV is provided, use rows matching Pool==sample (or all rows if your loader permits).
            # - Otherwise, treat the genomes listed in this sample's config as the in-pool list.
            if pool_design:
                inpool, _ = load_pool_design(
                    pool_design, cfg["sample"], default_genomes=list(cfg["genomes"].keys())
                )
            else:
                inpool = [g.upper() for g in cfg["genomes"].keys()]

            # Per-pool summary compares Low/High contamination only; no pools_for_sample arg needed.
            out = summarize_and_mark(
                workdir=Path(cfg["workdir"]),
                sample=cfg["sample"],
                inpool_genomes=inpool,
                xa_max=xa_max,
            )

            typer.echo(f"[summary] PDF: {out['pdf']}")
            typer.echo(f"[summary] HQ_BC: {out['hq_barcodes']}")
            typer.echo(f"[summary] Reads_to_discard: {out['reads_to_discard']}")
            typer.echo(
                f"[summary] winners={out['n_winners']} hq_bc={out['n_hq_bc']} "
                f"discard_reads={out['n_discard_reads']}"
            )


    if config:
        cfg = json.loads(Path(config).read_text())
        _do_one(cfg)
        raise typer.Exit()

    if inline_ready:
        cfg = _cfg_from_inline(sample, genome, bam, str(workdir), min_barcode_freq, chunk_size_cells)
        _do_one(cfg)
        raise typer.Exit()

    if configs:
        batch = _cfgs_from_tsv(configs, min_barcode_freq, chunk_size_cells)
        if not batch:
            typer.echo("[run] no configs found in TSV"); raise typer.Exit(0)
        for cfg in batch:
            typer.echo(f"[run] starting {cfg['sample']}")
            _do_one(cfg)
        raise typer.Exit()
