from __future__ import annotations
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import typer, glob, pandas as pd, numpy as np
import pysam, csv, json

app = typer.Typer(help="ambientmapper: local-first ambient cleaning pipeline")

# ----------------
# Config helpers
# ----------------
# --- add near the top of cli.py ---
from __future__ import annotations
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import typer, json, csv, re
from typing import Dict, List

# Import your step functions (adapt paths to match your repo)
from .extract import bam_to_qc
from .filtering import filter_qc_file
from .chunks import make_barcode_chunks
from .assign import assign_winners_for_chunk
from .merge import merge_chunk_outputs

app = typer.Typer(help="ambientmapper: local-first ambient cleaning pipeline")

def _clamp(n: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, n))

def _parse_csv_list(s: str | None) -> List[str]:
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

def _infer_genome_name(p: Path) -> str:
    # e.g., B73_scifiATAC...bam -> "B73"
    return re.split(r"[_.]", p.stem, maxsplit=1)[0]

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

def _run_pipeline(cfg: Dict[str, object], threads: int) -> None:
    """extract -> filter -> chunks -> assign -> merge (local; scheduler-agnostic)."""
    d = _cfg_dirs(cfg); _ensure_dirs(d)
    genomes = sorted(cfg["genomes"].items())

    # 10) extract (parallel over genomes)
    with ProcessPoolExecutor(max_workers=_clamp(threads, 1, len(genomes))) as ex:
        futs = [
            ex.submit(bam_to_qc, Path(bam), d["qc"] / f"{g}_QCMapping.txt")
            for g, bam in genomes
        ]
        for f in as_completed(futs): f.result()

    # 20) filter (parallel over genomes)
    minf = int(cfg["min_barcode_freq"])
    with ProcessPoolExecutor(max_workers=_clamp(threads, 1, len(genomes))) as ex:
        futs = []
        for g, _ in genomes:
            ip = d["qc"] / f"{g}_QCMapping.txt"
            op = d["filtered"] / f"filtered_{g}_QCMapping.txt"
            futs.append(ex.submit(filter_qc_file, ip, op, minf))
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
    """
    TSV columns: sample, genome, bam, workdir
    Rows with same 'sample' are grouped into one config.
    """
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

def load_cfg(path: Path):
    d = json.loads(Path(path).read_text())
    req = ["sample","genomes","min_barcode_freq","chunk_size_cells","workdir"]
    missing = [k for k in req if k not in d]
    if missing:
        raise typer.BadParameter(f"Missing keys in config: {missing}")
    return d

def dirs(cfg):
    root = Path(cfg["workdir"]) / cfg["sample"]
    return {
        "root": root,
        "qc": root / "qc",
        "filtered": root / "filtered_QCFiles",
        "chunks": root / "cell_map_ref_chunks",
        "final": root / "final",
    }

def ensure_dirs(d):
    for p in d.values():
        if isinstance(p, Path): p.mkdir(parents=True, exist_ok=True)

def _clamp(n, lo, hi): return max(lo, min(hi, n))

# ----------------
# Step 10: extract (BAM -> QC TSV)
# ----------------
def bam_to_qc(bam_path: Path, out_path: Path):
    with pysam.AlignmentFile(str(bam_path), "rb") as bam, open(out_path, "w", newline="") as out:
        w = csv.writer(out, delimiter="\t")
        for aln in bam.fetch(until_eof=True):
            if aln.is_unmapped or aln.is_secondary or aln.is_supplementary:
                continue
            read = aln.query_name
            mapq = int(aln.mapping_quality)
            ascore = aln.get_tag("AS") if aln.has_tag("AS") else ""
            nm = aln.get_tag("NM") if aln.has_tag("NM") else ""
            bc = ""
            if aln.has_tag("CB"): bc = aln.get_tag("CB")
            elif aln.has_tag("BC"): bc = aln.get_tag("BC")
            xa_count = 0
            if aln.has_tag("XA"):
                xa = aln.get_tag("XA")
                xa_count = xa.count(";") if isinstance(xa, str) else 0
            w.writerow([read, bc, mapq, ascore, nm, xa_count])

@app.command()
def extract(config: Path = typer.Option(..., "--config", "-c", exists=True, readable=True),
            threads: int = typer.Option(4, "--threads", "-t", min=1)):
    cfg = load_cfg(config); d = dirs(cfg); ensure_dirs(d)
    genomes = sorted(cfg["genomes"].items())
    maxw = _clamp(threads, 1, len(genomes))
    with ProcessPoolExecutor(max_workers=maxw) as ex:
        futs = []
        for gname, bam in genomes:
            futs.append(ex.submit(bam_to_qc, Path(bam), d["qc"]/f"{gname}_QCMapping.txt"))
        for f in as_completed(futs): f.result()
    typer.echo("[extract] done")

# ----------------
# Step 20: filter per genome
# ----------------
@app.command()
def filter(config: Path = typer.Option(..., "--config", "-c"),
           threads: int = typer.Option(4, "--threads", "-t", min=1)):
    import pandas as pd
    cfg = load_cfg(config); d = dirs(cfg); ensure_dirs(d)
    genomes = sorted(cfg["genomes"].keys()); minf = int(cfg["min_barcode_freq"])
    def _one(g):
        ip = d["qc"]/f"{g}_QCMapping.txt"
        op = d["filtered"]/f"filtered_{g}_QCMapping.txt"
        df = pd.read_csv(ip, sep="\t", names=["Read","BC","MAPQ","AS","NM","XAcount"], header=None)
        keep = set(df["BC"].value_counts()[lambda s: s>=minf].index)
        df = df[df["BC"].isin(keep)]
        df = (df.groupby(["Read","BC"], as_index=False)
                .agg(MAPQ=("MAPQ","max"), AS=("AS","max"), NM=("NM","min"), XAcount=("XAcount","max")))
        df.to_csv(op, sep="\t", index=False)
    maxw = _clamp(threads, 1, len(genomes))
    with ProcessPoolExecutor(max_workers=maxw) as ex:
        futs=[ex.submit(_one,g) for g in genomes]
        for f in as_completed(futs): f.result()
    typer.echo("[filter] done")

# ----------------
# Step 25: make BC chunks
# ----------------
@app.command()
def chunks(config: Path = typer.Option(..., "--config", "-c")):
    cfg = load_cfg(config); d = dirs(cfg); ensure_dirs(d)
    paths = sorted((d["filtered"]).glob("filtered_*_QCMapping.txt"))
    if not paths: 
        typer.echo("[chunks] no filtered files"); return
    import pandas as pd
    bc=set()
    for p in paths:
        for c in pd.read_csv(p, sep="\t", usecols=["BC"], chunksize=1_000_000):
            bc.update(c["BC"].dropna().astype(str))
    bc = sorted(bc); n = int(cfg["chunk_size_cells"])
    for i in range(0, len(bc), n):
        k = i//n + 1
        (d["chunks"]/f"{cfg['sample']}_cell_map_ref_chunk_{k}.txt").write_text("\n".join(bc[i:i+n]))
    typer.echo(f"[chunks] wrote {((len(bc)-1)//n)+1} chunk files")

# ----------------
# Step 30: assign winners (per chunk, parallel)
# ----------------
def _assign_one_chunk(filtered_dir: Path, chunk_file: Path, out_file: Path):
    import pandas as pd, numpy as np, os
    bc = {ln.strip() for ln in chunk_file.read_text().splitlines() if ln.strip()}
    qcs = sorted(filtered_dir.glob("filtered_*_QCMapping.txt"))
    rows=[]
    for path in qcs:
        genome = path.name.replace("filtered_","").replace("_QCMapping.txt","")
        for c in pd.read_csv(path, sep="\t", chunksize=1_000_000):
            sub = c[c["BC"].isin(bc)]
            if sub.empty: continue
            sub = sub.assign(Genome=genome)
            rows.append(sub[["Read","BC","Genome","AS","MAPQ","NM"]])
    big = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["Read","BC","Genome","AS","MAPQ","NM"])
    if big.empty:
        pd.DataFrame(columns=["Read","BC","Genome","AS","MAPQ","NM","delta_AS","is_tie"]).to_csv(out_file, sep="\t", index=False)
        return
    big = big.sort_values(["Read","AS","MAPQ","NM","Genome"], ascending=[True,False,False,True,True])
    top = big.groupby("Read", as_index=False).head(1).copy()
    nxt = (big.groupby("Read", as_index=False).nth(1)[["Read","AS","MAPQ","NM"]]
              .rename(columns={"AS":"AS2","MAPQ":"MAPQ2","NM":"NM2"}))
    top = top.merge(nxt, on="Read", how="left")
    top["delta_AS"] = top["AS"] - top["AS2"].fillna(-np.inf)
    dup_key = big[["Read","AS","MAPQ","NM"]].duplicated(keep=False)
    tied = set(big.loc[dup_key,"Read"])
    top["is_tie"] = top["Read"].isin(tied)
    top[["Read","BC","Genome","AS","MAPQ","NM","delta_AS","is_tie"]].to_csv(out_file, sep="\t", index=False)

@app.command()
def assign(config: Path = typer.Option(..., "--config", "-c"),
           threads: int = typer.Option(4, "--threads", "-t", min=1)):
    cfg = load_cfg(config); d = dirs(cfg); ensure_dirs(d)
    chunks = sorted(d["chunks"].glob(f"{cfg['sample']}_cell_map_ref_chunk_*.txt"))
    if not chunks: 
        typer.echo("[assign] no chunk files"); return
    maxw = _clamp(threads, 1, len(chunks))
    with ProcessPoolExecutor(max_workers=maxw) as ex:
        futs=[]
        for ch in chunks:
            out = d["chunks"]/ch.name.replace("_cell_map_ref_chunk_","_cell_genotype_reads_chunk_")
            futs.append(ex.submit(_assign_one_chunk, d["filtered"], ch, out))
        for f in as_completed(futs): f.result()
    typer.echo("[assign] done")

# ----------------
# Step 40: merge
# ----------------
@app.command()
def merge(config: Path = typer.Option(..., "--config", "-c")):
    cfg = load_cfg(config); d = dirs(cfg); ensure_dirs(d)
    paths = sorted(d["chunks"].glob(f"{cfg['sample']}_cell_genotype_reads_chunk_*.txt"))
    dfs = [pd.read_csv(p, sep="\t") for p in paths if p.exists() and p.stat().st_size>0]
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(
        columns=["Read","BC","Genome","AS","MAPQ","NM","delta_AS","is_tie"]
    )
    out = d["final"]/f"{cfg['sample']}_per_read_winner.tsv.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False, compression="gzip")
    typer.echo(f"[merge] wrote {out} rows={len(df)}")

# ----------------
# End-to-end
# ----------------
@app.command()
def run(
    # mode A: single JSON (still supported)
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
):
    """
    Run the pipeline.

    Modes:
      (A) --config path.json
      (B) --sample + --genome "A,B,..." + --bam "/p/a.bam,/p/b.bam" + --workdir /out
      (C) --configs samples.tsv   (columns: sample, genome, bam, workdir)
    """
    modes_used = sum([
        1 if config else 0,
        1 if (sample and (genome or bam or workdir)) else 0,
        1 if configs else 0,
    ])
    if modes_used != 1:
        raise typer.BadParameter("Choose exactly one mode: --config OR (--sample/--genome/--bam/--workdir) OR --configs")

    # Mode A: JSON
    if config:
        cfg = json.loads(Path(config).read_text())
        _run_pipeline(cfg, threads)
        typer.echo(f"[run] {cfg['sample']} complete")
        raise typer.Exit()

    # Mode B: inline single
    if sample and genome and bam and workdir:
        cfg = _cfg_from_inline(sample, genome, bam, str(workdir),
                               min_barcode_freq, chunk_size_cells)
        _run_pipeline(cfg, threads)
        typer.echo(f"[run] {cfg['sample']} complete")
        raise typer.Exit()

    # Mode C: TSV batch
    if configs:
        batch = _cfgs_from_tsv(configs, min_barcode_freq, chunk_size_cells)
        if not batch:
            typer.echo("[run] no configs found in TSV"); raise typer.Exit(0)
        for cfg in batch:
            typer.echo(f"[run] starting {cfg['sample']}")
            _run_pipeline(cfg, threads)
            typer.echo(f"[run] {cfg['sample']} complete")
        raise typer.Exit()

            
