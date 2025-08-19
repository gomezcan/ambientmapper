from __future__ import annotations
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import typer, glob, pandas as pd, numpy as np
import pysam, csv, json

app = typer.Typer(help="ambientmapper: local-first ambient cleaning pipeline")

# ----------------
# Config helpers
# ----------------
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
def run(config: Path = typer.Option(..., "--config", "-c"),
        threads: int = typer.Option(8, "--threads", "-t", min=1)):
    extract.callback(config=config, threads=threads)
    filter.callback(config=config, threads=threads)
    chunks.callback(config=config)
    assign.callback(config=config, threads=threads)
    merge.callback(config=config)
    typer.echo("[run] complete")
