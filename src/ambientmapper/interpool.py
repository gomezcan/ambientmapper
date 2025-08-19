# src/ambientmapper/interpool.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple

def _parse_samples_tsv(tsv: Path) -> Dict[str, Tuple[Path, List[str]]]:
    """
    Parse a samples.tsv with columns: sample, genome, bam, workdir
    Returns: { sample: (workdir, [genomes...]) }
    """
    import pandas as pd
    df = pd.read_csv(tsv, sep="\t")
    need = {"sample","genome","bam","workdir"}
    if need - set(df.columns):
        raise ValueError(f"TSV must include columns: {sorted(need)}")
    df["sample"] = df["sample"].astype(str)
    df["workdir"] = df["workdir"].apply(lambda p: str(Path(p).expanduser().resolve()))
    out: Dict[str, Tuple[Path, List[str]]] = {}
    for s, gdf in df.groupby("sample"):
        workdirs = set(gdf["workdir"].unique())
        if len(workdirs) != 1:
            raise ValueError(f"Sample '{s}' has multiple workdirs in TSV.")
        w = Path(next(iter(workdirs)))
        genomes = sorted(set(gdf["genome"].astype(str).str.upper()))
        out[s] = (w, genomes)
    return out

def _load_winners(workdir: Path, sample: str):
    import pandas as pd
    p = workdir / sample / "final" / f"{sample}_per_read_winner.tsv.gz"
    if not p.exists():
        return None
    return pd.read_csv(p, sep="\t", low_memory=False)

def interpool_summary(configs_tsv: Path, outdir: Path | None = None) -> Dict[str, str]:
    """
    Build cross-pool summaries from multiple per-pool runs.
    Writes:
      - interpool_bc_counts.tsv         (BCs per AssignedGenome x sample)
      - interpool_read_composition.tsv  (winner-read fractions per Genome x sample)
      - interpool_summary.pdf           (two heatmaps)
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.backends.backend_pdf as mpdf

    pools = _parse_samples_tsv(configs_tsv)
    # choose default outdir if not provided
    if outdir is None:
        any_wd = next(iter(pools.values()))[0]
        outdir = any_wd / "interpool"
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)

    # collect per-pool artifacts
    bc_counts_rows = []
    read_comp_rows = []

    for sample, (wd, inpool_genomes) in pools.items():
        winners = _load_winners(wd, sample)
        if winners is None or winners.empty:
            continue
        cols = {c.lower(): c for c in winners.columns}
        need = ["bc","genome","as","read"]
        if any(k not in cols for k in need):
            # skip malformed
            continue

        winners["Genome"] = winners[cols["genome"]].astype(str).str.upper()
        winners["BC"] = winners[cols["bc"]].astype(str)
        winners["AS"] = pd.to_numeric(winners[cols["as"]], errors="coerce")
        winners["Read"] = winners[cols["read"]].astype(str)

        # per-BC assignment by AS_mean (like per-pool summary)
        agg = (winners.groupby(["BC","Genome"], as_index=False)
                        .agg(AS_Total=("AS","sum"), n_Read=("Read","count")))
        agg["AS_mean"] = agg["AS_Total"] / agg["n_Read"]
        assign = (agg.sort_values(["BC","AS_mean","n_Read","Genome"],
                                  ascending=[True,False,False,True])
                     .groupby("BC", as_index=False).head(1)
                     .rename(columns={"Genome":"AssignedGenome"}))

        # BC counts per assigned genome (per sample)
        bc_counts = (assign.groupby("AssignedGenome").size()
                          .reset_index(name="n").assign(sample=sample))
        bc_counts_rows.append(bc_counts)

        # read composition: winner-read counts per genome (per sample)
        rc = (winners.groupby("Genome").size()
                      .reset_index(name="n_reads").assign(sample=sample))
        tot = rc["n_reads"].sum()
        if tot > 0:
            rc["frac_reads"] = rc["n_reads"] / tot
        read_comp_rows.append(rc)

    if not bc_counts_rows or not read_comp_rows:
        raise FileNotFoundError("No usable winners found for any sample in TSV.")

    bc_counts_df = pd.concat(bc_counts_rows, ignore_index=True)
    read_comp_df = pd.concat(read_comp_rows, ignore_index=True)

    # write TSVs
    bc_path = outdir / "interpool_bc_counts.tsv"
    rc_path = outdir / "interpool_read_composition.tsv"
    bc_counts_df.to_csv(bc_path, sep="\t", index=False)
    read_comp_df.to_csv(rc_path, sep="\t", index=False)

    # build heatmaps
    pdf_path = outdir / "interpool_summary.pdf"
    pdf = mpdf.PdfPages(str(pdf_path))

    # Heatmap 1: BC counts per assigned genome x sample
    genomes = sorted(bc_counts_df["AssignedGenome"].unique())
    samples = sorted(bc_counts_df["sample"].unique())
    mat = np.zeros((len(genomes), len(samples)), dtype=float)
    g2i = {g:i for i,g in enumerate(genomes)}
    s2j = {s:j for j,s in enumerate(samples)}
    for _, row in bc_counts_df.iterrows():
        mat[g2i[row["AssignedGenome"]], s2j[row["sample"]]] = row["n"]

    fig1 = plt.figure(figsize=(max(5, len(samples)*0.6), max(4, len(genomes)*0.3)))
    ax1 = fig1.gca()
    im = ax1.imshow(mat, aspect="auto")
    ax1.set_yticks(range(len(genomes))); ax1.set_yticklabels(genomes, fontsize=7)
    ax1.set_xticks(range(len(samples))); ax1.set_xticklabels(samples, fontsize=7, rotation=90)
    ax1.set_title("BCs per AssignedGenome × Sample")
    for i,g in enumerate(genomes):
        for j,s in enumerate(samples):
            v = int(mat[i,j])
            ax1.text(j, i, str(v), ha="center", va="center", fontsize=6, color=("white" if v else "black"))
    fig1.colorbar(im, ax=ax1, label="BCs")
    pdf.savefig(fig1); plt.close(fig1)

    # Heatmap 2: winner-read composition per genome × sample (fractions)
    genomes2 = sorted(read_comp_df["Genome"].unique())
    samples2 = sorted(read_comp_df["sample"].unique())
    mat2 = np.zeros((len(genomes2), len(samples2)), dtype=float)
    g2i2 = {g:i for i,g in enumerate(genomes2)}
    s2j2 = {s:j for j,s in enumerate(samples2)}
    for _, row in read_comp_df.iterrows():
        mat2[g2i2[row["Genome"]], s2j2[row["sample"]]] = float(row["frac_reads"])

    fig2 = plt.figure(figsize=(max(5, len(samples2)*0.6), max(4, len(genomes2)*0.3)))
    ax2 = fig2.gca()
    im2 = ax2.imshow(mat2, aspect="auto")
    ax2.set_yticks(range(len(genomes2))); ax2.set_yticklabels(genomes2, fontsize=7)
    ax2.set_xticks(range(len(samples2))); ax2.set_xticklabels(samples2, fontsize=7, rotation=90)
    ax2.set_title("Winner-read composition (fraction) per Sample")
    fig2.colorbar(im2, ax=ax2, label="Fraction of reads")
    pdf.savefig(fig2); plt.close(fig2)

    pdf.close()

    return {"bc_counts": str(bc_path), "read_comp": str(rc_path), "pdf": str(pdf_path)}
