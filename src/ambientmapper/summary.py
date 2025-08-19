# src/ambientmapper/summary.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple

def _load_winners(sample_dir: Path, sample: str) -> "pandas.DataFrame":
    import pandas as pd
    final = sample_dir / "final" / f"{sample}_per_read_winner.tsv.gz"
    if final.exists():
        df = pd.read_csv(final, sep="\t", low_memory=False)
        return df
    # fallback to chunk files
    chunks = sorted((sample_dir / "cell_map_ref_chunks").glob(f"{sample}_cell_genotype_reads_chunk_*.txt"))
    dfs = [pd.read_csv(p, sep="\t") for p in chunks if p.exists() and p.stat().st_size > 0]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def _load_pool_design(pool_tsv: Optional[Path], sample: str, default_genomes: list[str]) -> Tuple[list[str], list[str]]:
    """
    Returns (inpool_genomes, pools_for_sample). If pool_tsv is None, inpool=default_genomes and pool=[sample].
    Expect TSV with at least columns: Genome, Pool; optional: Plate
    """
    import pandas as pd
    if pool_tsv is None:
        return [g.upper() for g in default_genomes], [sample]
    pdf = pd.read_csv(pool_tsv, sep="\t", header=0)
    cols = {c.lower(): c for c in pdf.columns}
    # normalize columns
    genome_col = cols.get("genome")
    pool_col = cols.get("pool")
    plate_col = cols.get("plate")  # optional
    if not (genome_col and pool_col):
        raise ValueError("Pool design TSV must have at least columns: Genome, Pool (and optional Plate).")
    df = pdf.copy()
    # If Plate present, filter to Plate==sample. Else assume all rows belong to this sample.
    if plate_col:
        df = df[df[plate_col].astype(str).str.upper() == sample.upper()]
    inpool = sorted(df[genome_col].astype(str).str.upper().unique())
    pools = sorted(df[pool_col].astype(str).unique())
    if not pools:
        pools = [sample]
    return inpool, pools

def summarize_and_mark(
    workdir: Path,
    sample: str,
    inpool_genomes: list[str],
    xa_max: int = 2,
    out_dirname: str = "Plots",
) -> dict:
    """
    Build summaries, plots, and read-discard list.
    Returns dict with output paths.
    """
    import pandas as pd
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt

    sdir = workdir / sample
    winners = _load_winners(sdir, sample)
    if winners.empty:
        raise FileNotFoundError("No winner tables found. Run `ambientmapper merge` (or `run`) first.")

    # normalize column names and required ones
    colmap = {c.lower(): c for c in winners.columns}
    need = ["read","bc","genome","as","mapq"]
    missing = [k for k in need if k not in colmap]
    if missing:
        raise ValueError(f"Winners table is missing required columns: {missing} (found: {list(winners.columns)})")

    # Optional XA filter if present and xa_max>=0
    if "xacount" in colmap and xa_max >= 0:
        winners = winners[winners[colmap["xacount"]].fillna(0) <= xa_max].copy()

    # add Pool (simplify: single pool == sample)
    winners["Pool"] = sample
    winners["Genome"] = winners[colmap["genome"]].astype(str).str.upper()
    winners["BC"] = winners[colmap["bc"]].astype(str)
    winners["AS"] = pd.to_numeric(winners[colmap["as"]], errors="coerce")
    winners["Read"] = winners[colmap["read"]].astype(str)

    # -------- per-BC genotype scoring (AS_mean) --------
    agg = (winners
           .groupby(["BC","Genome","Pool"], as_index=False)
           .agg(AS_Total=("AS","sum"), n_Read=("Read","count")))
    agg["AS_mean"] = agg["AS_Total"] / agg["n_Read"]
    total_n = agg.groupby("BC", as_index=False)["n_Read"].sum().rename(columns={"n_Read":"Total_n"})
    agg = agg.merge(total_n, on="BC", how="left")

    # best assignment per BC
    genotype_assignment = (agg.sort_values(["BC","AS_mean","n_Read"], ascending=[True,False,False])
                             .groupby("BC", as_index=False).head(1)
                             .rename(columns={"Genome":"AssignedGenome"}))

    # InPool check (use provided inpool_genomes)
    genotype_assignment["Contaminated"] = np.where(
        genotype_assignment["AssignedGenome"].isin([g.upper() for g in inpool_genomes]),
        1, 0
    )

    # -------- plots --------
    out_dir = sdir / out_dirname
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{sample}_contamination_correction_summary.pdf"

    # ECDF of AS_mean (InPool vs Not_InPool)
    # attach BC-level contamination to per-BC rows
    agg_bc = agg.merge(genotype_assignment[["BC","Contaminated"]], on="BC", how="left")
    inpool_as = agg_bc.loc[agg_bc["Contaminated"]==1, "AS_mean"].dropna().values
    outpool_as = agg_bc.loc[agg_bc["Contaminated"]==0, "AS_mean"].dropna().values
    ks_as = stats.ks_2samp(inpool_as, outpool_as, alternative="two-sided")

    def _ecdf(x):
        x = np.sort(np.asarray(x))
        y = np.arange(1, len(x)+1) / len(x) if len(x) else np.array([])
        return x, y

    x1,y1 = _ecdf(inpool_as)
    x2,y2 = _ecdf(outpool_as)

    # ECDF of Total reads per BC
    inpool_n = agg_bc.loc[agg_bc["Contaminated"]==1, "Total_n"].dropna().values
    outpool_n = agg_bc.loc[agg_bc["Contaminated"]==0, "Total_n"].dropna().values
    ks_n = stats.ks_2samp(inpool_n, outpool_n, alternative="two-sided")
    xn1,yn1 = _ecdf(inpool_n)
    xn2,yn2 = _ecdf(outpool_n)

    # contamination rate per BC (fraction of reads not matching AssignedGenome)
    win_g = winners[["BC","Read","Genome"]]
    assigned = genotype_assignment[["BC","AssignedGenome","Total_n"]]
    merged = win_g.merge(assigned, on="BC", how="left")
    merged["mismatch"] = (merged["Genome"] != merged["AssignedGenome"]).astype(int)
    contam = (merged.groupby("BC", as_index=False)
                     .agg(Total_reads=("Read","count"),
                          Mismatch_reads=("mismatch","sum")))
    contam["Contamination_Rate"] = contam["Mismatch_reads"] / contam["Total_reads"]
    contam = contam.merge(genotype_assignment[["BC","Contaminated","Pool"]], on="BC", how="left")

    # Heatmap counts of HQ_BC per Pool x Genome
    HQ_BC = genotype_assignment[genotype_assignment["Contaminated"]==1].copy()
    heat = (HQ_BC.groupby(["AssignedGenome","Pool"]).size()
                  .reset_index(name="n"))

    # -------- assemble PDF --------
    import matplotlib.backends.backend_pdf as mpdf
    pdf = mpdf.PdfPages(str(pdf_path))

    # Plot 1: ECDF AS_mean
    fig1 = plt.figure(figsize=(4,3))
    ax = fig1.gca()
    ax.plot(x1, y1, drawstyle="steps-post", label="InPool")
    ax.plot(x2, y2, drawstyle="steps-post", label="Not_InPool")
    ax.set_title("Cum. Dist.\nAS mean")
    ax.set_xlabel("AS mean"); ax.set_ylabel("Cumulative probability")
    ax.legend(loc="lower right", fontsize=8)
    ax.text(0.98, 0.1,
            f"KS D={ks_as.statistic:.3f}\np={ks_as.pvalue:.2e}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    pdf.savefig(fig1); plt.close(fig1)

    # Plot 2: ECDF Total reads per BC (log x in labels is not ECDF-friendly; keep linear ECDF)
    fig2 = plt.figure(figsize=(4,3))
    ax = fig2.gca()
    ax.plot(xn1, yn1, drawstyle="steps-post", label="InPool")
    ax.plot(xn2, yn2, drawstyle="steps-post", label="Not_InPool")
    ax.set_title("Cum. Dist.\nTotal reads per BC")
    ax.set_xlabel("Reads"); ax.set_ylabel("Cumulative probability")
    ax.legend(loc="lower right", fontsize=8)
    ax.text(0.98, 0.1,
            f"KS D={ks_n.statistic:.3f}\np={ks_n.pvalue:.2e}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    pdf.savefig(fig2); plt.close(fig2)

    # Plot 3: Hexbin (only BCs correctly assigned, contamination vs reads)
    sub = contam.merge(genotype_assignment[["BC","Contaminated","Pool"]], on=["BC","Contaminated","Pool"])
    sub = sub[sub["Contaminated"]==1]
    fig3 = plt.figure(figsize=(6,3))
    ax = fig3.gca()
    if not sub.empty:
        hb = ax.hexbin(sub["Total_reads"], sub["Contamination_Rate"], gridsize=20, mincnt=1)
        ax.set_xlabel("Reads per BC"); ax.set_ylabel("Contamination rate")
        ax.set_title("Contamination within BCs correctly assigned")
        fig3.colorbar(hb, ax=ax, label="Density")
    else:
        ax.text(0.5,0.5,"No InPool BCs", ha="center")
    pdf.savefig(fig3); plt.close(fig3)

    # Plot 4: Heatmap of HQ_BC counts (AssignedGenome x Pool)
    fig4 = plt.figure(figsize=(5,4))
    ax = fig4.gca()
    if not heat.empty:
        # build a matrix
        import numpy as np
        genomes = sorted(heat["AssignedGenome"].unique())
        pools = sorted(heat["Pool"].unique())
        mat = np.zeros((len(genomes), len(pools)), dtype=int)
        g2i = {g:i for i,g in enumerate(genomes)}
        p2j = {p:j for j,p in enumerate(pools)}
        for _,row in heat.iterrows():
            mat[g2i[row["AssignedGenome"]], p2j[row["Pool"]]] = row["n"]
        im = ax.imshow(mat, aspect="auto")
        ax.set_yticks(range(len(genomes))); ax.set_yticklabels(genomes, fontsize=7)
        ax.set_xticks(range(len(pools)));  ax.set_xticklabels(pools, fontsize=7, rotation=90)
        ax.set_title("Barcode (BC) per Pool (Best AS mean)")
        for i,g in enumerate(genomes):
            for j,p in enumerate(pools):
                val = mat[i,j]
                ax.text(j, i, str(val), ha="center", va="center", fontsize=6, color="white" if val else "black")
        fig4.colorbar(im, ax=ax, label="BCs")
    else:
        ax.text(0.5,0.5,"No HQ_BC", ha="center")
    pdf.savefig(fig4); plt.close(fig4)

    pdf.close()

    # -------- outputs --------
    # HQ_BC table
    hq_path = sdir / f"{sample}_BCs_PASS_by_mapping.csv"
    HQ_BC[["BC","AssignedGenome","Total_n","Pool"]].to_csv(hq_path, index=False)

    # Reads to discard (winner genome != assigned BC genome)
    to_filter = winners[["Read","BC","Genome"]].merge(
        genotype_assignment[["BC","AssignedGenome","Pool"]],
        on="BC", how="left"
    )
    bad = to_filter[to_filter["Genome"] != to_filter["AssignedGenome"]].copy()
    bad.rename(columns={"Genome":"Genome.Mapped","AssignedGenome":"Genome.Assigned"}, inplace=True)
    bad = bad.sort_values(["BC","Read"])
    discard_path = sdir / f"Reads_to_discard_{sample}.csv"
    bad.to_csv(discard_path, index=False)

    return {
        "pdf": str(pdf_path),
        "hq_barcodes": str(hq_path),
        "reads_to_discard": str(discard_path),
        "n_winners": int(len(winners)),
        "n_hq_bc": int(len(HQ_BC)),
        "n_discard_reads": int(len(bad)),
    }

def summarize_cli(
    config_json: Path,
    pool_design: Optional[Path] = None,
    xa_max: int = 2,
) -> dict:
    import json
    cfg = json.loads(Path(config_json).read_text())
    workdir = Path(cfg["workdir"]).expanduser().resolve()
    sample = cfg["sample"]
    # if no pool design, default in-pool genomes are the genomes in config
    inpool, _ = _load_pool_design(pool_design, sample, default_genomes=list(cfg["genomes"].keys()))
    return summarize_and_mark(workdir, sample, inpool_genomes=inpool, xa_max=xa_max)
