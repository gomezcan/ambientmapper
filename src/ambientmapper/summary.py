# src/ambientmapper/summary.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Tuple, Dict

def _load_winners(sample_dir: Path, sample: str) -> "pandas.DataFrame":
    import pandas as pd
    final = sample_dir / "final" / f"{sample}_per_read_winner.tsv.gz"
    if final.exists():
        return pd.read_csv(final, sep="\t", low_memory=False)
    # fallback to chunk files
    chunks = sorted((sample_dir / "cell_map_ref_chunks").glob(f"{sample}_cell_genotype_reads_chunk_*.txt"))
    dfs = [pd.read_csv(p, sep="\t") for p in chunks if p.exists() and p.stat().st_size > 0]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

def _load_pool_design(pool_tsv: Optional[Path], sample: str, default_genomes: list[str]) -> Tuple[list[str], list[str]]:
    """
    Returns (inpool_genomes, pools_for_sample).
    If pool_tsv is None, inpool = default_genomes (uppercased) and pools = [sample].
    Expect TSV with at least columns: Genome, Pool; optional: Plate
    """
    import pandas as pd
    if pool_tsv is None:
        # whole-sample = one pool; in-pool = all genomes listed for this sample
        return [g.upper() for g in default_genomes], [sample]

    pdf = pd.read_csv(pool_tsv, sep="\t", header=0)
    cols = {c.lower(): c for c in pdf.columns}
    genome_col = cols.get("genome")
    pool_col   = cols.get("pool")
    if not (genome_col and pool_col):
        raise ValueError("Pool design TSV must include columns: Genome, Pool (Plate optional).")

    # 🔒 enforce: this run's sample IS the pool id
    df = pdf[pdf[pool_col].astype(str).str.upper() == sample.upper()].copy()
    if df.empty:
        raise ValueError(
            f"No rows in pool design where Pool == '{sample}'. "
            "Remember: sample name must equal Pool ID for this run."
        )

    inpool = sorted(df[genome_col].astype(str).str.upper().unique())
    pools  = [sample]  # single active pool for this run
    return inpool, pools

def summarize_and_mark(
    workdir: Path,
    sample: str,
    inpool_genomes: list[str],
    xa_max: int = 2,
    out_dirname: str = "Plots",
    contam_thresh_for_plots: float = 0.20,
    pools_for_sample: list[str] | None = None,  # None => no-pool mode
) -> Dict[str, object]:
    """
    Build summaries, plots, and read-discard list. Returns paths & counts.
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.backends.backend_pdf as mpdf
    from scipy import stats

    sdir = workdir / sample
    winners = _load_winners(sdir, sample)
    if winners.empty:
        raise FileNotFoundError("No winner tables found. Run `ambientmapper merge` (or `run`) first.")

    # Normalize columns
    colmap = {c.lower(): c for c in winners.columns}
    need = ["read", "bc", "genome", "as"]
    missing = [k for k in need if k not in colmap]
    if missing:
        raise ValueError(f"Winners table missing required columns: {missing}. Found: {list(winners.columns)}")

    # Optional XA filter if available
    if xa_max >= 0 and "xacount" in colmap:
        winners = winners[winners[colmap["xacount"]].fillna(0) <= xa_max].copy()

    winners["Pool"]   = sample  # single-pool default
    winners["Genome"] = winners[colmap["genome"]].astype(str).str.upper()
    winners["BC"]     = winners[colmap["bc"]].astype(str)
    winners["AS"]     = pd.to_numeric(winners[colmap["as"]], errors="coerce")
    winners["Read"]   = winners[colmap["read"]].astype(str)

    # -------- per-BC genotype scoring (AS_mean) --------
    agg = (winners
           .groupby(["BC","Genome","Pool"], as_index=False)
           .agg(AS_Total=("AS","sum"), n_Read=("Read","count")))
    agg["AS_mean"] = agg["AS_Total"] / agg["n_Read"]
    total_n = agg.groupby("BC", as_index=False)["n_Read"].sum().rename(columns={"n_Read":"Total_n"})
    agg = agg.merge(total_n, on="BC", how="left")

    # Best assignment per BC (break ties by n_Read desc, then Genome lex)
    genotype_assignment = (agg.sort_values(["BC","AS_mean","n_Read","Genome"],
                                           ascending=[True, False, False, True])
                             .groupby("BC", as_index=False).head(1)
                             .rename(columns={"Genome":"AssignedGenome"}))

    # Contaminated flag relative to in-pool genomes (if pools are meaningful)
    genotype_assignment["Contaminated"] = np.where(
        genotype_assignment["AssignedGenome"].isin([g.upper() for g in inpool_genomes]),
        1, 0
    )

    # -------- read-level mismatch & BC contamination rate --------
    win_g = winners[["BC","Read","Genome"]]
    assigned = genotype_assignment[["BC","AssignedGenome","Total_n"]]
    merged = win_g.merge(assigned, on="BC", how="left")
    merged["mismatch"] = (merged["Genome"] != merged["AssignedGenome"]).astype(int)
    contam = (merged.groupby("BC", as_index=False)
                     .agg(Total_reads=("Read","count"),
                          Mismatch_reads=("mismatch","sum")))
    contam["Contamination_Rate"] = contam["Mismatch_reads"] / contam["Total_reads"]

    # Attach contamination to per-BC AS table
    agg_bc = agg.merge(contam[["BC","Contamination_Rate","Total_reads"]], on="BC", how="left")

    # Pools vs no-pool grouping for plots
    has_pools = bool(pools_for_sample and len(pools_for_sample) > 0)
    if has_pools:
        agg_bc = agg_bc.merge(genotype_assignment[["BC","Contaminated","Pool"]], on="BC", how="left")
        group_mask_in  = agg_bc["Contaminated"].fillna(0).astype(int) == 1
        group_mask_out = ~group_mask_in
        label_in, label_out = "InPool", "Not_InPool"
    else:
        agg_bc["Group"] = (agg_bc["Contamination_Rate"].fillna(0.0) <= float(contam_thresh_for_plots))\
                            .map({True: "LowContam", False: "HighContam"})
        group_mask_in  = agg_bc["Group"] == "LowContam"
        group_mask_out = agg_bc["Group"] == "HighContam"
        label_in, label_out = "LowContam", "HighContam"

    # ---------- ECDF helpers ----------
    def _ecdf(x):
        x = np.sort(np.asarray(x))
        y = np.arange(1, len(x)+1) / len(x) if len(x) else np.array([])
        return x, y

    def _safe_ks(a, b):
        a = np.asarray(a); b = np.asarray(b)
        if a.size == 0 or b.size == 0:
            class R: pass
            r = R(); r.statistic = float("nan"); r.pvalue = float("nan")
            return r
        return stats.ks_2samp(a, b, alternative="two-sided")

    in_as  = agg_bc.loc[group_mask_in,  "AS_mean"].dropna().values
    out_as = agg_bc.loc[group_mask_out, "AS_mean"].dropna().values
    in_n   = agg_bc.loc[group_mask_in,  "Total_n"].dropna().values
    out_n  = agg_bc.loc[group_mask_out, "Total_n"].dropna().values

    ks_as = _safe_ks(in_as, out_as)
    ks_n  = _safe_ks(in_n,  out_n)

    x1,y1   = _ecdf(in_as);   x2,y2   = _ecdf(out_as)
    xn1,yn1 = _ecdf(in_n);    xn2,yn2 = _ecdf(out_n)

    # ---------- PDF ----------
    out_dir = (workdir / sample / out_dirname)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{sample}_contamination_correction_summary.pdf"
    pdf = mpdf.PdfPages(str(pdf_path))

    # Plot 1: ECDF AS_mean
    fig1 = plt.figure(figsize=(4,3)); ax = fig1.gca()
    ax.plot(x1, y1, drawstyle="steps-post", label=label_in)
    ax.plot(x2, y2, drawstyle="steps-post", label=label_out)
    ax.set_title("Cum. Dist.\nAS mean"); ax.set_xlabel("AS mean"); ax.set_ylabel("Cumulative probability")
    ax.legend(loc="lower right", fontsize=8)
    ax.text(0.98, 0.1, f"KS D={ks_as.statistic:.3f}\np={ks_as.pvalue:.2e}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    pdf.savefig(fig1); plt.close(fig1)

    # Plot 2: ECDF Total reads per BC
    fig2 = plt.figure(figsize=(4,3)); ax = fig2.gca()
    ax.plot(xn1, yn1, drawstyle="steps-post", label=label_in)
    ax.plot(xn2, yn2, drawstyle="steps-post", label=label_out)
    ax.set_title("Cum. Dist.\nTotal reads per BC"); ax.set_xlabel("Reads"); ax.set_ylabel("Cumulative probability")
    ax.legend(loc="lower right", fontsize=8)
    ax.text(0.98, 0.1, f"KS D={ks_n.statistic:.3f}\np={ks_n.pvalue:.2e}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    pdf.savefig(fig2); plt.close(fig2)

    # Plot 3: Hexbin contamination vs reads (all BCs)
    fig3 = plt.figure(figsize=(6,3)); ax = fig3.gca()
    if not contam.empty:
        hb = ax.hexbin(contam["Total_reads"], contam["Contamination_Rate"], gridsize=20, mincnt=1)
        ax.set_xlabel("Reads per BC"); ax.set_ylabel("Contamination rate")
        ax.set_title("BC contamination vs reads")
        fig3.colorbar(hb, ax=ax, label="Density")
    else:
        ax.text(0.5,0.5,"No BCs", ha="center")
    pdf.savefig(fig3); plt.close(fig3)

    # Plot 4: heatmap (if multiple pools) else bar chart per AssignedGenome
    fig4 = plt.figure(figsize=(5,4)); ax = fig4.gca()
    if has_pools and len(set(pools_for_sample)) > 1:
        # Build simple heatmap using counts per (AssignedGenome, Pool)
        heat = (genotype_assignment.groupby(["AssignedGenome","Pool"])
                                   .size().reset_index(name="n"))
        genomes = sorted(heat["AssignedGenome"].unique())
        pools   = sorted(heat["Pool"].unique())
        mat = np.zeros((len(genomes), len(pools)), dtype=int)
        g2i = {g:i for i,g in enumerate(genomes)}
        p2j = {p:j for j,p in enumerate(pools)}
        for _, row in heat.iterrows():
            mat[g2i[row["AssignedGenome"]], p2j[row["Pool"]]] = int(row["n"])
        im = ax.imshow(mat, aspect="auto")
        ax.set_yticks(range(len(genomes))); ax.set_yticklabels(genomes, fontsize=7)
        ax.set_xticks(range(len(pools)));  ax.set_xticklabels(pools, fontsize=7, rotation=90)
        ax.set_title("Barcode (BC) per AssignedGenome × Pool")
        for i,g in enumerate(genomes):
            for j,p in enumerate(pools):
                val = mat[i,j]
                ax.text(j, i, str(val), ha="center", va="center", fontsize=6,
                        color=("white" if val else "black"))
        fig4.colorbar(im, ax=ax, label="BCs")
    else:
        counts = (genotype_assignment.groupby("AssignedGenome")
                  .size().reset_index(name="n").sort_values("n", ascending=False))
        ax.bar(counts["AssignedGenome"], counts["n"])
        ax.set_title("Barcodes per AssignedGenome")
        ax.set_ylabel("BC count")
        ax.tick_params(axis='x', labelrotation=90, labelsize=7)
    pdf.savefig(fig4); plt.close(fig4)

    pdf.close()

    # -------- outputs --------
    # HQ_BC = BCs assigned to an in-pool genome
    hq = genotype_assignment[genotype_assignment["Contaminated"]==1].copy()
    hq_path = sdir / f"{sample}_BCs_PASS_by_mapping.csv"
    hq[["BC","AssignedGenome","Total_n","Pool"]].to_csv(hq_path, index=False)

    # Reads to discard: winner genome != assigned BC genome
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
        "n_hq_bc": int(len(hq)),
        "n_discard_reads": int(len(bad)),
    }

def summarize_cli(config_json: Path, pool_design: Optional[Path] = None, xa_max: int = 2) -> Dict[str, object]:
    import json
    cfg = json.loads(Path(config_json).read_text())
    workdir = Path(cfg["workdir"]).expanduser().resolve()
    sample = cfg["sample"]
    inpool, pools = _load_pool_design(pool_design, sample, default_genomes=list(cfg["genomes"].keys()))
    return summarize_and_mark(
        workdir, sample,
        inpool_genomes=inpool,
        xa_max=xa_max,
        pools_for_sample=pools if len(pools) > 0 else None,
    )
