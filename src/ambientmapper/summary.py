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

    
# src/ambientmapper/summary.py (patches)
def summarize_and_mark(
    workdir: Path,
    sample: str,
    inpool_genomes: list[str],
    xa_max: int = 2,
    out_dirname: str = "Plots",
    contam_thresh_for_plots: float = 0.20,   # NEW
    
    pools_for_sample: list[str] | None = None,  # pass from loader; None => no-pool mode
    ) -> dict:
    
    # After building `genotype_assignment` and `agg` (per-BC AS_mean table), build read-level mismatch
    win_g = winners[["BC","Read","Genome"]]
    assigned = genotype_assignment[["BC","AssignedGenome","Total_n"]]

    out_dir = (workdir / sample / out_dirname) # NEW
    out_dir.mkdir(parents=True, exist_ok=True) # NEW
    pdf_path = out_dir / f"{sample}_contamination_correction_summary.pdf" # NEW

    # mark "Contaminated" (only meaningful with pools, but harmless otherwise)
    import numpy as np
    genotype_assignment["Contaminated"] = np.where(
        genotype_assignment["AssignedGenome"].astype(str).str.upper()
        .isin([g.upper() for g in inpool_genomes]),
        1, 0
    )

    merged = win_g.merge(assigned, on="BC", how="left")
    merged["mismatch"] = (merged["Genome"] != merged["AssignedGenome"]).astype(int)
    contam = (merged.groupby("BC", as_index=False)
                     .agg(Total_reads=("Read","count"),
                          Mismatch_reads=("mismatch","sum")))
    contam["Contamination_Rate"] = contam["Mismatch_reads"] / contam["Total_reads"]

    # Attach BC-level contamination to the AS_mean table
    agg_bc = agg.merge(contam[["BC","Contamination_Rate","Total_reads"]], on="BC", how="left")
    # Attach the assignment classification (with/without pool design)
    has_pools = bool(pools_for_sample and len(pools_for_sample) > 0)
    if has_pools:
        # earlier logic: Contaminated = 1 if AssignedGenome is within in-pool genomes
        # (you already set Contaminated above using inpool_genomes)
        agg_bc = agg_bc.merge(genotype_assignment[["BC","Contaminated","Pool"]], on="BC", how="left")
        groups_col = "Contaminated"
        labels = {1: "InPool", 0: "Not_InPool"}
        group_mask_in = agg_bc[groups_col].fillna(0).astype(int) == 1
        group_mask_out = ~group_mask_in
        x_label_group_in = "InPool"
        x_label_group_out = "Not_InPool"
    else:
        # NEW: no-pool mode -> split BCs by contamination rate
        agg_bc["Group"] = (agg_bc["Contamination_Rate"].fillna(0.0) <= float(contam_thresh_for_plots)).map(
            {True: "LowContam", False: "HighContam"}
        )
        groups_col = "Group"
        group_mask_in = agg_bc["Group"] == "LowContam"
        group_mask_out = agg_bc["Group"] == "HighContam"
        x_label_group_in = "LowContam"
        x_label_group_out = "HighContam"

    # ---------- ECDFs ----------
    import numpy as np
    from scipy import stats
    def _safe_ks(a, b):    
        a = np.asarray(a); b = np.asarray(b)
        if a.size == 0 or b.size == 0:
            # return NaNs instead of raising
            class R: pass
            r = R(); r.statistic = float("nan"); r.pvalue = float("nan")
            return r
    return stats.ks_2samp(a, b, alternative="two-sided")
    
    def _ecdf(x):
        x = np.sort(np.asarray(x))
        y = np.arange(1, len(x)+1) / len(x) if len(x) else np.array([])
        return x, y

    in_as  = agg_bc.loc[group_mask_in,  "AS_mean"].dropna().values
    out_as = agg_bc.loc[group_mask_out, "AS_mean"].dropna().values
    ks_as = stats.ks_2samp(in_as, out_as, alternative="two-sided")

    in_n  = agg_bc.loc[group_mask_in,  "Total_n"].dropna().values
    out_n = agg_bc.loc[group_mask_out, "Total_n"].dropna().values
    ks_n = stats.ks_2samp(in_n, out_n, alternative="two-sided")

    ks_as = _safe_ks(in_as, out_as)
    ks_n  = _safe_ks(in_n,  out_n)

    x1,y1 = _ecdf(in_as); x2,y2 = _ecdf(out_as)
    xn1,yn1 = _ecdf(in_n); xn2,yn2 = _ecdf(out_n)

    # ---------- PDF pages ----------
    import matplotlib.pyplot as plt
    import matplotlib.backends.backend_pdf as mpdf
    pdf = mpdf.PdfPages(str(pdf_path))

    # Plot 1: ECDF AS_mean
    fig1 = plt.figure(figsize=(4,3)); ax = fig1.gca()
    ax.plot(x1, y1, drawstyle="steps-post", label=x_label_group_in)
    ax.plot(x2, y2, drawstyle="steps-post", label=x_label_group_out)
    ax.set_title("Cum. Dist.\nAS mean"); ax.set_xlabel("AS mean"); ax.set_ylabel("Cumulative probability")
    ax.legend(loc="lower right", fontsize=8)
    ax.text(0.98, 0.1, f"KS D={ks_as.statistic:.3f}\np={ks_as.pvalue:.2e}",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8)
    pdf.savefig(fig1); plt.close(fig1)

    # Plot 2: ECDF Total reads per BC
    fig2 = plt.figure(figsize=(4,3)); ax = fig2.gca()
    ax.plot(xn1, yn1, drawstyle="steps-post", label=x_label_group_in)
    ax.plot(xn2, yn2, drawstyle="steps-post", label=x_label_group_out)
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

    # Plot 4: if pools exist -> heatmap (AssignedGenome x Pool); else -> bar chart (AssignedGenome)
    fig4 = plt.figure(figsize=(5,4)); ax = fig4.gca()
    if has_pools:
        # existing heatmap code here (unchanged) ...
        pass
    else:
        counts = (genotype_assignment.groupby("AssignedGenome").size()
                  .reset_index(name="n").sort_values("n", ascending=False))
        ax.bar(counts["AssignedGenome"], counts["n"])
        ax.set_title("Barcodes per AssignedGenome")
        ax.set_ylabel("BC count")
        ax.tick_params(axis='x', labelrotation=90, labelsize=7)
    pdf.savefig(fig4); plt.close(fig4)

    pdf.close()

def summarize_cli(config_json: Path, pool_design: Optional[Path] = None, xa_max: int = 2) -> dict:
    import json
    cfg = json.loads(Path(config_json).read_text())
    workdir = Path(cfg["workdir"]).expanduser().resolve()
    sample = cfg["sample"]
    inpool, pools = _load_pool_design(pool_design, sample, default_genomes=list(cfg["genomes"].keys()))
    return summarize_and_mark(
        workdir, sample,
        inpool_genomes=inpool,
        xa_max=xa_max,
        pools_for_sample=pools if len(pools) > 0 else None,  # tell the plotter if we have pooling
    )

