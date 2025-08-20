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
    Returns (inpool_genomes, pools_for_sample). For per-pool summary we still
    use inpool_genomes to mark 'PASS' BCs, but we DO NOT plot In/Out here.
    If pool_tsv is None, inpool = default_genomes (uppercased) and pools = [sample].
    """
    import pandas as pd
    if pool_tsv is None:
        return [g.upper() for g in default_genomes], [sample]
    pdf = pd.read_csv(pool_tsv, sep="\t", header=0)
    cols = {c.lower(): c for c in pdf.columns}
    genome_col = cols.get("genome")
    pool_col   = cols.get("pool")
    if not (genome_col and pool_col):
        raise ValueError("Pool design TSV must include columns: Genome, Pool (Plate optional).")
    df = pdf[pdf[pool_col].astype(str).str.upper() == sample.upper()].copy()
    if df.empty:
        raise ValueError(f"No rows in pool design where Pool == '{sample}'.")
    inpool = sorted(df[genome_col].astype(str).str.upper().unique())
    pools  = [sample]
    return inpool, pools

def summarize_and_mark(
    workdir: Path,
    sample: str,
    inpool_genomes: list[str],
    xa_max: int = 2,
    out_dirname: str = "Plots",
    # Grouping controls for plots (per-pool):
    group_mode: str = "quantile",          # "quantile" | "threshold"
    contam_quantile: float = 0.20,         # compare bottom q vs top (1-q)
    contam_thresh_for_plots: float = 0.20, # used if group_mode == "threshold"
) -> Dict[str, object]:
    """
    Per-pool summary: compare low- vs high-contamination BCs (no InPool/Not_InPool plotting).
    Builds PDF + PASS BC list + reads-to-discard list. Returns paths & counts.
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

    winners["Genome"] = winners[colmap["genome"]].astype(str).str.upper()
    winners["BC"]     = winners[colmap["bc"]].astype(str)
    winners["AS"]     = pd.to_numeric(winners[colmap["as"]], errors="coerce")
    winners["Read"]   = winners[colmap["read"]].astype(str)
    winners["Pool"]   = sample  # metadata only

    # -------- per-BC genotype scoring (AS_mean) --------
    agg = (winners
           .groupby(["BC","Genome"], as_index=False)
           .agg(AS_Total=("AS","sum"), n_Read=("Read","count")))
    agg["AS_mean"] = agg["AS_Total"] / agg["n_Read"]
    total_n = agg.groupby("BC", as_index=False)["n_Read"].sum().rename(columns={"n_Read":"Total_n"})
    agg = agg.merge(total_n, on="BC", how="left")

    # 1) Sort by your preferred ranking
    #    Primary key: AS_mean desc
    #    Tie-breakers: n_Read desc, then Genome asc for determinism
    ranked = (agg.sort_values(["BC", "AS_mean", "n_Read", "Genome"],
                          ascending=[True, False, False, True])
              .assign(rank=lambda d: d.groupby("BC").cumcount()))

    # 2) First and second per BC from that explicit rank
    top1 = (ranked[ranked["rank"] == 0]
          .rename(columns={"Genome": "AssignedGenome",
                           "AS_mean": "AS_mean_1"}))
    top1 = top1[["BC", "AssignedGenome", "AS_mean_1", "Total_n"]]
    
    top2 = (ranked[ranked["rank"] == 1][["BC", "AS_mean"]]
          .rename(columns={"AS_mean": "AS_mean_2"}))
    
    # 3) Merge and compute the gap
    assigned = top1.merge(top2, on="BC", how="left")
    assigned["AS_mean_2"] = assigned["AS_mean_2"].fillna(-np.inf)
    assigned["AS_gap"] = assigned["AS_mean_1"] - assigned["AS_mean_2"]

    # Mark 'PASS' BCs relative to in-pool genomes (list = genomes used for this pool/sample)
    assigned["Contaminated"] = np.where(
        assigned["AssignedGenome"].isin([g.upper() for g in inpool_genomes]),
        1, 0
    )

    # -------- read-level mismatch & BC contamination rate --------
    win_g = winners[["BC","Read","Genome"]]
    merged = win_g.merge(assigned[["BC","AssignedGenome","Total_n"]], on="BC", how="left")
    merged["mismatch"] = (merged["Genome"] != merged["AssignedGenome"]).astype(int)
    contam = (merged.groupby("BC", as_index=False)
                     .agg(Total_reads=("Read","count"),
                          Mismatch_reads=("mismatch","sum")))
    contam["Contamination_Rate"] = contam["Mismatch_reads"] / contam["Total_reads"]

    # Attach contamination to assignment table
    agg_bc = assigned.merge(contam[["BC","Contamination_Rate","Total_reads"]], on="BC", how="left")

    # -------- grouping for plots: LowContam vs HighContam only --------
    if group_mode.lower() == "quantile":
        q = float(contam_quantile)
        q = min(max(q, 0.0), 0.49)  # keep sensible
        c = agg_bc["Contamination_Rate"].fillna(0.0)
        lo_thr = c.quantile(q) if len(c) else 0.0
        hi_thr = c.quantile(1.0 - q) if len(c) else 0.0
        group_low  = agg_bc["Contamination_Rate"] <= lo_thr
        group_high = agg_bc["Contamination_Rate"] >= hi_thr
        label_low, label_high = f"LowContam (≤Q{q:.2f})", f"HighContam (≥Q{1.0-q:.2f})"
    else:
        thr = float(contam_thresh_for_plots)
        group_low  = agg_bc["Contamination_Rate"].fillna(0.0) <= thr
        group_high = agg_bc["Contamination_Rate"].fillna(0.0) >  thr
        label_low, label_high = f"LowContam (≤{thr:.2f})", f"HighContam (>{thr:.2f})"

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
    
    def _new_fig(figsize):
        # Prefer constrained layout (Matplotlib 3.4+); fall back to tight_layout
        fig, ax = plt.subplots(figsize=figsize)
        try:
            fig.set_constrained_layout(True)
        except Exception:
            fig.tight_layout(pad=0.6)
        return fig, ax

    def _add_ks_box(ax, ks):
        # Top-left KS annotation with a soft box to avoid overlap
        ax.text(
            0.02, 0.98,
            f"KS D={getattr(ks, 'statistic', float('nan')):.3f}\n"
            f"p={getattr(ks, 'pvalue', float('nan')):.2e}",
            transform=ax.transAxes,
            ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.25", alpha=0.15, linewidth=0),
            fontsize=8,
        )
    
    import numpy as np  # ensure numpy is available for helpers
    import matplotlib.pyplot as plt
    import matplotlib.backends.backend_pdf as mpdf
    
        # vectors
    in_as   = agg_bc.loc[group_low,  "AS_mean_1"].dropna().values
    out_as  = agg_bc.loc[group_high, "AS_mean_1"].dropna().values
    in_n    = agg_bc.loc[group_low,  "Total_n"].dropna().values
    out_n   = agg_bc.loc[group_high, "Total_n"].dropna().values
    in_gap  = agg_bc.loc[group_low,  "AS_gap"].dropna().values
    out_gap = agg_bc.loc[group_high, "AS_gap"].dropna().values

    ks_as  = _safe_ks(in_as,  out_as)
    ks_n   = _safe_ks(in_n,   out_n)
    ks_gap = _safe_ks(in_gap, out_gap)

    x1,y1    = _ecdf(in_as);   x2,y2    = _ecdf(out_as)
    xn1,yn1  = _ecdf(in_n);    xn2,yn2  = _ecdf(out_n)
    xg1,yg1  = _ecdf(in_gap);  xg2,yg2  = _ecdf(out_gap)

    # ---------- PDF ----------
    out_dir = (workdir / sample / out_dirname)
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{sample}_contamination_correction_summary.pdf"
    pdf = mpdf.PdfPages(str(pdf_path))

    # Plot 1: ECDF AS_mean (of assigned genome per BC)
    fig1, ax = _new_fig((4.8, 3.2))
    ax.plot(x1, y1, drawstyle="steps-post", label=label_low)
    ax.plot(x2, y2, drawstyle="steps-post", label=label_high)
    ax.set_title("Cum. Dist. AS_mean (assigned)", pad=6)
    ax.set_xlabel("AS_mean", labelpad=2)
    ax.set_ylabel("Cumulative probability", labelpad=2)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    _add_ks_box(ax, ks_as)
    ax.margins(x=0.02, y=0.02)
    ax.tick_params(labelsize=8)
    pdf.savefig(fig1, bbox_inches="tight"); plt.close(fig1)

    # Plot 2: ECDF Total reads per BC
    fig2, ax = _new_fig((4.8, 3.2))
    ax.plot(xn1, yn1, drawstyle="steps-post", label=label_low)
    ax.plot(xn2, yn2, drawstyle="steps-post", label=label_high)
    ax.set_title("Cum. Dist. Reads per BC", pad=6)
    ax.set_xlabel("Reads", labelpad=2)
    ax.set_ylabel("Cumulative probability", labelpad=2)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    _add_ks_box(ax, ks_n)
    ax.margins(x=0.02, y=0.02)
    ax.tick_params(labelsize=8)
    pdf.savefig(fig2, bbox_inches="tight"); plt.close(fig2)

    # Plot 3: ECDF AS_gap (confidence gap between top-1 and top-2 AS_mean)
    fig3, ax = _new_fig((4.8, 3.2))
    ax.plot(xg1, yg1, drawstyle="steps-post", label=label_low)
    ax.plot(xg2, yg2, drawstyle="steps-post", label=label_high)
    ax.set_title("Cum. Dist. AS_gap (top1 - top2)", pad=6)
    ax.set_xlabel("AS_gap", labelpad=2)
    ax.set_ylabel("Cumulative probability", labelpad=2)
    ax.legend(loc="lower right", frameon=False, fontsize=8)
    _add_ks_box(ax, ks_gap)
    ax.margins(x=0.02, y=0.02)
    ax.tick_params(labelsize=8)
    pdf.savefig(fig3, bbox_inches="tight"); plt.close(fig3)

    # Plot 4: Hexbin contamination vs reads (all BCs)
    fig4, ax = _new_fig((6.2, 3.6))
    if not contam.empty:
        hb = ax.hexbin(contam["Total_reads"], contam["Contamination_Rate"],
                       gridsize=20, mincnt=1)
        cbar = fig4.colorbar(hb, ax=ax, shrink=0.85, pad=0.02)
        cbar.set_label("Density", fontsize=8)
        ax.set_title("BC contamination vs reads", pad=6)
        ax.set_xlabel("Reads per BC", labelpad=2)
        ax.set_ylabel("Contamination rate", labelpad=2)
    else:
        ax.text(0.5, 0.5, "No BCs", ha="center", va="center")
    ax.tick_params(labelsize=8)
    pdf.savefig(fig4, bbox_inches="tight"); plt.close(fig4)
    
    # Plot 5: Bar chart of counts per AssignedGenome
    fig5, ax = _new_fig((6.2, 4.0))
    counts = (assigned.groupby("AssignedGenome")
              .size().reset_index(name="n").sort_values("n", ascending=False))
    if not counts.empty:
        ax.bar(counts["AssignedGenome"], counts["n"])
    ax.set_title("Barcodes per AssignedGenome", pad=6)
    ax.set_ylabel("BC count", labelpad=2)
    ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    for tick in ax.get_xticklabels():
        tick.set_ha("right")
    pdf.savefig(fig5, bbox_inches="tight"); 
    plt.close(fig5)
    pdf.close()

    # -------- outputs --------
    # PASS BCs = assigned to a genome that is in the per-pool genome list
    hq = assigned[assigned["AssignedGenome"].isin([g.upper() for g in inpool_genomes])].copy()
    hq_path = sdir / f"{sample}_BCs_PASS_by_mapping.csv"
    hq[["BC","AssignedGenome","Total_n","AS_mean_1","AS_gap"]].to_csv(hq_path, index=False)

    # Reads to discard: winner genome != assigned BC genome
    to_filter = winners[["Read","BC","Genome"]].merge(
        assigned[["BC","AssignedGenome"]],
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
        # plotting groups are contamination-based only in per-pool summary
    )
