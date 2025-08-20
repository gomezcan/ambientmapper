# src/ambientmapper/plate.py
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Tuple
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as mpdf


def load_sample_to_wells(file_path: Path) -> Dict[str, List[str]]:
    """
    Parse a TSV mapping pools/samples to well ranges.

    File format (comments and blanks allowed):
      # Pool   Wells
      A1      A1-12,B1-12
      B3      E3-6,F1-8,G2-9
      ...

    Returns: { "A1": ["A1","A2",...], "B3": [...], ... }
    """
    sample_to_wells: Dict[str, List[str]] = {}
    with open(file_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "\t" not in line:
                raise ValueError(f"Bad line (expect TAB): {line!r}")
            sample, ranges_str = line.split("\t", 1)
            sample = sample.strip()
            wells = sample_to_wells.setdefault(sample, [])
            for r in ranges_str.split(","):
                r = r.strip().replace(" ", "")
                if not r:
                    continue
                m = re.match(r"^([A-Ha-h])(\d+)(?:-(\d+))?$", r)
                if not m:
                    raise ValueError(f"Bad well/range: {r!r}")
                row = m.group(1).upper()
                start = int(m.group(2))
                end = int(m.group(3)) if m.group(3) else start
                if start > end:
                    raise ValueError(f"Start>end in {r!r}")
                for i in range(start, end + 1):
                    if i < 1 or i > 12:
                        raise ValueError(f"Column out of 1..12 in {r!r}")
                    wells.append(f"{row}{i}")
    return sample_to_wells


def _load_winners(sample_dir: Path, sample: str) -> pd.DataFrame:
    """
    Load winners for a pool/sample (either merged or per-chunk).
    """
    final = sample_dir / "final" / f"{sample}_per_read_winner.tsv.gz"
    if final.exists():
        return pd.read_csv(final, sep="\t", low_memory=False)
    chunks = sorted((sample_dir / "cell_map_ref_chunks").glob(f"{sample}_cell_genotype_reads_chunk_*.txt"))
    dfs = [pd.read_csv(p, sep="\t") for p in chunks if p.exists() and p.stat().st_size > 0]
    return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()


def _bc_contamination_from_winners(winners: pd.DataFrame, xa_max: int = 2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute per-BC assignment (by AS_mean) and contamination rate
    using the standard logic (no pool design required).

    Returns:
      assigned   : DataFrame with columns [BC, AssignedGenome, AS_mean_1, AS_mean_2, AS_gap, Total_n]
      contam_bc  : DataFrame with columns [BC, Total_reads, Mismatch_reads, Contamination_Rate]
    """
    # Normalize columns
    colmap = {c.lower(): c for c in winners.columns}
    need = ["read", "bc", "genome", "as"]
    missing = [k for k in need if k not in colmap]
    if missing:
        raise ValueError(f"Winners table missing required columns: {missing}. Found: {list(winners.columns)}")

    if xa_max >= 0 and "xacount" in colmap:
        winners = winners[winners[colmap["xacount"]].fillna(0) <= xa_max].copy()

    w = pd.DataFrame({
        "Read": winners[colmap["read"]].astype(str),
        "BC": winners[colmap["bc"]].astype(str),
        "Genome": winners[colmap["genome"]].astype(str).str.upper(),
        "AS": pd.to_numeric(winners[colmap["as"]], errors="coerce"),
    })

    # Per-BC/per-Genome: AS_total, n_Read, AS_mean; Total_n per BC
    agg = (w.groupby(["BC", "Genome"], as_index=False)
             .agg(AS_Total=("AS", "sum"), n_Read=("Read", "count")))
    agg["AS_mean"] = agg["AS_Total"] / agg["n_Read"]
    total_n = (agg.groupby("BC", as_index=False)["n_Read"]
                 .sum().rename(columns={"n_Read": "Total_n"}))
    agg = agg.merge(total_n, on="BC", how="left")

    # Rank within BC by AS_mean desc, then n_Read desc, then Genome asc (deterministic)
    ranked = (agg.sort_values(["BC", "AS_mean", "n_Read", "Genome"],
                              ascending=[True, False, False, True])
                .assign(rank=lambda d: d.groupby("BC").cumcount()))

    top1 = (ranked[ranked["rank"] == 0]
            .rename(columns={"Genome": "AssignedGenome", "AS_mean": "AS_mean_1"}))
    top1 = top1[["BC", "AssignedGenome", "AS_mean_1", "Total_n"]]

    top2 = (ranked[ranked["rank"] == 1][["BC", "AS_mean"]]
            .rename(columns={"AS_mean": "AS_mean_2"}))

    assigned = top1.merge(top2, on="BC", how="left")
    assigned["AS_mean_2"] = assigned["AS_mean_2"].fillna(-np.inf)
    assigned["AS_gap"] = assigned["AS_mean_1"] - assigned["AS_mean_2"]

    # Read-level mismatch to compute contamination per BC
    win_g = w[["BC", "Read", "Genome"]]
    merged = win_g.merge(assigned[["BC", "AssignedGenome"]], on="BC", how="left")
    merged["mismatch"] = (merged["Genome"] != merged["AssignedGenome"]).astype(int)
    contam_bc = (merged.groupby("BC", as_index=False)
                        .agg(Total_reads=("Read", "count"),
                             Mismatch_reads=("mismatch", "sum")))
    contam_bc["Contamination_Rate"] = contam_bc["Mismatch_reads"] / contam_bc["Total_reads"]

    return assigned, contam_bc


def _make_density_pdf(per_bc: pd.DataFrame, out_pdf: Path) -> None:
    """
    Overlay density/hist of contamination distributions per sample.
    """
    pdf = mpdf.PdfPages(str(out_pdf))
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for sample, sub in per_bc.groupby("Sample"):
        vals = sub["Contamination_Rate"].dropna().values
        if len(vals) == 0:
            continue
        # Histogram with density=True keeps deps minimal (no seaborn)
        ax.hist(vals, bins=50, density=True, alpha=0.35, label=str(sample))
    ax.set_title("Contamination distributions per pool")
    ax.set_xlabel("Contamination rate per BC")
    ax.set_ylabel("Density")
    if len(per_bc["Sample"].unique()) <= 15:
        ax.legend(loc="upper right", fontsize=8, frameon=False)
    fig.tight_layout()
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)
    pdf.close()


def _make_plate_heatmap(sample_to_wells: Dict[str, List[str]],
                        per_sample: pd.DataFrame,
                        out_pdf: Path,
                        value_col: str = "median_contam") -> None:
    """
    Draw a 96-well (A–H x 1–12) heatmap where each well is colored by the
    contamination metric of its assigned sample/pool.
    """
    # Map sample -> value
    val = dict(zip(per_sample["Sample"].astype(str), per_sample[value_col].astype(float)))

    rows = list("ABCDEFGH")
    cols = list(range(1, 13))
    grid = np.full((len(rows), len(cols)), np.nan, dtype=float)

    for sample, wells in sample_to_wells.items():
        v = val.get(str(sample), np.nan)
        for w in wells:
            m = re.match(r"^([A-H])(\d{1,2})$", w)
            if not m:
                continue
            r = rows.index(m.group(1))
            c = int(m.group(2)) - 1
            if 0 <= r < 8 and 0 <= c < 12:
                grid[r, c] = v

    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    im = ax.imshow(grid, aspect="equal")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label(value_col.replace("_", " ").title(), fontsize=9)

    ax.set_xticks(range(12)); ax.set_xticklabels(cols)
    ax.set_yticks(range(8));  ax.set_yticklabels(rows)
    ax.set_title("Plate heatmap — contamination by pool (median per BC)")
    # Annotate wells with sample IDs if few unique samples to avoid clutter
    unique_samples = list(sample_to_wells.keys())
    if len(unique_samples) <= 24:
        inv_map = {}
        for s, wells in sample_to_wells.items():
            for w in wells:
                inv_map[w] = s
        for r, R in enumerate(rows):
            for c, C in enumerate(cols):
                w = f"{R}{C}"
                s = inv_map.get(w)
                if s is not None:
                    ax.text(c, r, s, ha="center", va="center", fontsize=6,
                            color="white" if np.isfinite(grid[r, c]) and grid[r, c] > np.nanmedian(grid) else "black")
    fig.tight_layout()
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def plate_summary(workdir: Path, plate_map: Path, outdir: Path | None = None,
                  xa_max: int = 2) -> Dict[str, str]:
    """
    Build plate-level summaries:

    - per-BC table with contamination for every pool on the plate
    - per-sample summary table (mean/median contamination, counts)
    - density PDF
    - plate heatmap PDF

    Returns dict of output paths.
    """
    workdir = Path(workdir).expanduser().resolve()
    sample_to_wells = load_sample_to_wells(plate_map)
    samples = list(sample_to_wells.keys())
    outroot = Path(outdir) if outdir else (workdir / "PlateSummary")
    outroot.mkdir(parents=True, exist_ok=True)

    all_bc_rows: List[pd.DataFrame] = []
    per_sample_rows = []

    for s in samples:
        sdir = workdir / s
        winners = _load_winners(sdir, s)
        if winners.empty:
            # skip silently but record
            continue
        assigned, contam = _bc_contamination_from_winners(winners, xa_max=xa_max)
        bc = assigned.merge(contam, on="BC", how="left")
        bc.insert(0, "Sample", s)
        all_bc_rows.append(bc)

        vals = bc["Contamination_Rate"].dropna().values
        per_sample_rows.append({
            "Sample": s,
            "n_BC": int(bc["BC"].nunique()),
            "n_reads": int(contam["Total_reads"].sum() if len(contam) else 0),
            "mean_contam": float(np.mean(vals)) if len(vals) else np.nan,
            "median_contam": float(np.median(vals)) if len(vals) else np.nan,
            "p95_contam": float(np.quantile(vals, 0.95)) if len(vals) else np.nan,
        })

    if not all_bc_rows:
        raise FileNotFoundError("No winners found for any pools in the plate map.")

    per_bc = pd.concat(all_bc_rows, ignore_index=True)
    per_sample = pd.DataFrame(per_sample_rows)

    # Write outputs
    bc_path = outroot / "plate_per_bc.tsv.gz"
    per_bc.to_csv(bc_path, sep="\t", index=False, compression="gzip")

    sm_path = outroot / "plate_per_sample.tsv"
    per_sample.to_csv(sm_path, sep="\t", index=False)

    # Plots
    dens_pdf = outroot / "plate_density.pdf"
    _make_density_pdf(per_bc, dens_pdf)

    heat_pdf = outroot / "plate_heatmap.pdf"
    _make_plate_heatmap(sample_to_wells, per_sample, heat_pdf, value_col="median_contam")

    return {
        "per_bc": str(bc_path),
        "per_sample": str(sm_path),
        "density_pdf": str(dens_pdf),
        "heatmap_pdf": str(heat_pdf),
    }


def plate_summary_cli(workdir: Path, plate_map: Path, outdir: Path | None, xa_max: int = 2) -> Dict[str, str]:
    """
    Thin CLI wrapper.
    """
    return plate_summary(workdir=workdir, plate_map=plate_map, outdir=outdir, xa_max=xa_max)
