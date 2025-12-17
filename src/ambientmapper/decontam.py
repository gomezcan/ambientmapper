# src/ambientmapper/decontam.py
from __future__ import annotations

import gzip
import json
import glob
from pathlib import Path
from typing import Dict, Optional, List

import pandas as pd
import typer

from .data import get_default_layout_path
from .utils import (
    load_sample_to_wells,
    load_barcode_layout,
    build_well_to_sample,
    build_barcode_to_sample,
)

app = typer.Typer(
    help="Generate barcode + read-level decontamination decisions.",
    add_completion=False,
    invoke_without_command=True,
    no_args_is_help=True,
)

# -------------------------
# Helpers
# -------------------------
def _infer_sample_name_from_cells_calls(cells_calls: Path) -> str:
    bn = cells_calls.name
    return bn.replace("_cells_calls.tsv.gz", "").replace("_cells_calls.tsv", "")


def _infer_sample_root_from_cells_calls(cells_calls: Path) -> Optional[Path]:
    """
    Expect: <workdir>/<sample>/final/<sample>_cells_calls.tsv(.gz)
    Return: <workdir>/<sample> (the sample root) or None.
    """
    p = cells_calls.resolve()
    if p.parent.name != "final":
        return None
    sample_root = p.parent.parent
    return sample_root if sample_root.exists() else None


def _default_assign_glob(sample_root: Path, sample_name: str) -> str:
    return str(sample_root / "cell_map_ref_chunks" / f"{sample_name}_chunk*_filtered.tsv.gz")


def _col_first(df: pd.DataFrame, cols: list[str]) -> Optional[str]:
    for c in cols:
        if c in df.columns:
            return c
    return None


def _norm_str(x) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if s.lower() == "nan":
        return ""
    return s


def _row_candidates_from_calls(row, candidate_cols: list[str]) -> list[str]:
    out: list[str] = []
    for c in candidate_cols:
        v = _norm_str(getattr(row, c, ""))
        if v and v not in out:
            out.append(v)
    return out


def _design_key(barcode: str, mode: str, n: int) -> str:
    """
    Extract the barcode key used for design matching + per-barcode rule maps.

    mode:
      - before-dash : take part before first '-'
      - full        : entire string
      - last        : last n characters of before-dash part
      - first       : first n characters of before-dash part
    """
    s = str(barcode).strip()
    if mode == "full":
        base = s
    else:
        base = s.split("-")[0]

    # normalize: remove separators used in layout (and sometimes in barcodes)
    base = base.replace("_", "").replace(":", "").replace(".", "")
    
    if mode == "before-dash":
        return base

    if n <= 0:
        return base

    if mode == "last":
        return base[-n:]
    if mode == "first":
        return base[:n]

    return base


def _build_bckey_to_expected(
    layout_path: Path,
    design_path: Path,
    *,
    design_bc_mode: str,
    design_bc_n: int,
) -> Dict[str, str]:
    """
    Build bc_key -> expected_genome map.

    Supports:
      (1) TSV with header: barcode  expected_genome
      (2) Pool ranges file: genome <tab> wells (A1-4,B1-4,...), using a 96-well layout
    """
    design_path = Path(design_path)
    layout_path = Path(layout_path)

    # Try direct barcode mapping TSV first
    try:
        df = pd.read_csv(design_path, sep="\t", dtype=str)
    except Exception:
        df = None

    if df is not None and {"barcode", "expected_genome"}.issubset(df.columns):
        df = df[["barcode", "expected_genome"]].astype(str)
        df["barcode"] = df["barcode"].map(lambda x: _design_key(x, design_bc_mode, design_bc_n))
        return dict(zip(df["barcode"], df["expected_genome"]))

    # Fallback: pool ranges + layout (sample names are expected genomes)
    sample_to_wells = load_sample_to_wells(design_path)
    well_to_sample = build_well_to_sample(sample_to_wells)
    well_to_barcode = load_barcode_layout(layout_path)
    bc_to_sample = build_barcode_to_sample(well_to_barcode, well_to_sample)

    out: Dict[str, str] = {}
    for bc, sample in bc_to_sample.items():
        k = _design_key(bc, design_bc_mode, design_bc_n)
        out[str(k)] = str(sample)
    return out


@app.callback(invoke_without_command=True)
def decontam_cmd(
    cells_calls: Path = typer.Option(
        ...,
        exists=True,
        readable=True,
        help="*_cells_calls.tsv(.gz) from genotyping.",
    ),
    out_dir: Path = typer.Option(
        ...,
        help="Output directory for decontam artifacts.",
    ),
    # Design / Layout
    design_file: Optional[Path] = typer.Option(
        None,
        "--design-file",
        help="Optional design file. If provided enables design-aware mode.",
        readable=True,
    ),
    layout_file: str = typer.Option(
        "DEFAULT",
        "--layout-file",
        help='Layout path for well-range design files, or "DEFAULT" (bundled).',
    ),
    strict_design_drop_mismatch: bool = typer.Option(
        True,
        "--strict-design-drop-mismatch/--no-strict-design-drop-mismatch",
        help="If strict, drop whole barcode when expected genome is not among candidates.",
    ),
    # Assignment inputs
    assignments: Optional[Path] = typer.Option(
        None,
        "--assignments",
        exists=True,
        readable=True,
        help="Single assignment chunk path (tsv/tsv.gz).",
    ),
    assign_glob: Optional[str] = typer.Option(
        None,
        "--assign-glob",
        help="Glob for assign chunks (e.g. 'SM2/cell_map_ref_chunks/SM2_chunk*_filtered.tsv.gz').",
    ),
    # Column overrides (assign outputs)
    read_id_col: str = typer.Option("Read", "--read-id-col", help="Column for Read ID"),
    barcode_col: str = typer.Option("BC", "--barcode-col", help="Column for Barcode"),
    genome_col: str = typer.Option("Genome", "--genome-col", help="Column for Genome"),
    class_col: str = typer.Option("assigned_class", "--class-col", help="Column for classification (winner/ambiguous)."),
    p_as_col: str = typer.Option("p_as", "--p-as-col", help="Column for assignment p-value"),
    # Logic configuration
    decontam_alpha: Optional[float] = typer.Option(
        None,
        "--decontam-alpha",
        help="If set, redefine 'winner' using p_as <= alpha (where p_as exists).",
    ),
    require_p_as: bool = typer.Option(
        False,
        "--require-p-as",
        help="If True and --decontam-alpha is set, treat missing p_as as non-winner.",
    ),
    # Misc
    sample_name: Optional[str] = typer.Option(None, "--sample-name", help="Output prefix."),
    chunksize: int = typer.Option(1_000_000, "--chunksize", help="Pandas chunk size."),
    # Design barcode key selection
    design_bc_mode: str = typer.Option(
        "before-dash",
        "--design-bc-mode",
        help="Barcode key extraction for design matching: before-dash|last|first|full",
    ),
    design_bc_n: int = typer.Option(
        10,
        "--design-bc-n",
        help="N used when --design-bc-mode is last/first (e.g. 10 for last 10 bp).",
    ),
):
    cells_calls = cells_calls.expanduser().resolve()
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if sample_name is None:
        sample_name = _infer_sample_name_from_cells_calls(cells_calls)

    sample_root = _infer_sample_root_from_cells_calls(cells_calls)

    # Validate optional design_file if provided
    if design_file is not None:
        design_file = Path(design_file).expanduser().resolve()
        if not design_file.exists():
            raise FileNotFoundError(f"--design-file not found: {design_file}")

    # Auto-assign-glob if user didn't provide read-level inputs
    if assignments is None and (assign_glob is None or not assign_glob.strip()):
        if sample_root is not None:
            assign_glob = _default_assign_glob(sample_root, sample_name)
            typer.echo(f"[decontam] auto --assign-glob: {assign_glob}")

    # ------------------------------------------------------------------
    # 1) Load per-barcode calls & build schema
    # ------------------------------------------------------------------
    typer.echo(f"[decontam] Loading cell calls: {cells_calls}")
    calls = pd.read_csv(cells_calls, sep="\t", dtype=str)

    if "barcode" not in calls.columns:
        raise ValueError(f"{cells_calls} missing required column: 'barcode'")

    best_col = _col_first(calls, ["best_genome", "top_genome", "genome_top1"])
    if best_col is None:
        raise ValueError(
            f"{cells_calls} missing a best-genome column. Expected one of: best_genome, top_genome, genome_top1"
        )

    if {"genome_top1", "genome_top2", "genome_top3"}.issubset(calls.columns):
        candidate_cols = ["genome_top1", "genome_top2", "genome_top3"]
    else:
        candidate_cols = []
        for c in ["top_genome", "best_genome", "genome_1", "genome_2"]:
            if c in calls.columns and c not in candidate_cols:
                candidate_cols.append(c)

    calls["barcode"] = calls["barcode"].astype(str)
    calls["bc_key"] = calls["barcode"].map(lambda x: _design_key(x, design_bc_mode, design_bc_n))

    # ------------------------------------------------------------------
    # 1b) Build design map (optional)
    # ------------------------------------------------------------------
    bckey_to_expected: Optional[Dict[str, str]] = None
    if design_file is not None:
        typer.echo(f"[decontam] Strategy A: Design-Aware ({design_file})")

        if layout_file == "DEFAULT":
            layout_path = get_default_layout_path()
        else:
            layout_path = Path(layout_file).expanduser().resolve()
            if not layout_path.exists():
                raise FileNotFoundError(f"--layout-file not found: {layout_path}")

        bckey_to_expected = _build_bckey_to_expected(
            layout_path,
            design_file,
            design_bc_mode=design_bc_mode,
            design_bc_n=design_bc_n,
        )
        if not bckey_to_expected:
            raise ValueError(f"Design parsing produced an empty bc->expected map: {design_file}")
    else:
        typer.echo(f"[decontam] Strategy B: Agnostic (Allowed = {best_col})")

    # ------------------------------------------------------------------
    # 2) Per-barcode allowed genome decision (keyed by bc_key)
    # ------------------------------------------------------------------
    allowed_genome_map: Dict[str, str] = {}
    barcodes_to_drop_set: set[str] = set()

    barcode_summary_rows: List[dict] = []
    barcodes_to_drop_rows: List[dict] = []
    per_barcode_rules_rows: List[dict] = []

    for row in calls.itertuples(index=False):
        bc_full = _norm_str(getattr(row, "barcode", ""))
        bc_key = _norm_str(getattr(row, "bc_key", ""))

        best = _norm_str(getattr(row, best_col, ""))
        cands = _row_candidates_from_calls(row, candidate_cols)

        expected: Optional[str] = None
        allowed: str = ""
        action = "keep_cleaned"
        notes = ""

        if not bc_key:
            # truly broken barcode string
            action = "drop_barcode"
            notes = "empty_bc_key"
        elif bckey_to_expected is not None:
            expected = _norm_str(bckey_to_expected.get(bc_key))
            if not expected:
                if strict_design_drop_mismatch:
                    action = "drop_barcode"
                    notes = "unknown_in_design"
                else:
                    action = "keep_cleaned"
                    allowed = ""
                    notes = "unknown_in_design_keep"
            elif expected in cands:
                allowed = expected
                notes = "design_match_candidate"
            else:
                if strict_design_drop_mismatch:
                    action = "drop_barcode"
                    notes = "design_mismatch_strict"
                else:
                    allowed = expected
                    notes = "design_mismatch_lenient"
        else:
            # Agnostic: do NOT drop barcodes with empty best; just skip mismatch logic for them
            if not best:
                allowed = ""
                notes = "no_best_genome_keep"
            else:
                allowed = best
                notes = f"agnostic_{best_col}"

        if action == "drop_barcode":
            barcodes_to_drop_set.add(bc_key)
            barcodes_to_drop_rows.append({"barcode": bc_full, "bc_key": bc_key, "reason": notes})
        elif allowed:
            allowed_genome_map[bc_key] = allowed
            per_barcode_rules_rows.append({"barcode": bc_full, "bc_key": bc_key, "drop_genome_ne": allowed})

        barcode_summary_rows.append(
            {
                "barcode": bc_full,
                "bc_key": bc_key,
                "best_genome_col": best_col,
                "best_genome": best,
                "candidates": ",".join(cands),
                "expected_genome": expected,
                "allowed_genome": allowed,
                "action": action,
                "notes": notes,
                "strategy": "A" if (bckey_to_expected is not None) else "B",
            }
        )

    # ------------------------------------------------------------------
    # 3) Identify assignment files
    # ------------------------------------------------------------------
    input_files: List[Path] = []

    if assignments is not None:
        input_files.append(Path(assignments).expanduser().resolve())

    if assign_glob is not None:
        assign_glob = assign_glob.strip() or None

    if assign_glob:
        found = sorted(glob.glob(assign_glob, recursive=True))
        if not found:
            raise FileNotFoundError(
                f"[decontam] No assignment files matched: {assign_glob}\n"
                f"Expected files like: {sample_name}_chunk*_filtered.tsv.gz under cell_map_ref_chunks/."
            )
        input_files.extend([Path(f).resolve() for f in found])

    input_files = sorted(set(input_files))

    if not input_files:
        typer.echo("[decontam] No assignment files found. Skipping read-level logic.")

    # ------------------------------------------------------------------
    # 4) Process read assignments → reads_to_drop.tsv.gz
    # ------------------------------------------------------------------
    reads_to_drop_path: Optional[Path] = None
    reads_total_map: Dict[str, int] = {}
    reads_dropped_map: Dict[str, int] = {}
    warned_missing_p_as = False

    if input_files:
        reads_to_drop_path = out_dir / f"{sample_name}_reads_to_drop.tsv.gz"
        typer.echo(f"[decontam] Processing {len(input_files)} assignment files...")

        with gzip.open(reads_to_drop_path, "wt") as fh:
            fh.write("read_id\tbarcode\tbc_key\tallowed_genome\twinner_genome\tp_as\treason\n")

            for fp in input_files:
                typer.echo(f"  -> Streaming {fp.name}")

                for chunk in pd.read_csv(fp, sep="\t", chunksize=chunksize, dtype=str):
                    required_in_chunk = {read_id_col, barcode_col, genome_col, class_col}
                    missing_cols = required_in_chunk - set(chunk.columns)
                    if missing_cols:
                        raise ValueError(f"File {fp.name} missing columns: {sorted(missing_cols)}")

                    bc_full_series = chunk[barcode_col].astype(str)
                    bc_key_series = bc_full_series.map(lambda x: _design_key(x, design_bc_mode, design_bc_n))

                    allowed_series = bc_key_series.map(allowed_genome_map)

                    # Barcode-level drop: drop all reads for those barcodes
                    mask_drop_barcode = bc_key_series.isin(barcodes_to_drop_set)

                    # Winner definition
                    class_series = chunk[class_col].astype(str)
                    is_winner_default = (class_series == "winner")

                    has_p_col = (p_as_col in chunk.columns)
                    p_raw = chunk[p_as_col] if has_p_col else pd.Series(pd.NA, index=chunk.index)

                    if decontam_alpha is not None and not has_p_col and not warned_missing_p_as:
                        typer.echo(
                            f"[decontam] Warning: --decontam-alpha set but column '{p_as_col}' not found "
                            f"in {fp.name}; falling back to '{class_col}' for those files.",
                            err=True,
                        )
                        warned_missing_p_as = True

                    if decontam_alpha is not None and has_p_col:
                        p_vals = pd.to_numeric(p_raw, errors="coerce")
                        has_p = p_vals.notna()
                        is_winner_override = has_p & (p_vals <= float(decontam_alpha))
                        if require_p_as:
                            is_confident = is_winner_override
                        else:
                            is_confident = is_winner_override | ((~has_p) & is_winner_default)
                    else:
                        is_confident = is_winner_default

                    # IMPORTANT: if allowed is missing/empty, do not drop mismatches (we cannot define mismatch)
                    mask_no_allowed = allowed_series.isna() | (allowed_series.astype(str) == "")

                    winner_genome = chunk[genome_col].astype(str)

                    # mismatch only applies when we have allowed_genome
                    mask_mismatch = (~mask_drop_barcode) & (~mask_no_allowed) & (winner_genome != allowed_series)

                    mask_mismatch_conf = mask_mismatch & is_confident
                    mask_drop_read = mask_drop_barcode | mask_mismatch_conf

                    # Totals by bc_key
                    counts_total = bc_key_series.value_counts()
                    for b, c in counts_total.items():
                        reads_total_map[b] = reads_total_map.get(b, 0) + int(c)

                    if not mask_drop_read.any():
                        continue

                    counts_dropped = bc_key_series[mask_drop_read].value_counts()
                    for b, c in counts_dropped.items():
                        reads_dropped_map[b] = reads_dropped_map.get(b, 0) + int(c)

                    reasons = pd.Series("", index=chunk.index, dtype=object)
                    reasons.loc[mask_drop_barcode] = "drop_barcode"

                    if decontam_alpha is not None and has_p_col:
                        p_vals2 = pd.to_numeric(p_raw, errors="coerce")
                        has_p2 = p_vals2.notna()
                        mask_p_conf = mask_mismatch_conf & has_p2 & (p_vals2 <= float(decontam_alpha))
                        mask_class_conf = mask_mismatch_conf & ~mask_p_conf
                        reasons.loc[mask_p_conf] = "mismatch_winner_pvalue"
                        reasons.loc[mask_class_conf] = "mismatch_winner_class"
                    else:
                        reasons.loc[mask_mismatch_conf] = "mismatch_winner"

                    out_df = pd.DataFrame(
                        {
                            "read_id": chunk[read_id_col].astype(str),
                            "barcode": bc_full_series,
                            "bc_key": bc_key_series,
                            "allowed_genome": allowed_series.fillna(""),
                            "winner_genome": winner_genome,
                            "p_as": p_raw.fillna("").astype(str),
                            "reason": reasons,
                        }
                    )
                    out_df.loc[mask_drop_read].to_csv(fh, sep="\t", header=False, index=False)

        typer.echo(f"[decontam] Wrote {reads_to_drop_path}")

    # ------------------------------------------------------------------
    # 5) Finalize summary stats (keyed by bc_key)
    # ------------------------------------------------------------------
    barcode_summary = pd.DataFrame.from_records(barcode_summary_rows)
    barcode_summary["reads_total"] = barcode_summary["bc_key"].map(reads_total_map).fillna(0).astype(int)
    barcode_summary["reads_dropped"] = barcode_summary["bc_key"].map(reads_dropped_map).fillna(0).astype(int)

    mask_nz = barcode_summary["reads_total"] > 0
    barcode_summary["fraction_removed"] = 0.0
    barcode_summary.loc[mask_nz, "fraction_removed"] = (
        barcode_summary.loc[mask_nz, "reads_dropped"] / barcode_summary.loc[mask_nz, "reads_total"]
    )

    # ------------------------------------------------------------------
    # 6) Write artifacts
    # ------------------------------------------------------------------
    barcode_summary_path = out_dir / f"{sample_name}_barcode_summary.tsv.gz"
    barcodes_to_drop_path = out_dir / f"{sample_name}_barcodes_to_drop.tsv.gz"
    per_barcode_rules_path = out_dir / f"{sample_name}_per_barcode_rules.tsv.gz"
    params_path = out_dir / f"{sample_name}_decontam_params.json"

    barcode_summary.to_csv(barcode_summary_path, sep="\t", index=False)
    pd.DataFrame.from_records(barcodes_to_drop_rows).to_csv(barcodes_to_drop_path, sep="\t", index=False)
    pd.DataFrame.from_records(per_barcode_rules_rows).to_csv(per_barcode_rules_path, sep="\t", index=False)

    params = {
        "cells_calls": str(cells_calls),
        "out_dir": str(out_dir),
        "design_file": str(design_file) if design_file else None,
        "layout_file": layout_file,
        "strict_design_drop_mismatch": strict_design_drop_mismatch,
        "assignments": str(assignments) if assignments else None,
        "assign_glob": assign_glob,
        "read_id_col": read_id_col,
        "barcode_col": barcode_col,
        "genome_col": genome_col,
        "class_col": class_col,
        "p_as_col": p_as_col,
        "decontam_alpha": decontam_alpha,
        "require_p_as": require_p_as,
        "chunksize": chunksize,
        "reads_to_drop": str(reads_to_drop_path) if reads_to_drop_path else None,
        "best_col": best_col,
        "candidate_cols": candidate_cols,
        "design_bc_mode": design_bc_mode,
        "design_bc_n": design_bc_n,
    }
    with params_path.open("w") as f:
        json.dump(params, f, indent=2)

    typer.echo(f"[decontam] Wrote {barcode_summary_path}")
    typer.echo(f"[decontam] Wrote {barcodes_to_drop_path}")
    typer.echo(f"[decontam] Wrote {per_barcode_rules_path}")
    typer.echo(f"[decontam] Wrote {params_path}")
    typer.echo("[decontam] Done.")
