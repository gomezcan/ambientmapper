from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd  # <- FIX
from .normalization import canonicalize_bc_seq_sample


def filter_qc_file(
    in_path: Path,
    out_path: Path,
    min_freq: int,
    sample_hint: Optional[str] = None,
) -> int:
    """
    Read a QC mapping TSV (from extract), normalize BC to '<seq>-<sample>',
    keep BCs with frequency >= min_freq, collapse duplicate (Read,BC) rows,
    and write a filtered TSV.

    Returns number of rows written.
    """
    # Read extract output (no header)
    df = pd.read_csv(
        in_path,
        sep="\t",
        names=["Read", "BC", "MAPQ", "AS", "NM", "XAcount"],
        header=None,
        dtype={"Read": str, "BC": str},
        low_memory=False,
    )

    # Normalize BC to '<seq>-<sample>'
    df["BC"] = df["BC"].fillna("").astype(str).map(
        lambda s: canonicalize_bc_seq_sample(s, sample_hint=sample_hint)
    )

    # Keep only barcodes with sufficient frequency
    bc_counts = df["BC"].value_counts()
    keep = set(bc_counts[bc_counts >= int(min_freq)].index)
    df = df[df["BC"].isin(keep)]

    # Collapse duplicate (Read, BC) per genome file
    df = (
        df.groupby(["Read", "BC"], as_index=False)
          .agg(MAPQ=("MAPQ", "max"),
               AS=("AS", "max"),
               NM=("NM", "min"),
               XAcount=("XAcount", "max"))
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    return int(len(df))
