from __future__ import annotations
from pathlib import Path
from typing import Union
from .normalization import canonicalize_bc_seq_sample

USECOLS = ["Read", "BC", "MAPQ", "AS", "NM", "XAcount"]

def filter_qc_file(in_path: Path, out_path: Path, min_freq: int, sample_hint: str | None = None) -> int:
    df = pd.read_csv(in_path, sep="\t", names=["Read","BC","MAPQ","AS","NM","XAcount"], header=None)
    # Normalize BCs here, too (covers pre-existing QC files)
    df["BC"] = df["BC"].astype(str).map(lambda s: canonicalize_bc_seq_sample(s, sample_hint=sample_hint))

    keep = set(df["BC"].value_counts()[lambda s: s >= int(min_freq)].index)
    df = df[df["BC"].isin(keep)]
    df = (df.groupby(["Read","BC"], as_index=False)
            .agg(MAPQ=("MAPQ","max"), AS=("AS","max"), NM=("NM","min"), XAcount=("XAcount","max")))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    return len(df)
