from __future__ import annotations
from pathlib import Path
from typing import Union

USECOLS = ["Read", "BC", "MAPQ", "AS", "NM", "XAcount"]

def filter_qc_file(in_path: Union[str, Path], out_path: Union[str, Path], min_barcode_freq: int) -> int:
    import pandas as pd
    df = pd.read_csv(in_path, sep="\t", names=USECOLS, header=None,
                     dtype={"Read": "string", "BC": "string"}, low_memory=False)
    for col in ("MAPQ","AS","NM","XAcount"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["BC"].notna() & (df["BC"].str.len() > 0)]
    keep = set(df["BC"].value_counts()[lambda s: s >= int(min_barcode_freq)].index)
    if not keep:
        pd.DataFrame(columns=USECOLS).to_csv(out_path, sep="\t", index=False)
        return 0
    df = df[df["BC"].isin(keep)]
    out = (df.groupby(["Read","BC"], as_index=False)
             .agg(MAPQ=("MAPQ","max"), AS=("AS","max"), NM=("NM","min"), XAcount=("XAcount","max")))
    out.to_csv(out_path, sep="\t", index=False)
    return int(len(out))
