from __future__ import annotations
from pathlib import Path
import pandas as pd
import numpy as np
import glob
from typing import Iterable

def _subset_stream(path: str | Path, bc_set: set[str], genome: str) -> Iterable[pd.DataFrame]:
    for chunk in pd.read_csv(path, sep="\t", chunksize=1_000):
        sub = chunk[chunk["BC"].isin(bc_set)]
        if sub.empty:
            continue
        sub = sub.assign(Genome=genome)
        yield sub[["Read","BC","Genome","AS","MAPQ","NM"]]

def assign_winners_for_chunk(filtered_dir: str | Path, chunk_file: str | Path, out_file: str | Path) -> int:
    bc_set = {ln.strip() for ln in Path(chunk_file).read_text().splitlines() if ln.strip()}
    qcs = sorted(glob.glob(str(Path(filtered_dir) / "filtered_*_QCMapping.txt")))
    rows: list[pd.DataFrame] = []
    for path in qcs:
        genome = Path(path).name.replace("filtered_","").replace("_QCMapping.txt","")
        rows.extend(_subset_stream(path, bc_set, genome))
    big = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["Read","BC","Genome","AS","MAPQ","NM"])
    out_file = Path(out_file)
    if big.empty:
        pd.DataFrame(columns=["Read","BC","Genome","AS","MAPQ","NM","delta_AS","is_tie"]).to_csv(out_file, sep="\t", index=False)
        return 0

    big = big.sort_values(["Read","AS","MAPQ","NM","Genome"], ascending=[True,False,False,True,True])
    top = big.groupby("Read", as_index=False).head(1).copy()

    nxt = (big.groupby("Read", as_index=False)
             .nth(1)[["Read","AS","MAPQ","NM"]]
             .rename(columns={"AS":"AS2","MAPQ":"MAPQ2","NM":"NM2"}))
    top = top.merge(nxt, on="Read", how="left")
    top["delta_AS"] = top["AS"] - top["AS2"].fillna(-np.inf)

    dup_key = big[["Read","AS","MAPQ","NM"]].duplicated(keep=False)
    tied = set(big.loc[dup_key, "Read"])
    top["is_tie"] = top["Read"].isin(tied)

    top[["Read","BC","Genome","AS","MAPQ","NM","delta_AS","is_tie"]].to_csv(out_file, sep="\t", index=False)
    return len(top)
