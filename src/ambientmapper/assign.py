from __future__ import annotations
from pathlib import Path

def assign_winners_for_chunk(filtered_dir: Path, chunk_file: Path, out_file: Path) -> int:
    import pandas as pd, numpy as np
    bc = {ln.strip() for ln in chunk_file.read_text().splitlines() if ln.strip()}
    qcs = sorted(filtered_dir.glob("filtered_*_QCMapping.txt"))
    rows=[]
    for path in qcs:
        genome = path.name.replace("filtered_","").replace("_QCMapping.txt","")
        for c in pd.read_csv(path, sep="\t", chunksize=1_000_000):
            sub = c[c["BC"].isin(bc)]
            if sub.empty: continue
            sub = sub.assign(Genome=genome)
            rows.append(sub[["Read","BC","Genome","AS","MAPQ","NM"]])
    big = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["Read","BC","Genome","AS","MAPQ","NM"])
    if big.empty:
        pd.DataFrame(columns=["Read","BC","Genome","AS","MAPQ","NM","delta_AS","is_tie"]).to_csv(out_file, sep="\t", index=False)
        return 0
    big = big.sort_values(["Read","AS","MAPQ","NM","Genome"], ascending=[True,False,False,True,True])
    top = big.groupby("Read", as_index=False).head(1).copy()
    nxt = (big.groupby("Read", as_index=False).nth(1)[["Read","AS","MAPQ","NM"]]
              .rename(columns={"AS":"AS2","MAPQ":"MAPQ2","NM":"NM2"}))
    top = top.merge(nxt, on="Read", how="left")
    top["delta_AS"] = top["AS"] - top["AS2"].fillna(-np.inf)
    dup_key = big[["Read","AS","MAPQ","NM"]].duplicated(keep=False)
    tied = set(big.loc[dup_key,"Read"])
    top["is_tie"] = top["Read"].isin(tied)
    top[["Read","BC","Genome","AS","MAPQ","NM","delta_AS","is_tie"]].to_csv(out_file, sep="\t", index=False)
    return int(len(top))
