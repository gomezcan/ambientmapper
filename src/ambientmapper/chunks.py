from __future__ import annotations
from pathlib import Path
import pandas as pd
import glob

def make_barcode_chunks(filtered_dir: str | Path, out_dir: str | Path, sample: str, chunk_size: int) -> int:
    paths = sorted(glob.glob(str(Path(filtered_dir) / "filtered_*_QCMapping.txt")))
    if not paths:
        return 0
    bc_all = set()
    for p in paths:
        for c in pd.read_csv(p, sep="\t", usecols=["BC"], chunksize=1_000):
            bc_all.update(c["BC"].dropna().astype(str))
    bc_all = sorted(bc_all)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    n_chunks = 0
    for i in range(0, len(bc_all), chunk_size):
        n_chunks += 1
        with open(Path(out_dir) / f"{sample}_cell_map_ref_chunk_{n_chunks}.txt", "w") as f:
            f.write("\n".join(bc_all[i:i+chunk_size]))
    return n_chunks
