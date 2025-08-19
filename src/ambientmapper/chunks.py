from __future__ import annotations
from pathlib import Path

def make_barcode_chunks(filtered_dir: Path, chunks_dir: Path, sample: str, chunk_size: int) -> int:
    import pandas as pd
    chunks_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(filtered_dir.glob("filtered_*_QCMapping.txt"))
    if not paths:
        return 0
    bc = set()
    for p in paths:
        for c in pd.read_csv(p, sep="\t", usecols=["BC"], chunksize=1_000_000):
            bc.update(c["BC"].dropna().astype(str))
    bc = sorted(bc)
    n = int(chunk_size)
    k = 0
    for i in range(0, len(bc), n):
        k += 1
        (chunks_dir / f"{sample}_cell_map_ref_chunk_{k}.txt").write_text("\n".join(bc[i:i+n]))
    return k
