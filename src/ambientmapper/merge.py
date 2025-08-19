from __future__ import annotations
from pathlib import Path

def merge_chunk_outputs(chunks_dir: Path, sample: str, out_path: Path) -> int:
    import pandas as pd
    paths = sorted(chunks_dir.glob(f"{sample}_cell_genotype_reads_chunk_*.txt"))
    dfs = [pd.read_csv(p, sep="\t") for p in paths if p.exists() and p.stat().st_size > 0]
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(
        columns=["Read","BC","Genome","AS","MAPQ","NM","delta_AS","is_tie"]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False, compression="gzip")
    return int(len(df))
