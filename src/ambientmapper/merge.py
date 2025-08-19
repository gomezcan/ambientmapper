from __future__ import annotations
from pathlib import Path
import glob

def merge_chunk_outputs(chunks_dir: str | Path, sample: str, out_gz: str | Path) -> int:
    import pandas as pd
    paths = sorted(glob.glob(str(Path(chunks_dir) / f"{sample}_cell_genotype_reads_chunk_*.txt")))
    dfs = [pd.read_csv(p, sep="\t") for p in paths if Path(p).stat().st_size > 0]
    df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame(
        columns=["Read","BC","Genome","AS","MAPQ","NM","delta_AS","is_tie"]
    )
    out = Path(out_gz)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, sep="\t", index=False, compression="gzip")
    return len(df)
