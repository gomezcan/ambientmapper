from __future__ import annotations
from pathlib import Path
import csv

def bam_to_qc(bam_path: Path, out_path: Path) -> int:
    import pysam  # lazy
    n = 0
    with pysam.AlignmentFile(str(bam_path), "rb") as bam, open(out_path, "w", newline="") as out:
        w = csv.writer(out, delimiter="\t")
        for aln in bam.fetch(until_eof=True):
            if aln.is_unmapped or aln.is_secondary or aln.is_supplementary:
                continue
            read = aln.query_name
            mapq = int(aln.mapping_quality)
            ascore = aln.get_tag("AS") if aln.has_tag("AS") else ""
            nm = aln.get_tag("NM") if aln.has_tag("NM") else ""
            bc = aln.get_tag("CB") if aln.has_tag("CB") else (aln.get_tag("BC") if aln.has_tag("BC") else "")
            xa_count = 0
            if aln.has_tag("XA"):
                xa = aln.get_tag("XA")
                xa_count = xa.count(";") if isinstance(xa, str) else 0
            w.writerow([read, bc, mapq, ascore, nm, xa_count]); n += 1
    return n
