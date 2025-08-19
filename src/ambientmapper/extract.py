from __future__ import annotations
from pathlib import Path
import csv


def bam_to_qc(bam_path: str | Path, out_path: str | Path) -> None:
    import pysam
    bam_path, out_path = str(bam_path), str(out_path)
    with pysam.AlignmentFile(bam_path, "rb") as bam, open(out_path, "w", newline="") as out:
        w = csv.writer(out, delimiter="\t")
        # columns: Read, BC, MAPQ, AS, NM, XAcount
        for aln in bam.fetch(until_eof=True):
            if aln.is_unmapped or aln.is_secondary or aln.is_supplementary:
                continue
            read = aln.query_name
            mapq = int(aln.mapping_quality)
            ascore = aln.get_tag("AS") if aln.has_tag("AS") else ""
            nm = aln.get_tag("NM") if aln.has_tag("NM") else ""
            bc = ""
            if aln.has_tag("CB"):
                bc = aln.get_tag("CB")
            elif aln.has_tag("BC"):
                bc = aln.get_tag("BC")
            xa_count = 0
            if aln.has_tag("XA"):
                xa = aln.get_tag("XA")  # semicolon-separated alt hits
                xa_count = xa.count(";") if isinstance(xa, str) else 0
            w.writerow([read, bc, mapq, ascore, nm, xa_count])
