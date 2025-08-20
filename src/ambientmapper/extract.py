# src/ambientmapper/extract.py
from pathlib import Path
import csv, pysam
from .normalization import canonicalize_bc_seq_sample_force


def bam_to_qc(bam_path: Path, out_path: Path, sample_hint: str | None = None):
    """
    Write per-read QC with normalized BC = '<seq>-<sample_name>'.
    """
    with pysam.AlignmentFile(str(bam_path), "rb") as bam, open(out_path, "w", newline="") as out:
        w = csv.writer(out, delimiter="\t")
        for aln in bam.fetch(until_eof=True):
            if aln.is_unmapped or aln.is_secondary or aln.is_supplementary:
                continue
            read = aln.query_name
            mapq = int(aln.mapping_quality)
            ascore = aln.get_tag("AS") if aln.has_tag("AS") else ""
            nm = aln.get_tag("NM") if aln.has_tag("NM") else ""
            bc_raw = ""
            if aln.has_tag("CB"): bc_raw = aln.get_tag("CB")
            elif aln.has_tag("BC"): bc_raw = aln.get_tag("BC")
            # NEW: normalize to seq-sample
            bc = canonicalize_bc_seq_sample_force(bc_raw, sample_name)

            xa_count = 0
            if aln.has_tag("XA"):
                xa = aln.get_tag("XA")
                xa_count = xa.count(";") if isinstance(xa, str) else 0

            w.writerow([read, bc, mapq, ascore, nm, xa_count])
