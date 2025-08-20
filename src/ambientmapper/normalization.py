from __future__ import annotations
from typing import Optional
import re

def canonicalize_bc_seq_sample_force(raw: str, sample: str) -> str:
    """
    Force a barcode string to '<seq>-<sample>', using the *provided* sample name.
    Example:
      'SEQ-B73Mo17_rep1_A1_B73v5_scifiATAC' -> 'SEQ-B73Mo17_rep1'  (if sample='B73Mo17_rep1')
      'SEQ'                                 -> 'SEQ-B73Mo17_rep1'
    """
    if not isinstance(raw, str) or not raw:
        return ""
    seq = raw.split("-", 1)[0]  # left side before the first '-'
    sample_clean = re.sub(r"\s+", "_", str(sample).strip())  # sanitize spaces
    return f"{seq}-{sample_clean}"
