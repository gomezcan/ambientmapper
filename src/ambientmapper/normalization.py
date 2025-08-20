from __future__ import annotations
from typing import Optional

def canonicalize_bc_seq_sample(raw: str, sample_hint: Optional[str] = None) -> str:
    """
    Normalize a barcode string to '<seq>-<sample>'.

    Input examples:
      'SEQ-B73Mo17_rep1_A1_B73v5_scifiATAC' -> 'SEQ-B73Mo17_rep1'
      'SEQ-B73Mo17_rep1'                    -> 'SEQ-B73Mo17_rep1'
      'SEQ'                                 -> 'SEQ' (no '-')

    Heuristic:
      - Split at the first '-' into LEFT (seq) and RIGHT.
      - If RIGHT starts with sample_hint (if provided), keep that sample_hint.
      - Else take the first '_' token from RIGHT as the sample token.
      - Return LEFT + '-' + sample.
    """
    if not isinstance(raw, str) or not raw:
        return ""
    if "-" not in raw:
        return raw

    left, right = raw.split("-", 1)
    right = right.strip("_")

    if sample_hint:
        # exact prefix match (case sensitive by default; change if needed)
        if right.startswith(sample_hint):
            return f"{left}-{sample_hint}"

    # default: first underscore-delimited token is the sample
    toks = [t for t in right.split("_") if t]
    sample = toks[0] if toks else ""
    return f"{left}-{sample}" if sample else left
