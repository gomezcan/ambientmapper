# src/ambientmapper/filtering.py
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional
import sys

from .normalization import canonicalize_bc_seq_sample_force


def filter_qc_file(
    in_path: Path,
    out_path: Path,
    min_freq: int,
    sample_name: Optional[str] = None,
) -> int:
    """
    Stream QCMapping file in two passes:

      Pass 1) Count barcodes (after optional normalization to '<seq>-<sample>'),
              build keep-set of barcodes with count >= min_freq.

      Pass 2) Re-stream file, emit rows whose barcode is in keep-set
              directly to output. No in-memory aggregation.

    Input format (no header), tab-delimited:
      Read  BC  MAPQ  AS  NM  XAcount  [frag_loc]

    Output format (WITH header), tab-delimited:
      Read  BC  MAPQ  AS  NM  XAcount  frag_loc

    Returns:
      total rows written (excluding header). May include duplicate
      (Read, BC) pairs from input; downstream stages (assign, chunks)
      handle dedup independently.

    Memory: O(unique_barcodes) for the keep-set, typically 100-200 MB.
    """
    in_path = Path(in_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    min_freq = int(min_freq)

    # -------------------------
    # Pass 1: count (normalized) barcodes
    # -------------------------
    bc_counts: Dict[str, int] = defaultdict(int)

    # Small optimization: if sample_name is provided, normalization is needed;
    # otherwise, we assume BC is already in desired form.
    do_norm = bool(sample_name)

    with in_path.open("r") as f:
        for line in f:
            if not line:
                continue
            line = line.rstrip("\n")
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                continue

            bc = parts[1]
            if do_norm:
                # canonicalize handles empty/malformed; keep it robust
                bc = canonicalize_bc_seq_sample_force(bc or "", sample_name)  # type: ignore[arg-type]

            # intern to reduce memory if many repeats
            bc = sys.intern(bc)
            bc_counts[bc] += 1

    keep = {bc for bc, c in bc_counts.items() if c >= min_freq}

    if not keep:
        # Write empty file with header, matching previous behavior
        with out_path.open("w") as out:
            out.write("Read\tBC\tMAPQ\tAS\tNM\tXAcount\tfrag_loc\n")
        return 0

    # Allow GC to reclaim counts dict memory before heavy pass 2
    bc_counts.clear()

    # -------------------------
    # Pass 2: stream-filter and emit
    # -------------------------
    def _to_int(x: str, default: int) -> int:
        try:
            return int(x)
        except Exception:
            return default

    n_written = 0

    with in_path.open("r") as f, out_path.open("w") as out:
        out.write("Read\tBC\tMAPQ\tAS\tNM\tXAcount\tfrag_loc\n")

        for line in f:
            if not line:
                continue
            line = line.rstrip("\n")
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 6:
                continue

            bc = parts[1]
            if do_norm:
                bc = canonicalize_bc_seq_sample_force(bc or "", sample_name)  # type: ignore[arg-type]

            if bc not in keep:
                continue

            # Sanitize metrics (robustness against malformed values)
            mapq = _to_int(parts[2], 0)
            alsc = _to_int(parts[3], 0)
            nm   = _to_int(parts[4], 10**9)
            xac  = _to_int(parts[5], 0)
            frag_loc = parts[6] if len(parts) >= 7 else ""

            out.write(f"{parts[0]}\t{bc}\t{mapq}\t{alsc}\t{nm}\t{xac}\t{frag_loc}\n")
            n_written += 1

    return n_written
