# src/ambientmapper/filtering.py
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Tuple
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

      Pass 2) Re-stream file, keep only rows whose barcode is in keep-set,
              and collapse duplicates by (Read, BC) with:
                MAPQ=max, AS=max, NM=min, XAcount=max.

    Input format (no header), tab-delimited:
      Read  BC  MAPQ  AS  NM  XAcount

    Output format (WITH header), tab-delimited:
      Read  BC  MAPQ  AS  NM  XAcount

    Returns:
      number of unique (Read, BC) pairs written (rows in output, excluding header).

    Notes on memory:
      - Pass 1 stores counts per unique barcode.
      - Pass 2 stores an aggregation dict per unique (Read, BC) that survives filtering.
        If min_freq is too low, this dict can still become large; for extreme datasets,
        implement sharded aggregation or external sort.
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
            out.write("Read\tBC\tMAPQ\tAS\tNM\tXAcount\n")
        return 0

    # Allow GC to reclaim counts dict memory before heavy pass 2
    bc_counts.clear()

    # -------------------------
    # Pass 2: filter + collapse by (Read, BC)
    # -------------------------
    # key: (Read, BC) -> (MAPQ, AS, NM, XAcount)
    agg: Dict[Tuple[str, str], Tuple[int, int, int, int]] = {}

    def _to_int(x: str, default: int) -> int:
        try:
            return int(x)
        except Exception:
            return default

    with in_path.open("r") as f:
        for line in f:
            if not line:
                continue
            line = line.rstrip("\n")
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 6:
                continue

            read = parts[0]
            bc = parts[1]
            if do_norm:
                bc = canonicalize_bc_seq_sample_force(bc or "", sample_name)  # type: ignore[arg-type]

            # Quick reject if BC not kept
            bc = sys.intern(bc)
            if bc not in keep:
                continue

            # Parse metrics
            mapq = _to_int(parts[2], 0)
            alsc = _to_int(parts[3], 0)
            nm = _to_int(parts[4], 10**9)
            xac = _to_int(parts[5], 0)

            # Intern read too (often repeated), helps memory when collapsing duplicates
            read = sys.intern(read)

            key = (read, bc)
            prev = agg.get(key)
            if prev is None:
                agg[key] = (mapq, alsc, nm, xac)
            else:
                pm, pa, pn, px = prev
                # MAPQ=max, AS=max, NM=min, XAcount=max
                if mapq < pm:
                    mapq = pm
                if alsc < pa:
                    alsc = pa
                if nm > pn:
                    nm = pn
                if xac < px:
                    xac = px
                agg[key] = (mapq, alsc, nm, xac)

    # -------------------------
    # Write output
    # -------------------------
    with out_path.open("w") as out:
        out.write("Read\tBC\tMAPQ\tAS\tNM\tXAcount\n")
        for (read, bc), (mapq, alsc, nm, xac) in agg.items():
            out.write(f"{read}\t{bc}\t{mapq}\t{alsc}\t{nm}\t{xac}\n")

    return int(len(agg))
