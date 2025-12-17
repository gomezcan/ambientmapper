# ambientmapper/utils.py

from __future__ import annotations

import csv
from typing import Dict, List, Tuple, Optional
from pathlib import Path


import pandas as pd


class PlateDesign:
    """
    Resolve expected genome per barcode using:
      - a Tn5 layout (well -> genome or well -> sample)
      - a design file (plate/well/sample design)

    For now this is a simple stub that expects the design file itself
    to already contain a mapping: barcode -> expected_genome.

    You can expand this later to actually use the 96-well layout.
    """

    def __init__(self, layout_path: Path, design_path: Path):
        self.layout_path = Path(layout_path)
        self.design_path = Path(design_path)
        self._bc_to_genome: Dict[str, str] = self._load_design()

    def _load_design(self) -> Dict[str, str]:
        """
        Expect design TSV with at least:
            barcode    expected_genome

        Return dict: barcode -> expected_genome
        """
        df = pd.read_csv(self.design_path, sep="\t")
        expected_cols = {"barcode", "expected_genome"}
        missing = expected_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"Design file {self.design_path} missing columns: {', '.join(sorted(missing))}"
            )
        return dict(zip(df["barcode"], df["expected_genome"]))

    def get_expected_genome(self, barcode: str) -> Optional[str]:
        return self._bc_to_genome.get(barcode)



def load_sample_to_wells(file_path: Path | str) -> Dict[str, List[str]]:
    """
    Parse a 'Pool<TAB>Ranges' text file into {sample: [well_id, ...]}.

    Accepts ranges like 'A1-12,B1-12' (case-insensitive rows).
    """
    sample_to_wells: Dict[str, List[str]] = {}
    with open(file_path, "r") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            sample, ranges_str = line.split("\t", 1)
            wells = sample_to_wells.setdefault(sample.strip(), [])
            for r in ranges_str.split(","):
                r = r.strip().replace(" ", "")
                if not r:
                    continue
                row = r[0].upper()
                rest = r[1:]
                if "-" in rest:
                    start_s, end_s = rest.split("-", 1)
                else:
                    start_s = end_s = rest
                start, end = int(start_s), int(end_s)
                if start > end:
                    raise ValueError(f"Start > end in '{r}'")
                wells.extend(f"{row}{i}" for i in range(start, end + 1))
    return sample_to_wells


def load_barcode_layout(layout_file: Path | str) -> Dict[str, str]:
    """
    Load a 96-well Tn5 barcode layout TSV into {WellID -> split_bc}.
    Expects first row to be column headers, first column to be row letters.
    """
    well_to_barcode: Dict[str, str] = {}
    with open(layout_file, "r") as f:
        reader = csv.reader(f, delimiter="\t")
        col_names = next(reader)[1:]
        for row in reader:
            row_label = row[0]
            for i, bc in enumerate(row[1:]):
                well_to_barcode[f"{row_label}{col_names[i]}"] = bc
    return well_to_barcode


def build_well_to_sample(sample_to_wells: Dict[str, List[str]]) -> Dict[str, str]:
    """Invert sample→wells to well→sample."""
    return {w: s for s, wells in sample_to_wells.items() for w in wells}


def build_barcode_to_sample(
    well_to_barcode: Dict[str, str],
    well_to_sample: Dict[str, str],
) -> Dict[str, str]:
    """Map split barcodes to sample names using well→barcode + well→sample."""
    return {bc: well_to_sample[well] for well, bc in well_to_barcode.items() if well in well_to_sample}
