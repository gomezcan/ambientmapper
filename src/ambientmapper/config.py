from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json

@dataclass
class SampleConfig:
    sample: str
    genomes: dict  # {genome_name: bam_path}
    min_barcode_freq: int
    chunk_size_cells: int
    workdir: str

    @classmethod
    def load(cls, path: str | Path) -> "SampleConfig":
        d = json.loads(Path(path).read_text())
        return cls(**d)

    # convenient paths
    def dir_root(self) -> Path: return Path(self.workdir) / self.sample
    def dir_qc(self) -> Path: return self.dir_root() / "qc"
    def dir_filtered(self) -> Path: return self.dir_root() / "filtered_QCFiles"
    def dir_chunks(self) -> Path: return self.dir_root() / "cell_map_ref_chunks"
    def dir_final(self) -> Path: return self.dir_root() / "final"

    def ensure_dirs(self) -> None:
        for p in [self.dir_qc(), self.dir_filtered(), self.dir_chunks(), self.dir_final()]:
            p.mkdir(parents=True, exist_ok=True)
