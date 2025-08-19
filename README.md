# ambientmapper
Ambient contamination cleaning for multi-genome mappings (single-cell, scifi-ATAC)

![tests](https://github.com/gomezcan/ambientmapper/actions/workflows/test.yml/badge.svg)

## Requirements

**OS:** Linux or macOS (Apple Silicon and x86_64 supported)

**Python:** 3.9–3.12 (tested in CI)

**CPU/RAM:** scales with your data; start with ≥8 threads and ≥16–32 GB RAM for comfortable chunking

**Disk:** enough for temporary QC TSVs and the final *.tsv.gz (can be 10s–100s of GB for large runs)

**External tools (optional):**
- samtools (optional) – not required by ambientmapper, but handy for sanity checks
  
**Not required:** GPU/CUDA
Parallelism is controlled by the CLI flag --threads (used for both per-genome and per-chunk steps). Chunk size is set in your sample JSON (chunk_size_cells).

## Install
### Option A — New conda environment (recommended)
```bash
mamba create -n ambientmapper -c conda-forge -c bioconda python=3.11 pip samtools
mamba activate ambientmapper

git clone https://github.com/gomezcan/ambientmapper.git
cd ambientmapper
pip install -e .

# Verify the install
ambientmapper --help
python - <<'PY'
import ambientmapper; print("ambientmapper", getattr(ambientmapper, "__version__", "n/a"))
PY

```
### Option B — Install into an existing conda env
```bash
conda activate <your-env>
# (optional) add channels once per machine
conda config --add channels conda-forge
conda config --add channels bioconda
conda config --set channel_priority strict

# install (from the repo you cloned)
pip install -e /path/to/ambientmapper
# or straight from GitHub
pip install "git+https://github.com/gomezcan/ambientmapper.git"
```

## Use
### one-shot, local, threads=N
```bash
ambientmapper run --config configs/example.json --threads 16
```
this produces:

```
<workdir>/<sample>/
  qc/                         # per-genome QC TSVs
  filtered_QCFiles/           # per-genome filtered/collapsed TSVs
  cell_map_ref_chunks/        # per-chunk winner TSVs
  final/<sample>_per_read_winner.tsv.gz   # merged result
```

### or stepwise
```bash
ambientmapper extract -c configs/example.json -t 8

ambientmapper filter  -c configs/example.json -t 8

ambientmapper chunks  -c configs/example.json

ambientmapper assign  -c configs/example.json -t 16

ambientmapper merge   -c configs/example.json
```



