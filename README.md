# ambientmapper
Ambient contamination cleaning for multi-genome mappings (single-cell, scifi-ATAC)

![tests](https://github.com/gomezcan/ambientmapper/actions/workflows/tests.yml/badge.svg)

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
### one-sample, at least two genome (no JSON file needed), threads=N
```bash
ambientmapper run \
  --sample SAMPLE_NAME \
  --genome GENOME1,GENOME2 \
  --bam /path_to_bam1.bam,path_to_bam2.bam \
  --workdir ./ambient_out \
  --min-barcode-freq 10 \
  --chunk-size-cells 5000 \
  --threads 16
```
this produces:

```
<ambient_out>/<SAMPLE_NAME>/
  qc/                         # per-genome QC TSVs
  filtered_QCFiles/           # per-genome filtered/collapsed TSVs
  cell_map_ref_chunks/        # per-chunk winner TSVs
  final/<sample>_per_read_winner.tsv.gz   # merged result
```
***Final columns:*** Read, BC, Genome, AS, MAPQ, NM, delta_AS, is_tie.

### one-sample, at least two genome (JSON mode), threads=N
```
ambientmapper run --config configs/example.json --threads 16

```
### one-sample, many genomes from a TSV (tab-separated), threads=N
#### # TSV columns: sample  genome  bam  workdir
```
ambientmapper run \
  --configs configs/samples.tsv \
  --min-barcode-freq 10 \
  --chunk-size-cells 5000 \
  --threads 16
```

### Quick test
```
# Quick local checks (copy/paste)

```bash
# help should show the new options
ambientmapper run --help

# dry sanity: create two empty dummy BAMs and run inline (just to test arg parsing)
# (the pipeline will start and likely fail earlier because they’re empty,
#  but it proves your flag parsing & file existence checks.)
touch /tmp/a.bam /tmp/b.bam
ambientmapper run \
  --sample TEST \
  --genome G1,G2 \
  --bam /tmp/a.bam,/tmp/b.bam \
  --workdir /tmp/amb_out \
  --threads 2
```

### or stepwise
```bash
ambientmapper extract -c configs/example.json -t 8

ambientmapper filter  -c configs/example.json -t 8

ambientmapper chunks  -c configs/example.json

ambientmapper assign  -c configs/example.json -t 16

## Troubleshooting
- Reduce --threads or chunk_size_cells if you hit RAM limits.
- Prefer fast local/scratch disk for workdir.
- If conda channels cause issues:
```
conda config --add channels conda-forge
conda config --add channels bioconda
conda config --set channel_priority strict

```


ambientmapper merge   -c configs/example.json
```



