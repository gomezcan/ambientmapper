# ambientmapper
Ambient contamination cleaning for multi-genome mapping (single-cell, scifi-ATAC)

![tests](https://github.com/gomezcan/ambientmapper/actions/workflows/tests.yml/badge.svg)

## Requirements

**OS:** Linux or macOS (Apple Silicon and x86_64 supported)\
**Python:** 3.9–3.12 (tested in CI)\
**CPU/RAM:** scales with your data; start with ≥8 threads and ≥16–32 GB RAM for comfortable chunking\
**Disk:** enough for temporary QC TSVs and the final *.tsv.gz (can be 10s–100s of GB for large runs)\
**External tools (optional):**\
  -samtools (optional) – not required by ambientmapper, but handy for sanity checks\
**Not required:** GPU/CUDA\
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

## Output directory layout
all outputs live under: 

```perl
<workdir>/<sample>/
├─ qc/                              # per-genome raw QC
├─ filtered_QCFiles/                # per-genome filtered/collapsed QC
├─ cell_map_ref_chunks/             # barcode chunks + per-chunk winners
├─ final/<sample>_per_read_winner.tsv.gz
├─ Plots/<sample>_contamination_correction_summary.pdf
├─ <sample>_BCs_PASS_by_mapping.csv           # BCs assigned to in-pool genome
└─ Reads_to_discard_<sample>.csv              # reads where winner ≠ assigned genome
```

## Running the pipeline
You can run **end-to-end** (*extract → filter → chunks → assign → merge*) and optionally **summarize**.

### One Step. Mode A — Inline one sample
```bash
ambientmapper run \
  --sample B73Mo17_rep1 \
  --genome B73,Mo17 \
  --bam /data/SC1_P1/B73.bam,/data/B73Mo17_rep1/Mo17.bam \
  --workdir /scratch/ambient_out \
  --min-barcode-freq 5 \
  --chunk-size-cells 1000 \
  --threads 16 \
  --with-summary \
  --xa-max 2
```

### One Step. Mode B — JSON config
configs/example.json
```json
{
  "sample": "B73Mo17_rep1",
  "genomes": {
    "B73":  "/data/B73Mo17_rep1/B73Mo17_rep1_A1_B73_scifiATAC.mq20.BC.rmdup.mm.bam",
    "Mo17": "/data/B73Mo17_rep1/B73Mo17_rep1_A1_Mo17_scifiATAC.mq20.BC.rmdup.mm.bam"
  },
  "min_barcode_freq": 10,
  "chunk_size_cells": 1000,
  "workdir": "/scratch/ambient_out"
}
```
run
```bash
ambientmapper run --config configs/SC1_P1.json \
  --threads 16 \
  --with-summary \
  --xa-max 2

```

### One Step. Mode C — TSV for many samples
configs/samples.tsv (tab-separated; one row per **(sample, genome)**)

```ts
sample        genome  bam                         workdir
B73Mo17_rep1  B73     /data/SC1_P1/B73.bam       /scratch/ambient_out
SB73Mo17_rep1 Mo17    /data/SC1_P1/Mo17.bam      /scratch/ambient_out
B73Mo17_rep1  B73     /data/SC1_P2/B73.bam       /scratch/ambient_out
B73Mo17_rep1  Mo17    /data/SC1_P2/Mo17.bam      /scratch/ambient_out
```
run
```bash
ambientmapper run \
  --configs configs/samples.tsv \
  --min-barcode-freq 10 \
  --chunk-size-cells 1000 \
  --threads 16 \
  --with-summary \
  --xa-max 2
```

## Two steps (recommended): per-pool then inter-pool

If your library is split into **pools** (e.g. plates/wells with subsets of genotypes), treat **each pool as its own sample.**
Run each pool end-to-end; the summary treats the genomes listed for that sample as the **in-pool** set.
Then, optionally, run an **inter-pool** comparison across multiple pools.


### 1) Per-pool run (one pool = one sample)
```bash
ambientmapper run \
  --sample B73Mo17_A \
  --genome B73,Mo17 \
  --bam /bam/B73.bam,/bam/Mo17.bam \
  --workdir ./ambient_out \
  --min-barcode-freq 10 \
  --chunk-size-cells 5000 \
  --threads 16 \
  --with-summary \
  --xa-max 2
```
Repeat for other pools (e.g. B73Mo17_A, B73Mo17_B, …).

### 2) Inter-pool comparison (after multiple pools are run)
Use a TSV listing all pool runs. **Headers must be lowercase** and include these columns:

configs/samples.tsv

```tsv
sample      genome   bam                 workdir
B73Mo17_A   B73     /bam/B73.bam        ./ambient_out
B73Mo17_A   Mo17    /bam/Mo17.bam       ./ambient_out
B73Mo17_B   B73     /bam/B73.bam        ./ambient_out
B73Mo17_B   W22     /bam/W22.bam        ./ambient_out
```
Note: Each sample line corresponds to one genome BAM mapping for that pool.

Now aggregate:
```bash
ambientmapper interpool \
  --configs configs/samples.tsv \
  --outdir ./ambient_out/interpool
```

This writes:
- interpool_bc_counts.tsv — #BCs per AssignedGenome × sample\
- interpool_read_composition.tsv — winner-read fractions per Genome × sample\
- interpool_summary.pdf — two heatmaps (BC counts and read composition)\

### No pool design?
If you don’t split by pools, just run per-sample. The summary treats the sample as **one big pool** and splits BCs into **LowContam vs HighContam** using contamination rate (default threshold *0.20*).


## dry sanity: 
create two empty dummy BAMs and run inline (just to test arg parsing).\ 
The pipeline will start and likely fail earlier because they’re empty,\
but it proves your flag parsing & file existence checks.

```bash
touch /tmp/a.bam /tmp/b.bam
ambientmapper run \
  --sample TEST \
  --genome G1,G2 \
  --bam /tmp/a.bam,/tmp/b.bam \
  --workdir /tmp/amb_out \
  --threads 2
```

### Step-by-step (debugging)
```bash
ambientmapper extract -c configs/example.json -t 8
ambientmapper filter  -c configs/example.json -t 8
ambientmapper chunks  -c configs/example.json
ambientmapper assign  -c configs/example.json -t 16
ambientmapper merge  -c configs/example.json -t 16

# after ambientmapper run / merge
ambientmapper summarize -c configs/example.json --xa-max 2
# (add --pool-design configs/pools.tsv if you have pools)**
```

#### Note on stepwise commands ####
The step-by-step subcommands (`extract`, `filter`, `chunks`, `assign`, `merge`, etc.) currently accept only **Mode A** (`--config JSON`).
If you prefer inline arguments (Mode B) or TSV (Mode C), please use the top-level `ambientmapper run` command, which supports all three modes.
Advanced parameter tuning (e.g. `alpha`, `k`, `assign.chunksize`) should be specified inside the JSON config under the "assign" block, and implemeted on the top-level options. 
  


### What the steps do (brief)

- **extract**
  From each genome BAM → per-read QC table with columns: `Read, BC, MAPQ, AS, NM, XAcount.`

- **filter**
  Keep barcodes (BCs) with frequency `≥ --min-barcode-freq.`
  Collapse duplicate (`Read`, `BC`) rows per genome using:
  - `MAPQ = max, AS = max, NM = min, XAcount = max.`

- **chunks**
  Build BC chunk files to cap memory (size controlled by `--chunk-size-cells`).

- **assign**
  Modular assign: learn edges (global), learn ECDFs (global), then score each chunk (parallel).
  - Uses AS, then MAPQ (desc), then NM (asc).
    Writes remove clear low quality aligments per-read
  - Note: `ambientmapper-assign` exposes low-level developer commands (learn-edges, learn-ecdfs, score-chunk) for debugging or advanced use.

- **merge**
  Combine per-chunk winners `→ final/<sample>_per_read_winner.tsv.gz.`

- **summarize** (*optional; per-pool*)
  Treat the current sample as **one pool**.
  - Compute assigned genome per BC (by AS_mean across that BC’s reads).
  - Contamination rate per BC = fraction of winner reads whose Genome ≠ AssignedGenome.
  - Produce a PDF with ECDFs (AS_mean, reads/BC), a contamination-vs-reads hexbin,\and counts (heatmap if multiple pools are provided; bar chart otherwise).
  - Write:
    - `<sample>_BCs_PASS_by_mapping.csv` (BCs whose assigned genome is considered in-pool for this run)
    - `Reads_to_discard_<sample>.csv` (winner reads where Genome ≠ AssignedGenome for that BC)

- **interpool** (*optional; cross-pool comparison after multiple per-pool runs*)
  Aggregate several samples (pools) to compare:
  - `interpool_bc_counts.tsv` — #BCs per **AssignedGenome × sample**
  - `interpool_read_composition.tsv` — winner-read **fractions per Genome × sample**
  - `interpool_summary.pdf` — two heatmaps (BC counts, read composition)

## Notes
  - BAMs should be coordinate-sorted and indexed; barcodes in CB or BC; alignment tags AS, NM expected; XA optional.
  - --xa-max 2 filters winners by XAcount ≤ 2. If your BAMs lack XA, set --xa-max -1.
  - --threads parallelizes per-genome (extract/filter) and per-chunk (assign). Adjust --chunk-size-cells to manage memory.

## Troubleshooting
- If you hit RAM limits:
    - Lower `--threads`
    - Lower `chunk_size_cells` (fewer cells per chunk).
    - Lower `assign.chunksize` (default 500,000 rows per read_csv call).

- Prefer fast local/scratch disk for workdir.
- If conda channels cause issues:
```
conda config --add channels conda-forge
conda config --add channels bioconda
conda config --set channel_priority strict
```
