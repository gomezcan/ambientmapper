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

## Output diractory layout
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

### Mode A — Inline one sample
```bash
ambientmapper run \
  --sample SC1_P1 \
  --genome B73,Mo17 \
  --bam /data/SC1_P1/B73.bam,/data/SC1_P1/Mo17.bam \
  --workdir /scratch/ambient_out \
  --min-barcode-freq 5 \
  --chunk-size-cells 1000 \
  --threads 16 \
  --with-summary \
  --xa-max 2
```

### Mode B — JSON config
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

### Mode C — TSV for many samples
configs/samples.tsv (tab-separated; one row per **(sample, genome)**)

```ts
sample  genome  bam                         workdir
SC1_P1  B73     /data/SC1_P1/B73.bam       /scratch/ambient_out
SC1_P1  Mo17    /data/SC1_P1/Mo17.bam      /scratch/ambient_out
SC1_P2  B73     /data/SC1_P2/B73.bam       /scratch/ambient_out
SC1_P2  Mo17    /data/SC1_P2/Mo17.bam      /scratch/ambient_out
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

## Pool design (optional)
If your library is split into **pools** (e.g., plates/wells with subsets of genotypes), provide a TSV so summary plots compare **InPool vs Not_InPool.**

configs/pools.tsv
```tsv
Genome  Pool    Plate
B73     P01_A    P01
Mo17    P02_B    P01
```
Use:

```bash
ambientmapper run --config configs/SC1_P1.json \
  --with-summary \
  --pool-design configs/pools.tsv
```
If **no pool design** is given, the sample is treated as **one big pool** with all listed genomes. The summary splits BCs into **LowContam vs HighContam** based on contamination rate (default threshold 0.20).

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


## What the steps do (brief)
- extract: from each genome BAM → per-read QC table (Read, BC, MAPQ, AS, NM, XAcount).
- filter: keep BCs with frequency ≥ --min-barcode-freq; collapse duplicates per (Read, BC) by MAPQ=max, AS=max, NM=min, XAcount=max).
-chunks: build BC chunk files to limit memory.
- assign: for reads in each chunk, pick the genome with best score (AS, MAPQ, NM) → per-read winners.
- merge: combine per-chunk winners → <sample>_per_read_winner.tsv.gz.
- summarize (optional):
  - per-BC assigned genome (by AS_mean),
  - contamination rate per BC = fraction of winner reads whose genome ≠ assigned genome,
  - PDF with ECDFs, hexbin, and counts (heatmap if there are multiple pools; bar chart otherwise),
  - *_BCs_PASS_by_mapping.csv (BCs assigned to in-pool genome)
  - Reads_to_discard_*.csv (reads whose winner genome ≠ that BC’s assigned genome).

## Notes
  - BAMs should be coordinate-sorted and indexed; barcodes in CB or BC; alignment tags AS, NM expected; XA optional.\
  -  --xa-max 2 filters winners by XAcount ≤ 2. If your BAMs lack XA, set --xa-max -1.\
  -  --threads parallelizes per-genome (extract/filter) and per-chunk (assign). Adjust --chunk-size-cells to manage memory.\

## Troubleshooting
- Reduce --threads or chunk_size_cells if you hit RAM limits.
- Prefer fast local/scratch disk for workdir.
- If conda channels cause issues:
```
conda config --add channels conda-forge
conda config --add channels bioconda
conda config --set channel_priority strict
```
