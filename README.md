# ambientmapper

Ambient contamination–aware multi-genome genotyping for single-cell libraries (scifi-ATAC, sci-ATAC, 10x-style, etc.)

![tests](https://github.com/gomezcan/ambientmapper/actions/workflows/tests.yml/badge.svg)

`ambientmapper` takes reads mapped to multiple reference genomes and:

1. Normalizes barcodes across genomes.
2. Builds ECDF-based models of winner-vs-runner-up score deltas (AS / MAPQ / NM).
3. Scores each read across all genomes, including a dominance filter for locally ambiguous alignments.
4. Computes per-read posteriors L(r,g) and estimates an ambient profile across genomes.
5. Performs per-barcode model selection (single vs doublet vs ambiguous) under an ambient-aware likelihood.
6. Produces a read-level drop list and cleaned BAM files with ambient-assigned reads removed.

---

## Table of Contents

- [Full pipeline overview](#full-pipeline-overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Running the pipeline](#running-the-pipeline)
  - [Mode A — Inline, one sample](#mode-a--inline-one-sample)
  - [Mode B — JSON config per sample](#mode-b--json-config-per-sample)
  - [Mode C — TSV for many samples](#mode-c--tsv-for-many-samples)
  - [Resumable runs and targeted re-runs](#resumable-runs-and-targeted-re-runs)
- [Step-by-step reference](#step-by-step-reference)
  - [1. extract](#1-extract--per-genome-qc-tables)
  - [2. filter](#2-filter--barcode-filtering--duplicate-collapse)
  - [3. chunks](#3-chunks--barcode-chunking)
  - [4. assign](#4-assign--ecdf-based-dominance-filter--per-read-winners)
  - [5. genotyping](#5-genotyping--ambient-aware-per-cell-calls)
- [Decontamination](#decontamination)
  - [decontam — build drop list and barcode policy](#decontam--build-drop-list-and-barcode-policy)
  - [clean-bams — filter BAM files using drop list](#clean-bams--filter-bam-files-using-drop-list)
- [Quality control and cross-pool analysis](#quality-control-and-cross-pool-analysis)
  - [summary — per-pool QC report (Python API)](#summary--per-pool-qc-report-python-api)
  - [interpool — cross-pool BC count and read composition](#interpool--cross-pool-bc-count-and-read-composition)
  - [plate — 96-well plate heatmap](#plate--96-well-plate-heatmap)
- [Output directory layout](#output-directory-layout)
- [Advanced parameter tuning](#advanced-parameter-tuning)
  - [Global run options](#global-run-options)
  - [Assign parameters](#assign-parameters)
  - [Genotyping parameters](#genotyping-parameters)
  - [Decontam parameters](#decontam-parameters)
  - [Environment variables](#environment-variables)
- [Troubleshooting / FAQ](#troubleshooting--faq)
- [Contributing and development](#contributing-and-development)

---

## Full pipeline overview

```
Core pipeline (ambientmapper run):
  extract → filter → chunks → assign → genotyping

Decontamination (run separately after genotyping):
  decontam → clean-bams

Optional QC utilities (run separately; summary is Python API only):
  summary | interpool | plate
```

---

## Requirements

**OS:** Linux or macOS (Apple Silicon and x86_64 tested in CI)

**Python:** 3.9–3.12

**CPU/RAM:**

* Scales with your data and number of genomes.
* For typical scifi-ATAC pools with 5–10 genomes:

  * Start with ≥8 threads and ≥16–32 GB RAM.
  * Use smaller `chunk_size_cells` and `assign.chunksize` if you hit memory limits.

**Disk:**

* Needs space for:

  * Per-genome QC TSVs (`qc/`, `filtered_QCFiles/`),
  * Chunk barcode lists and assign outputs,
  * Genotyping outputs under `final/`.
* Expect 10s–100s of GB for large experiments.

**External tools (optional):**

* `samtools` – not required by `ambientmapper` itself, but useful to prepare BAMs and for sanity checks.
* `pysam` is installed as a Python dependency. On Apple Silicon, prefer the conda-forge/bioconda build (`mamba install -c bioconda pysam`) over the pip build to avoid htslib compilation issues.

**Not required:** GPU / CUDA.

**BAM expectations:**

* Coordinate-sorted and indexed BAMs, one per genome.
* Per-read tags:

  * Barcodes in `CB` or `BC`.
  * Alignment score: `AS`.
  * Edit distance / mismatches: `NM`.
  * Mapping quality: `MAPQ`.
  * Optional multi-hit tag: `XA` (used to count alternative alignments).
* Reads from all genomes must share the same **raw barcode sequence**; `ambientmapper` will normalize barcodes across genomes.

---

## Installation

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

# install from local clone
pip install -e /path/to/ambientmapper
# or directly from GitHub
pip install "git+https://github.com/gomezcan/ambientmapper.git"
```

### Uninstall

```bash
pip uninstall -y ambientmapper
```

---

## Quick start

End-to-end example for a two-genome pool:

```bash
# 1. Run the core pipeline
ambientmapper run \
  --sample B73Mo17_rep1 \
  --genome B73,Mo17 \
  --bam /data/B73.bam,/data/Mo17.bam \
  --workdir /scratch/out \
  --threads 16

# 2. Build the drop list and barcode policy
ambientmapper decontam \
  --cells-calls /scratch/out/B73Mo17_rep1/final/B73Mo17_rep1_cells_calls.tsv.gz \
  --out-dir /scratch/out/B73Mo17_rep1/decontam/ \
  --ambiguous-policy top1_rescue \
  --threads 4

# 3. Clean the BAMs
ambientmapper clean-bams \
  --reads-to-drop /scratch/out/B73Mo17_rep1/decontam/B73Mo17_rep1_reads_to_drop.tsv.gz \
  --bam /data/B73.bam \
  --out-dir /scratch/out/B73Mo17_rep1/clean_bams/
```

---

## Running the pipeline

### Mode A — Inline, one sample

```bash
ambientmapper run \
  --sample B73Mo17_rep1 \
  --genome B73,Mo17 \
  --bam /data/B73Mo17_rep1/B73.bam,/data/B73Mo17_rep1/Mo17.bam \
  --workdir /scratch/ambient_out \
  --min-barcode-freq 10 \
  --chunk-size-cells 5000 \
  --threads 16 \
  --assign-xa-max 2
```

Key options (top-level):

* `--sample` — logical sample / pool name.
* `--genome` — comma-separated genome IDs.
* `--bam` — comma-separated BAM paths.
* `--workdir` — output root.
* `--min-barcode-freq` — minimum reads per barcode.
* `--chunk-size-cells` — approximate number of barcodes per chunk.
* `--threads` — global parallelism knob.
* `--assign-xa-max` — maximum allowed `XAcount`.

### Mode B — JSON config per sample

Create `configs/example.json`:

```json
{
  "sample": "B73Mo17_rep1",
  "workdir": "/scratch/ambient_out",
  "genomes": {
    "B73":  "/data/B73Mo17_rep1/B73Mo17_rep1_A1_B73_scifiATAC.mq20.BC.rmdup.mm.bam",
    "Mo17": "/data/B73Mo17_rep1/B73Mo17_rep1_A1_Mo17_scifiATAC.mq20.BC.rmdup.mm.bam"
  },
  "min_barcode_freq": 10,
  "chunk_size_cells": 5000,
  "assign": {
    "alpha": 0.05,
    "k": 10,
    "mapq_min": 20,
    "xa_max": 2,
    "chunksize": 500000,
    "batch_size": 32,
    "edges_workers": 8,
    "edges_max_reads": 5000000
  }
}
```

Run:
```bash
ambientmapper run \
  --config configs/example.json \
  --threads 16 \
  --assign-xa-max 2
```

### Mode C — TSV for many samples

Supply a TSV where each row is **one genome BAM for a given sample**. Required lowercase headers:

```
sample    genome    bam    workdir
```

Example:

```
sample        genome  bam                         workdir
PoolA         B73     /bam/B73.bam                /scratch/out
PoolA         Mo17    /bam/Mo17.bam               /scratch/out
PoolB         B73     /bam/B73.bam                /scratch/out
PoolB         W22     /bam/W22.bam                /scratch/out
```

Run:

```bash
ambientmapper run \
  --configs configs/samples.tsv \
  --min-barcode-freq 10 \
  --chunk-size-cells 5000 \
  --threads 16
```

Each distinct `sample` value is treated as an independent pipeline run under `<workdir>/<sample>/`.

### Resumable runs and targeted re-runs

`ambientmapper run` tracks completed steps using SHA-256 sentinel files in `_sentinels/`. Each sentinel fingerprints the config, parameters, and input files so that re-runs with changed parameters automatically invalidate the affected steps.

**Flags:**

| Flag | Description |
|---|---|
| `--resume / --no-resume` | (default: `--resume`) Skip steps whose sentinel is valid. Use `--no-resume` to ignore all sentinels and re-run everything. |
| `--force-steps <steps>` | Comma-separated list of steps to re-run even if their sentinel is valid. Downstream steps are unaffected unless their inputs change. Example: `--force-steps genotyping` |
| `--skip-to <step>` | Validate-only for all steps before `<step>` (materialize missing outputs, skip if present), then run normally from `<step>` onward. |
| `--only-steps <steps>` | Run only the named steps; all others are skipped unconditionally. Useful for re-running a single step after manual edits. |

Examples:

```bash
# Resume after an interrupted assign step
ambientmapper run --config cfg.json --threads 16 --resume

# Re-run only genotyping with a tighter BIC margin
ambientmapper run --config cfg.json --threads 8 \
  --only-steps genotyping \
  --genotyping-bic-margin 5.0

# Force re-run of assign + genotyping, skip extract/filter/chunks
ambientmapper run --config cfg.json --threads 16 \
  --force-steps assign,genotyping
```

---

## Step-by-step reference

Individual steps also accept `--config cfg.json` and run a single step in isolation. This is useful for debugging. Note that stepwise commands support **JSON config only** (no inline or TSV modes).

### 1. extract — per-genome QC tables

**Code:** `extract.py → bam_to_qc`

For each genome BAM:

* Scans mapped primary alignments (`!is_unmapped`, `!is_secondary`, `!is_supplementary`).
* Reads per alignment:

  * **Read** — read name (`query_name`).
  * **MAPQ** — mapping quality.
  * **AS** — alignment score (`AS` tag if present, else empty).
  * **NM** — mismatches/edits (`NM` tag if present, else empty).
  * **XAcount** — number of alternative alignments from the `XA` tag (0 if absent).
  * **BC** — barcode from `CB` or `BC`.
* Normalizes barcodes to `"<BCseq>-<sample>"` via `canonicalize_bc_seq_sample_force`.

**Writes:**

```
qc/<genome>_QCMapping.txt
# Columns: Read, BC, MAPQ, AS, NM, XAcount
```

```bash
ambientmapper extract --config cfg.json --threads 8
```

---

### 2. filter — barcode filtering + duplicate collapse

**Code:** `filtering.py → filter_qc_file`

For each QC file:

* Re-normalizes barcode to `"<BCseq>-<sample>"`.
* Counts reads per barcode and **keeps only barcodes with frequency ≥ min_barcode_freq**.
* Collapses duplicate `(Read, BC)` pairs using:

  * `MAPQ = max`
  * `AS = max`
  * `NM = min`
  * `XAcount = max`

**Writes:**

```
filtered_QCFiles/filtered_<genome>_QCMapping.txt
```

```bash
ambientmapper filter --config cfg.json --threads 8
```

---

### 3. chunks — barcode chunking

**Code:** `chunks.py → make_barcode_chunks`

* Scans all filtered QC files.
* Collects the **union** of all normalized barcodes (SQLite-backed deduplication for efficiency).
* Splits into chunks of size ≈ `chunk_size_cells` and writes a `manifest.json`.

Each chunk *k*:

```
cell_map_ref_chunks/<sample>_cell_map_ref_chunk_<k>.txt
```

(Just newline-separated barcodes.)

If downstream steps are invoked without prior chunking, a single-chunk dummy manifest is created automatically.

```bash
ambientmapper chunks --config cfg.json
```

---

### 4. assign — ECDF-based dominance filter + per-read winners

**Code:** `assign_streaming.py`

### Pass A — Learn winner score distributions (`learn_edges`)

**Performance:** By default, Pass A subsamples 50 000 barcodes and uses DuckDB
to scan each genome file exactly once. This reduces wall time from hours/days to
minutes for experiments with many genomes. Set `--assign-edges-subsample 0` to
use all barcodes (legacy behaviour).

For each `(Read, BC)` pair across genomes:

* Tracks **best1**, **best2**, **worst** alignments using a priority key:

  1. lower NM
  2. higher AS
  3. higher MAPQ
* Collects global histograms for `AS_best1`, `MAPQ_best1`.
* Computes **decile boundaries** (`k=10`) stored in:

```
ExplorationReadLevel/global_edges.npz
```

### Pass B — Learn Δ-score ECDFs (`learn_ecdfs_parallel` or `learn_ecdfs_batched`)

Computes:

* `dAS = AS_best1 - AS_best2`
* `dMAPQ = MAPQ_best1 - MAPQ_best2`

These are stratified by the winner-quality decile and histograms/ECDFs recorded in:

```
ExplorationReadLevel/global_ecdf.npz
```

### Pass C — Score each chunk (`score_chunk`)

For each chunk:

* Reconstructs per-read, per-genome evidence (**per_g**: AS, MAPQ, NM).
* Computes **p-values** using ΔAS/ΔMAPQ ECDFs (`p_as`, `p_mq`).
* Performs **winner vs ambiguous** decision:

  * Uses `alpha` (default 0.05) and NM differences.
* Runs **dominance filter** across all genomes.

**Outputs per chunk:**

```
raw_cell_map_ref_chunks/<sample>_chunk<k>_raw.tsv.gz
cell_map_ref_chunks/<sample>_chunk<k>_filtered.tsv.gz
```

Filtered tables include:

```
Read, BC, Genome, AS, MAPQ, NM, XAcount, p_as, p_mq, assigned_class
```

```bash
ambientmapper assign --config cfg.json --threads 16
```

**Debugging sub-flags** (skip already-computed model passes):

* `--skip-edges` — skip Pass A; models must already exist in `ExplorationReadLevel/`.
* `--skip-ecdf` — skip Pass B; ECDFs must already exist.
* `--only-score` — skip both learning passes and run only Pass C (score chunks).

---

### 5. genotyping — ambient-aware per-cell calls

**Code:** `genotyping.py` (CLI: `ambientmapper genotyping`)

Genotyping consumes `*_filtered.tsv.gz` files and runs a **two-pass streaming algorithm**.

### Pass 1 — Compute per-read posteriors + expected counts

* Schema normalization to:

  * `barcode`, `read_id`, `genome`, optional `AS`, `MAPQ`, `NM`, `p_as`, `p_mq`.
* Collapse to one row per `(barcode, read_id, genome)` using:

  * `AS=max`, `MAPQ=max`, `NM=min`, `p_* = min`.
* Compute fused score per row:

  * Robust z-scoring of AS, MAPQ, NM.
  * Weighted sum: `w_as`, `w_mapq`, `w_nm`.
  * P-value penalty via `p_as`, `p_mq`.
  * Convert to softmax with `beta` + ambient mass ⇒ `L(r,g)`.
* Accumulate:

  * `C(b,g) = Σ_r L(r,g)`
  * `n_reads(b)`
* Spill all posteriors to **barcode-hashed shards** for pass 2.

When `--genotyping-winner-only` is set, only reads flagged as `assigned_class == winner` are used in Pass 1, which increases signal purity at the cost of discarding ambiguously-mapped reads.

### Ambient estimation

Barcodes with `< 200` reads define the **low-read set**. Their `L(r,g)` values estimate the ambient profile η(g). Falls back to uniform η if needed.

### Candidate genomes per barcode

* Select **top-k genomes** (default 3) by expected counts C(b,g).

### Pass 2 — Per-barcode model selection

For each barcode:

* Reconstruct full `L_block = {L(r,g)}`.
* **Decision 1 (empty/noise):** Compare BIC_empty vs min(BIC_single, BIC_doublet) with `empty_bic_margin`. If the barcode is consistent with pure ambient, call it **empty**.
* **Decision 2 (single vs doublet):** Compare BIC_single vs BIC_doublet with `bic_margin` (ΔBIC gate).
* Apply purity sublabels: `single_clean`, `dirty_singlet`, `doublet`, `weak_doublet`, `indistinguishable`, `ambiguous`.

### Outputs

```
final/<sample>_cells_calls.tsv.gz
```

One row per barcode with fields:

* `call` ∈ {single_clean, dirty_singlet, doublet, weak_doublet, indistinguishable, ambiguous, empty}
* `genome_1`, `genome_2`
* `alpha` (ambient fraction), `rho` (doublet mixture)
* `purity`, `minor`
* `bic_single`, `bic_doublet`, `bic_best`, `delta_bic`
* `n_reads`, `n_effective`
* `concordance` (per-read argmax agreement with major genome)
* `p_top1`, `p_top2`, `p_top3`, `top3_sum`, `entropy`
* `top_genome`, `best_genome`
* `status_flag` (single / double / low_confidence / ambiguous)
* `suspect_multiplet`, `multiplet_reason`

```
final/<sample>_expected_counts_by_genome.csv
```

Dense C(b,g) matrix (barcodes × genomes) for QC and exploration.

```bash
ambientmapper genotyping --config cfg.json --threads 4
```

---

## Decontamination

### decontam — build drop list and barcode policy

**Code:** `decontam.py` (CLI: `ambientmapper decontam`)

Reads the genotyping calls file and the per-chunk assignment files to assign each barcode an **AllowedSet** — the set of genomes from which reads are considered valid for that barcode. Reads assigned to genomes outside the AllowedSet are written to a drop list. Processing is parallelized across assignment chunk files.

**AllowedSet construction:**

* For a **single-genome** barcode the AllowedSet is `{genome_1}` — only reads confidently mapping to that genome are kept.
* For a **doublet**, the AllowedSet depends on `--doublet-policy`: `top12` keeps reads from both top genomes, `top1` keeps only the dominant genome, `expected` uses a plate design file to look up the expected genome.
* For an **ambiguous** barcode, `--ambiguous-policy` controls whether reads are dropped outright or rescued to one or two genomes.

**Post-clean gating:** even after AllowedSet filtering, a barcode with too few remaining reads can be dropped entirely using `--min-reads-post-clean` and `--min-allowed-frac-post-clean`.

**Design-aware mode:** provide `--design-file` (barcode-to-expected-genome TSV or well-range format) to enable design-aware rescue and contamination binning.

**Minimal example:**

```bash
ambientmapper decontam \
  --cells-calls /out/SAMPLE/final/SAMPLE_cells_calls.tsv.gz \
  --out-dir /out/SAMPLE/decontam/ \
  --ambiguous-policy top1_rescue \
  --doublet-policy top12 \
  --min-reads-post-clean 10 \
  --threads 4
```

**Design-aware example:**

```bash
ambientmapper decontam \
  --cells-calls /out/SAMPLE/final/SAMPLE_cells_calls.tsv.gz \
  --out-dir /out/SAMPLE/decontam/ \
  --design-file /configs/plate_design.tsv \
  --ambiguous-policy design_rescue \
  --doublet-policy expected \
  --threads 4
```

**Key parameters:**

| Flag | Default | Description |
|---|---|---|
| `--cells-calls` | required | `*_cells_calls.tsv.gz` from genotyping |
| `--out-dir` | required | Output directory |
| `--ambiguous-policy` | `design_rescue` | `drop` \| `design_rescue` \| `top1_rescue` \| `top12_rescue` |
| `--doublet-policy` | `top12` | `top12` \| `top1` \| `expected` (requires `--design-file`) |
| `--indist-policy` | `top12` | `top12` \| `expected` (requires `--design-file`) |
| `--min-reads-post-clean` | `100` | Minimum reads remaining after cleaning to keep a barcode |
| `--min-allowed-frac-post-clean` | `0.90` | Minimum fraction of reads in AllowedSet to keep a barcode |
| `--design-file` | None | Barcode-to-expected-genome TSV or well-range file |
| `--decontam-alpha` | None | Optional p_as threshold to treat reads as confident winners |
| `--threads` | `1` | Parallel workers across assignment chunk files |
| `--assign-glob` | auto | Override automatic discovery of assignment chunk files |
| `--chunksize` | `1000000` | Row batch size when streaming assignment files |

**Outputs:**

```
<out-dir>/
├── {sample}_reads_to_drop.tsv.gz           # read_id column: QNAMEs to exclude from BAMs
├── {sample}_barcode_policy.tsv.gz          # per-barcode: call, AllowedSet, action, reason
├── {sample}_cells_calls.decontam.tsv.gz    # cells_calls + policy + post-clean metrics
├── {sample}_pre_barcode_genome_counts.tsv.gz
├── {sample}_post_barcode_genome_counts.tsv.gz
├── {sample}_pre_barcode_composition.tsv.gz
├── {sample}_post_barcode_composition.tsv.gz
├── {sample}_pre_contamination_bins.tsv.gz  # design mode only
└── {sample}_post_contamination_bins.tsv.gz # design mode only
```

---

### clean-bams — filter BAM files using drop list

**Code:** `clean_bams.py` (CLI: `ambientmapper clean-bams`)

Removes reads from one or more BAM files using the QNAME drop list produced by `decontam`. Reference-agnostic: the same drop list applies to all per-genome BAMs from the same sequencing run.

**Single-BAM usage:**

```bash
ambientmapper clean-bams \
  --reads-to-drop /out/SAMPLE/decontam/SAMPLE_reads_to_drop.tsv.gz \
  --bam /data/B73.bam \
  --out-dir /out/SAMPLE/clean_bams/
```

**Multi-BAM usage via map file:**

```bash
# bam_map.tsv (tab-separated: bam_id  bam_path)
# B73   /data/B73.bam
# Mo17  /data/Mo17.bam

ambientmapper clean-bams \
  --reads-to-drop /out/SAMPLE/decontam/SAMPLE_reads_to_drop.tsv.gz \
  --bam-map /configs/bam_map.tsv \
  --out-dir /out/SAMPLE/clean_bams/ \
  --out-suffix .clean.bam
```

**Key parameters:**

| Flag | Default | Description |
|---|---|---|
| `--reads-to-drop` | required | `reads_to_drop.tsv.gz` from decontam |
| `--bam` | — | Single BAM to clean (mutually exclusive with `--bam-map`) |
| `--bam-map` | — | TSV with `bam_id`, `bam_path` columns (mutually exclusive with `--bam`) |
| `--out-dir` | required | Output directory |
| `--out-suffix` | `.clean.bam` | Suffix appended to each output BAM filename stem |
| `--index / --no-index` | `--index` | Create `.bai` index for each cleaned BAM |
| `--log-every` | `2000000` | Progress log interval in reads (0 disables) |

**Outputs:**

```
<out-dir>/
├── {original_stem}.clean.bam
└── {original_stem}.clean.bam.bai   # if --index (default)
```

---

## Quality control and cross-pool analysis

### summary — per-pool QC report (Python API)

**Code:** `summary.py`

> **Note:** `summary` is not exposed as a `ambientmapper summary` CLI command. Invoke it as a Python function.

Produces a per-pool contamination QC PDF and a high-quality barcode CSV. Compares low-contamination vs high-contamination barcode groups using KS tests on AS_mean, reads/BC, and contamination rate distributions.

**Python usage:**

```python
from ambientmapper.summary import summarize_cli
from pathlib import Path

summarize_cli(
    config_json=Path("configs/example.json"),
    pool_design=None,   # or Path to pool design TSV
    xa_max=2,
)
```

**Outputs** (written to `<workdir>/<sample>/`):

```
<workdir>/<sample>/
├── {sample}_BCs_PASS_by_mapping.csv       # high-quality barcodes assigned to in-pool genomes
├── Reads_to_discard_{sample}.csv          # reads from wrong genome for each BC
└── Plots/
    └── {sample}_contamination_correction_summary.pdf   # 5-panel QC PDF
```

---

### interpool — cross-pool BC count and read composition

**Code:** `interpool.py` (CLI: `ambientmapper interpool`)

Produces cross-pool heatmaps of barcode counts and read composition for all pools defined in a samples TSV. The `--configs` argument uses the same format as `ambientmapper run --configs`.

```bash
ambientmapper interpool \
  --configs configs/samples.tsv \
  --outdir /scratch/interpool_out/
```

**Outputs:**

```
<outdir>/
├── interpool_bc_counts.tsv          # BCs per AssignedGenome × sample
├── interpool_read_composition.tsv   # winner-read fractions per Genome × sample
└── interpool_summary.pdf            # two heatmaps
```

---

### plate — 96-well plate heatmap

**Code:** `plate.py` (CLI: `ambientmapper plate`)

Produces a 96-well plate heatmap of per-well contamination rates and per-sample contamination statistics.

**Plate map format** (tab-separated `sample` and well ranges; `#` lines and blank lines ignored):

```
# Sample  Wells
B73       A1-12,B1-12
Mo17      C1-12,D1-12
```

```bash
ambientmapper plate \
  --workdir /scratch/out \
  --plate-map /configs/plate_map.tsv \
  --outdir /scratch/plate_out/ \
  --xa-max 2
```

**Outputs:**

```
<outdir>/
├── plate_per_bc.tsv.gz      # per-BC: BC, Total_reads, Mismatch_reads, Contamination_Rate
├── plate_per_sample.tsv     # per-sample summary stats (mean/median/p95 contamination)
├── plate_density.pdf        # contamination rate density plots
└── plate_heatmap.pdf        # 96-well heatmap
```

---

## Output directory layout

All outputs live under:

```
<workdir>/<sample>/
```

Typical layout for a completed run including decontamination:

```
<workdir>/<sample>/
├─ qc/                                   # per-genome raw QC (extract)
│   ├─ <genome1>_QCMapping.txt
│   └─ <genome2>_QCMapping.txt
├─ filtered_QCFiles/                     # per-genome filtered/collapsed QC (filter)
│   ├─ filtered_<genome1>_QCMapping.txt
│   └─ filtered_<genome2>_QCMapping.txt
├─ cell_map_ref_chunks/                  # barcode chunk files + per-chunk assign outputs
│   ├─ <sample>_cell_map_ref_chunk_1.txt
│   ├─ <sample>_cell_map_ref_chunk_2.txt
│   ├─ ...
│   ├─ <sample>_chunk1_filtered.tsv.gz
│   └─ <sample>_chunk2_filtered.tsv.gz
├─ raw_cell_map_ref_chunks/              # optional per-chunk raw scoring tables (assign)
│   ├─ <sample>_chunk1_raw.tsv.gz
│   └─ <sample>_chunk2_raw.tsv.gz
├─ ExplorationReadLevel/                 # global models learned from all chunks (assign)
│   ├─ global_edges.npz
│   └─ global_ecdf.npz
├─ final/                                # genotyping outputs
│   ├─ <sample>_cells_calls.tsv.gz       # main per-barcode genotyping calls
│   └─ <sample>_expected_counts_by_genome.csv
├─ decontam/                             # decontamination outputs
│   ├─ <sample>_reads_to_drop.tsv.gz
│   ├─ <sample>_barcode_policy.tsv.gz
│   ├─ <sample>_cells_calls.decontam.tsv.gz
│   ├─ <sample>_pre_barcode_genome_counts.tsv.gz
│   ├─ <sample>_post_barcode_genome_counts.tsv.gz
│   ├─ <sample>_pre_barcode_composition.tsv.gz
│   └─ <sample>_post_barcode_composition.tsv.gz
├─ clean_bams/                           # clean-bams outputs
│   ├─ <genome1>.clean.bam
│   └─ <genome1>.clean.bam.bai
├─ <sample>_BCs_PASS_by_mapping.csv     # high-quality barcodes (summary Python API)
├─ Reads_to_discard_<sample>.csv        # mismatched reads (summary Python API)
├─ Plots/                               # summary QC plots
│   └─ <sample>_contamination_correction_summary.pdf
└─ _sentinels/                          # per-step sentinel JSON files for resumable runs
    ├─ extract.ok.json
    ├─ filter.ok.json
    ├─ chunks.ok.json
    ├─ assign.ok.json
    └─ genotyping.ok.json
```

**Core outputs for downstream analysis:**

| File | Produced by |
|---|---|
| `final/<sample>_cells_calls.tsv.gz` | genotyping |
| `final/<sample>_expected_counts_by_genome.csv` | genotyping |
| `decontam/<sample>_reads_to_drop.tsv.gz` | decontam |
| `decontam/<sample>_cells_calls.decontam.tsv.gz` | decontam |
| `<sample>_BCs_PASS_by_mapping.csv` | summary (Python API) |

---

## Advanced parameter tuning

### Global run options

| Flag | Default | Description |
|---|---|---|
| `--threads` | `8` | Global parallelism |
| `--min-barcode-freq` | `10` | Barcode filter threshold |
| `--chunk-size-cells` | `5000` | Barcodes per chunk |
| `--resume / --no-resume` | `--resume` | Enable/disable sentinel-based skipping |
| `--force-steps` | — | Comma-separated steps to re-run regardless of sentinels |
| `--skip-to` | — | Run validate-only before this step, then run normally |
| `--only-steps` | — | Run only these steps; skip everything else |

### Assign parameters

Pass via JSON (`assign` block) or CLI prefix (`--assign-*`) when using `ambientmapper run`:

| Key / Flag | Default | Description |
|---|---|---|
| `assign.alpha` / `--assign-alpha` | `0.05` | FDR threshold for winner vs ambiguous classification |
| `assign.k` / `--assign-k` | `10` | Number of deciles for score stratification |
| `assign.mapq_min` / `--assign-mapq-min` | `20` | Minimum MAPQ to consider an alignment |
| `assign.xa_max` / `--assign-xa-max` | `2` | Maximum allowed alternative hits (`XAcount`); `-1` to disable |
| `assign.chunksize` / `--assign-chunksize` | — | Rows per streaming batch (reduces memory at cost of speed) |
| `assign.batch_size` / `--assign-batch-size` | `32` | Chunk files handled together in batched ECDF learning |
| `assign.edges_workers` / `--assign-edges-workers` | — | Workers for `learn_edges` (Pass A, Python fallback only) |
| `assign.edges_max_reads` / `--assign-edges-max-reads` | all | Cap on reads per genome during Pass A model learning (Python fallback only) |
| `assign.edges_subsample` / `--assign-edges-subsample` | `50000` | Subsample N barcodes for Pass A decile learning (0 = all). Dramatically reduces I/O for large experiments. |
| `assign.edges_duckdb` / `--assign-edges-duckdb` | `true` | Use DuckDB for Pass A edge learning. Falls back to Python if duckdb unavailable. |
| `assign.edges_duckdb_threads` / `--assign-edges-duckdb-threads` | `4` | DuckDB threads for Pass A edge learning. |
| `--ecdf-workers` | — | Workers for ECDF learning (Pass B) |

**Standalone assign sub-flags** (when calling `ambientmapper assign` directly):

* `--skip-edges` — skip Pass A (edges must already exist).
* `--skip-ecdf` — skip Pass B (ECDFs must already exist).
* `--only-score` — skip both learning passes; run only Pass C (score chunks).

### Genotyping parameters

Pass via `--genotyping-*` flags on `ambientmapper run`, or use `ambientmapper genotyping --help` for the standalone command.

**Score fusion:**

| Flag | Default | Description |
|---|---|---|
| `--genotyping-beta` | `1.0` | Softmax temperature for per-read score fusion |
| `--genotyping-w-as` | `1.0` | Weight for AS in score fusion |
| `--genotyping-w-mapq` | `1.0` | Weight for MAPQ |
| `--genotyping-w-nm` | `1.0` | Weight for NM (subtracted) |
| `--genotyping-w-confident` | `1.0` | Row weight for `winner` reads |
| `--genotyping-w-ambiguous` | `1.0` | Row weight for `ambiguous` reads |
| `--genotyping-ambient-const` | `0.001` | Per-read ambient pseudocount |
| `--genotyping-winner-only / --no-genotyping-winner-only` | on | Use only `assigned_class == winner` reads in Pass 1 |

**Calling thresholds:**

| Flag | Default | Description |
|---|---|---|
| `--genotyping-min-reads` | `5` | Min reads per barcode required for a call |
| `--genotyping-bic-margin` | `6.0` | ΔBIC required to call single vs doublet |
| `--genotyping-empty-bic-margin` | `10.0` | ΔBIC for calling a barcode empty/noise |
| `--genotyping-single-mass-min` | `0.6` | Min probability mass on top genome to call `single_clean` |
| `--genotyping-ratio-top1-top2-min` | `2.0` | Min top1/top2 score ratio to call `single_clean` |
| `--genotyping-doublet-minor-min` | `0.20` | Min minor fraction to call `doublet` (vs `weak_doublet`) |
| `--genotyping-topk-genomes` | `3` | Candidate genomes per barcode for Pass 2 |
| `--alpha-grid` | — | Step size for ambient fraction grid search |
| `--rho-grid` | — | Step size for doublet mixture grid search |
| `--max-alpha` | — | Maximum ambient fraction in grid search |

**Ambient estimation:**

| Flag | Default | Description |
|---|---|---|
| `--genotyping-eta-iters` | `2` | Ambient profile refinement iterations |
| `--genotyping-eta-seed-quantile` | — | Quantile of low-read barcodes used to seed ambient estimate |
| `--genotyping-empty-top1-max` | — | Max top1 mass for a barcode to seed the empty model |
| `--genotyping-empty-ratio12-max` | — | Max top1/top2 ratio for empty seeding |
| `--genotyping-empty-reads-max` | — | Max reads for empty seeding |
| `--genotyping-empty-seed-bic-min` | — | Min BIC_empty for a barcode to serve as an empty seed |
| `--genotyping-empty-tau-quantile` | — | Quantile of empty BIC values used to set the empty threshold |
| `--genotyping-empty-jsd-max` | — | Max JSD to ambient profile for calling a barcode empty |
| `--genotyping-jsd-normalize / --no-genotyping-jsd-normalize` | on | Normalize JSD computation |

**Multi-hit filter:**

| Flag | Default | Description |
|---|---|---|
| `--genotyping-max-hits` | None | Max near-best alternative hits allowed per read |
| `--genotyping-hits-delta-mapq` | None | MAPQ tolerance when counting near-best alternative hits |

**Performance:**

| Flag | Default | Description |
|---|---|---|
| `--genotyping-pass1-workers` | `1` | Parallel workers for Pass 1 posterior computation |
| `--genotyping-pass2-chunksize` | `200000` | Row batch size for Pass 2 streaming |
| `--genotyping-shards` | `32` | Temporary shard files for cross-barcode spill |
| `--genotyping-chunk-rows` | `5000000` | Row batch size for Pass 1 streaming |
| `--genotyping-threads` | — | Override thread count for the genotyping step |

### Decontam parameters

See the [decontam parameter table](#decontam--build-drop-list-and-barcode-policy) above.

### Environment variables

| Variable | Description |
|---|---|
| `AMM_CANARY_N` | If set to a positive integer N, runs the first N assign partitions serially before launching the full parallel execution. Causes a hard failure if any canary partition errors, catching configuration problems early without waiting for all workers. Example: `AMM_CANARY_N=2 ambientmapper run --config cfg.json` |

---

## Troubleshooting / FAQ

**Out of memory during `assign`**

Reduce `assign.chunksize` (rows held in memory per batch), `assign.edges_max_reads` (reads used for Pass A), or `--chunk-size-cells` (barcode batch size).

**`assign` step fails immediately on all chunks**

Set `AMM_CANARY_N=1` to run one partition serially with a full traceback. Verify all BAMs are coordinate-sorted, indexed, and contain `CB`/`BC`, `AS`, `NM`, and `MAPQ` tags.

**Genotyping produces mostly `ambiguous` calls**

Lower `--genotyping-bic-margin` to relax the evidence threshold. Check that `--genotyping-min-reads` is not too high. Try `--genotyping-winner-only` to restrict Pass 1 to high-confidence read assignments.

**`decontam` reports zero valid reads for many barcodes**

If `--ambiguous-policy drop` is set, all ambiguous barcodes are dropped entirely. Switch to `top1_rescue` or `top12_rescue`. Verify that `--assign-glob` (auto-discovered by default) is finding the correct chunk files.

**`clean-bams` crashes with a htslib / pysam error**

Ensure each BAM is coordinate-sorted and indexed (`.bam.bai` present). On Apple Silicon, install `pysam` via `mamba install -c bioconda pysam` instead of `pip`.

**Pipeline re-runs completed steps unexpectedly**

This happens when the config or parameters changed since the sentinel was written — expected behavior. Use `--no-resume` to disable sentinel checking entirely, or `--force-steps <step>` to selectively re-run.

---

## Contributing and development

```bash
# Install in editable mode with test dependencies
pip install -e ".[test]"

# Run the test suite
pytest tests/

# The CI badge at the top reflects .github/workflows/tests.yml
# running on ubuntu-latest across Python 3.9, 3.10, 3.11, 3.12.
```

For bugs and feature requests, open an issue at https://github.com/gomezcan/ambientmapper/issues.

For full CLI help on any command:

```bash
ambientmapper run --help
ambientmapper assign --help
ambientmapper genotyping --help
ambientmapper decontam --help
ambientmapper clean-bams --help
ambientmapper interpool --help
ambientmapper plate --help
```
