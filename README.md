# ambientmapper

Ambient contamination–aware multi-genome genotyping for single-cell libraries (scifi-ATAC, sci-ATAC, 10x-style, etc.)

![tests](https://github.com/gomezcan/ambientmapper/actions/workflows/tests.yml/badge.svg)

`ambientmapper` takes reads mapped to multiple reference genomes and:

1. Normalizes barcodes across genomes.
2. Builds ECDF-based models of winner-vs-runner-up score deltas (AS / MAPQ / NM).
3. Scores each read across all genomes, including a dominance filter for locally ambiguous alignments.
4. Computes per-read posteriors L(r,g) and estimates an ambient profile across genomes.
5. Performs per-barcode model selection (single vs doublet vs ambiguous) under an ambient-aware likelihood.

End-to-end pipeline:

```
extract → filter → chunks → assign → genotyping (→ optional summary/interpool, experimental)
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

## Output directory layout

All outputs live under:

```
<workdir>/<sample>/
```

Typical layout for a completed run:

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
│   ├─ <sample>_chunk1_filtered.tsv.gz   # assign-streaming outputs, one per chunk
│   └─ <sample>_chunk2_filtered.tsv.gz
├─ raw_cell_map_ref_chunks/              # optional per-chunk raw scoring tables (assign)
│   ├─ <sample>_chunk1_raw.tsv.gz
│   └─ <sample>_chunk2_raw.tsv.gz
├─ ExplorationReadLevel/                 # global models learned from all chunks
│   ├─ global_edges.npz                  # decile boundaries for AS & MAPQ
│   └─ global_ecdf.npz                   # per-decile ΔAS / ΔMAPQ ECDF model
├─ final/
│   ├─ <sample>_cells_calls.tsv.gz       # main per-barcode genotyping calls
│   ├─ <sample>_BCs_PASS_by_mapping.csv  # compact summary: barcode, AssignedGenome, call, purity, n_reads
│   ├─ <sample>_expected_counts_by_genome.csv  # expected counts C(b,g)
│   └─ generation.json                   # generation counter (used by some higher-level tools)
├─ _sentinels/                           # per-step sentinel JSON files for resumable runs
│   ├─ extract.ok
│   ├─ filter.ok
│   ├─ chunks.ok
│   ├─ assign.ok
│   └─ genotyping.ok
└─ (tmp_*/ directories are created during genotyping and removed on success)
```

The core outputs for downstream analysis are:

* `final/<sample>_cells_calls.tsv.gz`
* `final/<sample>_BCs_PASS_by_mapping.csv`
* `final/<sample>_expected_counts_by_genome.csv`

---

## Run whole pipeline
You can run `ambientmapper` in three main ways:

1. **Inline, single sample** (Mode A).
2. **JSON config per sample** (Mode B).
3. **TSV table for many samples** (Mode C).

In all cases, the pipeline executes:

```
extract → filter → chunks → assign → genotyping
```

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
  --xa-max 2
```

Key options (top-level):

* `--sample` — logical sample / pool name.
* `--genome` — comma-separated genome IDs.
* `--bam` — comma-separated BAM paths.
* `--workdir` — output root.
* `--min-barcode-freq` — minimum reads per barcode.
* `--chunk-size-cells` — approximate number of barcodes per chunk.
* `--threads` — global parallelism knob.
* `--xa-max` — maximum allowed `XAcount`.

### Mode B — JSON config per sample

Create ```configs/example.json:```

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
  --xa-max 2

```


### Mode C — TSV for many samples

You can supply a TSV where each row represents **one genome BAM for a given sample**. Required lowercase headers:

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

Each distinct ```sample``` value is treated as an independent pipeline run under ```<workdir>/<sample>/```.

## Run step-by-step debugging

Run steps individually using **JSON config only**:

```bash
ambientmapper extract   --config cfg.json --threads 8
ambientmapper filter    --config cfg.json --threads 8
ambientmapper chunks    --config cfg.json
ambientmapper assign    --config cfg.json --threads 16
ambientmapper genotyping \
  --assign "<glob>" \
  --outdir final/ \
  --sample SAMPLE \
  --threads 4
```

Notes:

* Stepwise commands do **not** support inline or TSV modes.
* Only `ambientmapper run` supports Modes A/B/C.
* Small empty BAMs can be used to validate CLI parsing.
  
---
# What each step does (current pipeline)

## 1) extract — per-genome QC tables

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

---

## 2) filter — barcode filtering + duplicate collapse

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

---

## 3) chunks — barcode chunking

**Code:** `chunks.py → make_barcode_chunks`

* Scans all filtered QC files.
* Collects the **union** of all normalized barcodes.
* Splits into chunks of size ≈ `chunk_size_cells`.

Each chunk *k*:

```
cell_map_ref_chunks/<sample>_cell_map_ref_chunk_<k>.txt
```

(Just newline-separated barcodes.)

If later steps run without existing chunks, a **dummy chunk + manifest.json** is created.

---

## 4) assign — ECDF-based dominance filter + per-read winners

**Code:** `assign_streaming.py`

### Pass A — Learn winner score distributions (`learn_edges_parallel`)

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

---

## 5) genotyping — ambient-aware per-cell calls

**Code:** `genotyping.py` (CLI: `ambientmapper genotyping`)

Genotyping consumes `*_filtered.tsv.gz` files and runs a **two-pass streaming algorithm**.

### Pass 1 — Compute per-read posteriors + expected counts

* Schema normalization to:

  * `barcode`, `read_id`, `genome`, optional `AS`, `MAPQ`, `NM`, `p_as`, `p_mq`.
* Collapse to one row per `(barcode, read_id, genome)` using:

  * `AS=max`, `MAPQ=max`, `NM=min`, `p_* = min`.
* Compute fused score per row:

  * Robust z-scoring of AS, MAPQ, NM.
  * Weighted sum: `w_as=0.5`, `w_mapq=1.0`, `w_nm=1.0`.
  * P-value penalty via `p_as`, `p_mq`.
  * Convert to softmax with `beta = 0.5` + ambient mass ⇒ `L(r,g)`.
* Accumulate:

  * `C(b,g) = Σ_r L(r,g)`
  * `n_reads(b)`
* Spill all posteriors to **barcode-hashed shards** for pass 2.

### Ambient estimation

Barcodes with `< 200` reads define the **low-read set**. Their `L(r,g)` values estimate the ambient profile η(g). Falls back to uniform η if needed.

### Candidate genomes per barcode

* Select **top-k genomes** (default 3) by expected counts C(b,g).

### Pass 2 — Per-barcode model selection

For each barcode:

* Reconstruct full `L_block = {L(r,g)}`.
* Test models:

  * **Single** genome + ambient (`alpha` grid 0→0.5).
  * **Doublet** mixture + ambient (`alpha` + `rho` grid).
* Compute composite log-likelihood, convert to **BIC**.
* Apply conservative calling:

  * **single** if ΔBIC ≥ margin & purity ≥ threshold.
  * **doublet** if ΔBIC ≥ margin & minor fraction ≥ threshold.
  * **indistinguishable** if near-tie between singles.
  * else **ambiguous**.

### Outputs

```
final/<sample>_cells_calls.tsv.gz
```

One row per barcode with fields:

* `call` ∈ {single, doublet, indistinguishable, ambiguous}
* `genome_1`, `genome_2`
* `alpha` (ambient fraction), `rho`(doublet mixture)
* `purity`, `minor`
* `bic_single`, `bic_doublet`, `bic_best`, `delta_bic`
* `n_reads`, `n_effective`
* `concordance` (per-read argmax agreement with major genome)
* `p_top1`, `p_top2`, `p_top3`, `top3_sum`, `entropy`
* `top_genome`, `best_genome`
* `status_flag` (single / double / low_confidence / ambiguous)
* `suspect_multiplet`, `multiplet_reason`

```
final/<sample>_BCs_PASS_by_mapping.csv
```

Barcode-level summary for downstream tools:

```
barcode,AssignedGenome,call,purity,n_reads
```

```
final/<sample>_expected_counts_by_genome.csv
```

Dense C(b,g) matrix (barcodes × genomes) for QC and exploration.


---

## Advanced parameter tuning

### Global options

* `--threads`: parallelism.
* `--min-barcode-freq`: barcode filter threshold.
* `--chunk-size-cells`: number of barcodes per chunk.

### Assign parameters (JSON or CLI overrides)
Assign-related knobs via JSON (`assign` block) or CLI prefix (`--assign-*`) control:

* `assign.alpha`: FDR threshold, per-read ECDF significance level for winner vs ambiguous (default 0.05).
* `assign.k`: number of deciles for score stratification (default 10).
* `assign.mapq_min`: minimum MAPQ to consider an alignment (default 20).
* `assign.xa_max`: maximum allowed alternative hits (`XAcount`) (default `2`; set to `-1` to disable).
* `assign.chunksize`: Number of rows read per chunk when streaming QC files to reduce memory usage (at the cost of processing time)
* `assign.batch_size`: number of chunk files handled together in batched learning.
* `assign.edges_workers`: number of workers for `learn_edges_parallel`.
* `assign.edges_max_reads`: optional cap on reads per genome when learning edges (defaul all reads).
* `assign.ecdf_workers`: ECDF workers.

### Genotyping parameters

* `--min_reads`: Minimum reads per barcode required for a confident call (default: 100). Note: reads are counted as pairs, so 100 reads correspond to 200 Tn5 insertions.
* `--beta`: softmax temperature for per-read score fusion.
* `--w_as`, `--w_mapq`, `--w_nm`: weights for AS, MAPQ, and NM.
* `--ambient_const`: per-read ambient mass before normalization.
* `--topk-genomes`: number of candidate genomes per barcode (default 3).
* `alpha_grid`, `rho_grid`, `max_alpha`, `bic_margin`, `single_mass_min`, `doublet_minor_min`, `near_tie_margin`:  control the ambient-aware model selection.
* `--chunk-rows`: streaming chunk size.
* `--shards`: number of spill shards.
* `--pass1-workers`: number of Pass1 workers.

For full details, see:

```bash
ambientmapper assign --help
ambientmapper genotyping --help
ambientmapper run --help
```


