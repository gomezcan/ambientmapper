# Changelog

All notable changes to `ambientmapper` will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — 0.2.0 (parquet-native pipeline)

The 0.2 series moves the per-genome QC files from TSV to Parquet across the
whole pipeline. The on-the-fly DuckDB CSV→Parquet conversion previously done
inside `assign` is being retired in favor of direct Parquet writes from
`extract` and `filter`, which:

- removes the segfault-prone large-TSV parsing during `assign`,
- cuts per-pool disk by ~75%,
- cuts per-pool peak memory by ~10×,
- requires no new dependencies (pyarrow and duckdb are already core deps).

The rollout is three staged PRs (see `plans/parquet-native-refactor.md`):

- **PR1** (this entry) — schema and helpers, no behavior change.
- **PR2** — `filter` writes Parquet directly; `chunks` reads Parquet.
- **PR3** — `extract` writes Parquet directly; retire implicit
  `_convert_to_parquet`; deprecate the `--prepare/--no-prepare` flag on
  `assign`.

### PR3 — Parquet-native `extract` + retire implicit `_convert_to_parquet`

Completes the parquet-native pipeline. Extract writes Parquet by default;
the implicit txt→parquet conversion previously fired at assign time is
removed from `pipeline.run`; the `--prepare` flag on `assign` is
deprecated.

#### Added

- `extract._bam_to_qc_parquet`: streams a BAM to typed Parquet conforming
  to `QC_PARQUET_SCHEMA`. Maintains 7 per-column accumulators flushed as
  a `pa.RecordBatch` every `row_group_size` rows (default 100,000).
  Atomic write via `.tmp` + `os.replace`. Empty-BAM case writes a valid
  schema-conformant empty parquet.
- `extract.bam_to_qc` now returns `int` (rows written). Backwards-
  compatible: existing callers ignore the return value.
- `extract --format {parquet,txt}` CLI flag (default `parquet`). `txt`
  selects the legacy header-less TSV writer.
- `tests/test_extract_parquet.py` (7 cases): schema conformance,
  empty-BAM, missing AS/NM tags → nulls, unmapped/secondary/supplementary
  filtering, row-group flush size, legacy TSV preserved, extension
  dispatch precedence.
- `tests/test_e2e_parquet.py`: load-bearing equivalence test — extract
  parquet → filter parquet vs. extract txt → filter parquet (same input
  BAM, same min_freq) must produce bit-equivalent filtered Parquet.

#### Changed

- `extract.bam_to_qc`: thin dispatcher on `out_path.suffix`. Old
  implementation preserved as `_bam_to_qc_tsv`.
- `cli.py extract`: defaults to Parquet output; `--format` flag added;
  sentinel `outputs` field records the chosen format. Sentinel hash
  unchanged (matches PR2 filter convention).
- `cli.py prepare`: docstring rewritten to flag it as a **MIGRATION TOOL**
  for pre-0.2 TSVs; echo message changed from "converting" to "migrating
  legacy".
- `cli.py assign --prepare/--no-prepare`: default flipped `True` → `False`.
  When `--prepare=True`, emits a yellow deprecation notice on stderr.
  The inline conversion now only fires if legacy TSV-only inputs are
  detected (via `discover_filtered_files`); on parquet-native pools the
  flag is a no-op with a "no legacy TSV detected" log.
- `pipeline.py _run_assign`: the implicit `_convert_to_parquet` call is
  **removed**. Replaced with a defensive warning that fires only when
  legacy TSV-only inputs are detected, pointing the user at
  `ambientmapper prepare` or `filter --no-resume`.
- `README.md`:
  - Disk-requirements blurb updated for ~25% smaller QC files.
  - `extract` reference section: output `.parquet`, typed columns
    documented, AS→`as_` rename explained, `--format txt` escape-hatch.
  - Directory tree shows `qc/<G>_QCMapping.parquet`.
  - New troubleshooting entry for the "legacy TSV inputs detected"
    warning + the `--prepare` deprecation.

#### Migration

- New pipelines run extract → filter → assign cleanly with no extra
  steps; no manual conversion needed.
- Pre-0.2 pools with TSV outputs: run `ambientmapper prepare` once to
  produce co-located Parquet, then continue with the new pipeline.
  Alternatively, `filter --no-resume` rewrites the TSV inputs from
  whatever extract left behind.
- Disk reclaim after migration:
  ```bash
  find <workdir>/<sample> \( -name '*_QCMapping.txt' -o -name 'filtered_*_QCMapping.txt' \) -delete
  ```
- `--prepare` survives one minor release for back-compat; will be removed
  in 0.3.

---

### PR2 — Parquet-native `filter` + parquet-aware `chunks`

Resolves the production segfault triggered when `_convert_to_parquet`
parsed multi-GB legacy TSVs inside `assign` (the largest pools were
hitting DuckDB CSV-reader crashes with no easy memory ceiling). With PR2,
`filter` writes Parquet directly so the inline conversion is unnecessary
on parquet inputs. The implicit conversion in `assign` still fires on
pure-TSV pools but becomes a no-op once filter has run; full removal is
in PR3.

#### Added

- `filtering._filter_qc_file_parquet`: DuckDB-backed single-SQL pipeline
  (`COUNT(*) OVER (PARTITION BY BC)` → filter by `min_freq` →
  `ORDER BY BC` → `COPY ... TO ... FORMAT PARQUET`). Memory bounded by
  the `duckdb_memory_limit` parameter (default `"16GB"`); external sort
  spills to `$TMPDIR`. Honors the `QC_PARQUET_SCHEMA` typed contract
  (int16/int32/string with explicit nullability).
- `filtering._filter_qc_file_parquet` empty-result handling: writes a
  schema-conformant zero-row Parquet via `pyarrow.ParquetWriter` when no
  barcodes pass `min_freq`. Downstream readers don't choke on empty files.
- `chunks._iter_barcodes_from_filtered`: parquet branch streams the `BC`
  column in row-group batches via `pyarrow.parquet.ParquetFile.iter_batches`,
  keeping memory constant regardless of file size.
- `cli.py filter --format {parquet,txt}` flag (default `parquet`).
  `txt` selects the legacy writer for back-compat / reproducing pre-0.2
  outputs.
- `tests/test_filter_parquet.py` (6 cases): parquet-in/parquet-out,
  TSV-in/parquet-out, empty-after-cutoff, BC normalization via DuckDB
  UDF, input-extension dispatch equivalence, legacy TSV-out path
  preserved.
- `tests/test_chunks_parquet.py` (7 cases): direct `_iter_barcodes_*`
  on parquet + TSV, `make_barcode_chunks` precedence (parquet preferred
  when complete; falls back to all-TSV when incomplete), empty-dir
  no-op, chunk size enforced.

#### Changed

- `filtering.filter_qc_file`: now a thin dispatcher over the parquet and
  TSV implementations based on `out_path.suffix`. Public signature kept
  backward-compatible; new keyword-only parameters
  (`duckdb_threads`, `duckdb_memory_limit`, `row_group_size`) supply
  the parquet path's tuning knobs with conservative defaults.
- `chunks.make_barcode_chunks`: replaces the hard-coded
  `filtered_*_QCMapping.txt` glob with
  `discover_filtered_files(filtered_dir)` so it transparently picks up
  whichever format is on disk. The all-or-nothing precedence rule
  (parquet wins when it covers every genome) prevents schema-cast
  surprises on mixed pools.
- `cli.py filter`: input format is auto-detected per genome (parquet
  preferred over txt). Output suffix derives from `--format`. The
  sentinel `outputs` field now records both the chosen `format` and
  the list of output `files`; the sentinel **hash** still excludes the
  format choice so existing sentinels remain valid (use `--no-resume`
  to switch a previously-filtered pool to a different format).
- `pipeline._run_filter`: mirrors the cli.py dispatch — honors a top-level
  `params.format` (defaults to `"parquet"`), prefers parquet input from
  extract.
- `README.md`: filter section + directory tree updated to reflect the
  new parquet output. Added the `--format txt` escape-hatch note.

#### Migration

- A user with existing TSV outputs from a pre-0.2 run can keep them and
  the precedence rule will continue to use them (assign already supports
  both formats). Re-running `filter --no-resume` overwrites with parquet.
- To switch a previously-filtered pool from TSV to parquet:
  `ambientmapper filter --config <cfg> --no-resume` (default
  `--format parquet`).
- Disk reclaim after parquet outputs are verified:
  `find <workdir>/<sample>/filtered_QCFiles -name 'filtered_*_QCMapping.txt' -delete`.

---

### PR1 — Schema constant and helper extraction (no behavior change)

#### Added

- `ambientmapper.extract.QC_PARQUET_SCHEMA`: locked PyArrow schema describing
  the columns, types, nullability, and ordering of QC parquet files produced
  by extract and filter. Columns: `Read, BC, MAPQ, as_, NM, XAcount, frag_loc`.
  Types: `int16/int32/string`. The `AS` BAM tag is renamed to `as_` so it does
  not collide with the SQL reserved word in DuckDB queries.
- `ambientmapper._filtered_io`: new module exposing the cross-stage helpers
  `genome_from_filename(path)` and `discover_filtered_files(filtered_dir)`.
  The latter implements the all-or-nothing precedence rule for choosing
  between legacy `.txt` and current `.parquet` inputs.
- `tests/test_qc_schema.py`: unit tests for the schema constant and the new
  `_filtered_io` helpers.

#### Changed

- `assign_streaming._genome_from_filename` and `_filtered_files` now re-export
  the new `_filtered_io` helpers under their original private names. No
  observable behavior change.
- `discover_filtered_files` (the lifted `_filtered_files`) now correctly
  returns parquet files when a directory contains only parquets (the original
  raised `FileNotFoundError` in that case). The fix is dormant pre-PR2
  because filter still always emits TSVs alongside any parquet.
