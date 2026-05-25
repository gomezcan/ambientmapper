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
