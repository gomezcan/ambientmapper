# Plan: Add resume logic to assign scoring step

## Context

The assign step scores 9,023 chunks sequentially (in batches of 50). With Root1's 26 genomes, each batch takes ~1,100-3,200s. At the current rate, the full scoring needs ~24h but the SLURM job has a 20h wall limit. When the job times out and is resubmitted, all already-completed chunks are reprocessed because there's no skip logic.

## File to modify

`/Volumes/lsa-amarand/fabio_home/Projects/00_Repos/ambientmapper/src/ambientmapper/assign_streaming.py`

## Two entry points need resume logic

### 1. `score_chunks_batched()` (line 1890) — batched DuckDB path

This is the hot path (used by the current Root1 job via `cli.py`). It processes N chunks in a single DuckDB scan.

**Strategy:** Before building the DuckDB query, check which chunks in the batch already have completed output. Remove them from the batch. If all chunks are done, return early with synthetic stats.

At line ~1920 (after `out_filtered_dir.mkdir`), before collecting BCs:

```python
# Resume: skip chunks whose filtered output already exists
done_stats = []
pending = []
for chf in chunk_files:
    tag = chf.stem
    suff = tag.split("_cell_map_ref_chunk_")[-1] if "_cell_map_ref_chunk_" in tag else tag
    base = f"{sample}_chunk{suff}"
    filt_out = out_filtered_dir / f"{base}_filtered.tsv.gz"
    if filt_out.exists() and filt_out.stat().st_size > 64:
        done_stats.append({"chunk": chf.name, "reads": -1, "winner": -1,
                           "ambig": -1, "ambig_bcs": -1, "elapsed": 0.0,
                           "resumed": True})
        _log(f"[assign/score] ● skip {chf.name} (output exists)", verbose)
    else:
        pending.append(chf)

if not pending:
    return done_stats

chunk_files = pending
```

Then at the end, combine: `return done_stats + stats`

### 2. `score_chunk()` (line 1593) — single-chunk CLI path

At line ~1630 (after `filt_out` is defined), before `bcs = _chunk_bcs(...)`:

```python
# Resume: skip if output already exists
if filt_out.exists() and filt_out.stat().st_size > 64:
    _log(f"[assign/score] ● skip {chunk_file.name} (output exists)", verbose)
    return {"chunk": chunk_file.name, "reads": -1, "winner": -1,
            "ambig": -1, "ambig_bcs": -1, "elapsed": 0.0, "resumed": True}
```

## Design notes

- **Size threshold (> 64 bytes):** Empty gzip files are ~20-38 bytes. A valid filtered output with at least a header is larger. This matches the pattern used in genotyping.py's `_iter_shard_chunks`.
- **Stats use -1 for resumed chunks:** Distinguishes "skipped" from "empty" (which has reads=0). The caller in `cli.py` already handles this fine — it just sums stats for display.
- **No `--no-resume` flag needed:** The existing `--no-resume` on the `assign` CLI command isn't wired to scoring. If the user wants a clean re-run, they can delete the `cell_map_ref_chunks/` directory. This keeps it simple.
- **Batched path is the priority:** The single-chunk path (`score_chunk`) is a convenience wrapper; the batched path is what the parallel pipeline uses.

## Verification

1. Reinstall on HPC: `pip install .`
2. Resubmit the Root1 job — it should skip steps 1-3 (existing sentinels) and resume scoring from chunk ~1,313
3. Log should show `● skip` messages for completed chunks and `▶`/`■` for new ones
