# Plan: Enhance `_reclass_within_topk()` with rescued flag + composite score

## Context

The current `_reclass_within_topk()` (genotyping.py:271) only uses AS to decide winner/ambiguous within the top-K genome context. This loses two valuable signals:

1. **The "rescued" flag from assign**: `_friend_rescue()` in assign_streaming.py labels ambiguous reads as "rescued" per-genome when they map within ±500bp of a winner read. This is discriminative — a read can be "rescued" in genome A but "ambiguous" in genome B. Currently, `_reclass_within_topk` overwrites this flag entirely.

2. **MAPQ and NM**: The assign step uses all three metrics (NM, AS, MAPQ) jointly, but the re-classification only compares AS. Reads with identical AS but different MAPQ/NM across top-K genomes are missed.

## File to modify

`/Volumes/lsa-amarand/fabio_home/Projects/00_Repos/ambientmapper/src/ambientmapper/genotyping.py`

## Changes

### 1. Replace `_reclass_within_topk` body (lines 271–308)

**Signature:** `_reclass_within_topk(df)` → `_reclass_within_topk(df, cfg: MergeConfig)`

**New three-tier classification logic (all vectorized numpy, no Python loops):**

**Tier 1 — Original winner:** If a read has exactly 1 genome with `assigned_class == "winner"` in the top-K set → preserve it. Rationale: the assign step already proved this read unambiguously belongs to that genome across all genomes.

**Tier 2 — Rescued flag:** For reads not resolved by Tier 1, count genomes with `assigned_class == "rescued"`. If exactly 1 → promote to winner. If 0 or 2+ → next tier. Rationale: per-genome spatial proximity to a winner is discriminative evidence, but only when it's unique to one genome.

**Tier 3 — Composite score:** For remaining reads, compute `score = AS * cfg.w_as + MAPQ * cfg.w_mapq - NM * cfg.w_nm` (same formula as `_compute_read_posteriors`, line 680). If exactly 1 genome has the best score → winner. If tied → ambiguous.

**Edge cases handled:**
- Winner in genome A + rescued in genome B → Tier 1 picks A (winner > rescued)
- Rescued in 2+ genomes → falls to Tier 3 (non-discriminative)
- All "ambiguous" (no rescued/winner) → Tier 3 only (backward-compatible path)
- Missing MAPQ/NM columns → return df unchanged (guard clause)

### 2. Update call site (line 1596)

```python
# Before:
df = _reclass_within_topk(df)
# After:
df = _reclass_within_topk(df, cfg)
```

### 3. Update docstring

Describe the three-tier priority system and the role of the rescued flag.

## What does NOT change

- `MergeConfig` — no new parameters (uses existing `w_as`, `w_mapq`, `w_nm`)
- `_compute_read_posteriors` — unchanged (already treats "rescued" like "winner")
- DuckDB/pandas reduction paths — "rescued" already survives the aggregation
- `_filter_promiscuous_ambiguous_reads` — only filters "ambiguous", not "rescued" (correct)
- No new CLI flags

## Backward compatibility

When all input `assigned_class` values are "ambiguous" (datasets without `_friend_rescue`, or old assign outputs): Tier 1 and Tier 2 resolve zero reads → Tier 3 applies composite score to all. With default weights (w_as=1.0, w_mapq=1.0, w_nm=1.0), the composite score adds MAPQ and NM discrimination beyond the current AS-only logic. This is the intended enhancement.

## Verification

1. After editing, reinstall on HPC: `pip install .` from the ambientmapper repo root
2. The currently running `01_step1-4_Root1.sh` produces assign outputs with the "rescued" flag — once complete, run genotyping with `--topk-reclass` to exercise the new logic
3. Check that the calls output shows more "single_clean" calls (rescued reads promoted to winners should improve singlet detection)
