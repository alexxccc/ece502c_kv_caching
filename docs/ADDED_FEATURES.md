# Added features (Cake + global cache, workload, experiments)

This note summarizes modules and scripts added after the original proposal
examples: integration of the Cake bidirectional scheduler with the policy-driven
GPU cache, synthetic workloads, and batch experiments with CSV plus plots.

## New Python modules (`src/kv_cache_sim/`)

### `prefill_simulation.py`

Connects scheduling decisions to **materializing** KV on the managed GPU tier:

- **`simulate_cake_prefill_with_global_cache`** — Runs
  `CakeBidirectionalScheduler` against `[gpu_tier, …cold_tiers]`, then walks
  chunks in **chunk-index order** and:
  - **RECOMPUTE**: `CacheManager.store` for the request chunk (may evict via policy).
  - **LOAD from GPU**: `CacheManager.access` (hit / LRU-style bookkeeping).
  - **LOAD from cold tier** (e.g. disk): `CacheManager.store` with the cold-tier
    chunk (pulls KV into GPU under the same `(cache_id, chunk_index)` key).

Returns `(CakeScheduleSummary, PrefillMetrics)` with estimated TTFT, recompute/load
counts, eviction count, and post-prefill GPU usage.

- **`simulate_linear_prefill_with_global_cache`** — Same materialization path for
  `RecomputeLoadScheduler` (per-chunk recompute vs load, **no** Cake two-front
  timeline). TTFT is approximated as the **sum** of per-chunk times
  (`LinearScheduleMode.SUM`) by default, for a sequential-prefill baseline.

- **`PrefillMetrics`** — One prefill’s accounting: `estimated_ttft_ms`,
`recompute_count`, `load_count`, `eviction_count`, `gpu_chunks_after`,
`gpu_used_bytes`.

- **`LinearScheduleMode`** — `SUM` or `MAX` for aggregating linear schedule times
  (default for the linear baseline is `SUM`).

### `workload.py`

Synthetic request streams for multi-request experiments:

- **`WorkloadConfig`** — `seed`, `num_requests`, `num_documents`, prompt length
  range, optional `arrival_gap_ms`, **`document_mix`** (`uniform`, `round_robin`,
  `skewed`) and **`skew_hot_probability`** for hot-document traffic.
- **`generate_requests`** — Document pool with optional **uniform**, strict
  **round-robin**, or **skewed** selection toward `doc-0`; each `cache_id` keeps a
  fixed prompt length per workload draw.
- **`seed_tier_from_request`** — Store all chunks for one `Request` on a raw
  `MemoryTier` (no eviction policy; use for warm disk).
- **`seed_disk_for_workload`** — For every `cache_id` in a request list, seeds
  the **longest** prompt seen for that id so cold storage can satisfy loads.
- **`make_disk_tier`** — Convenience constructor for a large disk-like tier.

## New examples (`examples/`)

| Script | Purpose |
|--------|--------|
| `cake_global_cache_prefill.py` | Short demo: two requests, shared `cache_id`, tight GPU + disk seed; prints TTFT, recompute/load, evictions. |
| `run_cake_global_experiment.py` | Batch comparison: **Cake** vs **linear** scheduling × **LTP / FIFO / LRU** on GPU; presets (`--preset`), **`--schedulers cake_only`**, **`--attention-mult`**, workload mix flags; CSV + bar charts (including total recompute/load). See **`docs/eviction_experiment_scenarios.md`**. |

## Dependencies

- **`requirements.txt`** — `matplotlib` (optional for figures; CSV is always written).

## Experiment outputs

`run_cake_global_experiment.py` writes to `--output-dir` (default: `results/`):

- `metrics.csv` — One row per (scenario, request) with TTFT, recompute/load counts,
  evictions, GPU occupancy after prefill.
- `summary_by_scenario.csv` — Aggregated mean TTFT and totals per scenario.
- `fig_mean_ttft.png` — Mean estimated TTFT by scenario.
- `fig_total_evictions.png` — Total GPU evictions across all requests by scenario.
- `fig_total_recompute.png` / `fig_total_load.png` — Aggressive totals for policy comparisons.

### Example commands

From the repo root (Linux/macOS):

```bash
export PYTHONPATH=src
pip install -r requirements.txt
python examples/cake_global_cache_prefill.py
python examples/run_cake_global_experiment.py
```

Custom run:

```bash
python examples/run_cake_global_experiment.py --seed 1 --num-requests 64 \
  --gpu-mb 4 --output-dir results/run1
```

## Design caveat

KV is written to the GPU tier in **three phases**: all **LOAD from GPU** (access)
first, then **LOAD from cold tiers** (store), then **RECOMPUTE** (store). Pure
chunk-index order would sometimes run RECOMPUTE before LOAD-from-GPU for another
chunk and evict data the scheduler assumed was still resident. TTFT estimates
still come only from the scheduler timeline (Cake: `max(compute_front, load_front)`),
not from this materialization order.

## Relation to original “next steps”

| Original next step | Where it lives |
|--------------------|----------------|
| Cake + global retention + Late-Token (or other) eviction | `simulate_cake_prefill_with_global_cache` + `CacheManager(policy=…)` |
| Synthetic workload | `workload.py`, `generate_requests` |
| Compare Cake+global vs Cake vs global-style baselines | `run_cake_global_experiment.py` (Cake vs linear; policies per GPU tier) |
| Report metrics and graphs | CSV + PNG under `results/` |
