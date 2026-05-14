# ECE 502C KV Caching Project

Group members: Alex Caulin-Cardo, Kutalp Dilber, Tyler Wilson

## Project Goal

This repository is for a simulation-based study of key-value (KV) cache
handling in LLM serving systems. The proposal focuses on the tradeoff between
recomputing KV cache and loading cached KV chunks from the memory hierarchy.

The main project idea is to evaluate a cost-aware cache policy that gives
higher priority to later token chunks. Later chunks are usually more expensive
to recompute, so the simulator will compare a Late-Token Priority policy
against simpler baselines such as compute-only, load-only, FIFO, and LRU. 

## Relationship to Cake

Cake shows that long-context prefill can be accelerated by using compute and
I/O in parallel: compute starts from the beginning of the prompt, while cache
loading starts from the end, and the two meet in the middle.

This project builds on Cake by studying a layer that Cake does not emphasize:
persistent global KV-cache management across requests. Instead of treating every
request independently, the simulator lets different requests share the same
`cache_id`, representing a repeated document, conversation prefix, or RAG
context whose KV chunks may remain in GPU/CPU memory.

The main research question becomes:

```text
How should a Cake-style compute/load scheduler interact with a global cache
policy that decides which reusable KV chunks stay in faster memory?
```

## Key Idea

Our idea is to extend the Cake scheduler idea by adding global cache-retention decisions
across requests. Cake decides how to compute and load KV cache within a single request, 
while our Late-Token Priority policy decides which reusable KV chunks should remain in faster 
memory for future requests. By combining the two, we are trying to reduce TTFT for 
repeated-prefix workloads.

## Changes Since Proposal
We specifically used the proposal feedback in these design choices:

- global cache keys via `cache_id`, separate from per-request `request_id`
- variable-size chunking hooks for block-size/paged-memory experiments
- explicit memory-tier bandwidth fields so later experiments can use measured
  GPU, CPU, disk, or network numbers

## Current Status

Implemented so far:

- `Request` / `KVChunk` / `MemoryTier` data models
- Global `cache_id` for cross-request KV reuse; per-request `request_id` for metrics
- `chunk_request_with_prefix_split`: splits a prompt into a **shared prefix region** (keyed by `cache_id`, reusable across requests that share the same document) and a **private suffix region** (keyed by `request_id`, unique to each request); variable-length boundary chunk ends exactly at `shared_prefix_tokens`
- Fixed-size and variable-size chunking helpers
- Policy-driven `CacheManager` with `store_replacing` (upgrades partial-coverage residents) and `MemoryTier.upsert` (never downgrades coverage)
- FIFO, LRU, and Late-Token Priority eviction policies
- Coverage-aware chunk lookup: a cache hit is only valid if the stored chunk covers the full needed token range (`end_token >= needed.end_token`)
- `CostModel`: attention cost scales with chunk position (later chunks are more expensive to recompute, matching Cake's core observation); load cost is byte-accurate from tier bandwidth
- `RecomputeLoadScheduler`: independent per-chunk recompute-vs-load decisions
- `CakeBidirectionalScheduler`: bidirectional two-front schedule with boolean-array progress tracking; I/O front skips uncached chunks at zero cost; Phase 2 sweeps remaining unscheduled chunks
- `simulate_cake_prefill_with_global_cache` / `simulate_linear_prefill_with_global_cache`: schedule chunks, then materialize KV onto a policy-managed GPU tier; **write-through on every RECOMPUTE** persists chunks to disk so later requests can LOAD instead of recompute; disk starts empty (causal, no pre-warming)
- `WorkloadConfig` / `generate_requests`: repeatable causal request streams; each request draws a variable-length shared prefix from a document pool plus an independent private suffix; supports `uniform`, `round_robin`, and `skewed` document-selection strategies
- Experiment driver `examples/run_cake_global_experiment.py`: Cake vs sequential baseline × FIFO / LRU / LTP across named presets; writes `results/metrics.csv`, `results/summary_by_scenario.csv`, five PNG figures, and `results/sim_debug.log`

Further documentation: [docs/ADDED_FEATURES.md](docs/ADDED_FEATURES.md).

## Repository Layout

```text
.
+-- docs/
|   +-- ADDED_FEATURES.md    # Cake+global cache integration, workload, experiments
+-- results/
|   +-- README.md            # experiments: outputs, presets, how to read CSVs & figures
+-- examples/
|   +-- basic_simulation.py
|   +-- cake_bidirectional_schedule.py
|   +-- cake_global_cache_prefill.py
|   +-- choose_recompute_vs_load.py
|   +-- compare_eviction_policies.py
|   +-- global_cache_reuse.py
|   +-- run_cake_global_experiment.py
+-- requirements.txt         # optional: matplotlib for experiment plots
+-- src/
|   +-- kv_cache_sim/
|       +-- __init__.py
|       +-- cache.py
|       +-- models.py
|       +-- policies.py
|       +-- prefill_simulation.py
|       +-- scheduler.py
|       +-- workload.py
|       +-- README.md
+-- README.md
```

## How to Run

Install the optional plotting dependency first:

```bash
pip install -r requirements.txt
```

**Run all presets (recommended)** — generates per-preset subfolders and cross-preset aggregate plots under `results/presets/`:

```powershell
# Windows
$env:PYTHONPATH="src"
python examples/run_cake_global_experiment.py --all-presets
```

```bash
# Linux / macOS
PYTHONPATH=src python examples/run_cake_global_experiment.py --all-presets
```

**Run a single named preset:**

```powershell
python examples/run_cake_global_experiment.py --preset hot_document_skew
```

**Run with default settings** (results go to `results/`):

```powershell
python examples/run_cake_global_experiment.py
```

Each run writes `metrics.csv`, `summary_by_scenario.csv`, five PNG figures, and `sim_debug.log` to its output directory. The `--all-presets` run additionally writes `metrics_all_presets.csv` and `summary_avg_across_presets.csv` to `results/presets/`.
