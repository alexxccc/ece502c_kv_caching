# `kv_cache_sim`

This package holds the simulator for the project.

## What Exists Now

- `Request`: a synthetic LLM serving request with `cache_id` (shared document key) and `shared_prefix_tokens` (how many prefix tokens this request borrows from its document).
- `KVChunk`: a contiguous piece of KV cache; `cache_key = (cache_id, chunk_index)`.
- `MemoryTier`: a storage tier (GPU, CPU, disk) with capacity and bandwidth; `upsert` never downgrades chunk coverage.
- `chunk_request_with_prefix_split`: splits a prompt into a shared-prefix region (keyed by `cache_id`) and a private-suffix region (keyed by `request_id`); variable boundary chunk ends exactly at `shared_prefix_tokens`.
- `chunk_request` / `chunk_request_with_sizes`: fixed-size and variable-size helpers.
- `CacheManager`: policy-driven cache with `store` (evict to fit), `store_replacing` (upgrade partial-coverage residents), and `access` (LRU hit accounting).
- `FIFOPolicy`: evicts the oldest stored chunk.
- `LRUPolicy`: evicts the least recently accessed chunk.
- `LateTokenPriorityPolicy`: evicts earlier prompt chunks first (lower `end_token` = lower priority).
- `CostModel`: recompute cost grows with token position (attention looks back further); load cost is byte-accurate from tier bandwidth.
- `RecomputeLoadScheduler`: independent per-chunk recompute-vs-load decisions with coverage-aware cache lookup.
- `CakeBidirectionalScheduler`: bidirectional two-front schedule; I/O front scans backward skipping uncached chunks; boolean array tracks committed chunks; Phase 2 sweeps remainder.
- `prefill_simulation`: `simulate_cake_prefill_with_global_cache` and `simulate_linear_prefill_with_global_cache`; write-through on every RECOMPUTE; disk starts empty (causal simulation); returns `PrefillMetrics` including `coverage_miss_count`.
- `workload`: `WorkloadConfig`, `generate_requests`; supports `uniform`, `round_robin`, `skewed` document selection; no pre-warming.

See also [docs/ADDED_FEATURES.md](../../docs/ADDED_FEATURES.md) in the repo root.

## Global Cache Identity

Requests have both a `request_id` and a `cache_id`.

- `request_id` identifies one user request for metrics.
- `cache_id` identifies the reusable KV-cache prefix shared across requests.

This distinction lets the simulator model persistent CPU/GPU cache reuse across
requests, which is the main improvement opportunity over treating each Cake
request independently.

## Layout

```text
src/kv_cache_sim/
+-- models.py             # core request/cache/memory objects
+-- cache.py              # policy-driven cache manager
+-- policies.py           # FIFO, LRU, Late-Token Priority
+-- scheduler.py          # recompute-vs-load and Cake-style scheduling
+-- prefill_simulation.py # Cake/linear + global cache materialization
+-- workload.py           # synthetic requests and cold-tier seeding
```

## Design Notes

The simulator should stay separate from any vLLM/LMCache integration work. That
keeps the project runnable on normal class hardware while still letting us model
the operating-systems questions from the proposal: scheduling, contention,
memory hierarchy behavior, and eviction policy.

A request now carries `shared_prefix_tokens` alongside `prompt_tokens`:

```python
request = Request(
    request_id="req-0",
    cache_id="doc-0",          # shared across requests that use the same document
    arrival_time_ms=0.0,
    prompt_tokens=1024,
    shared_prefix_tokens=768,  # first 768 tokens are the reusable document prefix
)
chunks = chunk_request_with_prefix_split(request, chunk_size_tokens=256)
# chunks 0-2 use cache_id="doc-0"  (shared, reusable)
# chunk  3   uses cache_id="req-0-private"  (private suffix, not shared)
```

## Source Code Notes

The scheduler module currently includes two schedulers:

- `RecomputeLoadScheduler`: makes independent per-chunk decisions.
- `CakeBidirectionalScheduler`: runs a Cake-style baseline with compute moving
  forward from the beginning and I/O loading backward from the end.

## Extension (implemented)

Cake-style scheduling is combined with global cache retention via
`simulate_cake_prefill_with_global_cache` and a `CacheManager` using policies
such as `LateTokenPriorityPolicy` (see `examples/cake_global_cache_prefill.py`
and `examples/run_cake_global_experiment.py`).
