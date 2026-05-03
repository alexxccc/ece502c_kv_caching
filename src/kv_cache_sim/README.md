# `kv_cache_sim`

This package holds the simulator for the project.

## What Exists Now

- `Request`: describes a synthetic LLM serving request.
- `KVChunk`: describes a contiguous piece of KV cache.
- `MemoryTier`: represents a storage tier such as GPU, CPU, or disk.
- `chunk_request`: splits a prompt into KV chunks for simulation.
- `chunk_request_with_sizes`: splits a prompt using variable chunk sizes.
- `CacheManager`: stores chunks in a tier and evicts when the tier is full.
- `FIFOPolicy`: evicts the oldest stored chunk.
- `LRUPolicy`: evicts the least recently accessed chunk.
- `LateTokenPriorityPolicy`: evicts earlier prompt chunks first.
- `CostModel`: estimates recompute and load time.
- `RecomputeLoadScheduler`: chooses recompute vs load for each chunk.
- `CakeBidirectionalScheduler`: simulates Cake's two-front schedule.
- `prefill_simulation`: after Cake or linear scheduling, materialize KV on a
  `CacheManager` GPU tier (and optional cold tiers); returns `PrefillMetrics`.
- `workload`: `WorkloadConfig`, `generate_requests`, and helpers to seed disk-like
  tiers for cold-load experiments.

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

The inputs of the current simulator are very simple and in the form:
    request = Request(
        request_id="request-001",
        cache_id="document-prefix-42",
        arrival_time_ms=0.0,
        prompt_tokens=4096,
    )
  It still allows for the patterns of the scheduler to be determined.

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
