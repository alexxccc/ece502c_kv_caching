# `kv_cache_sim`

This package will hold the simulator for the project.

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

## Global Cache Identity

Requests have both a `request_id` and a `cache_id`.

- `request_id` identifies one user request for metrics.
- `cache_id` identifies the reusable KV-cache prefix shared across requests.

This distinction lets the simulator model persistent CPU/GPU cache reuse across
requests, which is the main improvement opportunity over treating each Cake
request independently.

## Planned Modules

```text
src/kv_cache_sim/
├── models.py          # core request/cache/memory objects
├── cache.py           # policy-driven cache manager
├── policies.py        # FIFO, LRU, Late-Token Priority
├── scheduler.py       # recompute-vs-load decisions
├── workloads.py       # synthetic prompt and concurrency generation
├── metrics.py         # latency, throughput, hit-rate collection
└── experiments.py     # reusable experiment runner
```

## Design Notes

The simulator should stay separate from any vLLM/LMCache integration work. That
keeps the project runnable on normal class hardware while still letting us model
the operating-systems questions from the proposal: scheduling, contention,
memory hierarchy behavior, and eviction policy.

The next scheduler layer should include a Cake-style bidirectional baseline:
compute chunks from the beginning, load chunks from the end, and stop when the
two fronts meet. The project extension is to combine that with global cache
retention policies such as Late-Token Priority.
