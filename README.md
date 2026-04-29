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

The professor's proposal feedback is reflected in three design choices:

- global cache keys via `cache_id`, separate from per-request `request_id`
- variable-size chunking hooks for block-size/paged-memory experiments
- explicit memory-tier bandwidth fields so later experiments can use measured
  GPU, CPU, disk, or network numbers

## Current Status

This version contains the project scaffold, core simulator data models, the
first eviction policies, and a cost-aware recompute-vs-load scheduler.

Implemented so far:

- request and KV chunk data models
- global cache IDs for cross-request KV reuse
- memory-tier model for GPU, CPU, and disk-like storage
- cache-placement helpers
- fixed-size and variable-size chunking helpers
- policy-driven cache manager
- FIFO, LRU, and Late-Token Priority eviction policies
- cost model for recompute-vs-load scheduling
- scheduler that chooses whether each chunk should be recomputed or loaded
- a small example script that builds a toy cache state
- a policy comparison script that forces evictions
- a global-cache reuse example with two requests sharing one prefix
- a recompute-vs-load scheduling example

Next steps:

1. Add a Cake-style bidirectional scheduler using the same cost model.
2. Add synthetic workload generation for 4k-16k token prompts.
3. Add experiment runners and plotting scripts.
4. Add final-report metrics and graphs.

## Repository Layout

```text
.
+-- examples/
|   +-- basic_simulation.py
|   +-- choose_recompute_vs_load.py
|   +-- compare_eviction_policies.py
|   +-- global_cache_reuse.py
+-- src/
|   +-- kv_cache_sim/
|       +-- __init__.py
|       +-- cache.py
|       +-- models.py
|       +-- policies.py
|       +-- scheduler.py
|       +-- README.md
+-- README.md
```

## Running the Example

From this directory:

```powershell
python .\examples\basic_simulation.py
python .\examples\choose_recompute_vs_load.py
python .\examples\compare_eviction_policies.py
python .\examples\global_cache_reuse.py
```

The basic example prints a few synthetic KV chunks, places them into a GPU
memory tier, and shows how much capacity remains. The policy comparison example
fills a small GPU tier and shows which chunk each eviction policy removes. The
global-cache example shows two different requests reusing the same cached prefix
through a shared `cache_id`. The scheduler example estimates recompute and load
time for each chunk, then chooses the cheaper action.
