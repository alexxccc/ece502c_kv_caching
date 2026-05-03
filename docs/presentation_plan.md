# Presentation plan (current simulator)

Use this outline for a professor check-in or course presentation. It matches what is **implemented in the repo today** (no live vLLM / production stack).

---

## What “Cake” vs “linear” means in *our* code

Both modes use the **same** `CostModel` and the **same** tiers (GPU + disk). For each chunk, the model still asks: *recompute* or *load* from a tier? The difference is **how per-chunk work is turned into a prefill “TTFT” estimate** for that request.

| Mode | Scheduler in code | What it models |
|------|-------------------|----------------|
| **Cake** | `CakeBidirectionalScheduler` | **Bidirectional** prefill (as in the Cake paper’s idea): a **compute front** walks **forward** from chunk 0; an **I/O front** walks **backward** from the last chunk. The schedule interleaves the two. **Estimated TTFT** for the prefill is `max(compute_timeline, load_timeline)` — work can **overlap** in the model, so the bottleneck is the **slower** of the two “pipes.” |
| **Linear** | `RecomputeLoadScheduler` | **No** two-front schedule. Every chunk is decided **independently** (recompute vs load). There is **no** built-in overlap between compute and I/O in the timeline. For reporting, we set **estimated TTFT** to the **sum** of per-chunk times (`LinearScheduleMode.SUM`) — a **sequential** prefill upper bound / baseline, *not* the Cake parallel story. |

**Takeaway for slides:** *Cake* = parallel compute vs I/O **across the two fronts**; *linear* = same per-chunk decisions, but **no** bidirectional scheduling, and we report a **sum** as a simple **non-Cake** baseline for comparison on the **same** workload and **same** global cache.

**Caveat:** This is a **discrete-event style estimate**, not a packet-level or kernel-level schedule. It is enough to compare **policies and schedulers** on the same assumptions.

---

## Suggested talk length

- **10–12 minutes:** sections 1–4, one figure, one table, limitations.
- **20 minutes:** add live or recorded demo, deeper results, Q&A on design.

---

## 1. Setup (1–2 min)

- **Course / project goal:** study **KV cache** in LLM **inference**: recompute vs load, **prefix reuse** across requests, **limited** fast memory.
- **Reference paper:** Cake — *“Compute Or Load KV Cache? Why Not Both?”* — bidirectional **compute forward / load backward** to cut **TTFT** when prefix KV can be loaded from **slow** storage.
- **Our twist:** Cake focuses on **one** prefill and **compute vs I/O**. We add **global cache management on a small GPU** across **many** requests (same `cache_id` = shared prefix). **Eviction policy** (FIFO, LRU, Late-Token Priority) decides **what stays** for the **next** request.

---

## 2. Problem statement (1–2 min)

- Long prompts → expensive **prefill**; **prefix caching** avoids redoing work.
- Fast GPU memory is **small** → **evictions**; not everything can stay hot.
- **Research question (one sentence):** How does combining **Cake-style** prefill scheduling with a **global** eviction policy (especially **Late-Token Priority**) affect **estimated TTFT** and **recompute vs load** vs **FIFO/LRU** and vs a **linear** prefill baseline, under **repeated-prefix** traffic?

---

## 3. What we built (simulator) (2–3 min)

Keep this **diagram-first**:

| Piece | Role |
|-------|------|
| `Request` / `KVChunk` | Synthetic requests; chunks keyed by `(cache_id, chunk_index)`. |
| `MemoryTier` | GPU (managed + capacity-limited), disk (cold, seeded in experiments). |
| `CacheManager` + policy | **FIFO**, **LRU**, **Late-Token Priority** — eviction when storing into GPU. |
| `CostModel` | Estimates **recompute** time (grows with position) vs **load** time (bandwidth). |
| **Cake path** | `simulate_cake_prefill_with_global_cache` → TTFT from bidirectional schedule; then **materialize** KV onto GPU (counts evictions). |
| **Linear path** | `simulate_linear_prefill_with_global_cache` → TTFT as **sum** of chunk times; same materialization story. |
| **Workload** | `WorkloadConfig` + `generate_requests` — multiple documents, repeated `cache_id`s; `seed_disk_for_workload` optional cold Tier KV. |

**Optional one sentence:** Materialization applies GPU/cold stores in **phases** (GPU hits → cold loads → recomputes) so eviction order does not violate “scheduler assumed this chunk was still on GPU” for the same prefill.

---

## 4. Experiments you can show *today* (2–4 min)

**Artifacts (generate before the talk):**

```bash
export PYTHONPATH=src
pip install -r requirements.txt   # for plots

python examples/cake_global_cache_prefill.py
python examples/run_cake_global_experiment.py --output-dir results
```

| Output | Use in slides |
|--------|----------------|
| `results/summary_by_scenario.csv` | **Table:** mean TTFT, total evictions, total recompute/load per scenario. |
| `results/fig_mean_ttft.png` | **Bar chart:** mean estimated TTFT — Cake vs linear × policies. |
| `results/fig_total_evictions.png` | **Bar chart:** pressure on GPU — policies differ when capacity is tight. |
| `results/metrics.csv` | Backup / appendix; per-request drill-down if asked. |

**Scenarios (from `run_cake_global_experiment.py`):**  
`cake_ltp`, `cake_fifo`, `cake_lru`, `linear_ltp`, `linear_fifo` — same workload, fresh GPU + seeded disk each scenario.

**Knobs to mention:** `--gpu-mb`, `--num-requests`, `--num-documents`, `--seed`, prompt range — **reproducibility**.

---

## 5. Insights to argue (honest) (1–2 min)

- **Under pressure** (small GPU, many repeats), **eviction policy** and **Cake vs linear** can both move metrics; **under loose capacity**, curves may **collapse** — report that if it happens (still a valid result).
- **Late-Token Priority** is aligned with the cost model (later tokens **more expensive** to recompute); ask whether it reduces **recomputes** or **mean TTFT** vs FIFO on **your** CSV.
- **Simulation**, not production: **insights are qualitative + comparative**, not absolute latency claims.

---

## 6. Limitations & future work (1 min)

- Chunk-level abstraction; not per-layer KV or real kernels.
- Cost model is **parametric** — sweep or calibrate if you extend the report.
- Disk seeding assumes **cold tier already has** full-document KV — good for isolating **GPU** contention; say you could add **cold-start** requests without seed.

---

## 7. Slide outline (copy-paste skeleton)

1. Title + group + course.
2. Problem: long prefill, prefix reuse, small GPU.
3. Cake (paper): bidirectional compute/load; **our** extension: **cross-request** GPU cache + eviction.
4. Simulator diagram (`cache_id`, GPU, disk, policies).
5. **Cake vs linear** (this doc, top section).
6. Workload + setup (seed, documents, GPU MB).
7. **Results:** 2 figures + 1 table from `results/`.
8. Takeaways + limitations.
9. Thank you / questions.

---

## 8. Backup questions (short answers)

- **Is this vLLM?** No — **simulation** for OS/architecture-style tradeoffs.
- **How is prefix “detected”?** In the simulator, **`cache_id`** is explicit; real systems use **token-hash / radix tree** lookups (see prior discussion).
- **Why recompute if disk is seeded?** Scheduler picks **min cost** per chunk; recompute can still win if **cheaper** than reading slow storage for that chunk.

---

## File map for the presenter

| Path | Content |
|------|---------|
| `cake.pdf` | Cake reference paper (PDF in repo). |
| `docs/ADDED_FEATURES.md` | Module-level documentation for integration + experiments. |
| `docs/eviction_experiment_scenarios.md` | Presets and workloads that stress FIFO vs LRU vs Late-Token Priority; metrics beyond mean TTFT. |
| `examples/cake_global_cache_prefill.py` | Short 2-request demo. |
| `examples/run_cake_global_experiment.py` | Batch comparison + CSV + PNG. |
| `src/kv_cache_sim/prefill_simulation.py` | Cake vs linear + GPU materialization. |
| `src/kv_cache_sim/workload.py` | Synthetic requests + disk seeding. |
