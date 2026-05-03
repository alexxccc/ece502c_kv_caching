# Eviction-policy scenarios (insights-focused)

Default runs often show **Cake vs linear** clearly but **FIFO vs LRU vs Late-Token Priority (LTP)** only weakly on **mean TTFT** and **total eviction count**. That is expected when:

- GPU pressure is moderate,
- the workload treats documents similarly,
- **mean TTFT** is dominated by the **scheduler math**, not **which** chunk survived.

This guide defines **stress patterns** and **metrics** so you can argue policy differences with evidence.

---

## What to look at (better than mean TTFT alone)

| Metric | Why it helps eviction analysis |
|--------|--------------------------------|
| **Total recompute** (`summary_by_scenario.csv` / `fig_total_recompute.png`) | Policies that retain **useful** chunks should shift work from **recompute** toward **load** when disk is seeded. |
| **Total load** (`fig_total_load.png`) | Complement to recomputes; sum with workload choices reflects **hit structure**. |
| **Per-request rows** in `metrics.csv` | Spot bursts where one policy pays extra recomputes. |

**Interpretation caution:** Higher load is not automatically “better”—it depends whether loads are **cheap GPU hits** vs **cold disk**. Here all loads count together; pair with **recompute** and narrative.

---

## Workload dimensions (CLI)

Implemented in `WorkloadConfig` / `examples/run_cake_global_experiment.py`:

| Knob | Role |
|------|------|
| **`--gpu-mb`** | Primary eviction pressure. Try **2** MiB when chunk ≈ 1 MiB (1024 tokens × 1024 B/token). |
| **`--document-mix`** | `uniform` (default), **`round_robin`**, **`skewed`**. |
| **`--skew-hot-probability`** | With `skewed`, fraction of requests hitting **`doc-0`** (rest uniform over docs). Stresses **LRU vs LTP** differently. |
| **`--attention-mult`** | Scales `attention_ms_per_token_position`. **Raises cost of late-token recompute**—aligns LTP’s story (“protect tail”). |
| **`--schedulers cake_only`** | Drops linear baselines so figures emphasize **only** FIFO vs LRU vs LTP under Cake. |

---

## Named presets (`--preset`)

Bundles are applied **after** defaults; you can still override with explicit flags **before** `--preset`… actually presets overwrite — pass `--preset` last or rely on preset bundle.

| Preset | Intent |
|--------|--------|
| **`single_long_prefix`** | One document, **8192** tokens, **2 MiB** GPU → GPU holds only **2** chunks; **heavy churn** on one long prefix. Good for **which chunk survives**. |
| **`hot_document_skew`** | Many docs but **~92%** traffic on **`doc-0`** → **recency** favors LRU on the hot doc; LTP still prioritizes **positional** value. |
| **`alternating_two_docs`** | **`round_robin`** between two docs → periodic switching; **FIFO vs LRU** often diverge more than under uniform random. |
| **`steep_late_recompute_cost`** | **Large `attention_mult`** + tighter GPU → mistakes evicting **late** chunks hurt more in the **cost model**; LTP may separate from FIFO on **recompute totals**. |

Example:

```bash
export PYTHONPATH=src

python examples/run_cake_global_experiment.py \
  --preset hot_document_skew \
  --schedulers cake_only \
  --output-dir results/evict_hot_doc

python examples/run_cake_global_experiment.py \
  --preset single_long_prefix \
  --schedulers cake_only \
  --seed 0 \
  --output-dir results/evict_single_long
```

---

## Scenario-by-scenario narratives (what to say)

### A — Single long prefix, tiny GPU (`single_long_prefix`)

- **Setup:** One shared document, prompt ~8k tokens, GPU holds **two** chunks worth.
- **Hypothesis:** LTP keeps **later** chunks when forced to evict **early** ones; if **future** requests need **full** length, retaining **tail** may reduce **expensive** recomputes vs FIFO evicting “oldest insert” regardless of position.
- **Watch:** `total_recompute` for **cake_ltp** vs **cake_fifo** / **cake_lru**.

### B — Hot document (`hot_document_skew`)

- **Setup:** Most requests hit **`doc-0`**, others occasionally other docs.
- **Hypothesis:** **LRU** keeps lines touched recently → strong retention for **hot** doc’s chunks when traffic clusters. **LTP** ignores recency → may keep **positional** preference instead; differences show up in **which** chunks survive under misses for **cold** docs.
- **Watch:** Compare policies on **total_load** / **total_recompute** when alternating cold-doc phases (extend trace length if needed).

### C — Alternating two documents (`alternating_two_docs`)

- **Setup:** `doc-0`, `doc-1`, `doc-0`, … strict alternation.
- **Hypothesis:** **FIFO** evicts by insertion order; **LRU** reacts to **access** patterns each prefill. Creates **phase shifts** in GPU contents vs uniform traffic.
- **Watch:** Run longer **`--num-requests`** (e.g. 120); inspect **`metrics.csv`** for oscillating **recompute_count** per request.

### D — Steep late-token compute cost (`steep_late_recompute_cost`)

- **Setup:** **`attention_mult`** large (preset uses **120×** base attention term).
- **Hypothesis:** Scheduler prefers **load** over **recompute** more often for **late** chunks; losing late KV forces **expensive** recomputes—**LTP** (evict **early** chunks first) should reduce **tail** recomputes vs policies that evict **recent** or **FIFO** blocks that might include **late** chunks depending on order.
- **Watch:** **cake_ltp** **total_recompute** vs others; **mean TTFT** may move slightly.

---

## If policies still look identical

That is a **valid finding**:

1. Report that under **your** cost parameters and disk seed, **scheduler dominates** and eviction is **second-order**.
2. Narrow GPU further or lengthen prompts until **`gpu_chunks_after`** in CSV shows sustained **partial** occupancy.
3. Add **per-document** metrics (future work): aggregate recomputes for **`cache_id == doc-0`** only.

---

## Figure outputs (after code update)

Run produces:

- `fig_mean_ttft.png`
- `fig_total_evictions.png`
- **`fig_total_recompute.png`** — policy sensitivity
- **`fig_total_load.png`** — policy sensitivity

Same workload seed → comparable bars across scenarios.
