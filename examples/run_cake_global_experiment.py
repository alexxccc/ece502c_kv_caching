"""
Compare Cake + global cache (eviction policies) vs linear scheduling + global cache.

Requests are generated causally: each request draws a random prefix of a document
from the pool (shared portion) plus a private suffix.  No cache tier is pre-warmed;
the first request to touch a prefix must recompute it, and every recomputed chunk
is written through to an effectively-unbounded disk tier so later requests can load
rather than recompute.

Usage
-----
Single run (default or named preset):
    python examples/run_cake_global_experiment.py
    python examples/run_cake_global_experiment.py --preset hot_document_skew

All presets — per-preset subfolders + cross-preset aggregate plots:
    python examples/run_cake_global_experiment.py --all-presets
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
import logging
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from kv_cache_sim.cache import CacheManager
from kv_cache_sim.models import (
    CachePlacement,
    MemoryTier,
    Request,
    chunk_request_with_prefix_split,
)
from kv_cache_sim.policies import EvictionPolicy, FIFOPolicy, FrequencyTokenPriorityPolicy, LateTokenPriorityPolicy, LRUPolicy
from kv_cache_sim.prefill_simulation import (
    LinearScheduleMode,
    PrefillMetrics,
    simulate_cake_prefill_with_global_cache,
    simulate_linear_prefill_with_global_cache,
)
from kv_cache_sim.scheduler import CostModel
from kv_cache_sim.workload import (
    WorkloadConfig,
    generate_requests,
    make_disk_tier,
)

_DISK_CAPACITY_BYTES = 2**62


@dataclass(frozen=True)
class Scenario:
    name: str
    policy: EvictionPolicy
    use_cake: bool


def run_scenario(
    scenario: Scenario,
    requests: list[Request],
    chunk_size_tokens: int,
    bytes_per_token: int,
    gpu_capacity_bytes: int,
    gpu_bandwidth_bytes_per_ms: float,
    disk: MemoryTier,
    cost_model: CostModel,
) -> list[PrefillMetrics]:
    gpu = MemoryTier(
        name=CachePlacement.GPU,
        capacity_bytes=gpu_capacity_bytes,
        bandwidth_bytes_per_ms=gpu_bandwidth_bytes_per_ms,
    )
    cache = CacheManager(tier=gpu, policy=scenario.policy)
    metrics_list: list[PrefillMetrics] = []

    for request in requests:
        chunks = chunk_request_with_prefix_split(
            request, chunk_size_tokens, bytes_per_token
        )
        if scenario.use_cake:
            _, metrics = simulate_cake_prefill_with_global_cache(
                chunks,
                cache,
                cost_model,
                cold_tiers=[disk],
                write_through_tiers=[disk],
            )
        else:
            _, metrics = simulate_linear_prefill_with_global_cache(
                chunks,
                cache,
                cost_model,
                cold_tiers=[disk],
                write_through_tiers=[disk],
                aggregate=LinearScheduleMode.SUM,
            )
        metrics_list.append(metrics)

    return metrics_list


PRESETS: dict[str, dict[str, Any]] = {
    # One document; requests draw variable-depth slices of a very long context.
    "single_long_prefix": {
        "num_documents": 1,
        "num_requests": 64,
        "doc_length_min": 8192,
        "doc_length_max": 8192,
        "shared_prefix_min": 1024,
        "shared_prefix_max": 8192,
        "private_suffix_min": 128,
        "private_suffix_max": 256,
        "gpu_mb": 4,
        "document_mix": "uniform",
    },
    # Six documents with one dominant hot doc (92 % probability).
    "hot_document_skew": {
        "num_documents": 6,
        "num_requests": 128,
        "doc_length_min": 2048,
        "doc_length_max": 8192,
        "shared_prefix_min": 256,
        "shared_prefix_max": 4096,
        "private_suffix_min": 128,
        "private_suffix_max": 512,
        "gpu_mb": 8,
        "document_mix": "skewed",
        "skew_hot_probability": 0.92,
    },
    # Two equal-length documents served round-robin.
    "alternating_two_docs": {
        "num_documents": 2,
        "num_requests": 128,
        "doc_length_min": 4096,
        "doc_length_max": 4096,
        "shared_prefix_min": 512,
        "shared_prefix_max": 4096,
        "private_suffix_min": 128,
        "private_suffix_max": 256,
        "gpu_mb": 4,
        "document_mix": "round_robin",
    },
    # Inflated attention cost makes late-sequence recompute expensive.
    "steep_late_recompute_cost": {
        "attention_mult": 4.0,
        "gpu_mb": 8,
        "num_requests": 128,
    },
}


def _tight_ylim(ax: Any, values: list[float]) -> None:
    if not values or max(values) == 0:
        return
    lo, hi = min(values), max(values)
    pad = max((hi - lo) * 0.15, hi * 0.05)
    ax.set_ylim(bottom=max(0.0, lo - pad), top=hi + pad)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _make_scenarios(schedulers: str) -> tuple[Scenario, ...]:
    cake = (
        Scenario("cake_ltp", LateTokenPriorityPolicy(), True),
        Scenario("cake_fifo", FIFOPolicy(), True),
        Scenario("cake_lru", LRUPolicy(), True),
        Scenario("cake_fltp", FrequencyTokenPriorityPolicy(), True),
    )
    linear = (
        Scenario("linear_ltp", LateTokenPriorityPolicy(), False),
        Scenario("linear_fifo", FIFOPolicy(), False),
        Scenario("linear_lru", LRUPolicy(), False),
        Scenario("linear_fltp", FrequencyTokenPriorityPolicy(), False),
    )
    if schedulers == "cake_only":
        return cake
    if schedulers == "linear_only":
        return linear
    return cake + linear


def _run_one_preset(
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[dict[str, dict[str, float]], list[dict[str, Any]]]:
    """Run all scenarios for one configuration, save outputs, return summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-run log file.
    log_handler = logging.FileHandler(
        output_dir / "sim_debug.log", mode="w", encoding="utf-8"
    )
    log_handler.setLevel(logging.DEBUG)
    log_handler.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))
    root_logger = logging.getLogger("kv_cache_sim")
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(log_handler)

    config = WorkloadConfig(
        seed=args.seed,
        num_requests=args.num_requests,
        num_documents=args.num_documents,
        doc_length_min=args.doc_length_min,
        doc_length_max=args.doc_length_max,
        shared_prefix_tokens_min=args.shared_prefix_min,
        shared_prefix_tokens_max=args.shared_prefix_max,
        private_suffix_tokens_min=args.private_suffix_min,
        private_suffix_tokens_max=args.private_suffix_max,
        document_mix=args.document_mix,
        skew_hot_probability=args.skew_hot_probability,
    )
    requests = generate_requests(config)

    chunk_size = args.chunk_size
    bpt = args.bytes_per_token
    gpu_capacity = args.gpu_mb * 1024 * 1024
    disk_bw = args.disk_bandwidth_kb * 1024

    cost_model = CostModel(
        compute_ms_per_token=0.001,
        attention_ms_per_token_position=1e-6 * args.attention_mult,
    )

    scenarios = _make_scenarios(args.schedulers)

    csv_rows: list[dict[str, Any]] = []
    summary: dict[str, dict[str, float]] = {}

    for scenario in scenarios:
        disk = make_disk_tier(
            capacity_bytes=_DISK_CAPACITY_BYTES,
            bandwidth_bytes_per_ms=disk_bw,
        )
        metrics_list = run_scenario(
            scenario, requests, chunk_size, bpt, gpu_capacity, 900 * 1024 * 1024,
            disk, cost_model,
        )
        n = len(metrics_list)
        summary[scenario.name] = {
            "mean_ttft_ms": sum(m.estimated_ttft_ms for m in metrics_list) / max(n, 1),
            "total_evictions": float(sum(m.eviction_count for m in metrics_list)),
            "total_recompute": float(sum(m.recompute_count for m in metrics_list)),
            "total_load": float(sum(m.load_count for m in metrics_list)),
            "total_coverage_misses": float(sum(m.coverage_miss_count for m in metrics_list)),
        }
        for i, m in enumerate(metrics_list):
            csv_rows.append({
                "scenario": scenario.name,
                "use_cake": int(scenario.use_cake),
                "policy": scenario.policy.name,
                "request_index": i,
                "ttft_ms": m.estimated_ttft_ms,
                "recompute_count": m.recompute_count,
                "load_count": m.load_count,
                "eviction_count": m.eviction_count,
                "coverage_miss_count": m.coverage_miss_count,
                "gpu_chunks_after": m.gpu_chunks_after,
                "gpu_used_bytes": m.gpu_used_bytes,
            })

    write_csv(output_dir / "metrics.csv", csv_rows)
    write_csv(
        output_dir / "summary_by_scenario.csv",
        [{"scenario": k, **{mk: int(v) if mk != "mean_ttft_ms" else v
                            for mk, v in d.items()}}
         for k, d in summary.items()],
    )

    # Remove this run's handler so it doesn't accumulate across preset runs.
    root_logger.removeHandler(log_handler)
    log_handler.close()

    # Per-preset plots.
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return summary, csv_rows

    _save_per_preset_plots(plt, output_dir, summary)
    print(f"  Saved plots -> {output_dir}")
    return summary, csv_rows


def _save_per_preset_plots(plt: Any, out_dir: Path, summary: dict[str, dict[str, float]]) -> None:
    names = list(summary.keys())
    x = range(len(names))

    metrics_cfg = [
        ("mean_ttft_ms",          "fig_mean_ttft.png",         "Mean estimated TTFT (ms)",              "steelblue",    "Mean estimated TTFT by scenario"),
        ("total_evictions",       "fig_total_evictions.png",   "Total evictions (all requests)",        "coral",        "GPU evictions under partial shared-prefix workload"),
        ("total_recompute",       "fig_total_recompute.png",   "Total recompute operations",            "seagreen",     "Total recomputes (lower = more cache hits)"),
        ("total_load",            "fig_total_load.png",        "Total load operations",                 "mediumpurple", "Total loads from cache hierarchy"),
        ("total_coverage_misses", "fig_coverage_misses.png",   "Total coverage misses (partial chunks)", "darkorange",  "Coverage misses: stale partial chunks replaced"),
    ]

    for metric_key, fname, ylabel, color, title in metrics_cfg:
        values = [summary[k][metric_key] for k in names]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(x, values, color=color, edgecolor="black", linewidth=0.5)
        ax.set_xticks(list(x), names, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        _tight_ylim(ax, values)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)


def _save_aggregate_plots(
    plt: Any,
    out_dir: Path,
    all_summaries: dict[str, dict[str, dict[str, float]]],
) -> None:
    """Grouped bar charts: X = scenario, one bar group per preset, colored by preset."""
    preset_names = list(all_summaries.keys())
    # Gather union of scenario names in consistent order (from first preset).
    scenario_names = list(next(iter(all_summaries.values())).keys())

    n_scenarios = len(scenario_names)
    n_presets = len(preset_names)
    import numpy as np

    colors = plt.cm.tab10.colors  # type: ignore[attr-defined]
    bar_width = 0.8 / n_presets
    x = np.arange(n_scenarios)

    metrics_cfg = [
        ("mean_ttft_ms",          "agg_mean_ttft.png",         "Mean TTFT (ms)",                "Mean TTFT by scenario and preset"),
        ("total_evictions",       "agg_total_evictions.png",   "Total evictions",               "GPU evictions by scenario and preset"),
        ("total_recompute",       "agg_total_recompute.png",   "Total recomputes",              "Recomputes by scenario and preset"),
        ("total_load",            "agg_total_load.png",        "Total loads",                   "Loads by scenario and preset"),
        ("total_coverage_misses", "agg_coverage_misses.png",   "Total coverage misses",         "Coverage misses by scenario and preset"),
    ]

    for metric_key, fname, ylabel, title in metrics_cfg:
        fig, ax = plt.subplots(figsize=(11, 5))
        for pi, preset in enumerate(preset_names):
            values = [
                all_summaries[preset].get(sc, {}).get(metric_key, 0.0)
                for sc in scenario_names
            ]
            offsets = x + (pi - n_presets / 2 + 0.5) * bar_width
            ax.bar(offsets, values, width=bar_width,
                   color=colors[pi % len(colors)], edgecolor="black",
                   linewidth=0.4, label=preset)
        ax.set_xticks(x, scenario_names, rotation=25, ha="right")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(title="preset", fontsize=8, title_fontsize=8)
        all_values = [
            all_summaries[p].get(sc, {}).get(metric_key, 0.0)
            for p in preset_names for sc in scenario_names
        ]
        _tight_ylim(ax, all_values)
        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)
        print(f"  Wrote {out_dir / fname}")

    # One extra: average across presets per scenario.
    fig, ax = plt.subplots(figsize=(9, 4))
    avg_ttft = [
        sum(all_summaries[p].get(sc, {}).get("mean_ttft_ms", 0.0) for p in preset_names) / n_presets
        for sc in scenario_names
    ]
    ax.bar(range(n_scenarios), avg_ttft, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(n_scenarios), scenario_names, rotation=25, ha="right")
    ax.set_ylabel("Avg mean TTFT across presets (ms)")
    ax.set_title("Grand-average TTFT: averaged over all presets")
    _tight_ylim(ax, avg_ttft)
    fig.tight_layout()
    fig.savefig(out_dir / "agg_grand_avg_ttft.png", dpi=150)
    plt.close(fig)
    print(f"  Wrote {out_dir / 'agg_grand_avg_ttft.png'}")


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "results",
        help="Root directory for outputs",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-requests", type=int, default=128)
    parser.add_argument("--num-documents", type=int, default=8)
    parser.add_argument(
        "--preset", choices=list(PRESETS), default=None,
        help="Apply a named parameter bundle (overrides individual flags)",
    )
    parser.add_argument(
        "--all-presets", action="store_true",
        help="Run every preset, save per-preset subfolders, then write aggregate plots",
    )
    parser.add_argument("--doc-length-min", type=int, default=2048)
    parser.add_argument("--doc-length-max", type=int, default=8192)
    parser.add_argument("--shared-prefix-min", type=int, default=256)
    parser.add_argument("--shared-prefix-max", type=int, default=4096)
    parser.add_argument("--private-suffix-min", type=int, default=128)
    parser.add_argument("--private-suffix-max", type=int, default=512)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--bytes-per-token", type=int, default=1024)
    parser.add_argument(
        "--gpu-mb", type=int, default=16,
        help="GPU tier capacity in MiB (tight to force evictions)",
    )
    parser.add_argument(
        "--document-mix", choices=("uniform", "round_robin", "skewed"), default="uniform",
    )
    parser.add_argument("--skew-hot-probability", type=float, default=0.75)
    parser.add_argument("--attention-mult", type=float, default=1.0)
    parser.add_argument(
        "--disk-bandwidth-kb", type=int, default=512,
        help="Disk read bandwidth in KiB/ms (default 512 KiB/ms ~ 512 MB/s)",
    )
    parser.add_argument(
        "--schedulers", choices=("cake_and_linear", "cake_only", "linear_only"),
        default="cake_and_linear",
    )
    return parser


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()

    plt = None
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib is not installed; skipped figures. "
            "Install with: pip install matplotlib",
            file=sys.stderr,
        )

    if args.all_presets:
        agg_dir = args.output_dir / "presets"
        agg_dir.mkdir(parents=True, exist_ok=True)
        all_summaries: dict[str, dict[str, dict[str, float]]] = {}
        all_csv_rows: list[dict[str, Any]] = []

        for preset_name in PRESETS:
            print(f"\n=== Preset: {preset_name} ===")
            preset_args = copy.copy(args)
            for key, value in PRESETS[preset_name].items():
                setattr(preset_args, key, value)
            out = agg_dir / preset_name
            summary, csv_rows = _run_one_preset(preset_args, out)
            all_summaries[preset_name] = summary
            for row in csv_rows:
                all_csv_rows.append({"preset": preset_name, **row})
            print(f"  Done: {out}")

        # Combined metrics across all presets.
        write_csv(agg_dir / "metrics_all_presets.csv", all_csv_rows)

        # Per-scenario averages across presets.
        scenario_names = list(next(iter(all_summaries.values())).keys())
        metric_keys = ["mean_ttft_ms", "total_evictions", "total_recompute",
                       "total_load", "total_coverage_misses"]
        avg_summary_rows = []
        for sc in scenario_names:
            row: dict[str, Any] = {"scenario": sc}
            for mk in metric_keys:
                vals = [all_summaries[p][sc][mk] for p in all_summaries if sc in all_summaries[p]]
                row[f"avg_{mk}"] = sum(vals) / max(len(vals), 1)
            avg_summary_rows.append(row)
        write_csv(agg_dir / "summary_avg_across_presets.csv", avg_summary_rows)
        print(f"\n  Wrote {agg_dir / 'metrics_all_presets.csv'}")
        print(f"  Wrote {agg_dir / 'summary_avg_across_presets.csv'}")

        if plt is not None:
            print(f"\n=== Aggregate plots -> {agg_dir} ===")
            _save_aggregate_plots(plt, agg_dir, all_summaries)
        print("\nAll presets complete.")
        return

    # Single run (with optional --preset override).
    if args.preset:
        for key, value in PRESETS[args.preset].items():
            setattr(args, key, value)
        out_dir = args.output_dir / "presets" / args.preset
    else:
        out_dir = args.output_dir

    _run_one_preset(args, out_dir)
    print(f"Done. Results in {out_dir}")


if __name__ == "__main__":
    main()
