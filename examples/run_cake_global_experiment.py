"""
Compare Cake + global cache (eviction policies) vs linear scheduling + global cache.

Requests are generated causally: each request draws a random prefix of a document
from the pool (shared portion) plus a private suffix.  No cache tier is pre-warmed;
the first request to touch a prefix must recompute it, and every recomputed chunk
is written through to an effectively-unbounded disk tier so later requests can load
rather than recompute.

Writes CSV metrics and matplotlib figures under ``results/`` by default.
"""

from __future__ import annotations

import argparse
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
from kv_cache_sim.policies import EvictionPolicy, FIFOPolicy, LateTokenPriorityPolicy, LRUPolicy
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

# Disk is modelled as effectively unbounded — GPU-to-disk capacity ratios are
# typically orders of magnitude, so the disk never fills up in practice.
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


# Documented parameter bundles.  Each entry overrides the matching argparse
# dest names (underscored) on top of the CLI defaults.
PRESETS: dict[str, dict[str, Any]] = {
    # One document; requests draw variable-depth slices of a very long context.
    # GPU holds only part of the prefix, so policies compete over which portion
    # of the document is worth retaining vs recomputing.
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
    # Six documents with one dominant hot doc (92 % probability).  Heavy skew
    # exposes how well each policy protects the hot prefix while cold docs
    # spill to disk and get reloaded.
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
    # Two equal-length documents served round-robin.  The cache must hold both
    # prefixes simultaneously or continuously thrash between them.
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
    # Inflated attention cost makes recomputing late-sequence tokens very
    # expensive.  Highlights scenarios where retaining high-index shared-prefix
    # chunks (LTP's intent) competes with the LRU advantage on private-suffix
    # eviction.
    "steep_late_recompute_cost": {
        "attention_mult": 120.0,
        "gpu_mb": 8,
        "num_requests": 128,
    },
}


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results",
        help="Directory for metrics.csv and PNG figures",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-requests", type=int, default=128)
    parser.add_argument("--num-documents", type=int, default=8)
    parser.add_argument(
        "--preset",
        choices=list(PRESETS),
        default=None,
        help="Apply a named parameter bundle (overrides individual flags)",
    )
    parser.add_argument(
        "--doc-length-min",
        type=int,
        default=2048,
        help="Minimum document length in tokens (the shared-prefix pool ceiling)",
    )
    parser.add_argument(
        "--doc-length-max",
        type=int,
        default=8192,
        help="Maximum document length in tokens",
    )
    parser.add_argument(
        "--shared-prefix-min",
        type=int,
        default=256,
        help="Minimum shared prefix tokens drawn per request (0 = no sharing)",
    )
    parser.add_argument(
        "--shared-prefix-max",
        type=int,
        default=4096,
        help="Maximum shared prefix tokens drawn per request (clamped to doc length)",
    )
    parser.add_argument(
        "--private-suffix-min",
        type=int,
        default=128,
        help="Minimum private suffix tokens per request",
    )
    parser.add_argument(
        "--private-suffix-max",
        type=int,
        default=512,
        help="Maximum private suffix tokens per request",
    )
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--bytes-per-token", type=int, default=1024)
    parser.add_argument(
        "--gpu-mb",
        type=int,
        default=16,
        help="GPU tier capacity in MiB (tight to force evictions)",
    )
    parser.add_argument(
        "--document-mix",
        choices=("uniform", "round_robin", "skewed"),
        default="uniform",
        help="How requests pick a document: uniform random, round-robin, or skewed hot doc",
    )
    parser.add_argument(
        "--skew-hot-probability",
        type=float,
        default=0.75,
        help="With skewed mix, probability that a request targets doc-0",
    )
    parser.add_argument(
        "--attention-mult",
        type=float,
        default=1.0,
        help="Multiplier on attention_ms_per_token_position",
    )
    parser.add_argument(
        "--disk-bandwidth-kb",
        type=int,
        default=512,
        help="Disk read bandwidth in KiB/ms (default 512 KiB/ms ≈ 512 MB/s)",
    )
    parser.add_argument(
        "--schedulers",
        choices=("cake_and_linear", "cake_only", "linear_only"),
        default="cake_and_linear",
    )
    args = parser.parse_args()

    if args.preset:
        for key, value in PRESETS[args.preset].items():
            setattr(args, key, value)

    # Route all kv_cache_sim DEBUG messages to a log file in the output dir.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _log_handler = logging.FileHandler(args.output_dir / "sim_debug.log", mode="w", encoding="utf-8")
    _log_handler.setLevel(logging.DEBUG)
    _log_handler.setFormatter(logging.Formatter("%(name)s %(levelname)s %(message)s"))
    logging.getLogger("kv_cache_sim").setLevel(logging.DEBUG)
    logging.getLogger("kv_cache_sim").addHandler(_log_handler)

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
    disk_bw = args.disk_bandwidth_kb * 1024  # KiB/ms → bytes/ms

    cost_model = CostModel(
        compute_ms_per_token=0.001,
        attention_ms_per_token_position=1e-6 * args.attention_mult,
    )

    cake_scenarios = (
        Scenario("cake_ltp", LateTokenPriorityPolicy(), True),
        Scenario("cake_fifo", FIFOPolicy(), True),
        Scenario("cake_lru", LRUPolicy(), True),
    )
    linear_scenarios = (
        Scenario("linear_ltp", LateTokenPriorityPolicy(), False),
        Scenario("linear_fifo", FIFOPolicy(), False),
    )
    if args.schedulers == "cake_only":
        scenarios = cake_scenarios
    elif args.schedulers == "linear_only":
        scenarios = linear_scenarios
    else:
        scenarios = cake_scenarios + linear_scenarios

    csv_rows: list[dict[str, Any]] = []
    summary_by_scenario: dict[str, dict[str, float]] = {}

    for scenario in scenarios:
        # Fresh disk per scenario — causal, starts empty.
        disk = make_disk_tier(
            capacity_bytes=_DISK_CAPACITY_BYTES,
            bandwidth_bytes_per_ms=disk_bw,
        )

        metrics_list = run_scenario(
            scenario,
            requests,
            chunk_size,
            bpt,
            gpu_capacity,
            900 * 1024 * 1024,
            disk,
            cost_model,
        )

        n = len(metrics_list)
        summary_by_scenario[scenario.name] = {
            "mean_ttft_ms": sum(m.estimated_ttft_ms for m in metrics_list) / max(n, 1),
            "total_evictions": float(sum(m.eviction_count for m in metrics_list)),
            "total_recompute": float(sum(m.recompute_count for m in metrics_list)),
            "total_load": float(sum(m.load_count for m in metrics_list)),
            "total_coverage_misses": float(
                sum(m.coverage_miss_count for m in metrics_list)
            ),
        }

        for i, m in enumerate(metrics_list):
            csv_rows.append(
                {
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
                }
            )

    out_dir = args.output_dir
    write_csv(out_dir / "metrics.csv", csv_rows)
    write_csv(
        out_dir / "summary_by_scenario.csv",
        [
            {
                "scenario": name,
                "mean_ttft_ms": d["mean_ttft_ms"],
                "total_evictions": int(d["total_evictions"]),
                "total_recompute": int(d["total_recompute"]),
                "total_load": int(d["total_load"]),
                "total_coverage_misses": int(d["total_coverage_misses"]),
            }
            for name, d in summary_by_scenario.items()
        ],
    )

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print(
            "matplotlib is not installed; skipped figures. "
            "Install with: pip install matplotlib",
            file=sys.stderr,
        )
        print(f"Wrote {out_dir / 'metrics.csv'}", file=sys.stderr)
        return

    names = list(summary_by_scenario.keys())
    x = range(len(names))
    means = [summary_by_scenario[k]["mean_ttft_ms"] for k in names]
    evs = [summary_by_scenario[k]["total_evictions"] for k in names]
    recs = [summary_by_scenario[k]["total_recompute"] for k in names]
    loads = [summary_by_scenario[k]["total_load"] for k in names]
    cov_misses = [summary_by_scenario[k]["total_coverage_misses"] for k in names]

    def _tight_ylim(ax: Any, values: list[float]) -> None:
        """Zoom y-axis to the data range so bar differences are visible.

        Uses 15 % of the spread as padding on each side; falls back to
        5 % of the max when all values are identical.  Always clamps the
        bottom to 0 so bars can't appear to float above the axis.
        """
        if not values or max(values) == 0:
            return
        lo, hi = min(values), max(values)
        pad = max((hi - lo) * 0.15, hi * 0.05)
        ax.set_ylim(bottom=max(0.0, lo - pad), top=hi + pad)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x, means, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_xticks(list(x), names, rotation=25, ha="right")
    ax.set_ylabel("Mean estimated TTFT (ms)")
    ax.set_title("Per-request prefill: mean estimated TTFT by scenario")
    _tight_ylim(ax, means)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_mean_ttft.png", dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.bar(x, evs, color="coral", edgecolor="black", linewidth=0.5)
    ax2.set_xticks(list(x), names, rotation=25, ha="right")
    ax2.set_ylabel("Total evictions (all requests)")
    ax2.set_title("GPU evictions under partial shared-prefix workload")
    _tight_ylim(ax2, evs)
    fig2.tight_layout()
    fig2.savefig(out_dir / "fig_total_evictions.png", dpi=150)
    plt.close(fig2)

    fig3, ax3 = plt.subplots(figsize=(9, 4))
    ax3.bar(x, recs, color="seagreen", edgecolor="black", linewidth=0.5)
    ax3.set_xticks(list(x), names, rotation=25, ha="right")
    ax3.set_ylabel("Total recompute operations")
    ax3.set_title("Policy sensitivity: total recomputes (lower = more cache hits)")
    _tight_ylim(ax3, recs)
    fig3.tight_layout()
    fig3.savefig(out_dir / "fig_total_recompute.png", dpi=150)
    plt.close(fig3)

    fig4, ax4 = plt.subplots(figsize=(9, 4))
    ax4.bar(x, loads, color="mediumpurple", edgecolor="black", linewidth=0.5)
    ax4.set_xticks(list(x), names, rotation=25, ha="right")
    ax4.set_ylabel("Total load operations (GPU + cold tiers)")
    ax4.set_title("Policy sensitivity: total loads from cache hierarchy")
    _tight_ylim(ax4, loads)
    fig4.tight_layout()
    fig4.savefig(out_dir / "fig_total_load.png", dpi=150)
    plt.close(fig4)

    fig5, ax5 = plt.subplots(figsize=(9, 4))
    ax5.bar(x, cov_misses, color="darkorange", edgecolor="black", linewidth=0.5)
    ax5.set_xticks(list(x), names, rotation=25, ha="right")
    ax5.set_ylabel("Total coverage misses (partial-chunk replacements)")
    ax5.set_title("Coverage misses: stale partial chunks replaced by larger recomputes")
    _tight_ylim(ax5, cov_misses)
    fig5.tight_layout()
    fig5.savefig(out_dir / "fig_coverage_misses.png", dpi=150)
    plt.close(fig5)

    print(f"Wrote {out_dir / 'metrics.csv'}")
    print(f"Wrote {out_dir / 'summary_by_scenario.csv'}")
    for name in (
        "fig_mean_ttft.png",
        "fig_total_evictions.png",
        "fig_total_recompute.png",
        "fig_total_load.png",
        "fig_coverage_misses.png",
    ):
        print(f"Wrote {out_dir / name}")


if __name__ == "__main__":
    main()
