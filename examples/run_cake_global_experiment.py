"""
Compare Cake + global cache (eviction policies) vs linear scheduling + global cache.

Writes CSV metrics and matplotlib figures under ``results/`` by default.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from kv_cache_sim.cache import CacheManager
from kv_cache_sim.models import CachePlacement, MemoryTier, Request, chunk_request
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
    seed_disk_for_workload,
)


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
        chunks = chunk_request(request, chunk_size_tokens, bytes_per_token)
        if scenario.use_cake:
            _, metrics = simulate_cake_prefill_with_global_cache(
                chunks, cache, cost_model, cold_tiers=[disk]
            )
        else:
            _, metrics = simulate_linear_prefill_with_global_cache(
                chunks,
                cache,
                cost_model,
                cold_tiers=[disk],
                aggregate=LinearScheduleMode.SUM,
            )
        metrics_list.append(metrics)

    return metrics_list


# Override groups for ``--preset`` (see docs/eviction_experiment_scenarios.md).
PRESETS: dict[str, dict[str, Any]] = {
    "single_long_prefix": {
        "num_documents": 1,
        "prompt_min": 8192,
        "prompt_max": 8192,
        "gpu_mb": 2,
        "num_requests": 48,
        "document_mix": "uniform",
    },
    "hot_document_skew": {
        "num_documents": 6,
        "num_requests": 64,
        "gpu_mb": 2,
        "prompt_min": 2048,
        "prompt_max": 4096,
        "document_mix": "skewed",
        "skew_hot_probability": 0.92,
    },
    "alternating_two_docs": {
        "num_documents": 2,
        "num_requests": 56,
        "gpu_mb": 3,
        "prompt_min": 4096,
        "prompt_max": 4096,
        "document_mix": "round_robin",
    },
    "steep_late_recompute_cost": {
        "attention_mult": 120.0,
        "gpu_mb": 2,
        "num_requests": 48,
    },
}


def write_csv(
    path: Path,
    rows: list[dict[str, str | int | float]],
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
    parser.add_argument("--num-requests", type=int, default=48)
    parser.add_argument("--num-documents", type=int, default=6)
    parser.add_argument("--prompt-min", type=int, default=1024)
    parser.add_argument("--prompt-max", type=int, default=4096)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--bytes-per-token", type=int, default=1024)
    parser.add_argument(
        "--gpu-mb",
        type=int,
        default=4,
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
        default=0.85,
        help="With skewed mix, probability that a request targets doc-0",
    )
    parser.add_argument(
        "--attention-mult",
        type=float,
        default=1.0,
        help="Multiplier on attention_ms_per_token_position (raise to stress late-token recompute cost)",
    )
    parser.add_argument(
        "--schedulers",
        choices=("cake_and_linear", "cake_only", "linear_only"),
        default="cake_and_linear",
        help="Which scheduler groups to include (cake_only isolates eviction policies)",
    )
    parser.add_argument(
        "--preset",
        choices=list(PRESETS),
        default=None,
        help="Apply a documented parameter bundle for eviction-focused experiments",
    )
    args = parser.parse_args()

    if args.preset:
        for key, value in PRESETS[args.preset].items():
            setattr(args, key, value)

    config = WorkloadConfig(
        seed=args.seed,
        num_requests=args.num_requests,
        num_documents=args.num_documents,
        prompt_tokens_min=args.prompt_min,
        prompt_tokens_max=args.prompt_max,
        document_mix=args.document_mix,
        skew_hot_probability=args.skew_hot_probability,
    )
    requests = generate_requests(config)

    chunk_size = args.chunk_size
    bpt = args.bytes_per_token
    gpu_capacity = args.gpu_mb * 1024 * 1024

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

    csv_rows: list[dict[str, str | int | float]] = []
    summary_by_scenario: dict[str, dict[str, float]] = {}

    for scenario in scenarios:
        disk = make_disk_tier(
            capacity_bytes=512 * 1024 * 1024,
            bandwidth_bytes_per_ms=512 * 1024,
        )
        seed_disk_for_workload(disk, requests, chunk_size, bpt)

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

        total_ttft = sum(m.estimated_ttft_ms for m in metrics_list)
        total_ev = sum(m.eviction_count for m in metrics_list)
        n = len(metrics_list)
        summary_by_scenario[scenario.name] = {
            "mean_ttft_ms": total_ttft / max(n, 1),
            "total_evictions": float(total_ev),
            "total_recompute": float(sum(m.recompute_count for m in metrics_list)),
            "total_load": float(sum(m.load_count for m in metrics_list)),
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

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x, means, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x, names, rotation=25, ha="right")
    ax.set_ylabel("Mean estimated TTFT (ms)")
    ax.set_title("Per-request prefill: mean estimated TTFT by scenario")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_mean_ttft.png", dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(9, 4))
    ax2.bar(x, evs, color="coral", edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x, names, rotation=25, ha="right")
    ax2.set_ylabel("Total evictions (all requests)")
    ax2.set_title("GPU evictions when retaining shared-prefix KV under capacity")
    fig2.tight_layout()
    fig2.savefig(out_dir / "fig_total_evictions.png", dpi=150)
    plt.close(fig2)

    recs = [summary_by_scenario[k]["total_recompute"] for k in names]
    loads = [summary_by_scenario[k]["total_load"] for k in names]

    fig3, ax3 = plt.subplots(figsize=(9, 4))
    ax3.bar(x, recs, color="seagreen", edgecolor="black", linewidth=0.5)
    ax3.set_xticks(x, names, rotation=25, ha="right")
    ax3.set_ylabel("Total recompute operations (all chunks, all requests)")
    ax3.set_title("Policy sensitivity: total recomputes (lower often better when disk is warm)")
    fig3.tight_layout()
    fig3.savefig(out_dir / "fig_total_recompute.png", dpi=150)
    plt.close(fig3)

    fig4, ax4 = plt.subplots(figsize=(9, 4))
    ax4.bar(x, loads, color="mediumpurple", edgecolor="black", linewidth=0.5)
    ax4.set_xticks(x, names, rotation=25, ha="right")
    ax4.set_ylabel("Total load operations (GPU + cold tiers)")
    ax4.set_title("Policy sensitivity: total loads from cache hierarchy")
    fig4.tight_layout()
    fig4.savefig(out_dir / "fig_total_load.png", dpi=150)
    plt.close(fig4)

    print(f"Wrote {out_dir / 'metrics.csv'}")
    print(f"Wrote {out_dir / 'summary_by_scenario.csv'}")
    print(f"Wrote {out_dir / 'fig_mean_ttft.png'}")
    print(f"Wrote {out_dir / 'fig_total_evictions.png'}")
    print(f"Wrote {out_dir / 'fig_total_recompute.png'}")
    print(f"Wrote {out_dir / 'fig_total_load.png'}")


if __name__ == "__main__":
    main()
