"""Minimal Cake prefill + Late-Token global GPU cache (two requests, shared prefix)."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from kv_cache_sim.cache import CacheManager
from kv_cache_sim.models import CachePlacement, MemoryTier, Request, chunk_request
from kv_cache_sim.policies import LateTokenPriorityPolicy
from kv_cache_sim.prefill_simulation import simulate_cake_prefill_with_global_cache
from kv_cache_sim.scheduler import CostModel
from kv_cache_sim.workload import make_disk_tier, seed_disk_for_workload


def main() -> None:
    chunk_size = 1024
    bpt = 1024
    cache_id = "shared-doc"

    first = Request(
        request_id="r0",
        cache_id=cache_id,
        arrival_time_ms=0.0,
        prompt_tokens=4096,
        shared_prefix_tokens=4096,
    )
    second = Request(
        request_id="r1",
        cache_id=cache_id,
        arrival_time_ms=1.0,
        prompt_tokens=4096,
        shared_prefix_tokens=4096,
    )

    gpu = MemoryTier(
        name=CachePlacement.GPU,
        capacity_bytes=3 * chunk_size * bpt,
        bandwidth_bytes_per_ms=900 * 1024 * 1024,
    )
    disk = make_disk_tier(
        capacity_bytes=64 * 1024 * 1024,
        bandwidth_bytes_per_ms=512 * 1024,
    )
    seed_disk_for_workload(disk, [first], chunk_size, bpt)

    cache = CacheManager(tier=gpu, policy=LateTokenPriorityPolicy())
    cost = CostModel(
        compute_ms_per_token=0.001,
        attention_ms_per_token_position=1e-6,
    )

    for label, req in ("first", first), ("second", second):
        chunks = chunk_request(req, chunk_size, bpt)
        summary, metrics = simulate_cake_prefill_with_global_cache(
            chunks, cache, cost, cold_tiers=[disk]
        )
        print(f"--- {label} request ({req.request_id}) ---")
        print(f"estimated TTFT (ms): {metrics.estimated_ttft_ms:.2f}")
        print(f"recompute={metrics.recompute_count} load={metrics.load_count}")
        print(f"evictions this prefill: {metrics.eviction_count}")
        print(f"GPU chunks after: {metrics.gpu_chunks_after}")


if __name__ == "__main__":
    main()
