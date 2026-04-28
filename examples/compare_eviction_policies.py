"""Compare FIFO, LRU, and Late-Token Priority eviction behavior."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from kv_cache_sim.cache import CacheManager
from kv_cache_sim.models import CachePlacement, MemoryTier, Request, chunk_request
from kv_cache_sim.policies import FIFOPolicy, LRUPolicy, LateTokenPriorityPolicy


def build_gpu_cache(policy) -> CacheManager:
    return CacheManager(
        tier=MemoryTier(
            name=CachePlacement.GPU,
            capacity_bytes=3 * 1024 * 1024,
            bandwidth_bytes_per_ms=900 * 1024 * 1024,
        ),
        policy=policy,
    )


def run_policy_demo(policy) -> None:
    request = Request(
        request_id="request-001",
        arrival_time_ms=0.0,
        prompt_tokens=5 * 1024,
        cache_id="shared-prefix-001",
    )
    chunks = chunk_request(
        request=request,
        chunk_size_tokens=1024,
        bytes_per_token=1024,
    )

    cache = build_gpu_cache(policy)

    for chunk in chunks[:3]:
        cache.store(chunk)

    # Touch chunk 0 so LRU keeps it, while FIFO will still consider it oldest.
    cache.access("shared-prefix-001", 0)

    result = cache.store(chunks[3])
    remaining = sorted(chunk.chunk_index for chunk in cache.tier.chunks.values())
    evicted = [chunk.chunk_index for chunk in result.evicted_chunks]

    print(f"{policy.name}")
    print(f"  evicted chunks: {evicted}")
    print(f"  remaining chunks: {remaining}")


def main() -> None:
    for policy in (FIFOPolicy(), LRUPolicy(), LateTokenPriorityPolicy()):
        run_policy_demo(policy)


if __name__ == "__main__":
    main()
