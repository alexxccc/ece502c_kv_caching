"""Show how different requests can reuse the same persistent KV cache."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from kv_cache_sim.cache import CacheManager
from kv_cache_sim.models import CachePlacement, MemoryTier, Request, chunk_request
from kv_cache_sim.policies import LateTokenPriorityPolicy


def main() -> None:
    first_request = Request(
        request_id="request-a",
        cache_id="document-prefix-42",
        arrival_time_ms=0.0,
        prompt_tokens=2048,
    )
    second_request = Request(
        request_id="request-b",
        cache_id="document-prefix-42",
        arrival_time_ms=10.0,
        prompt_tokens=2048,
    )

    cache = CacheManager(
        tier=MemoryTier(
            name=CachePlacement.GPU,
            capacity_bytes=4 * 1024 * 1024,
            bandwidth_bytes_per_ms=900 * 1024 * 1024,
        ),
        policy=LateTokenPriorityPolicy(),
    )

    first_chunks = chunk_request(first_request, chunk_size_tokens=1024, bytes_per_token=1024)
    second_chunks = chunk_request(second_request, chunk_size_tokens=1024, bytes_per_token=1024)

    for chunk in first_chunks:
        cache.store(chunk)

    result = cache.store(second_chunks[0])

    print(f"first request id: {first_request.request_id}")
    print(f"second request id: {second_request.request_id}")
    print(f"shared cache id: {first_request.cache_id}")
    print(f"stored chunk came from: {result.stored_chunk.request_id}")
    print(f"evicted chunks: {len(result.evicted_chunks)}")
    print(f"gpu chunks stored: {len(cache.tier.chunks)}")


if __name__ == "__main__":
    main()
