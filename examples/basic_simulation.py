"""Basic smoke test for the KV-cache simulator scaffold."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from kv_cache_sim.models import CachePlacement, MemoryTier, Request, chunk_request


def main() -> None:
    request = Request(
        request_id="request-001",
        arrival_time_ms=0.0,
        prompt_tokens=4096,
        shared_prefix_tokens=2048,
        output_tokens=128,
    )

    chunks = chunk_request(
        request=request,
        chunk_size_tokens=1024,
        bytes_per_token=2048,
    )

    gpu_memory = MemoryTier(
        name=CachePlacement.GPU,
        capacity_bytes=8 * 1024 * 1024,
        bandwidth_bytes_per_ms=900 * 1024 * 1024,
    )

    for chunk in chunks[:2]:
        gpu_memory.store(chunk)

    print(f"Request: {request.request_id}")
    print(f"Prompt tokens: {request.prompt_tokens}")
    print(f"Generated chunks: {len(chunks)}")
    print(f"GPU chunks stored: {len(gpu_memory.chunks)}")
    print(f"GPU used bytes: {gpu_memory.used_bytes}")
    print(f"GPU free bytes: {gpu_memory.free_bytes}")
    print("Chunks by Late-Token Priority:")

    for chunk in gpu_memory.iter_chunks_by_priority():
        print(
            f"  chunk={chunk.chunk_index} "
            f"tokens={chunk.start_token}-{chunk.end_token} "
            f"priority={chunk.late_token_priority}"
        )


if __name__ == "__main__":
    main()
