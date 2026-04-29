"""
Demonstrate a Cake-style bidirectional compute/load schedule.

Simple input w/ 4 chunks.
"""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from kv_cache_sim.models import CachePlacement, MemoryTier, Request, chunk_request
from kv_cache_sim.scheduler import CakeBidirectionalScheduler, CostModel


def main() -> None:
    request = Request(
        request_id="request-001",
        cache_id="document-prefix-42",
        arrival_time_ms=0.0,
        prompt_tokens=4096,
    )
    chunks = chunk_request(request, chunk_size_tokens=1024, bytes_per_token=1024)

    disk = MemoryTier(
        name=CachePlacement.DISK,
        capacity_bytes=16 * 1024 * 1024,
        bandwidth_bytes_per_ms=512 * 1024,
    )

    # Cake assumes reusable prefix chunks are available to the I/O front.
    for chunk in chunks:
        disk.store(chunk)

    scheduler = CakeBidirectionalScheduler(
        memory_tiers=[disk],
        cost_model=CostModel(
            compute_ms_per_token=0.001,
            attention_ms_per_token_position=0.000001,
        ),
    )
    summary = scheduler.schedule_chunks(chunks)

    for operation in summary.operations:
        source = "compute" if operation.source_tier is None else operation.source_tier.value
        print(
            f"chunk={operation.chunk.chunk_index} "
            f"tokens={operation.chunk.start_token}-{operation.chunk.end_token} "
            f"action={operation.action.value} "
            f"source={source} "
            f"start={operation.start_time_ms:.2f} ms "
            f"end={operation.end_time_ms:.2f} ms"
        )

    print(f"recompute count: {summary.recompute_count}")
    print(f"load count: {summary.load_count}")
    print(f"compute front time: {summary.compute_front_time_ms:.2f} ms")
    print(f"load front time: {summary.load_front_time_ms:.2f} ms")
    print(f"estimated TTFT: {summary.total_estimated_ttft_ms:.2f} ms")


if __name__ == "__main__":
    main()
