"""Demonstrate cost-aware recompute-vs-load scheduling."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from kv_cache_sim.models import CachePlacement, MemoryTier, Request, chunk_request
from kv_cache_sim.scheduler import CostModel, RecomputeLoadScheduler


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
        bandwidth_bytes_per_ms=256 * 1024,
    )

    # Pretend only the first and last chunks are globally cached on disk.
    disk.store(chunks[0])
    disk.store(chunks[-1])

    scheduler = RecomputeLoadScheduler(
        memory_tiers=[disk],
        cost_model=CostModel(
            compute_ms_per_token=0.001,
            attention_ms_per_token_position=0.000001,
        ),
    )
    summary = scheduler.schedule_chunks(chunks)

    for decision in summary.decisions:
        load_time = (
            "not cached"
            if decision.load_time_ms is None
            else f"{decision.load_time_ms:.2f} ms"
        )
        source = "none" if decision.source_tier is None else decision.source_tier.value
        print(
            f"chunk={decision.chunk.chunk_index} "
            f"tokens={decision.chunk.start_token}-{decision.chunk.end_token} "
            f"action={decision.action.value} "
            f"compute={decision.compute_time_ms:.2f} ms "
            f"load={load_time} "
            f"source={source}"
        )

    print(f"recompute count: {summary.recompute_count}")
    print(f"load count: {summary.load_count}")
    print(f"total estimated time: {summary.total_estimated_time_ms:.2f} ms")


if __name__ == "__main__":
    main()
