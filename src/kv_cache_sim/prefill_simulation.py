"""Connect Cake-style prefill scheduling to global GPU cache retention."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from kv_cache_sim.cache import CacheManager
from kv_cache_sim.models import CachePlacement, KVChunk, MemoryTier
from kv_cache_sim.scheduler import (
    CakeBidirectionalScheduler,
    CakeScheduleSummary,
    CostModel,
    RecomputeLoadScheduler,
    ScheduleAction,
    ScheduleSummary,
)


class LinearScheduleMode(str, Enum):
    """How to aggregate per-chunk times for the linear (non-Cake) baseline."""

    SUM = "sum"
    MAX = "max"


@dataclass(frozen=True)
class PrefillMetrics:
    """One completed prefill with scheduling and cache-management accounting."""

    estimated_ttft_ms: float
    recompute_count: int
    load_count: int
    eviction_count: int
    gpu_chunks_after: int
    gpu_used_bytes: int


def _materialize_cake_to_gpu(
    summary: CakeScheduleSummary,
    request_chunks: list[KVChunk],
    gpu_cache: CacheManager,
    tiers_by_placement: dict[CachePlacement, MemoryTier],
) -> int:
    """Apply Cake operations to the managed GPU tier; return eviction count."""

    by_index = {op.chunk.chunk_index: op for op in summary.operations}
    return _materialize_schedule_entries_to_gpu(by_index, request_chunks, gpu_cache, tiers_by_placement)


def _materialize_schedule_entries_to_gpu(
    by_index: dict[int, object],
    request_chunks: list[KVChunk],
    gpu_cache: CacheManager,
    tiers_by_placement: dict[CachePlacement, MemoryTier],
) -> int:
    """Materialize in phases so RECOMPUTE stores cannot evict KV needed for GPU loads.

    Chunk-index order alone can process RECOMPUTE before LOAD-from-GPU for higher
    indices; eviction may remove chunks the scheduler expected to hit.
    """

    eviction_count = 0
    key_chunk = {c.chunk_index: c for c in request_chunks}
    gpu_name = gpu_cache.tier.name

    def action_src(idx: int) -> tuple[ScheduleAction, CachePlacement | None]:
        entry = by_index[idx]
        act = getattr(entry, "action")
        src = getattr(entry, "source_tier", None)
        return act, src

    indices = sorted(key_chunk.keys())

    def run_gpu_loads() -> None:
        for chunk_index in indices:
            action, source_tier = action_src(chunk_index)
            if action != ScheduleAction.LOAD or source_tier != gpu_name:
                continue
            cached = gpu_cache.access(key_chunk[chunk_index].cache_id, chunk_index)
            if cached is None:
                raise ValueError(
                    f"scheduled LOAD from GPU but chunk {chunk_index} is missing"
                )

    def run_cold_loads() -> None:
        nonlocal eviction_count
        for chunk_index in indices:
            action, source_tier = action_src(chunk_index)
            if action != ScheduleAction.LOAD or source_tier is None:
                continue
            if source_tier == gpu_name:
                continue
            cold_tier = _tier_for_placement(tiers_by_placement, source_tier)
            cache_key = key_chunk[chunk_index].cache_key
            stored_source = cold_tier.chunks.get(cache_key)
            if stored_source is None:
                raise ValueError(
                    f"scheduled LOAD from {source_tier.value} but chunk {cache_key} is not present"
                )
            result = gpu_cache.store(stored_source)
            eviction_count += len(result.evicted_chunks)

    def run_recomputes() -> None:
        nonlocal eviction_count
        for chunk_index in indices:
            action, _ = action_src(chunk_index)
            if action != ScheduleAction.RECOMPUTE:
                continue
            chunk = key_chunk[chunk_index]
            result = gpu_cache.store(chunk)
            eviction_count += len(result.evicted_chunks)

    for chunk_index in indices:
        action, source_tier = action_src(chunk_index)
        if action == ScheduleAction.LOAD and source_tier is None:
            raise ValueError("LOAD operation missing source_tier")

    run_gpu_loads()
    run_cold_loads()
    run_recomputes()
    return eviction_count


def _tier_for_placement(
    tiers_by_placement: dict[CachePlacement, MemoryTier], placement: CachePlacement
) -> MemoryTier:
    tier = tiers_by_placement.get(placement)
    if tier is None:
        raise ValueError(f"no MemoryTier registered for placement {placement.value}")
    return tier


def _tiers_dict(memory_tiers: list[MemoryTier]) -> dict[CachePlacement, MemoryTier]:
    out: dict[CachePlacement, MemoryTier] = {}
    for tier in memory_tiers:
        if tier.name in out:
            raise ValueError(f"duplicate memory tier for placement {tier.name.value}")
        out[tier.name] = tier
    return out


def simulate_cake_prefill_with_global_cache(
    chunks: list[KVChunk],
    gpu_cache: CacheManager,
    cost_model: CostModel,
    cold_tiers: list[MemoryTier] | None = None,
) -> tuple[CakeScheduleSummary, PrefillMetrics]:
    """Run Cake bidirectional scheduling, then materialize KV onto the GPU tier.

    The scheduler sees the GPU tier (from ``gpu_cache``) plus any optional cold
    tiers (CPU/disk) so the I/O front can load warm KV from slower storage.
    Recomputed chunks and loads from cold tiers call :meth:`CacheManager.store`;
    loads from the GPU tier call :meth:`CacheManager.access` for hit accounting.
    """

    cold_tiers = cold_tiers or []
    memory_tiers: list[MemoryTier] = [gpu_cache.tier, *cold_tiers]
    tiers_by_placement = _tiers_dict(memory_tiers)

    scheduler = CakeBidirectionalScheduler(memory_tiers=memory_tiers, cost_model=cost_model)
    summary = scheduler.schedule_chunks(chunks)
    eviction_count = _materialize_cake_to_gpu(
        summary, chunks, gpu_cache, tiers_by_placement
    )

    metrics = PrefillMetrics(
        estimated_ttft_ms=summary.total_estimated_ttft_ms,
        recompute_count=summary.recompute_count,
        load_count=summary.load_count,
        eviction_count=eviction_count,
        gpu_chunks_after=len(gpu_cache.tier.chunks),
        gpu_used_bytes=gpu_cache.tier.used_bytes,
    )
    return summary, metrics


def simulate_linear_prefill_with_global_cache(
    chunks: list[KVChunk],
    gpu_cache: CacheManager,
    cost_model: CostModel,
    cold_tiers: list[MemoryTier] | None = None,
    aggregate: LinearScheduleMode = LinearScheduleMode.SUM,
) -> tuple[ScheduleSummary, PrefillMetrics]:
    """Sequential per-chunk recompute/load choices (no Cake bidirectional TTFT)."""

    cold_tiers = cold_tiers or []
    memory_tiers: list[MemoryTier] = [gpu_cache.tier, *cold_tiers]
    tiers_by_placement = _tiers_dict(memory_tiers)

    scheduler = RecomputeLoadScheduler(memory_tiers=memory_tiers, cost_model=cost_model)
    summary = scheduler.schedule_chunks(chunks)
    eviction_count = _materialize_linear_to_gpu(
        summary, chunks, gpu_cache, tiers_by_placement
    )

    times = [decision.estimated_time_ms for decision in summary.decisions]
    if aggregate == LinearScheduleMode.SUM:
        ttft = sum(times)
    else:
        ttft = max(times) if times else 0.0

    metrics = PrefillMetrics(
        estimated_ttft_ms=ttft,
        recompute_count=summary.recompute_count,
        load_count=summary.load_count,
        eviction_count=eviction_count,
        gpu_chunks_after=len(gpu_cache.tier.chunks),
        gpu_used_bytes=gpu_cache.tier.used_bytes,
    )
    return summary, metrics


def _materialize_linear_to_gpu(
    summary: ScheduleSummary,
    request_chunks: list[KVChunk],
    gpu_cache: CacheManager,
    tiers_by_placement: dict[CachePlacement, MemoryTier],
) -> int:
    """Materialize linear scheduler decisions using the same phased order as Cake."""

    by_index = {decision.chunk.chunk_index: decision for decision in summary.decisions}
    return _materialize_schedule_entries_to_gpu(
        by_index, request_chunks, gpu_cache, tiers_by_placement
    )
