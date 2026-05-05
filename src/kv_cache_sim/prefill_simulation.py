"""Connect Cake-style prefill scheduling to global GPU cache retention."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Mapping

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

logger = logging.getLogger(__name__)


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
    coverage_miss_count: int
    gpu_chunks_after: int
    gpu_used_bytes: int


def _materialize_schedule_entries_to_gpu(
    by_index: Mapping[int, object],
    request_chunks: list[KVChunk],
    gpu_cache: CacheManager,
    tiers_by_placement: dict[CachePlacement, MemoryTier],
    write_through_tiers: list[MemoryTier],
) -> tuple[int, int]:
    """Materialize scheduled operations against the GPU cache and write-through tiers.

    Returns ``(eviction_count, coverage_miss_count)``.

    Execution is phased to avoid a subtle ordering hazard: RECOMPUTE operations
    could evict chunks that a later LOAD-from-GPU was counting on.  By running
    GPU loads first, those hits are secured before any eviction pressure.

        Phase 1 — GPU loads   (LOAD, source == GPU)
        Phase 2 — cold loads  (LOAD, source != GPU)
        Phase 3 — recomputes  (RECOMPUTE)

    On RECOMPUTE, the fresh chunk is written to the GPU tier via
    ``CacheManager.store_replacing`` (which upgrades a partially-covered
    resident version if one exists) and then upserted to every
    ``write_through_tiers`` entry so later requests can load rather than
    recompute.  The upsert never downgrades existing coverage on disk.
    """

    eviction_count = 0
    coverage_miss_count = 0
    key_chunk = {c.chunk_index: c for c in request_chunks}
    gpu_name = gpu_cache.tier.name

    def action_src(idx: int) -> tuple[ScheduleAction, CachePlacement | None]:
        entry = by_index[idx]
        act = getattr(entry, "action")
        src = getattr(entry, "source_tier", None)
        return act, src

    indices = sorted(key_chunk.keys())

    # Phase 1: secure GPU-resident loads before any eviction pressure.
    for chunk_index in indices:
        action, source_tier = action_src(chunk_index)
        if action != ScheduleAction.LOAD or source_tier != gpu_name:
            continue
        cached = gpu_cache.access(key_chunk[chunk_index].cache_id, chunk_index)
        if cached is None:
            raise ValueError(
                f"scheduled LOAD from GPU but chunk {chunk_index} is missing"
            )

    # Phase 2: cold-tier loads → bring into GPU.
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
                f"scheduled LOAD from {source_tier.value} but chunk {cache_key} "
                "is not present"
            )
        result = gpu_cache.store(stored_source)
        eviction_count += len(result.evicted_chunks)

    # Phase 3: recomputes → update GPU and write through to persistent tiers.
    for chunk_index in indices:
        action, _ = action_src(chunk_index)
        if action != ScheduleAction.RECOMPUTE:
            continue

        chunk = key_chunk[chunk_index]

        # Detect whether a partial entry existed (for coverage miss accounting).
        for tier in [gpu_cache.tier, *write_through_tiers]:
            resident = tier.chunks.get(chunk.cache_key)
            if resident is not None and resident.end_token < chunk.end_token:
                coverage_miss_count += 1
                logger.debug(
                    "recompute chunk (%s, %d): replacing partial coverage "
                    "end=%d with end=%d in %s",
                    chunk.cache_id,
                    chunk.chunk_index,
                    resident.end_token,
                    chunk.end_token,
                    tier.name.value,
                )
                break

        # Store on GPU, replacing any partial-coverage resident.
        result = gpu_cache.store_replacing(chunk)
        eviction_count += len(result.evicted_chunks)

        # Write-through: persist to cold tiers, never downgrading coverage.
        for wt_tier in write_through_tiers:
            stored = wt_tier.upsert(chunk)
            if stored is not None:
                logger.debug(
                    "write-through chunk (%s, %d) end=%d to %s",
                    chunk.cache_id,
                    chunk.chunk_index,
                    chunk.end_token,
                    wt_tier.name.value,
                )

    return eviction_count, coverage_miss_count


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
    write_through_tiers: list[MemoryTier] | None = None,
) -> tuple[CakeScheduleSummary, PrefillMetrics]:
    """Run Cake bidirectional scheduling, then materialise KV onto the GPU tier.

    ``cold_tiers`` are visible to the scheduler for LOAD decisions (e.g. disk).
    ``write_through_tiers`` receive a copy of every freshly RECOMPUTE'd chunk so
    later requests can LOAD instead of recompute — pass the same disk tier for
    both to get the causal write-through behaviour described in the design.

    Recomputed chunks and loads from cold tiers call
    :meth:`CacheManager.store_replacing` or :meth:`CacheManager.store`;
    loads from the GPU tier call :meth:`CacheManager.access` for hit
    accounting.
    """
    cold_tiers = cold_tiers or []
    write_through_tiers = write_through_tiers or []
    memory_tiers: list[MemoryTier] = [gpu_cache.tier, *cold_tiers]
    tiers_by_placement = _tiers_dict(memory_tiers)

    scheduler = CakeBidirectionalScheduler(
        memory_tiers=memory_tiers, cost_model=cost_model
    )
    summary = scheduler.schedule_chunks(chunks)

    by_index = {op.chunk.chunk_index: op for op in summary.operations}
    eviction_count, coverage_miss_count = _materialize_schedule_entries_to_gpu(
        by_index, chunks, gpu_cache, tiers_by_placement, write_through_tiers
    )

    metrics = PrefillMetrics(
        estimated_ttft_ms=summary.total_estimated_ttft_ms,
        recompute_count=summary.recompute_count,
        load_count=summary.load_count,
        eviction_count=eviction_count,
        coverage_miss_count=coverage_miss_count,
        gpu_chunks_after=len(gpu_cache.tier.chunks),
        gpu_used_bytes=gpu_cache.tier.used_bytes,
    )
    return summary, metrics


def simulate_linear_prefill_with_global_cache(
    chunks: list[KVChunk],
    gpu_cache: CacheManager,
    cost_model: CostModel,
    cold_tiers: list[MemoryTier] | None = None,
    write_through_tiers: list[MemoryTier] | None = None,
    aggregate: LinearScheduleMode = LinearScheduleMode.SUM,
) -> tuple[ScheduleSummary, PrefillMetrics]:
    """Sequential per-chunk recompute/load choices (no Cake bidirectional TTFT).

    ``write_through_tiers`` behaves identically to the Cake variant — every
    RECOMPUTE is persisted to those tiers so the causal simulation stays warm.
    """
    cold_tiers = cold_tiers or []
    write_through_tiers = write_through_tiers or []
    memory_tiers: list[MemoryTier] = [gpu_cache.tier, *cold_tiers]
    tiers_by_placement = _tiers_dict(memory_tiers)

    scheduler = RecomputeLoadScheduler(
        memory_tiers=memory_tiers, cost_model=cost_model
    )
    summary = scheduler.schedule_chunks(chunks)

    by_index = {d.chunk.chunk_index: d for d in summary.decisions}
    eviction_count, coverage_miss_count = _materialize_schedule_entries_to_gpu(
        by_index, chunks, gpu_cache, tiers_by_placement, write_through_tiers
    )

    times = [decision.estimated_time_ms for decision in summary.decisions]
    ttft = sum(times) if aggregate == LinearScheduleMode.SUM else (max(times) if times else 0.0)

    metrics = PrefillMetrics(
        estimated_ttft_ms=ttft,
        recompute_count=summary.recompute_count,
        load_count=summary.load_count,
        eviction_count=eviction_count,
        coverage_miss_count=coverage_miss_count,
        gpu_chunks_after=len(gpu_cache.tier.chunks),
        gpu_used_bytes=gpu_cache.tier.used_bytes,
    )
    return summary, metrics
