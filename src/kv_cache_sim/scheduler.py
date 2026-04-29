"""
Simple scheduler implementation
Cost-aware scheduling for recomputing vs loading KV-cache chunks.
Uses the CostModel to estimate compute and load times, and chooses the cheaper option.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from kv_cache_sim.models import CachePlacement, KVChunk, MemoryTier


class ScheduleAction(str, Enum):
    # How the simulator should obtain a KV chunk 

    RECOMPUTE = "recompute"
    LOAD = "load"


@dataclass(frozen=True)
class CostModel:
    """
    Parameters used to estimate compute and I/O load cost.

    The compute model captures Cake's main observation: later chunks are more
    expensive to compute because attention has to look back over more context.
    Loading cost is based on chunk size and memory-tier bandwidth.
    """

    compute_ms_per_token: float
    attention_ms_per_token_position: float
    compute_queue_delay_ms: float = 0.0
    load_queue_delay_ms: dict[CachePlacement, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.compute_ms_per_token < 0:
            raise ValueError("compute_ms_per_token must be non-negative")
        if self.attention_ms_per_token_position < 0:
            raise ValueError("attention_ms_per_token_position must be non-negative")
        if self.compute_queue_delay_ms < 0:
            raise ValueError("compute_queue_delay_ms must be non-negative")
        if any(delay < 0 for delay in self.load_queue_delay_ms.values()):
            raise ValueError("load queue delays must be non-negative")

    def estimate_compute_time_ms(self, chunk: KVChunk) -> float:
        # Estimate time to recompute a chunk during prefill.
        average_position = (chunk.start_token + chunk.end_token) / 2
        token_cost = chunk.token_count * self.compute_ms_per_token
        attention_cost = (
            chunk.token_count
            * average_position
            * self.attention_ms_per_token_position
        )
        # compute time = queue delay + token cost + attention cost
        return self.compute_queue_delay_ms + token_cost + attention_cost

    def estimate_load_time_ms(self, chunk: KVChunk, tier: MemoryTier) -> float:
        # Estimate time to load a cached chunk from a memory tier
        queue_delay = self.load_queue_delay_ms.get(tier.name, 0.0)
        # load time = queue delay + chunk size / bandwidth
        return queue_delay + tier.estimate_load_time_ms(chunk)


@dataclass(frozen=True)
class ScheduleDecision:
    # Scheduler decision for one KV chunk

    chunk: KVChunk
    action: ScheduleAction
    estimated_time_ms: float
    compute_time_ms: float
    load_time_ms: float | None
    source_tier: CachePlacement | None
    reason: str


@dataclass(frozen=True)
class ScheduleSummary:
    # Aggregate view of a list of scheduling decisions

    decisions: tuple[ScheduleDecision, ...]

    @property
    def recompute_count(self) -> int:
        return sum(
            decision.action == ScheduleAction.RECOMPUTE
            for decision in self.decisions
        )

    @property
    def load_count(self) -> int:
        return sum(decision.action == ScheduleAction.LOAD for decision in self.decisions)

    @property
    def total_estimated_time_ms(self) -> float:
        return sum(decision.estimated_time_ms for decision in self.decisions)


class RecomputeLoadScheduler:
    # Choose whether each KV chunk should be recomputed or loaded

    def __init__(
        self,
        memory_tiers: list[MemoryTier],
        cost_model: CostModel,
    ) -> None:
        if not memory_tiers:
            raise ValueError("memory_tiers must contain at least one tier")

        self.memory_tiers = memory_tiers
        self.cost_model = cost_model

    def find_cached_chunk(self, chunk: KVChunk) -> tuple[MemoryTier, KVChunk] | None:
        """Find a chunk in the configured memory tiers by global cache key."""

        for tier in self.memory_tiers:
            cached_chunk = tier.chunks.get(chunk.cache_key)
            if cached_chunk is not None:
                return tier, cached_chunk

        return None

    def choose_for_chunk(self, chunk: KVChunk) -> ScheduleDecision:
        # Choose the cheaper way to obtain one KV chunk

        compute_time_ms = self.cost_model.estimate_compute_time_ms(chunk)
        cached = self.find_cached_chunk(chunk)

        if cached is None:
            return ScheduleDecision(
                chunk=chunk,
                action=ScheduleAction.RECOMPUTE,
                estimated_time_ms=compute_time_ms,
                compute_time_ms=compute_time_ms,
                load_time_ms=None,
                source_tier=None,
                reason="chunk is not cached in any configured tier",
            )

        source_tier, cached_chunk = cached
        load_time_ms = self.cost_model.estimate_load_time_ms(cached_chunk, source_tier)

        if load_time_ms <= compute_time_ms:
            return ScheduleDecision(
                chunk=chunk,
                action=ScheduleAction.LOAD,
                estimated_time_ms=load_time_ms,
                compute_time_ms=compute_time_ms,
                load_time_ms=load_time_ms,
                source_tier=source_tier.name,
                reason="estimated load time is less than recompute time",
            )

        return ScheduleDecision(
            chunk=chunk,
            action=ScheduleAction.RECOMPUTE,
            estimated_time_ms=compute_time_ms,
            compute_time_ms=compute_time_ms,
            load_time_ms=load_time_ms,
            source_tier=source_tier.name,
            reason="estimated recompute time is less than load time",
        )

    def schedule_chunks(self, chunks: list[KVChunk]) -> ScheduleSummary:
        # Schedule a batch of chunks independently
        return ScheduleSummary(
            decisions=tuple(self.choose_for_chunk(chunk) for chunk in chunks)
        )
