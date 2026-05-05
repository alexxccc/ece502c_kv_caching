"""
Simple scheduler implementation
Cost-aware scheduling for recomputing vs loading KV-cache chunks.
Uses the CostModel to estimate compute and load times, and chooses the cheaper option.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

from kv_cache_sim.models import CachePlacement, KVChunk, MemoryTier

logger = logging.getLogger(__name__)


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
    Loading cost is based on chunk size and memory-tier bandwidth — both are
    byte-accurate, scaling with the actual token count of each chunk so that
    variable-size boundary chunks are costed correctly.
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
        # Estimate time to load a cached chunk from a memory tier.
        # Uses the *needed* chunk's size_bytes (byte-accurate for variable chunks).
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
            decision.action == ScheduleAction.RECOMPUTE for decision in self.decisions
        )

    @property
    def load_count(self) -> int:
        return sum(decision.action == ScheduleAction.LOAD for decision in self.decisions)

    @property
    def total_estimated_time_ms(self) -> float:
        return sum(decision.estimated_time_ms for decision in self.decisions)


@dataclass(frozen=True)
class CakeOperation:
    """One operation on Cake's compute or I/O front."""

    chunk: KVChunk
    action: ScheduleAction
    start_time_ms: float
    end_time_ms: float
    source_tier: CachePlacement | None
    reason: str = ""

    @property
    def duration_ms(self) -> float:
        return self.end_time_ms - self.start_time_ms


@dataclass(frozen=True)
class CakeScheduleSummary:
    """Timeline produced by the Cake-style bidirectional scheduler."""

    operations: tuple[CakeOperation, ...]
    compute_front_time_ms: float
    load_front_time_ms: float

    @property
    def total_estimated_ttft_ms(self) -> float:
        return max(self.compute_front_time_ms, self.load_front_time_ms)

    @property
    def recompute_count(self) -> int:
        return sum(
            operation.action == ScheduleAction.RECOMPUTE for operation in self.operations
        )

    @property
    def load_count(self) -> int:
        return sum(
            operation.action == ScheduleAction.LOAD for operation in self.operations
        )


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
        """Find chunk in tiers; only return a hit if cached version fully covers
        the needed token range (end_token >= chunk.end_token).

        The search stops at the first tier that holds the cache key because
        write-through keeps GPU and disk in sync — if the first hit has
        insufficient coverage, no later tier will have a better version.
        """
        for tier in self.memory_tiers:
            cached_chunk = tier.chunks.get(chunk.cache_key)
            if cached_chunk is not None:
                if cached_chunk.end_token >= chunk.end_token:
                    return tier, cached_chunk
                # Partial coverage found in this tier; log and report miss.
                logger.debug(
                    "chunk (%s, %d): partial coverage in %s - "
                    "cached end=%d, need end=%d -> recompute",
                    chunk.cache_id,
                    chunk.chunk_index,
                    tier.name.value,
                    cached_chunk.end_token,
                    chunk.end_token,
                )
                return None
        return None

    def choose_for_chunk(self, chunk: KVChunk) -> ScheduleDecision:
        # Choose the cheaper way to obtain one KV chunk

        compute_time_ms = self.cost_model.estimate_compute_time_ms(chunk)
        cached = self.find_cached_chunk(chunk)

        if cached is None:
            # Determine whether a partial entry existed for the reason string.
            partial = any(
                t.chunks.get(chunk.cache_key) is not None for t in self.memory_tiers
            )
            reason = (
                "partial coverage found — recomputing for missing tokens"
                if partial
                else "chunk is not cached in any configured tier"
            )
            return ScheduleDecision(
                chunk=chunk,
                action=ScheduleAction.RECOMPUTE,
                estimated_time_ms=compute_time_ms,
                compute_time_ms=compute_time_ms,
                load_time_ms=None,
                source_tier=None,
                reason=reason,
            )

        source_tier, cached_chunk = cached
        # Use the needed chunk's size_bytes for load cost (byte-accurate).
        load_time_ms = self.cost_model.estimate_load_time_ms(chunk, source_tier)

        if load_time_ms <= compute_time_ms:
            return ScheduleDecision(
                chunk=chunk,
                action=ScheduleAction.LOAD,
                estimated_time_ms=load_time_ms,
                compute_time_ms=compute_time_ms,
                load_time_ms=load_time_ms,
                source_tier=source_tier.name,
                reason=f"load from {source_tier.name.value} faster than recompute",
            )

        return ScheduleDecision(
            chunk=chunk,
            action=ScheduleAction.RECOMPUTE,
            estimated_time_ms=compute_time_ms,
            compute_time_ms=compute_time_ms,
            load_time_ms=load_time_ms,
            source_tier=source_tier.name,
            reason=f"recompute faster than load from {source_tier.name.value}",
        )

    def schedule_chunks(self, chunks: list[KVChunk]) -> ScheduleSummary:
        # Schedule a batch of chunks independently
        return ScheduleSummary(
            decisions=tuple(self.choose_for_chunk(chunk) for chunk in chunks)
        )


class CakeBidirectionalScheduler:
    """
    Simulates Cake's two-front compute/load schedule.

    The I/O front starts at the last chunk and scans backward, skipping any
    chunk that is not in cache (or has insufficient coverage) with zero time
    cost — those are left for the compute front to handle.  Once the I/O
    front identifies a loadable chunk it compares the projected load finish
    time against the projected compute finish time; it claims the chunk only
    if loading would complete no later than compute.

    A boolean ``scheduled`` array tracks which chunks have been committed to
    one of the two fronts.  When the fronts cross (load_index < compute_index)
    Phase 1 ends and Phase 2 sweeps the remaining unscheduled chunks in order,
    assigning RECOMPUTE to each.  This handles the sparse case correctly:
    cached chunks can appear anywhere in the sequence and non-cached chunks
    are never wasted in the I/O pipeline.

    TTFT is ``max(compute_front_time_ms, load_front_time_ms)``.
    """

    def __init__(
        self,
        memory_tiers: list[MemoryTier],
        cost_model: CostModel,
    ) -> None:
        if not memory_tiers:
            raise ValueError("memory_tiers must contain at least one tier")

        self.memory_tiers = memory_tiers
        self.cost_model = cost_model

    def _find_cached_with_coverage(
        self, chunk: KVChunk
    ) -> tuple[MemoryTier, KVChunk] | None:
        """Return the first tier that holds chunk with sufficient end_token coverage.

        Stops at the first tier that owns the cache key; if its coverage is
        insufficient no other tier will have a better version (write-through
        invariant keeps all tiers in sync for the same key).
        """
        for tier in self.memory_tiers:
            cached_chunk = tier.chunks.get(chunk.cache_key)
            if cached_chunk is not None:
                if cached_chunk.end_token >= chunk.end_token:
                    return tier, cached_chunk
                logger.debug(
                    "chunk (%s, %d): partial coverage in %s - "
                    "cached end=%d, need end=%d",
                    chunk.cache_id,
                    chunk.chunk_index,
                    tier.name.value,
                    cached_chunk.end_token,
                    chunk.end_token,
                )
                return None
        return None

    def schedule_chunks(self, chunks: list[KVChunk]) -> CakeScheduleSummary:
        """Build a Cake-style bidirectional schedule for prompt chunks.

        Phase 1 — bidirectional scan:
          The I/O front scans backward, silently skipping non-loadable chunks.
          When it finds a loadable chunk it compares projected timelines;
          the cheaper front advances.  Phase 1 ends when the fronts cross.

        Phase 2 — compute sweep:
          Any chunk not yet scheduled (skipped by the I/O front or left in the
          middle after the fronts crossed) is assigned RECOMPUTE in index order,
          appended to the compute timeline.
        """
        ordered_chunks = sorted(chunks, key=lambda c: c.chunk_index)
        n = len(ordered_chunks)
        if n == 0:
            return CakeScheduleSummary(
                operations=(),
                compute_front_time_ms=0.0,
                load_front_time_ms=0.0,
            )

        scheduled: list[bool] = [False] * n
        operations: list[CakeOperation] = []

        compute_time_ms = 0.0
        load_time_ms = 0.0
        compute_index = 0
        load_index = n - 1

        # ── Phase 1: bidirectional scan ──────────────────────────────────────
        while compute_index <= load_index:
            # Scan backward from load_index to find the next loadable chunk.
            # Non-loadable chunks are skipped at zero cost (Phase 2 handles them).
            while load_index >= compute_index:
                load_candidate = ordered_chunks[load_index]
                cached = self._find_cached_with_coverage(load_candidate)
                if cached is not None:
                    break
                load_index -= 1  # skip — not in cache or insufficient coverage

            if load_index < compute_index:
                break  # fronts have crossed; no more loadable chunks in range

            load_candidate = ordered_chunks[load_index]
            source_tier = cached[0]  # type: ignore[index]  # cached is not None here

            # Use the needed chunk's size for byte-accurate load cost.
            load_next_ms = load_time_ms + self.cost_model.estimate_load_time_ms(
                load_candidate, source_tier
            )
            compute_next_ms = compute_time_ms + self.cost_model.estimate_compute_time_ms(
                ordered_chunks[compute_index]
            )

            if load_next_ms <= compute_next_ms:
                # I/O front claims this chunk.
                start = load_time_ms
                load_time_ms = load_next_ms
                scheduled[load_index] = True
                operations.append(
                    CakeOperation(
                        chunk=load_candidate,
                        action=ScheduleAction.LOAD,
                        start_time_ms=start,
                        end_time_ms=load_time_ms,
                        source_tier=source_tier.name,
                        reason=f"loaded from {source_tier.name.value}",
                    )
                )
                load_index -= 1
            else:
                # Compute front is cheaper right now; advance it.
                compute_candidate = ordered_chunks[compute_index]
                start = compute_time_ms
                compute_time_ms += self.cost_model.estimate_compute_time_ms(
                    compute_candidate
                )
                scheduled[compute_index] = True
                operations.append(
                    CakeOperation(
                        chunk=compute_candidate,
                        action=ScheduleAction.RECOMPUTE,
                        start_time_ms=start,
                        end_time_ms=compute_time_ms,
                        source_tier=None,
                        reason="recompute faster than load from available tier",
                    )
                )
                compute_index += 1

        # ── Phase 2: compute sweep for all remaining unscheduled chunks ──────
        for i in range(n):
            if scheduled[i]:
                continue
            chunk = ordered_chunks[i]

            # Build a descriptive reason for logging / metrics.
            cached_any = self._find_cached_with_coverage(chunk)
            if cached_any is not None:
                reason = "compute front reached chunk before I/O front"
            else:
                partial = any(
                    t.chunks.get(chunk.cache_key) is not None
                    for t in self.memory_tiers
                )
                reason = (
                    "partial coverage — recomputing for missing tokens"
                    if partial
                    else "not in any cache tier"
                )

            start = compute_time_ms
            compute_time_ms += self.cost_model.estimate_compute_time_ms(chunk)
            scheduled[i] = True
            operations.append(
                CakeOperation(
                    chunk=chunk,
                    action=ScheduleAction.RECOMPUTE,
                    start_time_ms=start,
                    end_time_ms=compute_time_ms,
                    source_tier=None,
                    reason=reason,
                )
            )

        assert all(scheduled), "BUG: some chunks were not scheduled"

        return CakeScheduleSummary(
            operations=tuple(operations),
            compute_front_time_ms=compute_time_ms,
            load_front_time_ms=load_time_ms,
        )
