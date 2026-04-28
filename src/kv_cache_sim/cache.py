"""Cache manager that applies eviction policies to a memory tier."""

from __future__ import annotations

from dataclasses import dataclass

from kv_cache_sim.models import KVChunk, MemoryTier
from kv_cache_sim.policies import EvictionPolicy


@dataclass(frozen=True)
class CacheStoreResult:
    """Result of trying to store one chunk in a managed cache tier."""

    stored_chunk: KVChunk
    evicted_chunks: tuple[KVChunk, ...]


class CacheManager:
    """Policy-driven wrapper around a single memory tier."""

    def __init__(self, tier: MemoryTier, policy: EvictionPolicy) -> None:
        self.tier = tier
        self.policy = policy

    def access(self, cache_id: str, chunk_index: int) -> KVChunk | None:
        """Return a cached chunk and update policy state if it exists."""

        chunk = self.tier.chunks.get((cache_id, chunk_index))
        if chunk is None:
            return None

        self.policy.record_access(chunk)
        return chunk

    def store(self, chunk: KVChunk) -> CacheStoreResult:
        """Store a chunk, evicting policy-selected victims as needed."""

        if chunk.size_bytes > self.tier.capacity_bytes:
            raise ValueError(
                f"chunk {chunk.chunk_index} is larger than "
                f"{self.tier.name.value} capacity"
            )

        existing = self.access(chunk.cache_id, chunk.chunk_index)
        if existing is not None:
            return CacheStoreResult(stored_chunk=existing, evicted_chunks=())

        evicted_chunks: list[KVChunk] = []
        while not self.tier.can_store(chunk):
            victim = self.policy.choose_victim(self.tier)
            removed = self.tier.remove(victim.cache_id, victim.chunk_index)
            self.policy.record_remove(removed)
            evicted_chunks.append(removed)

        stored_chunk = self.tier.store(chunk)
        self.policy.record_store(stored_chunk)
        return CacheStoreResult(
            stored_chunk=stored_chunk,
            evicted_chunks=tuple(evicted_chunks),
        )

    def contains(self, cache_id: str, chunk_index: int) -> bool:
        """Return whether the managed tier contains a chunk."""

        return self.tier.contains(cache_id, chunk_index)
