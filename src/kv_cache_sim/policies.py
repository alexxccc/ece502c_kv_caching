"""Eviction policies for simulated KV-cache tiers."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from kv_cache_sim.models import KVChunk, MemoryTier


ChunkKey = tuple[str, int]


def chunk_key(chunk: KVChunk) -> ChunkKey:
    """Return the stable dictionary key for a chunk."""

    return chunk.cache_key


class EvictionPolicy(ABC):
    """Interface for policies that choose which cache chunk to evict."""

    name: str

    def record_store(self, chunk: KVChunk) -> None:
        """Update policy state after a chunk is stored."""

    def record_access(self, chunk: KVChunk) -> None:
        """Update policy state after a cache hit/access."""

    def record_remove(self, chunk: KVChunk) -> None:
        """Update policy state after a chunk leaves the tier."""

    @abstractmethod
    def choose_victim(self, tier: MemoryTier) -> KVChunk:
        """Choose a chunk to evict from the given tier."""


class FIFOPolicy(EvictionPolicy):
    """Evict the chunk that has been stored for the longest time."""

    name = "fifo"

    def choose_victim(self, tier: MemoryTier) -> KVChunk:
        if not tier.chunks:
            raise ValueError("cannot choose a FIFO victim from an empty tier")

        return next(iter(tier.chunks.values()))


class LRUPolicy(EvictionPolicy):
    """Evict the least recently used chunk."""

    name = "lru"

    def __init__(self) -> None:
        self._clock = 0
        self._last_access: dict[ChunkKey, int] = {}

    def record_store(self, chunk: KVChunk) -> None:
        self.record_access(chunk)

    def record_access(self, chunk: KVChunk) -> None:
        self._clock += 1
        self._last_access[chunk_key(chunk)] = self._clock

    def record_remove(self, chunk: KVChunk) -> None:
        self._last_access.pop(chunk_key(chunk), None)

    def choose_victim(self, tier: MemoryTier) -> KVChunk:
        if not tier.chunks:
            raise ValueError("cannot choose an LRU victim from an empty tier")

        return min(
            tier.chunks.values(),
            key=lambda chunk: self._last_access.get(chunk_key(chunk), -1),
        )


class LateTokenPriorityPolicy(EvictionPolicy):
    """Evict earlier chunks before later chunks.

    This is the first implementation of the project proposal's main idea. A
    lower chunk index means the chunk represents earlier prompt tokens, which
    are treated as cheaper to recompute than later prompt tokens.
    """

    name = "late_token_priority"

    def choose_victim(self, tier: MemoryTier) -> KVChunk:
        if not tier.chunks:
            raise ValueError("cannot choose an LTP victim from an empty tier")

        return min(
            tier.chunks.values(),
            key=lambda chunk: (
                chunk.late_token_priority,
                chunk.chunk_index,
                chunk.cache_id
            ),
        )


class FrequencyTokenPriorityPolicy(EvictionPolicy):
    """Evict by log(1 + end_token) * (2 + access_count), lowest score first.

    Combines two causal signals without omniscient knowledge of which chunks
    belong to a shared prefix:

    - log(1 + end_token): diminishing-returns recompute cost — later tokens
      are more expensive but a chunk at position 8000 is not 4x as precious
      as one at 2000.
    - (2 + access_count): reuse evidence — frequently accessed chunks are
      likely from a shared prefix.  The +2 offset keeps newly inserted chunks
      (access_count=0) at score 2*log(1+end_token) rather than zero, so fresh
      chunks are not immediately the top eviction target.

    Ties broken by insertion order (oldest evicted first).
    """

    name = "fltp"

    def __init__(self) -> None:
        self._access_count: dict[ChunkKey, int] = {}
        self._insert_order: dict[ChunkKey, int] = {}
        self._clock = 0

    def record_store(self, chunk: KVChunk) -> None:
        key = chunk_key(chunk)
        self._access_count[key] = 0
        self._clock += 1
        self._insert_order[key] = self._clock

    def record_access(self, chunk: KVChunk) -> None:
        key = chunk_key(chunk)
        self._access_count[key] = self._access_count.get(key, 0) + 1

    def record_remove(self, chunk: KVChunk) -> None:
        key = chunk_key(chunk)
        self._access_count.pop(key, None)
        self._insert_order.pop(key, None)

    def _score(self, chunk: KVChunk) -> tuple[float, int]:
        key = chunk_key(chunk)
        keep = math.log1p(chunk.end_token) * math.log(2 + self._access_count.get(key, 0))
        age = self._insert_order.get(key, 0)
        return keep, age  # evict min keep; break ties by evicting oldest (min age)

    def choose_victim(self, tier: MemoryTier) -> KVChunk:
        if not tier.chunks:
            raise ValueError("cannot choose an FTP victim from an empty tier")

        return min(tier.chunks.values(), key=self._score)
