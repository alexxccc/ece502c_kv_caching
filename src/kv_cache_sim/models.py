"""Core data models for the KV-cache simulation.

The first simulator layer is intentionally small: it defines requests, KV cache
chunks, and memory tiers without committing to one scheduling policy. Later
modules can build compute-only, load-only, LRU, and Late-Token Priority behavior
on top of these classes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, Sequence


class CachePlacement(str, Enum):
    """Where a KV chunk currently lives in the simulated memory hierarchy."""

    GPU = "gpu"
    CPU = "cpu"
    DISK = "disk"
    NOT_CACHED = "not_cached"


@dataclass(frozen=True)
class Request:
    """A synthetic LLM serving request.

    Attributes:
        request_id: Stable ID for metrics and debugging.
        cache_id: Stable ID for the reusable prefix/KV cache object. Different
            requests can share a cache_id when they reuse the same prefix.
        arrival_time_ms: Simulated arrival time.
        prompt_tokens: Total prompt length.
        shared_prefix_tokens: Number of prompt tokens that may reuse cache.
        output_tokens: Number of tokens to decode after prefill.
    """

    request_id: str
    arrival_time_ms: float
    prompt_tokens: int
    cache_id: str | None = None
    shared_prefix_tokens: int = 0
    output_tokens: int = 1

    def __post_init__(self) -> None:
        if self.cache_id is None:
            object.__setattr__(self, "cache_id", self.request_id)
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if not self.cache_id:
            raise ValueError("cache_id must be non-empty")
        if self.arrival_time_ms < 0:
            raise ValueError("arrival_time_ms must be non-negative")
        if self.prompt_tokens <= 0:
            raise ValueError("prompt_tokens must be positive")
        if self.shared_prefix_tokens < 0:
            raise ValueError("shared_prefix_tokens must be non-negative")
        if self.shared_prefix_tokens > self.prompt_tokens:
            raise ValueError("shared_prefix_tokens cannot exceed prompt_tokens")
        if self.output_tokens <= 0:
            raise ValueError("output_tokens must be positive")


@dataclass(frozen=True)
class KVChunk:
    """A contiguous chunk of KV cache for part of a reusable prefix."""

    request_id: str
    cache_id: str
    chunk_index: int
    start_token: int
    end_token: int
    size_bytes: int
    placement: CachePlacement = CachePlacement.NOT_CACHED

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if not self.cache_id:
            raise ValueError("cache_id must be non-empty")
        if self.chunk_index < 0:
            raise ValueError("chunk_index must be non-negative")
        if self.start_token < 0:
            raise ValueError("start_token must be non-negative")
        if self.end_token <= self.start_token:
            raise ValueError("end_token must be greater than start_token")
        if self.size_bytes <= 0:
            raise ValueError("size_bytes must be positive")

    @property
    def token_count(self) -> int:
        """Number of prompt tokens represented by this chunk."""

        return self.end_token - self.start_token

    @property
    def late_token_priority(self) -> int:
        """Simple priority signal used by the proposed LTP policy later."""

        return self.end_token

    @property
    def cache_key(self) -> tuple[str, int]:
        """Global cache key used to reuse chunks across requests."""

        return (self.cache_id, self.chunk_index)

    def with_placement(self, placement: CachePlacement) -> "KVChunk":
        """Return a copy of this chunk with an updated cache placement."""

        return KVChunk(
            request_id=self.request_id,
            cache_id=self.cache_id,
            chunk_index=self.chunk_index,
            start_token=self.start_token,
            end_token=self.end_token,
            size_bytes=self.size_bytes,
            placement=placement,
        )


@dataclass
class MemoryTier:
    """A simulated memory tier such as GPU memory, CPU memory, or disk."""

    name: CachePlacement
    capacity_bytes: int
    bandwidth_bytes_per_ms: float
    chunks: Dict[tuple[str, int], KVChunk] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.name == CachePlacement.NOT_CACHED:
            raise ValueError("MemoryTier name must be a real cache placement")
        if self.capacity_bytes <= 0:
            raise ValueError("capacity_bytes must be positive")
        if self.bandwidth_bytes_per_ms <= 0:
            raise ValueError("bandwidth_bytes_per_ms must be positive")

    @property
    def used_bytes(self) -> int:
        """Total bytes occupied by chunks in this tier."""

        return sum(chunk.size_bytes for chunk in self.chunks.values())

    @property
    def free_bytes(self) -> int:
        """Remaining capacity in this tier."""

        return self.capacity_bytes - self.used_bytes

    def can_store(self, chunk: KVChunk) -> bool:
        """Return whether the tier has enough free capacity for a chunk."""

        return chunk.size_bytes <= self.free_bytes

    def store(self, chunk: KVChunk) -> KVChunk:
        """Store a chunk in this tier and return its updated representation."""

        if not self.can_store(chunk):
            raise ValueError(
                f"{self.name.value} cannot store chunk {chunk.chunk_index}: "
                f"needs {chunk.size_bytes} bytes, has {self.free_bytes} bytes"
            )

        placed_chunk = chunk.with_placement(self.name)
        self.chunks[placed_chunk.cache_key] = placed_chunk
        return placed_chunk

    def remove(self, cache_id: str, chunk_index: int) -> KVChunk:
        """Remove and return a chunk from this tier."""

        return self.chunks.pop((cache_id, chunk_index))

    def contains(self, cache_id: str, chunk_index: int) -> bool:
        """Return whether the tier contains the given cache/chunk pair."""

        return (cache_id, chunk_index) in self.chunks

    def estimate_load_time_ms(self, chunk: KVChunk) -> float:
        """Estimate how long it takes to load a chunk from this tier."""

        return chunk.size_bytes / self.bandwidth_bytes_per_ms

    def upsert(self, chunk: KVChunk) -> KVChunk | None:
        """Store chunk only if it has strictly greater end_token coverage than any
        existing entry at the same cache key.  Never downgrades an already-cached
        version.  Returns the newly stored chunk, or None if the existing entry
        already covers at least as many tokens.
        """
        existing = self.chunks.get(chunk.cache_key)
        if existing is not None and existing.end_token >= chunk.end_token:
            return None
        if existing is not None:
            # Remove old entry so capacity accounting is based on the net change.
            del self.chunks[chunk.cache_key]
        return self.store(chunk)

    def iter_chunks_by_priority(self, reverse: bool = False) -> Iterable[KVChunk]:
        """Iterate over chunks ordered by Late-Token Priority.

        Lower priority chunks are earlier in the prompt and are cheaper eviction
        candidates for the proposed policy.
        """

        return iter(
            sorted(
                self.chunks.values(),
                key=lambda chunk: chunk.late_token_priority,
                reverse=reverse,
            )
        )


def chunk_request_with_prefix_split(
    request: Request,
    chunk_size_tokens: int,
    bytes_per_token: int,
) -> list[KVChunk]:
    """Split a request into shared-prefix chunks and private-suffix chunks.

    Chunks covering [0, shared_prefix_tokens) use ``request.cache_id`` (the
    document id) so they can be reused across requests that reference the same
    document.  Chunks covering [shared_prefix_tokens, prompt_tokens) use a
    per-request cache id (``<request_id>-private``) so they are never matched
    against another request's cached data.

    The last shared chunk may be smaller than ``chunk_size_tokens`` when
    ``shared_prefix_tokens`` is not an exact multiple of the chunk size —
    this is the variable-size boundary block.  All chunk indices are
    continuous across both regions (they do not restart at the private
    boundary) so ``by_index`` dicts in the materialisation layer never
    collide between the two regions.

    When ``shared_prefix_tokens == 0`` the entire prompt is private.
    When ``shared_prefix_tokens == prompt_tokens`` the entire prompt is
    treated as a potentially-reusable shared prefix (degenerate case).
    """

    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be positive")
    if bytes_per_token <= 0:
        raise ValueError("bytes_per_token must be positive")

    private_cache_id = f"{request.request_id}-private"
    chunks: list[KVChunk] = []
    chunk_index = 0
    pos = 0

    # Shared prefix region — uses the document cache_id.
    while pos < request.shared_prefix_tokens:
        end = min(pos + chunk_size_tokens, request.shared_prefix_tokens)
        token_count = end - pos
        chunks.append(
            KVChunk(
                request_id=request.request_id,
                cache_id=request.cache_id,
                chunk_index=chunk_index,
                start_token=pos,
                end_token=end,
                size_bytes=token_count * bytes_per_token,
            )
        )
        chunk_index += 1
        pos = end

    # Private suffix region — unique cache_id per request.
    while pos < request.prompt_tokens:
        end = min(pos + chunk_size_tokens, request.prompt_tokens)
        token_count = end - pos
        chunks.append(
            KVChunk(
                request_id=request.request_id,
                cache_id=private_cache_id,
                chunk_index=chunk_index,
                start_token=pos,
                end_token=end,
                size_bytes=token_count * bytes_per_token,
            )
        )
        chunk_index += 1
        pos = end

    return chunks


def chunk_request(
    request: Request,
    chunk_size_tokens: int,
    bytes_per_token: int,
) -> list[KVChunk]:
    """Split a request prompt into fixed-size KV chunks."""

    if chunk_size_tokens <= 0:
        raise ValueError("chunk_size_tokens must be positive")
    if bytes_per_token <= 0:
        raise ValueError("bytes_per_token must be positive")

    chunks: list[KVChunk] = []
    for chunk_index, start_token in enumerate(
        range(0, request.prompt_tokens, chunk_size_tokens)
    ):
        end_token = min(start_token + chunk_size_tokens, request.prompt_tokens)
        token_count = end_token - start_token
        chunks.append(
            KVChunk(
                request_id=request.request_id,
                cache_id=request.cache_id,
                chunk_index=chunk_index,
                start_token=start_token,
                end_token=end_token,
                size_bytes=token_count * bytes_per_token,
            )
        )

    return chunks


def chunk_request_with_sizes(
    request: Request,
    chunk_sizes_tokens: Sequence[int],
    bytes_per_token: int,
) -> list[KVChunk]:
    """Split a request prompt into variable-size KV chunks.

    This supports experiments inspired by paged-memory systems such as vLLM,
    where the block size may be tuned to the request/KV-cache length.
    """

    if not chunk_sizes_tokens:
        raise ValueError("chunk_sizes_tokens must contain at least one size")
    if any(size <= 0 for size in chunk_sizes_tokens):
        raise ValueError("all chunk sizes must be positive")
    if bytes_per_token <= 0:
        raise ValueError("bytes_per_token must be positive")
    if sum(chunk_sizes_tokens) < request.prompt_tokens:
        raise ValueError("chunk sizes must cover the full prompt")

    chunks: list[KVChunk] = []
    start_token = 0
    for chunk_index, size_tokens in enumerate(chunk_sizes_tokens):
        if start_token >= request.prompt_tokens:
            break

        end_token = min(start_token + size_tokens, request.prompt_tokens)
        token_count = end_token - start_token
        chunks.append(
            KVChunk(
                request_id=request.request_id,
                cache_id=request.cache_id,
                chunk_index=chunk_index,
                start_token=start_token,
                end_token=end_token,
                size_bytes=token_count * bytes_per_token,
            )
        )
        start_token = end_token

    return chunks
