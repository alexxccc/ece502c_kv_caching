"""Synthetic workload generation for KV-cache experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass

from kv_cache_sim.models import CachePlacement, MemoryTier, Request, chunk_request


@dataclass(frozen=True)
class WorkloadConfig:
    """Controls repeatable random request streams for simulations."""

    seed: int
    num_requests: int
    num_documents: int
    prompt_tokens_min: int
    prompt_tokens_max: int
    arrival_gap_ms: float = 0.0
    document_mix: str = "uniform"
    """How to pick ``cache_id`` per request: uniform, round_robin, or skewed."""

    skew_hot_probability: float = 0.85
    """With ``skewed`` mix, probability of choosing ``doc-0`` (rest split uniformly)."""

    def __post_init__(self) -> None:
        if self.num_requests <= 0:
            raise ValueError("num_requests must be positive")
        if self.num_documents <= 0:
            raise ValueError("num_documents must be positive")
        if self.prompt_tokens_min <= 0 or self.prompt_tokens_max < self.prompt_tokens_min:
            raise ValueError("invalid prompt token range")
        if self.arrival_gap_ms < 0:
            raise ValueError("arrival_gap_ms must be non-negative")
        if self.document_mix not in ("uniform", "round_robin", "skewed"):
            raise ValueError("document_mix must be uniform, round_robin, or skewed")
        if not 0.0 <= self.skew_hot_probability <= 1.0:
            raise ValueError("skew_hot_probability must be in [0, 1]")


def generate_requests(config: WorkloadConfig) -> list[Request]:
    """Sample a list of requests with shared cache IDs across a document pool."""

    rng = random.Random(config.seed)
    doc_ids = [f"doc-{i}" for i in range(config.num_documents)]
    doc_prompt_tokens = {
        doc_id: rng.randint(config.prompt_tokens_min, config.prompt_tokens_max)
        for doc_id in doc_ids
    }

    requests: list[Request] = []
    for i in range(config.num_requests):
        if config.document_mix == "round_robin":
            cache_id = doc_ids[i % len(doc_ids)]
        elif config.document_mix == "skewed":
            if rng.random() < config.skew_hot_probability:
                cache_id = doc_ids[0]
            else:
                cache_id = rng.choice(doc_ids)
        else:
            cache_id = rng.choice(doc_ids)
        prompt_tokens = doc_prompt_tokens[cache_id]
        requests.append(
            Request(
                request_id=f"req-{i}",
                cache_id=cache_id,
                arrival_time_ms=i * config.arrival_gap_ms,
                prompt_tokens=prompt_tokens,
                shared_prefix_tokens=prompt_tokens,
            )
        )
    return requests


def seed_tier_from_request(
    tier: MemoryTier,
    request: Request,
    chunk_size_tokens: int,
    bytes_per_token: int,
) -> None:
    """Place every chunk for ``request`` onto ``tier`` (used for cold storage warm-up)."""

    for chunk in chunk_request(request, chunk_size_tokens, bytes_per_token):
        if not tier.can_store(chunk):
            raise ValueError(
                f"{tier.name.value} tier cannot seed chunk {chunk.chunk_index}: "
                f"capacity {tier.capacity_bytes}, chunk {chunk.size_bytes} bytes"
            )
        tier.store(chunk)


def seed_disk_for_workload(
    disk: MemoryTier,
    requests: list[Request],
    chunk_size_tokens: int,
    bytes_per_token: int,
) -> None:
    """Pre-load cold storage with KV for the longest prompt seen per ``cache_id``."""

    by_cache: dict[str, int] = {}
    for request in requests:
        prev = by_cache.get(request.cache_id, 0)
        by_cache[request.cache_id] = max(prev, request.prompt_tokens)

    for cache_id, prompt_tokens in by_cache.items():
        synthetic = Request(
            request_id=f"seed-{cache_id}",
            cache_id=cache_id,
            arrival_time_ms=0.0,
            prompt_tokens=prompt_tokens,
            shared_prefix_tokens=prompt_tokens,
        )
        seed_tier_from_request(disk, synthetic, chunk_size_tokens, bytes_per_token)


def make_disk_tier(capacity_bytes: int, bandwidth_bytes_per_ms: float) -> MemoryTier:
    """Helper for a large disk-like tier used in Cake cold-load experiments."""

    return MemoryTier(
        name=CachePlacement.DISK,
        capacity_bytes=capacity_bytes,
        bandwidth_bytes_per_ms=bandwidth_bytes_per_ms,
    )
