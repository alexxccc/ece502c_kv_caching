"""Synthetic workload generation for KV-cache experiments."""

from __future__ import annotations

import random
from dataclasses import dataclass

from kv_cache_sim.models import CachePlacement, MemoryTier, Request


@dataclass(frozen=True)
class WorkloadConfig:
    """Controls repeatable causal request streams for simulations.

    Documents are generated first as a pool of "source material" with fixed
    lengths.  Each request then draws a random prefix of one document (the
    shared, potentially-reusable portion) plus an independent private suffix
    (unique to that request).  No cache tier is pre-warmed; the very first
    request to touch each prefix must recompute it, and later requests can
    benefit from what prior requests left behind in the cache hierarchy.

    Attributes:
        seed: RNG seed for reproducibility.
        num_requests: Total requests to generate.
        num_documents: Size of the document pool.
        doc_length_min / doc_length_max: Range for each document's token length.
        shared_prefix_tokens_min / shared_prefix_tokens_max: Range for how many
            tokens a request borrows from its chosen document (always starts at
            token 0).  Clamped to the actual document length at generation time.
        private_suffix_tokens_min / private_suffix_tokens_max: Range for the
            per-request private suffix that follows the shared prefix.
        document_mix: How requests choose a document — ``uniform`` (random),
            ``round_robin``, or ``skewed`` (hot-document bias).
        skew_hot_probability: With ``skewed`` mix, probability of choosing
            the first document (doc-0).
    """

    seed: int
    num_requests: int
    num_documents: int
    doc_length_min: int
    doc_length_max: int
    shared_prefix_tokens_min: int
    shared_prefix_tokens_max: int
    private_suffix_tokens_min: int
    private_suffix_tokens_max: int
    document_mix: str = "uniform"
    """How to pick ``cache_id`` per request: uniform, round_robin, or skewed."""

    skew_hot_probability: float = 0.85
    """With ``skewed`` mix, probability of choosing ``doc-0`` (rest split uniformly)."""

    def __post_init__(self) -> None:
        if self.num_requests <= 0:
            raise ValueError("num_requests must be positive")
        if self.num_documents <= 0:
            raise ValueError("num_documents must be positive")
        if self.doc_length_min <= 0 or self.doc_length_max < self.doc_length_min:
            raise ValueError("invalid doc_length range")
        if self.shared_prefix_tokens_min < 0:
            raise ValueError("shared_prefix_tokens_min must be non-negative")
        if self.shared_prefix_tokens_max < self.shared_prefix_tokens_min:
            raise ValueError("shared_prefix_tokens_max must be >= min")
        if self.private_suffix_tokens_min <= 0:
            raise ValueError("private_suffix_tokens_min must be positive")
        if self.private_suffix_tokens_max < self.private_suffix_tokens_min:
            raise ValueError("private_suffix_tokens_max must be >= min")
        if self.document_mix not in ("uniform", "round_robin", "skewed"):
            raise ValueError("document_mix must be uniform, round_robin, or skewed")
        if not 0.0 <= self.skew_hot_probability <= 1.0:
            raise ValueError("skew_hot_probability must be in [0, 1]")


def generate_requests(config: WorkloadConfig) -> list[Request]:
    """Generate a causal stream of requests with partial shared prefixes.

    Step 1 — build a fixed document pool: each document gets a random length
    drawn from [doc_length_min, doc_length_max].

    Step 2 — for each request pick a document according to ``document_mix``,
    draw ``shared_prefix_tokens`` uniformly from the intersection of
    [shared_prefix_tokens_min, shared_prefix_tokens_max] and [0, doc_length],
    draw a private suffix length, and assemble the Request.

    The resulting Request has:
      - ``cache_id``  = the document id (shared-prefix cache key root)
      - ``prompt_tokens`` = shared_prefix_tokens + private_suffix_tokens
      - ``shared_prefix_tokens`` = the prefix length drawn above

    No disk or GPU tier is seeded.  All caching is causal: a chunk only
    appears in any tier after the request that first computed it is processed.
    """
    rng = random.Random(config.seed)
    doc_ids = [f"doc-{i}" for i in range(config.num_documents)]
    doc_lengths: dict[str, int] = {
        doc_id: rng.randint(config.doc_length_min, config.doc_length_max)
        for doc_id in doc_ids
    }

    requests: list[Request] = []
    for i in range(config.num_requests):
        if config.document_mix == "round_robin":
            doc_id = doc_ids[i % len(doc_ids)]
        elif config.document_mix == "skewed":
            if rng.random() < config.skew_hot_probability:
                doc_id = doc_ids[0]
            else:
                doc_id = rng.choice(doc_ids)
        else:
            doc_id = rng.choice(doc_ids)

        doc_length = doc_lengths[doc_id]

        # Clamp the shared prefix range to [0, doc_length].
        prefix_lo = min(config.shared_prefix_tokens_min, doc_length)
        prefix_hi = min(config.shared_prefix_tokens_max, doc_length)
        shared_prefix_tokens = rng.randint(prefix_lo, prefix_hi)

        private_tokens = rng.randint(
            config.private_suffix_tokens_min,
            config.private_suffix_tokens_max,
        )
        prompt_tokens = shared_prefix_tokens + private_tokens

        requests.append(
            Request(
                request_id=f"req-{i}",
                cache_id=doc_id,
                arrival_time_ms=0.0,
                prompt_tokens=prompt_tokens,
                shared_prefix_tokens=shared_prefix_tokens,
            )
        )

    return requests


def make_disk_tier(capacity_bytes: int, bandwidth_bytes_per_ms: float) -> MemoryTier:
    """Create a disk-like cold storage tier.

    Pass a very large ``capacity_bytes`` (e.g. ``2**62``) to model an
    effectively unbounded persistent store — realistic given that disk capacity
    is typically orders of magnitude larger than GPU memory.
    """
    return MemoryTier(
        name=CachePlacement.DISK,
        capacity_bytes=capacity_bytes,
        bandwidth_bytes_per_ms=bandwidth_bytes_per_ms,
    )
