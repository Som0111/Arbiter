"""Semantic cache: reuse answers for near-duplicate prompts, isolated per (tenant, tier)."""
import itertools
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from google import genai
from google.genai import types

from arbiter.config import get_settings

EMBED_MODEL = "gemini-embedding-001"
EMBED_DIMS = 768


@lru_cache
def _client() -> genai.Client:
    return genai.Client(api_key=get_settings().gemini_api_key)


def embed(text: str) -> list[float]:
    result = _client().models.embed_content(
        model=EMBED_MODEL,
        contents=text,
        config=types.EmbedContentConfig(
            task_type="SEMANTIC_SIMILARITY", output_dimensionality=EMBED_DIMS
        ),
    )
    return list(result.embeddings[0].values)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


@dataclass
class CacheEntry:
    embedding: list[float]
    response: str
    model: str
    cost_usd: float
    created_at: float  # time.time()


class SemanticCache:
    """Per-(tenant, tier) buckets, each an OrderedDict kept in recency order (front = LRU).

    max_size bounds the TOTAL entries across all buckets; when full, the globally least
    recently used entry is evicted (the oldest `last_used` tick among the bucket fronts).
    """

    def __init__(self, ttl_seconds: float = 3600, similarity_threshold: float = 0.92,
                 max_size: int = 500):
        # bucket key -> {entry_id: [last_used_tick, CacheEntry]}
        self._store: dict[tuple[str, str], OrderedDict[int, list]] = {}
        self.ttl = ttl_seconds
        self.threshold = similarity_threshold
        self.max_size = max_size
        self.evictions_total = 0  # LRU evictions only; TTL expiry is not counted
        self._ids = itertools.count()
        self._ticks = itertools.count()
        self._lock = threading.Lock()

    def get(self, tenant_id: str, tier: str, query_embedding: list[float]) -> CacheEntry | None:
        """Best entry in this (tenant, tier) bucket with similarity >= threshold, else None.

        A hit becomes the most recently used entry."""
        with self._lock:
            self._purge_expired()
            bucket = self._store.get((tenant_id, tier), {})
            best_id, best_sim = None, self.threshold
            for entry_id, (_, entry) in bucket.items():
                sim = cosine_similarity(query_embedding, entry.embedding)
                if sim >= best_sim:
                    best_id, best_sim = entry_id, sim
            if best_id is None:
                return None
            bucket.move_to_end(best_id)
            bucket[best_id][0] = next(self._ticks)
            return bucket[best_id][1]

    def put(self, tenant_id: str, tier: str, query_embedding: list[float], entry: CacheEntry):
        # query_embedding is the key; entry.embedding is what get() compares against.
        entry.embedding = query_embedding
        with self._lock:
            self._purge_expired()
            while self._size() >= self.max_size and self._size() > 0:
                self._evict_lru()
            bucket = self._store.setdefault((tenant_id, tier), OrderedDict())
            bucket[next(self._ids)] = [next(self._ticks), entry]

    def size(self) -> int:
        with self._lock:
            return self._size()

    def _size(self) -> int:
        return sum(len(b) for b in self._store.values())

    def _evict_lru(self) -> None:
        key = min((k for k, b in self._store.items() if b),
                  key=lambda k: next(iter(self._store[k].values()))[0])
        self._store[key].popitem(last=False)
        if not self._store[key]:
            del self._store[key]
        self.evictions_total += 1

    def _purge_expired(self) -> None:
        cutoff = time.time() - self.ttl
        for key in list(self._store):
            bucket = self._store[key]
            for entry_id in [i for i, (_, e) in bucket.items() if e.created_at < cutoff]:
                del bucket[entry_id]
            if not bucket:
                del self._store[key]
