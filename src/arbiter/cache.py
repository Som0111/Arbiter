"""Semantic cache: reuse answers for near-duplicate prompts, isolated per (tenant, tier)."""
import threading
import time
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
    def __init__(self, ttl_seconds: float = 3600, similarity_threshold: float = 0.92):
        self._store: dict[tuple[str, str], list[CacheEntry]] = {}
        self.ttl = ttl_seconds
        self.threshold = similarity_threshold
        self._lock = threading.Lock()

    def get(self, tenant_id: str, tier: str, query_embedding: list[float]) -> CacheEntry | None:
        """Best entry in this (tenant, tier) bucket with similarity >= threshold, else None."""
        with self._lock:
            self._purge_expired()
            best, best_sim = None, self.threshold
            for entry in self._store.get((tenant_id, tier), []):
                sim = cosine_similarity(query_embedding, entry.embedding)
                if sim >= best_sim:
                    best, best_sim = entry, sim
            return best

    def put(self, tenant_id: str, tier: str, query_embedding: list[float], entry: CacheEntry):
        # query_embedding is the key; entry.embedding is what get() compares against.
        entry.embedding = query_embedding
        with self._lock:
            self._store.setdefault((tenant_id, tier), []).append(entry)

    def size(self) -> int:
        with self._lock:
            return sum(len(b) for b in self._store.values())

    def _purge_expired(self) -> None:
        cutoff = time.time() - self.ttl
        for key, bucket in self._store.items():
            self._store[key] = [e for e in bucket if e.created_at >= cutoff]
