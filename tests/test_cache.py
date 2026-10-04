import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from arbiter.cache import CacheEntry, SemanticCache, cosine_similarity, embed

V = [1.0, 0.0, 0.0]


def entry(vec=V, age=0.0):
    return CacheEntry(vec, "cached answer", "groq/llama-3.1-8b-instant", 0.002, time.time() - age)


@pytest.fixture
def cache():
    c = SemanticCache(ttl_seconds=3600, similarity_threshold=0.92)
    c.put("tenant_a", "simple", V, entry())
    return c


def test_identical_query_hits(cache):
    hit = cache.get("tenant_a", "simple", V)
    assert hit is not None and hit.response == "cached answer"


def test_similar_query_above_threshold_hits(cache):
    assert cache.get("tenant_a", "simple", [1.0, 0.1, 0.0]) is not None  # cos ~0.995


def test_different_tenant_misses(cache):
    assert cache.get("tenant_b", "simple", V) is None


def test_different_tier_misses(cache):
    assert cache.get("tenant_a", "complex", V) is None


def test_expired_entry_misses_and_is_purged():
    c = SemanticCache(ttl_seconds=60)
    c.put("tenant_a", "simple", V, entry(age=61))
    assert c.get("tenant_a", "simple", V) is None
    assert c.size() == 0


def test_below_threshold_misses(cache):
    assert cache.get("tenant_a", "simple", [0.0, 1.0, 0.0]) is None  # orthogonal
    assert cache.get("tenant_a", "simple", [1.0, 0.5, 0.0]) is None  # cos ~0.894 < 0.92


def test_best_match_wins():
    c = SemanticCache()
    c.put("t", "simple", [1.0, 0.2, 0.0], CacheEntry([], "near", "m", 0.0, time.time()))
    c.put("t", "simple", V, CacheEntry([], "exact", "m", 0.0, time.time()))
    assert c.get("t", "simple", V).response == "exact"


def test_cosine_similarity_basics():
    assert cosine_similarity(V, V) == pytest.approx(1.0)
    assert cosine_similarity(V, [0.0, 1.0, 0.0]) == pytest.approx(0.0)
    assert cosine_similarity(V, [0.0, 0.0, 0.0]) == 0.0  # zero vector must not divide by zero


def test_embed_calls_current_embedding_model():
    result = SimpleNamespace(embeddings=[SimpleNamespace(values=[0.1, 0.2])])
    with patch("arbiter.cache._client") as client:
        client.return_value.models.embed_content.return_value = result
        assert embed("hello") == [0.1, 0.2]
    kwargs = client.return_value.models.embed_content.call_args.kwargs
    assert kwargs["model"] == "gemini-embedding-001"
    assert kwargs["contents"] == "hello"
