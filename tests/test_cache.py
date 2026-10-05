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


# ---- LRU eviction ----

def basis(n, size=6):
    return [1.0 if i == n else 0.0 for i in range(size)]


def filled(max_size=3, n=3, tenant="t", tier="simple"):
    c = SemanticCache(max_size=max_size)
    for i in range(n):
        c.put(tenant, tier, basis(i), CacheEntry([], f"answer {i}", "m", 0.0, time.time()))
    return c


def test_default_max_size_is_500():
    assert SemanticCache().max_size == 500


def test_inserting_past_max_size_evicts_the_lru_entry():
    c = filled(max_size=3, n=3)
    c.put("t", "simple", basis(3), CacheEntry([], "answer 3", "m", 0.0, time.time()))
    assert c.size() == 3 and c.evictions_total == 1
    assert c.get("t", "simple", basis(0)) is None  # entry 0 was least recently used
    for kept in (1, 2, 3):
        assert c.get("t", "simple", basis(kept)).response == f"answer {kept}"


def test_cache_hit_refreshes_recency():
    c = filled(max_size=3, n=3)
    assert c.get("t", "simple", basis(0)).response == "answer 0"  # touch the oldest
    c.put("t", "simple", basis(3), CacheEntry([], "answer 3", "m", 0.0, time.time()))
    assert c.get("t", "simple", basis(1)) is None  # entry 1 is now the LRU and is evicted
    assert c.get("t", "simple", basis(0)).response == "answer 0"


def test_eviction_is_global_across_buckets():
    c = SemanticCache(max_size=3)
    c.put("a", "simple", basis(0), CacheEntry([], "a0", "m", 0.0, time.time()))
    c.put("b", "complex", basis(0), CacheEntry([], "b0", "m", 0.0, time.time()))
    c.put("a", "simple", basis(1), CacheEntry([], "a1", "m", 0.0, time.time()))
    c.get("a", "simple", basis(0))  # a0 is now fresher than b0
    c.put("c", "standard", basis(0), CacheEntry([], "c0", "m", 0.0, time.time()))
    assert c.get("b", "complex", basis(0)) is None  # b0 was the global LRU
    assert c.get("a", "simple", basis(0)) is not None and c.get("c", "standard", basis(0))
    assert c.size() == 3


def test_expired_entries_are_dropped_before_evicting_live_ones():
    c = SemanticCache(max_size=2, ttl_seconds=60)
    c.put("t", "simple", basis(0), entry(basis(0), age=61))  # already expired
    c.put("t", "simple", basis(1), entry(basis(1)))
    c.put("t", "simple", basis(2), entry(basis(2)))
    assert c.evictions_total == 0 and c.size() == 2
    assert c.get("t", "simple", basis(1)) is not None


def test_miss_does_not_change_recency_or_counters():
    c = filled(max_size=3, n=3)
    assert c.get("t", "simple", basis(5)) is None
    assert c.get("other", "simple", basis(0)) is None
    assert (c.size(), c.evictions_total) == (3, 0)
