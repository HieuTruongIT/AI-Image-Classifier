# tests/test_cache.py
"""
Tests for the LRU cache module.
"""

import time
import pytest
from app.cache import LRUCache, compute_image_hash


class TestLRUCache:
    """Test suite for LRU cache implementation."""

    def test_cache_basic_operations(self):
        """Test basic get/set operations."""
        cache = LRUCache(max_size=10, ttl_seconds=60)
        
        # Set and get
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        # Get non-existent
        assert cache.get("nonexistent") is None

    def test_cache_max_size(self):
        """Test cache evicts oldest entries when full."""
        cache = LRUCache(max_size=3, ttl_seconds=60)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        cache.set("key4", "value4")  # Should evict key1
        
        assert cache.get("key1") is None
        assert cache.get("key2") == "value2"
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_cache_ttl_expiration(self):
        """Test entries expire after TTL."""
        cache = LRUCache(max_size=10, ttl_seconds=0.1)  # 100ms TTL
        
        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"
        
        time.sleep(0.2)  # Wait for expiration
        assert cache.get("key1") is None

    def test_cache_lru_ordering(self):
        """Test LRU eviction order."""
        cache = LRUCache(max_size=3, ttl_seconds=60)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.set("key3", "value3")
        
        # Access key1 to make it most recently used
        cache.get("key1")
        
        # Add new key, should evict key2 (least recently used)
        cache.set("key4", "value4")
        
        assert cache.get("key1") == "value1"
        assert cache.get("key2") is None
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_cache_stats(self):
        """Test cache statistics."""
        cache = LRUCache(max_size=10, ttl_seconds=60)
        
        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("nonexistent")  # Miss
        
        stats = cache.get_stats()
        assert stats["size"] == 1
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate"] == pytest.approx(0.6667, rel=0.01)

    def test_cache_clear(self):
        """Test cache clearing."""
        cache = LRUCache(max_size=10, ttl_seconds=60)
        
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        cache.clear()
        
        assert cache.get("key1") is None
        assert cache.get("key2") is None
        stats = cache.get_stats()
        assert stats["size"] == 0
        assert stats["hits"] == 0
        assert stats["misses"] == 0

    def test_cache_delete(self):
        """Test deleting individual entries."""
        cache = LRUCache(max_size=10, ttl_seconds=60)
        
        cache.set("key1", "value1")
        assert cache.delete("key1") is True
        assert cache.get("key1") is None
        assert cache.delete("nonexistent") is False

    def test_cache_update_existing(self):
        """Test updating existing entry."""
        cache = LRUCache(max_size=10, ttl_seconds=60)
        
        cache.set("key1", "value1")
        cache.set("key1", "updated_value")
        assert cache.get("key1") == "updated_value"


class TestImageHash:
    """Test suite for image hashing."""

    def test_compute_hash(self):
        """Test image hash computation."""
        data1 = b"image data 1"
        data2 = b"image data 2"
        
        hash1 = compute_image_hash(data1)
        hash2 = compute_image_hash(data2)
        
        assert len(hash1) == 32
        assert len(hash2) == 32
        assert hash1 != hash2

    def test_hash_deterministic(self):
        """Test hash is deterministic for same input."""
        data = b"test image data"
        
        hash1 = compute_image_hash(data)
        hash2 = compute_image_hash(data)
        
        assert hash1 == hash2

    def test_hash_different_for_different_input(self):
        """Test hash is different for different inputs."""
        hashes = [compute_image_hash(f"data{i}".encode()) for i in range(10)]
        assert len(set(hashes)) == 10  # All unique
