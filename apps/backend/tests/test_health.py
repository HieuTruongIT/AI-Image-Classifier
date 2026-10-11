# tests/test_health.py
"""
Tests for health and info endpoints.
"""

import pytest


class TestHealthEndpoints:
    """Test suite for health check endpoints."""

    def test_root_endpoint(self, client):
        """Test root endpoint returns API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert data["service"] == "VisionScale API"
        assert "version" in data
        assert "docs" in data

    def test_health_endpoint(self, client):
        """Test health check returns ok status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_loaded"] is True

    def test_version_endpoint(self, client):
        """Test version endpoint returns correct info."""
        response = client.get("/version")
        assert response.status_code == 200
        data = response.json()
        assert "api_version" in data
        assert "model_version" in data
        assert "features" in data
        assert isinstance(data["features"], list)
        # Verify key features are listed
        assert "image_classification" in data["features"]
        assert "batch_prediction" in data["features"]
        assert "async_jobs" in data["features"]

    def test_metrics_endpoint(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/plain")
        # Check for expected metric names
        assert "visionscale" in response.text or "request" in response.text.lower()


class TestCacheEndpoints:
    """Test suite for cache management endpoints."""

    def test_cache_stats(self, client):
        """Test cache stats endpoint."""
        response = client.get("/cache/stats")
        assert response.status_code == 200
        data = response.json()
        assert "enabled" in data
        assert "size" in data
        assert "max_size" in data
        assert "hits" in data
        assert "misses" in data
        assert "hit_rate" in data

    def test_cache_clear(self, client):
        """Test cache clear endpoint."""
        response = client.post("/cache/clear")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
