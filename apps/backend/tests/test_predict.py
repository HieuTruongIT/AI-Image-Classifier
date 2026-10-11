# tests/test_predict.py
"""
Tests for prediction endpoints.
"""

import io
import pytest
from PIL import Image


class TestPredictEndpoint:
    """Test suite for /predict endpoint."""

    def test_predict_valid_image(self, client, sample_image_file):
        """Test prediction with a valid image."""
        response = client.post(
            "/predict?top_k=5",
            files={"file": sample_image_file}
        )
        assert response.status_code == 200
        data = response.json()
        
        # Check required fields
        assert "label" in data
        assert "confidence" in data
        assert "top_k" in data
        assert "model_version" in data
        
        # Validate types
        assert isinstance(data["label"], str)
        assert isinstance(data["confidence"], float)
        assert 0 <= data["confidence"] <= 1
        assert isinstance(data["top_k"], list)
        assert len(data["top_k"]) == 5

    def test_predict_top_k_variations(self, client, sample_image_bytes):
        """Test different top_k values."""
        for k in [1, 3, 5, 10]:
            response = client.post(
                f"/predict?top_k={k}",
                files={"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data["top_k"]) == k

    def test_predict_invalid_top_k(self, client, sample_image_bytes):
        """Test invalid top_k values are rejected."""
        # top_k = 0 should fail
        response = client.post(
            "/predict?top_k=0",
            files={"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        )
        assert response.status_code == 422  # Validation error

        # top_k > 10 should fail
        response = client.post(
            "/predict?top_k=20",
            files={"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        )
        assert response.status_code == 422

    def test_predict_non_image_file(self, client, invalid_file):
        """Test rejection of non-image files."""
        response = client.post(
            "/predict",
            files={"file": invalid_file}
        )
        assert response.status_code == 415  # Unsupported media type

    def test_predict_corrupted_image(self, client, corrupted_image):
        """Test handling of corrupted image data."""
        response = client.post(
            "/predict",
            files={"file": corrupted_image}
        )
        assert response.status_code == 400  # Bad request

    def test_predict_no_file(self, client):
        """Test error when no file is provided."""
        response = client.post("/predict")
        assert response.status_code == 422  # Validation error

    def test_predict_large_image(self, client, large_image_bytes):
        """Test prediction with larger image (should resize)."""
        response = client.post(
            "/predict?top_k=3",
            files={"file": ("large.jpg", io.BytesIO(large_image_bytes), "image/jpeg")}
        )
        assert response.status_code == 200
        data = response.json()
        assert "label" in data

    def test_predict_png_image(self, client):
        """Test prediction with PNG image format."""
        img = Image.new("RGB", (224, 224), color=(0, 255, 0))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        buffer.seek(0)
        
        response = client.post(
            "/predict",
            files={"file": ("test.png", buffer, "image/png")}
        )
        assert response.status_code == 200

    def test_predict_response_has_inference_time(self, client, sample_image_bytes):
        """Test that response includes inference time."""
        response = client.post(
            "/predict",
            files={"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        )
        assert response.status_code == 200
        data = response.json()
        assert "inference_ms" in data
        # inference_ms should be 0 if cache hit, or positive if computed
        assert data["inference_ms"] >= 0

    def test_predict_cache_behavior(self, client, sample_image_bytes):
        """Test that identical images return cached results."""
        # First request
        response1 = client.post(
            "/predict",
            files={"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        )
        assert response1.status_code == 200
        
        # Second identical request
        response2 = client.post(
            "/predict",
            files={"file": ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")}
        )
        assert response2.status_code == 200
        data2 = response2.json()
        
        # Should be a cache hit
        assert data2["cache_hit"] is True
        assert data2["inference_ms"] == 0
