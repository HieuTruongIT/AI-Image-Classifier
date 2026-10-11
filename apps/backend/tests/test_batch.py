# tests/test_batch.py
"""
Tests for batch prediction and async job endpoints.
"""

import io
import time
import pytest
from PIL import Image


def create_test_image(color=(128, 128, 128)):
    """Helper to create test images."""
    img = Image.new("RGB", (100, 100), color=color)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer


class TestBatchPrediction:
    """Test suite for /predict/batch endpoint."""

    def test_batch_predict_single_image(self, client):
        """Test batch prediction with single image."""
        files = [("files", ("img1.jpg", create_test_image(), "image/jpeg"))]
        response = client.post("/predict/batch?top_k=3", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["successful"] == 1
        assert data["failed"] == 0
        assert len(data["results"]) == 1

    def test_batch_predict_multiple_images(self, client):
        """Test batch prediction with multiple images."""
        files = [
            ("files", ("img1.jpg", create_test_image((255, 0, 0)), "image/jpeg")),
            ("files", ("img2.jpg", create_test_image((0, 255, 0)), "image/jpeg")),
            ("files", ("img3.jpg", create_test_image((0, 0, 255)), "image/jpeg")),
        ]
        response = client.post("/predict/batch?top_k=3", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert data["successful"] == 3
        assert data["failed"] == 0
        assert len(data["results"]) == 3

    def test_batch_predict_mixed_valid_invalid(self, client):
        """Test batch with mix of valid and invalid files."""
        files = [
            ("files", ("valid.jpg", create_test_image(), "image/jpeg")),
            ("files", ("invalid.txt", io.BytesIO(b"not image"), "text/plain")),
            ("files", ("valid2.jpg", create_test_image(), "image/jpeg")),
        ]
        response = client.post("/predict/batch", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert data["successful"] == 2
        assert data["failed"] == 1
        
        # Check that failed result has error
        failed_result = next(r for r in data["results"] if r["error"])
        assert failed_result["filename"] == "invalid.txt"

    def test_batch_predict_empty(self, client):
        """Test batch with no files."""
        response = client.post("/predict/batch", files=[])
        assert response.status_code == 422  # Validation error - no files

    def test_batch_predict_has_total_ms(self, client):
        """Test that batch response includes total processing time."""
        files = [("files", ("img1.jpg", create_test_image(), "image/jpeg"))]
        response = client.post("/predict/batch", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert "total_ms" in data
        assert data["total_ms"] >= 0

    def test_batch_result_structure(self, client):
        """Test that batch results have correct structure."""
        files = [("files", ("test.jpg", create_test_image(), "image/jpeg"))]
        response = client.post("/predict/batch?top_k=5", files=files)
        
        assert response.status_code == 200
        result = response.json()["results"][0]
        
        assert "index" in result
        assert "filename" in result
        assert "label" in result
        assert "confidence" in result
        assert "top_k" in result
        assert result["error"] is None


class TestAsyncJobs:
    """Test suite for async job endpoints."""

    def test_job_submit(self, client):
        """Test submitting an async job."""
        files = [("files", ("img1.jpg", create_test_image(), "image/jpeg"))]
        response = client.post("/jobs/submit?top_k=3", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert "job_id" in data
        assert "status" in data
        assert "total_images" in data
        assert "poll_url" in data
        assert data["total_images"] == 1

    def test_job_status(self, client):
        """Test getting job status."""
        # Submit a job
        files = [("files", ("img1.jpg", create_test_image(), "image/jpeg"))]
        submit_response = client.post("/jobs/submit", files=files)
        job_id = submit_response.json()["job_id"]
        
        # Get status
        response = client.get(f"/jobs/{job_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == job_id
        assert "status" in data
        assert "progress" in data
        assert "total_images" in data
        assert "processed_images" in data

    def test_job_not_found(self, client):
        """Test getting status of non-existent job."""
        response = client.get("/jobs/nonexistent-job-id")
        assert response.status_code == 404

    def test_list_jobs(self, client):
        """Test listing jobs."""
        response = client.get("/jobs")
        assert response.status_code == 200
        data = response.json()
        assert "jobs" in data
        assert "stats" in data
        assert isinstance(data["jobs"], list)

    def test_job_delete(self, client):
        """Test deleting/cancelling a job."""
        # Submit a job
        files = [("files", ("img1.jpg", create_test_image(), "image/jpeg"))]
        submit_response = client.post("/jobs/submit", files=files)
        job_id = submit_response.json()["job_id"]
        
        # Wait a moment for it to complete
        time.sleep(0.5)
        
        # Delete the job
        response = client.delete(f"/jobs/{job_id}")
        assert response.status_code == 200

    def test_job_results_not_ready(self, client):
        """Test getting results before job completes."""
        # Submit a larger batch that takes time
        files = [("files", (f"img{i}.jpg", create_test_image(), "image/jpeg")) for i in range(5)]
        submit_response = client.post("/jobs/submit", files=files)
        job_id = submit_response.json()["job_id"]
        
        # Immediately try to get results (may still be processing)
        response = client.get(f"/jobs/{job_id}/results")
        # Could be 200 (completed fast) or 400 (still processing)
        assert response.status_code in [200, 400]
