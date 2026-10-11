# tools/load_test.py
"""
Load Testing Script for VisionScale API using Locust.

Run with:
    locust -f tools/load_test.py --host=http://localhost:8000

Or headless:
    locust -f tools/load_test.py --host=http://localhost:8000 \
           --headless -u 10 -r 2 -t 60s --csv=results/load_test

Results will show:
- Requests per second (RPS)
- Response time percentiles (p50, p95, p99)
- Error rates
"""

import io
import os
import random
from locust import HttpUser, task, between
from PIL import Image
import numpy as np


def generate_random_image(width=224, height=224):
    """Generate a random RGB image for testing."""
    # Create random pixel data
    pixels = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(pixels, 'RGB')
    
    # Convert to bytes
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=85)
    buffer.seek(0)
    return buffer


def generate_gradient_image(width=224, height=224, seed=None):
    """Generate a gradient image (more realistic than random noise)."""
    if seed is not None:
        np.random.seed(seed)
    
    # Create gradient
    x = np.linspace(0, 1, width)
    y = np.linspace(0, 1, height)
    xv, yv = np.meshgrid(x, y)
    
    # Random colors for gradient
    r = (xv * 255 * random.random()).astype(np.uint8)
    g = (yv * 255 * random.random()).astype(np.uint8)
    b = ((xv + yv) / 2 * 255 * random.random()).astype(np.uint8)
    
    pixels = np.stack([r, g, b], axis=-1)
    img = Image.fromarray(pixels, 'RGB')
    
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=85)
    buffer.seek(0)
    return buffer


class VisionScaleUser(HttpUser):
    """
    Simulated user for load testing VisionScale API.
    
    Behaviors:
    - Mostly prediction requests (80%)
    - Some health checks (10%)
    - Some version checks (5%)
    - Some explain requests (5%)
    """
    
    # Wait 1-3 seconds between requests (simulates real users)
    wait_time = between(1, 3)
    
    def on_start(self):
        """Called when user starts. Pre-generate some test images."""
        self.test_images = [
            generate_gradient_image(seed=i) for i in range(5)
        ]
    
    @task(80)  # 80% of requests
    def predict_image(self):
        """Submit an image for classification."""
        # Pick a random pre-generated image
        img_buffer = random.choice(self.test_images)
        img_buffer.seek(0)  # Reset buffer position
        
        files = {"file": ("test.jpg", img_buffer, "image/jpeg")}
        
        with self.client.post(
            "/predict?top_k=5",
            files=files,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "label" in data and "confidence" in data:
                    response.success()
                else:
                    response.failure("Invalid response format")
            elif response.status_code == 429:
                response.failure("Rate limited")
            else:
                response.failure(f"Status {response.status_code}")
    
    @task(10)  # 10% of requests
    def health_check(self):
        """Check API health."""
        with self.client.get("/health", catch_response=True) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("status") == "ok":
                    response.success()
                else:
                    response.failure("Unhealthy status")
            else:
                response.failure(f"Status {response.status_code}")
    
    @task(5)  # 5% of requests
    def version_check(self):
        """Check API version."""
        self.client.get("/version")
    
    @task(5)  # 5% of requests
    def explain_image(self):
        """Request Grad-CAM explanation."""
        img_buffer = random.choice(self.test_images)
        img_buffer.seek(0)
        
        files = {"file": ("test.jpg", img_buffer, "image/jpeg")}
        
        with self.client.post(
            "/explain",
            files=files,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if "heatmap_base64" in data:
                    response.success()
                else:
                    response.failure("No heatmap in response")
            else:
                response.failure(f"Status {response.status_code}")


class CacheTestUser(HttpUser):
    """
    User that tests cache effectiveness by sending identical images.
    """
    
    wait_time = between(0.5, 1)
    
    def on_start(self):
        """Generate a single image to reuse (tests cache hits)."""
        self.cached_image = generate_gradient_image(seed=42)
    
    @task
    def predict_cached_image(self):
        """Send the same image repeatedly to test caching."""
        self.cached_image.seek(0)
        files = {"file": ("cached.jpg", self.cached_image, "image/jpeg")}
        self.client.post("/predict?top_k=3", files=files)


# ─────────────────────────────────────────────────────────────
# Standalone Runner (without locust web UI)
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import subprocess
    import sys
    
    print("""
╔══════════════════════════════════════════════════════════════╗
║            VisionScale Load Testing                          ║
╠══════════════════════════════════════════════════════════════╣
║  Usage:                                                       ║
║                                                               ║
║  1. Web UI Mode (recommended):                                ║
║     locust -f tools/load_test.py --host=http://localhost:8000 ║
║     Then open http://localhost:8089                           ║
║                                                               ║
║  2. Headless Mode:                                            ║
║     locust -f tools/load_test.py --host=http://localhost:8000 ║
║            --headless -u 10 -r 2 -t 60s                       ║
║                                                               ║
║  Parameters:                                                  ║
║    -u : Number of concurrent users                            ║
║    -r : Spawn rate (users per second)                         ║
║    -t : Test duration (e.g., 60s, 5m)                         ║
║                                                               ║
║  For production testing against HuggingFace:                  ║
║     locust -f tools/load_test.py \\                            ║
║            --host=https://prakhyatc-visionscale-backend.hf.space ║
╚══════════════════════════════════════════════════════════════╝
    """)
