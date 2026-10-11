# tests/conftest.py
"""
Pytest fixtures for VisionScale backend tests.
"""

import io
import pytest
from PIL import Image
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    """Create a test client for the FastAPI app."""
    from app.main import app
    with TestClient(app) as c:
        yield c


@pytest.fixture
def sample_image_bytes():
    """Generate a sample RGB image as bytes."""
    img = Image.new("RGB", (224, 224), color=(128, 64, 32))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer.getvalue()


@pytest.fixture
def sample_image_file(sample_image_bytes):
    """Return a file-like object for upload testing."""
    return ("test.jpg", io.BytesIO(sample_image_bytes), "image/jpeg")


@pytest.fixture
def invalid_file():
    """Return an invalid (non-image) file for testing."""
    return ("test.txt", io.BytesIO(b"not an image"), "text/plain")


@pytest.fixture
def corrupted_image():
    """Return corrupted image bytes."""
    return ("corrupt.jpg", io.BytesIO(b"corrupted image data"), "image/jpeg")


@pytest.fixture
def large_image_bytes():
    """Generate a larger image for testing."""
    img = Image.new("RGB", (1024, 1024), color=(255, 0, 0))
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=95)
    buffer.seek(0)
    return buffer.getvalue()
