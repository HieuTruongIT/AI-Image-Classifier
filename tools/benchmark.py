import io
import random

from locust import HttpUser, task, between
from PIL import Image
import numpy as np


def generate_image(seed):
    rng = np.random.default_rng(seed)
    pixels = rng.integers(
        0, 255,
        (224, 224, 3),
        dtype=np.uint8
    )

    img = Image.fromarray(pixels, "RGB")

    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    buffer.seek(0)

    return buffer


class BenchmarkUser(HttpUser):
    wait_time = between(1, 2)

    def on_start(self):
        self.images = [
            generate_image(i)
            for i in range(100)
        ]

    @task
    def predict(self):
        img = random.choice(self.images)
        img.seek(0)

        files = {
            "file": ("test.jpg", img, "image/jpeg")
        }

        with self.client.post(
            "/predict?top_k=5",
            files=files,
            catch_response=True,
        ) as response:

            if response.status_code == 200:
                response.success()
            else:
                response.failure(
                    f"HTTP {response.status_code}"
                )

