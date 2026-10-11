# VisionScale — Real-Time Image Classification as a Production-Ready ML Service

React • TypeScript • FastAPI • PyTorch • TorchScript • Hugging Face Spaces • Vercel • Docker • GitHub Actions

---

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE) ![Tech Stack](https://img.shields.io/badge/Stack-React%20|%20FastAPI%20|%20PyTorch-lightgrey)

## Table of Contents

- Executive Summary
- Problem Statement
- Non-Functional Requirements
- Real-World Use Cases
- Technical Solution Overview
- Architecture
- Data
  - Dataset Description
  - Data Preprocessing
- Model Design
  - Model Architecture
  - Training Strategy
- Evaluation & Metrics
- Key Engineering Decisions
- Performance & Results
- Deployment Considerations
- Repository Structure
- Getting Started
  - Prerequisites
  - Installation
  - Backend Setup
  - Frontend Setup
  - Training
  - Evaluation
  - Inference
- Results & Impact
- Roadmap
- Documentation
- License

---

## Executive Summary

- Problem: Many teams need a fast, reliable way to classify user-uploaded images (products, content, screenshots) without building an ML stack from scratch.  
- Why it matters: Accurate, low-latency image classification underpins search, recommendation, moderation, and analytics in consumer and enterprise products. Getting it wrong leads to poor UX, manual triage, or compliance risk.  
- Why this solution is non-trivial: VisionScale is not just a notebook model. It implements the full ML lifecycle: data → training → model versioning → TorchScript export → API service → web frontend → CI/CD → monitoring hooks, designed to be cloud-native and reproducible.  
- Who uses it:
  - Product teams needing a drop-in image classifier service
  - ML engineers prototyping and deploying models with minimal infra overhead
  - Startups and researchers demonstrating computer vision features in a realistic, production-like environment

---

## Quick Start with Docker

```bash
# Clone and start all services
git clone https://github.com/prakhyatc/visionscale.git
cd visionscale
docker-compose up -d

# Services available at:
# - Backend API:  http://localhost:8000
# - Frontend:     http://localhost:5173
# - Prometheus:   http://localhost:9090
# - Grafana:      http://localhost:3000 (admin/visionscale)
```

---

## Features

### Core ML Features
| Feature | Description |
|---------|-------------|
| **ImageNet-1K Classification** | 1000-class image classification using ResNet18 |
| **TorchScript Optimized** | CPU-optimized inference with <200ms p50 latency |
| **Grad-CAM Explainability** | Visualize what the model "sees" with heatmaps |
| **Batch Prediction** | Process up to 20 images in a single request |
| **Async Job Queue** | Submit large batches, poll for results |

### Production Features
| Feature | Description |
|---------|-------------|
| **API Key Authentication** | Token-based auth with usage tracking |
| **Rate Limiting** | Per-IP request limits (token bucket algorithm) |
| **Response Caching** | LRU cache with 1-hour TTL for identical images |
| **Prometheus Metrics** | Request rates, latency percentiles, error rates |
| **Grafana Dashboard** | Pre-built dashboard for monitoring |
| **Health Checks** | `/health` endpoint for load balancers |
| **OpenAPI/Swagger Docs** | Interactive API documentation at `/docs` |

### Frontend Features
| Feature | Description |
|---------|-------------|
| **Drag-and-Drop Upload** | Easy image upload interface |
| **Sample Image Gallery** | Test with pre-loaded sample images |
| **Dark/Light Mode** | Theme toggle with system preference detection |
| **Prediction History** | View past predictions in sidebar |
| **Grad-CAM Heatmaps** | Toggle heatmap overlay on images |
| **Export Results** | Download predictions as JSON |

### DevOps Features
| Feature | Description |
|---------|-------------|
| **Docker Compose** | One-command local development environment |
| **CI/CD Workflows** | GitHub Actions for automated deployment |
| **Load Testing** | Locust script for performance validation |
| **Model Quantization** | INT8 quantization script for optimization |

---

## Problem Statement

Teams often need to classify images (e.g., “is this a cat or dog?”, “what category is this product?”, “is this content safe?”) as part of a broader workflow. The challenge is not only achieving good accuracy, but doing so:

- Reliably (no flaky endpoints)
- At low latency (near real-time user feedback)
- With reproducible training and safe deployments
- Without requiring GPUs or heavyweight infra for small/medium workloads

Naive approaches (single-script Flask servers, unversioned models, ad-hoc training) typically fail because they:

- Mix training and inference logic, making deployment brittle.
- Don’t version models or track experiments, making regressions hard to diagnose.
- Ignore production concerns: cold starts, logging, monitoring, and CI/CD.
- Depend on local GPU-only workflows that don’t translate to CPU-only cloud hosting.

Constraints considered:

- Data quality: CIFAR-10 is used as a demonstration dataset, but the system is designed to support custom datasets with different class distributions and quality issues.
- Latency: Target <200 ms p50 inference latency for a single image on CPU-only hosts.
- Accuracy: Target >90% top-1 accuracy on the chosen dataset, with the option to trade off size vs. performance.
- Hardware: CPU-only deployment on Hugging Face Spaces (backend) and Vercel (frontend), with the ability to leverage GPUs during training (e.g., Google Colab Pro).
- Environment: Stateless containers, ephemeral filesystem, environment-based configuration.

---

## Non-Functional Requirements

- Performance
  - p50 inference latency < 200 ms on CPU-only backend for 224×224 images.
  - Handle bursts of ~10–20 concurrent requests without timeouts on default Space hardware.
- Scalability
  - Horizontally scalable via additional backend instances (Spaces, containers).
  - CDN-backed frontend (Vercel / CloudFront) for static assets.
- Reliability & Robustness
  - Health check endpoint (/health) and lightweight version endpoint (/version).
  - Graceful handling of invalid inputs (non-image files, corrupted content).
- Reproducibility
  - Training code in a dedicated ml/ module with pinned dependencies.
  - Model artifacts exported as TorchScript and versioned via MLflow/DVC + tags on Hugging Face Hub.
- Deployment Constraints
  - CPU-only runtime (no GPU assumptions during inference).
  - Docker-based backend image (Hugging Face Spaces: Docker SDK).
  - Frontend deployed via Vercel (static build + API integration).
  - Environment variables used for secrets and model configuration (no hard-coded paths).

---

## Real-World Use Cases

1. E-commerce Product Tagging
   - Context: Retail platform where sellers upload product photos.
   - Who benefits: Merchandising and search teams, who need consistent tags (“sneakers”, “kitchenware”, etc.) to power filters and recommendations.
   - Integration: VisionScale runs as a microservice; the product ingestion pipeline POSTs each new image to /predict, stores the predicted class, and uses it in search and recommendation features.

2. Content Moderation Pre-Filter
   - Context: Social platforms or forums where users upload images.
   - Who benefits: Trust & Safety teams and moderators.
   - Integration: VisionScale provides a classification endpoint; high-risk categories (e.g., “graphic content” in a custom model) are flagged for human review or blocked, while low-risk content flows through automatically.

3. Internal Knowledge Base / Screenshot Categorization
   - Context: Internal tools where engineers and support staff upload screenshots of dashboards, logs, or UI issues.
   - Who benefits: Support and SRE teams organizing incidents and feature requests.
   - Integration: Screenshots are automatically categorized (e.g., “billing issue”, “UI bug”, “infra dashboard”), enabling faster search and triage.

4. Prototype for Domain-Specific Models
   - Context: Teams planning a domain-specific vision model (e.g., medical imaging, manufacturing defects).
   - Who benefits: ML teams that want a working reference system to plug in their own dataset and model.
   - Integration: VisionScale serves as a template; only the training pipeline and labels change, while the deployment, API, and frontend remain largely unchanged.

5. Education & Internal ML Platform Demos
   - Context: Demonstrating ML platform capabilities or onboarding junior engineers.
   - Who benefits: ML platform teams and educators.
   - Integration: VisionScale showcases the full lifecycle: dataset → training → model registry → API → UI → monitoring, serving as a canonical example.

---

## Technical Solution Overview

At a high level:

1. User uploads an image via the React frontend.  
2. Frontend sends a multipart/form-data POST request to the FastAPI backend /predict endpoint.  
3. Backend:
   - Validates and decodes the image using Pillow.
   - Checks LRU cache for identical image (returns cached result if hit).
   - Applies the same transforms used during training (resize, crop, normalization).
   - Runs inference using a TorchScript-compiled ResNet18 model loaded from Hugging Face Hub or local cache.
   - Computes top-k probabilities and returns JSON (labels + confidences).
4. Frontend displays:
   - Predicted class.
   - Confidence score (with visual representation).
   - Grad-CAM heatmap visualization (optional).
   - Inference latency information.
5. API Endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | POST | Single image classification |
| `/predict/batch` | POST | Batch classification (up to 20 images) |
| `/explain` | POST | Grad-CAM heatmap generation |
| `/jobs/submit` | POST | Submit async batch job |
| `/jobs/{id}` | GET | Poll job status |
| `/jobs/{id}/results` | GET | Get completed job results |
| `/samples` | GET | Get sample images for testing |
| `/health` | GET | Health check |
| `/version` | GET | API and model version info |
| `/metrics` | GET | Prometheus metrics |
| `/cache/stats` | GET | Cache statistics |
| `/auth/keys` | GET/POST | API key management |
| `/docs` | GET | Swagger UI documentation |

---

## Architecture

The system is built as a small but complete ML platform:

- Data Ingestion
  - Training data pulled from public dataset sources (e.g., CIFAR-10) or a custom image directory.
  - Collected in ml/data/ or referenced from external storage.
- Preprocessing
  - Normalized transforms shared between training and inference (resize, center crop, normalization).
  - Implemented using torchvision.transforms.
- Model Pipeline
  - Fine-tuned ResNet18 (or EfficientNet) on the chosen dataset.
  - Exported to TorchScript for portable, efficient CPU inference.
  - Model weights and label map stored in a model repository (Hugging Face Hub) with semantic version tags.
- Inference Path
  - FastAPI app (Hugging Face Spaces Docker backend).
  - ModelService abstraction handles loading of TorchScript model and labels, and applying preprocessing.
  - Stateless inference; model loaded once per worker.
- Post-processing
  - Softmax probabilities.
  - Top-k selection.
  - Response JSON formatting with label and confidence.
- Evaluation Pipeline
  - Training scripts compute accuracy, confusion matrix, and per-class performance.
  - Results logged in MLflow/DVC and as text/JSON, optionally visualized in notebooks.

Diagram reference : 
**Diagram 1:** High-Level System Architecture (End-to-End)  
![High-Level System Architecture](docs/VisionScale%20System%20Architecure%20I.png)

**Diagram 2:** Training & Evaluation Pipeline (Offline)  
![Training & Evaluation Pipeline](docs/VisionScale%20System%20Architecture%20II.png)

**Diagram 3:** Inference & Runtime Flow (Request-Level)  
![Inference & Runtime Flow](docs/VisionScale%20System%20Architecture%20III.png)


---

## Data

### Dataset Description

- Source: CIFAR-10 (demo) and/or custom user-provided dataset.
- Size: 50,000 training images, 10,000 test images (for CIFAR-10).
- Labels: 10 classes, e.g., airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck.
- Characteristics:
  - 32×32 color images, low resolution.
  - Balanced class distribution across the 10 categories.
  - Diverse visual patterns but lacking fine-grained details due to small image size.
- Known Biases / Limitations:
  - CIFAR-10 is not representative of real-world product catalogues or safety-critical domains.
  - Images are small and artificially curated; performance will differ on high-resolution, noisy, or domain-specific data.
  - Demo metrics should be treated as prototype-level rather than production guarantees.

### Data Preprocessing

- Cleaning:
  - Validate that all training files are loadable images.
  - Drop or log corrupted files.
- Augmentations (training only):
  - Random horizontal flips.
  - Random crops and slight color jittering.
  - Optional Cutout/RandomErasing for regularization.
- Transforms (training + inference):
  - Resize to 256×256.
  - Center crop to 224×224.
  - Convert to tensor.
  - Normalize with ImageNet mean and std:
    - mean = (0.485, 0.456, 0.406)
    - std = (0.229, 0.224, 0.225)
- Why these steps:
  - Reuse standard ImageNet normalization to leverage pretrained weights effectively.
  - Augmentations improve generalization and reduce overfitting on small datasets.
  - Matching transforms between training and inference is critical for stable performance.

---

## Model Design

### Model Architecture

- Base: ResNet18 from torchvision.models (pretrained on ImageNet).
- Modifications:
  - Replace final fully connected (FC) layer with a new FC with output dimension equal to the number of classes (10 for CIFAR-10).
  - Optional dropout before final layer for additional regularization.
- Why this architecture:
  - ResNet18 is a well-understood backbone with a strong accuracy–speed trade-off.
  - Small enough for CPU-only deployment, yet expressive enough for non-trivial image tasks.
  - Easily replaceable with larger backbones (e.g., ResNet50, EfficientNet) if accuracy needs outweigh latency constraints.

### Training Strategy

- Loss: Cross-entropy loss for multi-class classification.
- Optimizer: SGD with momentum (0.9) or Adam; SGD preferred for more stable convergence when fine-tuning.
- Learning Rate Schedule:
  - StepLR or cosine annealing scheduler.
  - Typical config: base LR = 1e-3–1e-2, decay by factor of 10 every few epochs or cosine over the full training period.
- Regularization:
  - Weight decay (e.g., 1e-4).
  - Data augmentation as described above.
- Hardware:
  - Training: GPU-enabled environment (e.g., Google Colab Pro with T4/V100).
  - Inference: CPU-only on Hugging Face Spaces.
- Export:
  - After training, the model is converted to TorchScript (torch.jit.script / torch.jit.trace) and saved as model.torchscript.pt.
  - Label map (labels.txt) stored alongside the model in the model repository.

---

## Evaluation & Metrics

- Metrics Used:
  - Top-1 accuracy (primary metric for demo).
  - Top-5 accuracy (optional, useful for multi-label or ambiguous cases).
  - Confusion matrix to inspect per-class performance.
- Validation Strategy:
  - Train/validation split from training set (e.g., 45,000 train / 5,000 val).
  - Early stopping or LR scheduling based on validation accuracy.
- Representative Results (CIFAR-10 demo):
  - Top-1 accuracy: ~92–94% on test set.
  - Top-5 accuracy: ~99% on test set.
- Failure Cases & Limitations:
  - Confusion between visually similar classes (e.g., cat vs dog, truck vs automobile).
  - Sensitivity to strong image compression and unusual crops.
  - Limited robustness on out-of-distribution images (e.g., non-natural photos, line drawings).
  - Domain shift: performance will differ if used on real product images or medical scans without retraining.

---

## Key Engineering Decisions

1. TorchScript for Inference vs. Raw PyTorch
   - Chosen approach: Export fine-tuned PyTorch model to TorchScript and load it in the FastAPI backend.
   - Alternatives considered:
     - Load raw .pt checkpoints and rebuild the model architecture in code.
     - ONNX export + ONNX Runtime.
   - Trade-offs:
     - TorchScript keeps the PyTorch runtime while offering more controlled serialization and fewer code-path dependencies.
     - ONNX could provide better hardware portability, but adds another toolchain and requires careful ops coverage.
   - Mitigation:
     - Interface is designed so the inference wrapper can be swapped to ONNX or TensorRT if future performance requirements demand it.

2. CPU-only Inference vs. GPU-backed Serving
   - Chosen approach: Optimize for CPU-only inference on Hugging Face Spaces (free-tier, Docker Spaces).
   - Alternatives considered:
     - GPU-backed hosting (paid tier, Kubernetes cluster, or dedicated GPU instances).
   - Trade-offs:
     - CPU-only is cheaper and easier to maintain for moderate workloads, at the cost of higher per-request latency.
     - GPUs significantly reduce latency at higher monetary cost and with more complex infra.
   - Mitigation:
     - Use efficient backbone (ResNet18), minimal preprocessing overhead, and instance reuse to keep latency under the target threshold.
     - Architecture designed so the backend container can be deployed to GPU-backed infrastructure later without code changes.

3. React + FastAPI Separation vs. Monolithic Backend-Rendered UI
   - Chosen approach: Split into a React SPA frontend and a FastAPI backend.
   - Alternatives considered:
     - Monolithic MVC app (e.g., Django templates).
     - Single-page Streamlit/Gradio app combining UI + inference.
   - Trade-offs:
     - Separate frontend/backend increases complexity but mirrors real production systems (API-first).
     - Gradio/Streamlit are great for internal demos but are harder to skin, integrate, and scale in a multi-service environment.
   - Mitigation:
     - Use simple REST API between frontend and backend, making it easy to replace either side without rewriting the other.
     - Minimal but clean UI using TailwindCSS/Material UI, sufficient for demos and extensible for productization.

---

## Performance & Results

- Model Performance (CIFAR-10 demo):
  - Top-1 accuracy: ~92–94%.
  - Top-5 accuracy: ~99%.
- Inference Latency (CPU-only backend):
  - p50: ~150–200 ms per 224×224 image.
  - p95: ~300–400 ms under light concurrent load.
  - Measured using synthetic load tests (e.g., Locust) against the /predict endpoint.
- Model Size:
  - TorchScript model: ~40–50 MB (depending on backbone and quantization).
- Resource Usage:
  - Memory footprint: ~300–600 MB for FastAPI + model per container.
  - CPU usage: modest for single requests; scales linearly with traffic for synchronous inference.

---

## Deployment Considerations

- Production Deployment Flow:
  - Training pipeline produces a new TorchScript model and labels.txt.
  - Model and metadata pushed to a Hugging Face model repository with a semantic version/tag (e.g., v1, v2).
  - Backend reads MODEL_REPO, MODEL_REV, MODEL_FILE, LABELS_FILE from environment variables and loads the correct model at startup via huggingface_hub.
  - Frontend is built and deployed to Vercel, configured with NEXT_PUBLIC_API_URL (or similar) pointing to the backend’s public URL.
- Batch vs. Real-time Inference:
  - Primary mode is real-time per-image inference via the /predict endpoint.
  - Batch inference can be supported via a separate script or endpoint that processes image lists from storage.
- Hardware Requirements:
  - Backend: CPU-only container on Hugging Face Spaces (Docker Space), with at least 1–2 vCPUs and 1–2 GB RAM.
  - Frontend: Vercel static hosting (no special hardware).
- Monitoring & Retraining:
  - /metrics endpoint exposes Prometheus-compatible metrics (request count, latency, error rates).
  - Logs (predictions, errors, model version) can be shipped to a log aggregator.
  - Monitoring dashboards (Prometheus/Grafana) track:
    - Request rate.
    - Latency percentiles.
    - Error rates.
    - Model version in production.
  - Retraining triggered when:
    - Performance drops on key segments (detected via monitoring/evaluations).
    - New labeled data is available in production.

---

## Repository Structure

```
The repository is organized as a monorepo-style project with separate apps for backend and frontend, 
ML scripts, monitoring configs, and CI/CD workflows.
.
├── apps
│   ├── backend
│   │   ├── app
│   │   │   ├── main.py             # FastAPI entrypoint with all endpoints
│   │   │   ├── model_service.py    # TorchScript model loading & inference
│   │   │   ├── gradcam.py          # Grad-CAM explainability implementation
│   │   │   ├── cache.py            # LRU prediction cache
│   │   │   ├── rate_limit.py       # Token bucket rate limiter
│   │   │   ├── jobs.py             # Async job queue for batch processing
│   │   │   ├── auth.py             # API key authentication
│   │   │   └── metrics.py          # Prometheus metrics
│   │   ├── tests
│   │   │   ├── test_health.py      # Health endpoint tests
│   │   │   ├── test_predict.py     # Prediction endpoint tests
│   │   │   ├── test_batch.py       # Batch & async job tests
│   │   │   └── test_cache.py       # Cache module tests
│   │   ├── Dockerfile              # Docker image for Hugging Face Spaces
│   │   ├── requirements.txt        # Backend Python dependencies
│   │   └── pytest.ini              # Pytest configuration
│   └── frontend
│       ├── src
│       │   ├── App.tsx             # Root React component (full UI)
│       │   └── index.css           # Tailwind styles & animations
│       ├── Dockerfile.dev          # Development container
│       ├── package.json
│       └── tailwind.config.js
├── ml
│   ├── src
│   │   ├── train_cifar.py          # Training script with ImageNet normalization
│   │   ├── dataset.py              # Dataset & transform utilities
│   │   ├── export_imagenet.py      # ImageNet model export script
│   │   └── quantize_model.py       # INT8 quantization script
│   └── configs
│       └── cifar10_resnet18.yaml   # Training hyperparameters
├── monitoring
│   ├── grafana
│   │   ├── dashboard.json          # Pre-built Grafana dashboard
│   │   └── provisioning/           # Auto-provisioning configs
│   └── prometheus
│       └── prometheus.yml          # Prometheus scrape config
├── tools
│   ├── load_test.py                # Locust load testing script
│   └── requirements.txt            # Dev tool dependencies
├── .github
│   └── workflows
│       ├── deploy-backend.yml      # CI/CD to Hugging Face Spaces
│       └── deploy-frontend.yml     # CI/CD to Vercel
├── docs
│   ├── model-card.md               # Model documentation
│   └── *.png                       # Architecture diagrams
├── docker-compose.yml              # One-command local dev environment
├── README.md                       # This file
└── LICENSE


⸻

## Getting Started

### Prerequisites

- Backend:
  - Python 3.11+
  - pip / venv
- Frontend:
  - Node.js 18+ (or compatible LTS)
  - npm or yarn or pnpm
- Training (optional):
  - GPU-enabled environment (e.g., Google Colab Pro) recommended.
- Accounts:
  - Hugging Face account (for model hosting).
  - Vercel account (for frontend hosting), if deploying.

### Installation

Clone the repository:

```bash
git clone https://github.com/<your-username>/VisionScale.git
cd VisionScale
```

Backend Setup

```bash
cd apps/backend
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you plan to run inference with the existing TorchScript model from Hugging Face Hub, set the environment variables:

```bash
export MODEL_REPO="your-username/VisionScale-cifar10"
export MODEL_REV="v1"
export MODEL_FILE="model.torchscript.pt"
export LABELS_FILE="labels.txt"
```

Frontend Setup

```bash
cd apps/frontend
npm install            # or: yarn / pnpm
```

Training

From the repo root or ml/:

```bash
cd ml
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Example CIFAR-10 training
python src/train_cifar.py \
  --config configs/cifar10_resnet18.yaml \
  --epochs 20 \
  --out out/cifar10_resnet18
```

Export to TorchScript and write labels:

```bash
python src/export_torchscript.py \
  --checkpoint out/cifar10_resnet18/best.pth \
  --labels data/labels_cifar10.txt \
  --out out/artifacts/model.torchscript.pt
```

Upload the resulting model.torchscript.pt and labels.txt to your Hugging Face model repo (manually or using huggingface_hub CLI).

### Evaluation

Evaluation can be done via the training script or notebook:

```bash
python src/train_cifar.py \
  --config configs/cifar10_resnet18.yaml \
  --evaluate-only \
  --checkpoint out/cifar10_resnet18/best.pth
```

This computes accuracy and optionally dumps confusion matrix and per-class stats.

### Inference

Run Backend Locally

```bash
cd apps/backend
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Test the API:

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@/path/to/image.jpg"
```

Run Frontend Locally

```bash
cd apps/frontend
npm run dev   # or yarn dev / pnpm dev
```

Configure the frontend (e.g., .env.local):

```
VITE_API_URL=http://localhost:8000
```

Open the app in the browser at http://localhost:5173 (or the port shown in the dev server) and upload an image.

### Running Tests

```bash
cd apps/backend
source .venv/bin/activate
pip install pytest httpx
pytest -v
```

Test coverage includes:
- Health and info endpoints
- Single and batch prediction
- Async job submission and polling
- Cache operations
- Error handling for invalid inputs

### Load Testing

```bash
pip install locust
locust -f tools/load_test.py --host=http://localhost:8000
# Open http://localhost:8089 to control the test
```

---

## Results & Impact

- Ready-to-deploy image classification service that abstracts training → export → serving.
- Reference implementation for model versioning, API-first architecture, CI/CD, health checks and monitoring.
- Enables fast iteration: swap datasets/backbones, run A/B tests, and roll back via versioned artifacts.
- Promotes MLE maturity: reproducible experiments, automated deployments, observable production behavior.

---

## Roadmap

### Completed
- ✅ Full model card (evaluation, intended use, limitations)
- ✅ UX improvements (drag‑and‑drop, prediction history)
- ✅ Dark/light mode theme toggle
- ✅ Grad-CAM explainability visualization
- ✅ Batch prediction API
- ✅ Async job queue for large batches
- ✅ Response caching (LRU)
- ✅ Rate limiting
- ✅ API key authentication
- ✅ Prometheus metrics & Grafana dashboard
- ✅ Docker Compose for local development
- ✅ Load testing with Locust
- ✅ Comprehensive test suite

### In Progress
- 🔄 Redis-backed distributed caching
- 🔄 OpenTelemetry distributed tracing

### Planned
- Multi-model A/B testing and routing
- Webhook notifications for async jobs
- ONNX export for broader hardware support
- Kubernetes manifests (HPA, PDB, ConfigMaps)
- Model drift detection
- Automated retraining pipelines

---

## Documentation

- Model card: docs/model-card.md — intended use, training data, evaluation, limitations.
- Experiment tracking: MLflow / DVC for runs, metrics, artifacts.
- Reproducibility:
  - Hyperparameters in ml/configs/*.yaml
  - Dependencies pinned in requirements.txt / pyproject.toml
  - Training scripts runnable end‑to‑end with minimal manual steps

---

## License

This project is licensed under the MIT License. See the LICENSE file for details.