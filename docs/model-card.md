# Model Card: VisionScale CIFAR-10 Classifier

## Model Details

| Property | Value |
|----------|-------|
| **Model Name** | VisionScale CIFAR-10 ResNet18 |
| **Version** | v1.0.0 |
| **Architecture** | ResNet18 (torchvision) |
| **Framework** | PyTorch 2.2+ / TorchScript |
| **License** | MIT |
| **Maintainer** | [Your Name] |
| **Last Updated** | June 2025 |

### Description

A fine-tuned ResNet18 model for 10-class image classification, exported to TorchScript for CPU-optimized inference. The model serves as the backbone for the VisionScale production ML inference service.

---

## Intended Use

### Primary Use Cases

- **Image Classification API**: Real-time classification of user-uploaded images via REST endpoint
- **Content Categorization**: Automated tagging for products, screenshots, or user content
- **ML Platform Demo**: Reference implementation for model versioning, deployment, and monitoring

### Users

- Product teams needing drop-in image classification
- ML engineers prototyping vision models
- Educators demonstrating ML lifecycle concepts

### Out-of-Scope Uses

- Medical diagnosis or safety-critical decisions
- Fine-grained classification (e.g., dog breeds, car models)
- Images significantly different from natural photos (documents, x-rays, satellite imagery)

---

## Training Data

### Dataset

| Property | Value |
|----------|-------|
| **Name** | CIFAR-10 |
| **Source** | [University of Toronto](https://www.cs.toronto.edu/~kriz/cifar.html) |
| **Training Samples** | 50,000 |
| **Test Samples** | 10,000 |
| **Image Size** | 32×32 (upscaled to 224×224) |
| **Classes** | 10 |

### Classes

```
airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
```

### Data Preprocessing

**Training Transforms:**
- Random resized crop (224×224, scale 0.7–1.0)
- Random horizontal flip
- Normalize (ImageNet mean/std)

**Inference Transforms:**
- Resize to 256×256
- Center crop to 224×224
- Normalize with ImageNet statistics:
  - Mean: `(0.485, 0.456, 0.406)`
  - Std: `(0.229, 0.224, 0.225)`

> **Important**: Training and inference transforms use identical normalization to prevent accuracy drift.

---

## Training Procedure

### Hyperparameters

| Parameter | Value |
|-----------|-------|
| Optimizer | AdamW |
| Learning Rate | 1e-3 |
| Weight Decay | 0.01 |
| Scheduler | Cosine Annealing |
| Batch Size | 64 |
| Epochs | 20 |
| Pretrained | ImageNet weights |

### Hardware

- **Training**: GPU (CUDA / MPS / Colab T4)
- **Inference**: CPU-only (Hugging Face Spaces)

### Export

Model exported via `torch.jit.script()` for portable, dependency-free inference.

---

## Evaluation

### Metrics

| Metric | Value |
|--------|-------|
| **Top-1 Accuracy** | ~92–94% |
| **Top-5 Accuracy** | ~99% |
| **p50 Latency (CPU)** | <200ms |
| **p95 Latency (CPU)** | ~300–400ms |

### Confusion Matrix Observations

Common misclassifications:
- `cat` ↔ `dog` (visually similar animals)
- `automobile` ↔ `truck` (similar vehicle shapes)
- `deer` ↔ `horse` (four-legged animals)

### Limitations

1. **Low-resolution training data**: CIFAR-10 images are 32×32; model may struggle with fine details
2. **Limited classes**: Only 10 categories; out-of-distribution images return arbitrary predictions
3. **Domain shift**: Performance degrades on non-natural images (drawings, renders, documents)
4. **Class imbalance handling**: Not evaluated on imbalanced datasets
5. **Adversarial robustness**: No adversarial training or evaluation performed

---

## Ethical Considerations

### Bias & Fairness

- CIFAR-10 contains limited diversity in image sources and contexts
- No demographic or geographic diversity analysis performed
- Model should not be used for decisions affecting individuals

### Privacy

- Model trained on public dataset with no PII
- Inference API does not store uploaded images beyond request lifecycle

### Environmental Impact

- Training: ~1 hour on consumer GPU (minimal carbon footprint)
- Inference: CPU-only, ~0.5W per request

---

## Deployment

### Hosting

| Component | Platform |
|-----------|----------|
| Backend API | Hugging Face Spaces (Docker) |
| Frontend | Vercel |
| Model Registry | Hugging Face Hub |

### Model Versioning

Models are tagged with semantic versions on Hugging Face Hub:
- `v1` — Initial CIFAR-10 release
- `main` — Latest development version

### Environment Variables

```bash
MODEL_REPO=prakhyatc/visionscale-cifar10
MODEL_REV=v1
MODEL_FILE=model.torchscript.pt
LABELS_FILE=labels.txt
```

---

## Monitoring

### Prometheus Metrics

| Metric | Type | Description |
|--------|------|-------------|
| `VisionScale_requests_total` | Counter | Total HTTP requests by route/method/status |
| `VisionScale_errors_total` | Counter | Error responses by route/status |
| `VisionScale_request_duration_ms` | Histogram | End-to-end request latency |
| `VisionScale_inference_duration_ms` | Histogram | Model inference time only |

### Health Endpoints

- `GET /health` — Liveness check
- `GET /version` — API and model version info
- `GET /metrics` — Prometheus-format metrics

---

## Citation

If you use this model, please cite:

```bibtex
@misc{visionscale2025,
  title={VisionScale: Production ML Inference Service},
  year={2025},
  url={https://github.com/prakhyatc/visionscale}
}
```

---

## Changelog

| Version | Date | Changes |
|---------|------|---------|
| v1.2.0 | Feb 2026 | Add batch API, async jobs, API key auth, tests |
| v1.1.0 | Feb 2026 | Add caching, rate limiting, Grafana dashboard |
| v1.0.0 | June 2025 | Initial release with ImageNet ResNet18 |

---

## Contact

For questions or issues, please open a GitHub issue.
