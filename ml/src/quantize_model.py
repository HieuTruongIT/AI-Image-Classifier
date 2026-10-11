# ml/src/quantize_model.py
"""
Model Optimization for ResNet18.

This script provides multiple optimization strategies:
1. TorchScript optimization (works everywhere)
2. INT8 Dynamic Quantization (requires supported backend)

TorchScript optimization typically provides 1.2-1.5x speedup on CPU.
INT8 quantization can provide 2-3x speedup but requires QNNPACK or FBGEMM backend.
"""

import os
import sys
import time
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

# ImageNet normalization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def load_model():
    """Load pretrained ResNet18."""
    print("Loading pretrained ResNet18...")
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    return model


def try_quantize_dynamic(model):
    """
    Try INT8 dynamic quantization.
    
    Returns (quantized_model, success).
    May fail on platforms without quantization support (e.g., Mac ARM).
    """
    print("Attempting INT8 dynamic quantization...")
    
    try:
        # Try to set QNNPACK backend (works on some ARM devices)
        if hasattr(torch.backends, 'quantized'):
            torch.backends.quantized.engine = 'qnnpack'
    except Exception:
        pass
    
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            quantized = torch.quantization.quantize_dynamic(
                model,
                {torch.nn.Linear},  # Only quantize Linear layers (more portable)
                dtype=torch.qint8
            )
        return quantized, True
    except RuntimeError as e:
        print(f"Quantization not supported on this platform: {e}")
        return model, False


def optimize_torchscript(model):
    """
    Create an optimized TorchScript model.
    
    TorchScript optimization includes:
    - Graph optimization
    - Operator fusion
    - Dead code elimination
    """
    print("Creating optimized TorchScript model...")
    
    # Create sample input for tracing
    sample_input = torch.randn(1, 3, 224, 224)
    
    # Trace the model (better optimization than scripting for ResNet)
    with torch.inference_mode():
        traced = torch.jit.trace(model, sample_input)
    
    # Optimize the traced model
    traced = torch.jit.optimize_for_inference(traced)
    
    return traced


def benchmark_model(model, name, num_runs=100, warmup=20):
    """Benchmark model inference time."""
    # Create dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    
    # Warmup
    print(f"Warming up {name}...")
    with torch.inference_mode():
        for _ in range(warmup):
            _ = model(dummy_input)
    
    # Benchmark
    print(f"Benchmarking {name} ({num_runs} runs)...")
    times = []
    with torch.inference_mode():
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = model(dummy_input)
            end = time.perf_counter()
            times.append((end - start) * 1000)  # ms
    
    times = np.array(times)
    return {
        "mean_ms": times.mean(),
        "std_ms": times.std(),
        "p50_ms": np.percentile(times, 50),
        "p95_ms": np.percentile(times, 95),
        "p99_ms": np.percentile(times, 99),
    }


def get_model_size_mb(model_path):
    """Get model file size in MB."""
    if os.path.exists(model_path):
        return os.path.getsize(model_path) / (1024 * 1024)
    return 0


def verify_accuracy(original, optimized, num_tests=10):
    """Verify optimized model produces correct outputs."""
    matches = 0
    confidence_diffs = []
    
    for _ in range(num_tests):
        x = torch.randn(1, 3, 224, 224)
        
        with torch.inference_mode():
            orig_out = original(x)
            opt_out = optimized(x)
        
        orig_pred = orig_out.argmax(dim=1).item()
        opt_pred = opt_out.argmax(dim=1).item()
        
        if orig_pred == opt_pred:
            matches += 1
        
        orig_conf = torch.softmax(orig_out, dim=1).max().item()
        opt_conf = torch.softmax(opt_out, dim=1).max().item()
        confidence_diffs.append(abs(orig_conf - opt_conf))
    
    return {
        "accuracy_match": matches / num_tests,
        "mean_confidence_diff": np.mean(confidence_diffs),
        "max_confidence_diff": np.max(confidence_diffs),
    }


def main():
    print("=" * 60)
    print("ResNet18 Model Optimization")
    print("=" * 60)
    
    # Load model
    original_model = load_model()
    
    # Create output directory
    out_dir = "apps/backend/app/models"
    os.makedirs(out_dir, exist_ok=True)
    
    # 1) Create optimized TorchScript model
    optimized_model = optimize_torchscript(original_model)
    
    # Save optimized TorchScript model
    ts_path = f"{out_dir}/model.torchscript.pt"
    optimized_model.save(ts_path)
    print(f"Saved TorchScript model: {ts_path}")
    
    # 2) Try INT8 quantization (may not work on all platforms)
    quantized_model, quantized_success = try_quantize_dynamic(original_model)
    
    if quantized_success:
        # Save quantized model using torch.jit.script
        try:
            scripted_quantized = torch.jit.script(quantized_model)
            quant_path = f"{out_dir}/model.quantized.pt"
            scripted_quantized.save(quant_path)
            print(f"Saved quantized model: {quant_path}")
        except Exception as e:
            print(f"Could not save quantized model: {e}")
            quantized_success = False
    
    # Benchmark
    print("\n--- Latency Benchmark ---")
    
    orig_bench = benchmark_model(original_model, "Original PyTorch")
    ts_bench = benchmark_model(optimized_model, "Optimized TorchScript")
    
    print(f"\nOriginal PyTorch:")
    print(f"  Mean: {orig_bench['mean_ms']:.2f}ms, P50: {orig_bench['p50_ms']:.2f}ms, P95: {orig_bench['p95_ms']:.2f}ms")
    
    print(f"\nOptimized TorchScript:")
    print(f"  Mean: {ts_bench['mean_ms']:.2f}ms, P50: {ts_bench['p50_ms']:.2f}ms, P95: {ts_bench['p95_ms']:.2f}ms")
    
    ts_speedup = orig_bench['mean_ms'] / ts_bench['mean_ms']
    print(f"\nTorchScript Speedup: {ts_speedup:.2f}x")
    
    if quantized_success:
        quant_bench = benchmark_model(quantized_model, "INT8 Quantized")
        print(f"\nINT8 Quantized:")
        print(f"  Mean: {quant_bench['mean_ms']:.2f}ms, P50: {quant_bench['p50_ms']:.2f}ms, P95: {quant_bench['p95_ms']:.2f}ms")
        quant_speedup = orig_bench['mean_ms'] / quant_bench['mean_ms']
        print(f"\nQuantized Speedup: {quant_speedup:.2f}x")
    
    # Verify accuracy
    print("\n--- Accuracy Verification ---")
    ts_accuracy = verify_accuracy(original_model, optimized_model)
    print(f"TorchScript - Predictions match: {ts_accuracy['accuracy_match']*100:.0f}%")
    print(f"TorchScript - Max confidence diff: {ts_accuracy['max_confidence_diff']:.6f}")
    
    # Model sizes
    print("\n--- Model Sizes ---")
    ts_size = get_model_size_mb(ts_path)
    print(f"TorchScript model: {ts_size:.2f} MB")
    
    if quantized_success and os.path.exists(f"{out_dir}/model.quantized.pt"):
        quant_size = get_model_size_mb(f"{out_dir}/model.quantized.pt")
        print(f"Quantized model:   {quant_size:.2f} MB")
    
    print("\n" + "=" * 60)
    print("Optimization Complete!")
    print(f"TorchScript speedup: {ts_speedup:.2f}x")
    if quantized_success:
        print(f"Quantization: Available (run with USE_QUANTIZED=true)")
    else:
        print(f"Quantization: Not available on this platform")
    print("=" * 60)
    
    return ts_speedup


if __name__ == "__main__":
    main()
