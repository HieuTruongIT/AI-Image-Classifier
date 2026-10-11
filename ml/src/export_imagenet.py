# ml/src/export_imagenet.py
"""
Export ImageNet pretrained ResNet18 to TorchScript for production inference.
No training required — uses torchvision pretrained weights (1000 classes).
"""

import os
import torch
from torchvision import models

def main():
    print("Loading pretrained ResNet18 (ImageNet weights)...")
    
    # Load pretrained model with ImageNet weights
    weights = models.ResNet18_Weights.IMAGENET1K_V1
    model = models.resnet18(weights=weights)
    model.eval()
    
    print(f"Model loaded. Classes: {len(weights.meta['categories'])}")
    
    # Export to TorchScript
    print("Exporting to TorchScript...")
    scripted = torch.jit.script(model)
    
    # Create output directory
    out_dir = "apps/backend/app/models"
    os.makedirs(out_dir, exist_ok=True)
    
    # Save TorchScript model
    model_path = f"{out_dir}/model.torchscript.pt"
    scripted.save(model_path)
    print(f"Saved TorchScript model: {model_path}")
    
    # Save labels (ImageNet 1000 classes)
    labels_path = f"{out_dir}/labels.txt"
    categories = weights.meta["categories"]
    with open(labels_path, "w") as f:
        for cat in categories:
            f.write(f"{cat}\n")
    print(f"Saved labels ({len(categories)} classes): {labels_path}")
    
    # Verify model size
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"Model size: {size_mb:.1f} MB")
    
    # Quick sanity check
    print("\nRunning sanity check...")
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        output = scripted(dummy_input)
    print(f"Output shape: {output.shape} (expected: [1, 1000])")
    print("Export complete!")

if __name__ == "__main__":
    main()
