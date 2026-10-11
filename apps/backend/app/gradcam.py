# apps/backend/app/gradcam.py
"""
Grad-CAM (Gradient-weighted Class Activation Mapping) implementation.
Generates visual explanations for CNN predictions by highlighting
important regions in the input image.
"""

from __future__ import annotations
import base64
import io
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

# ImageNet normalization
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


class GradCAM:
    """
    Grad-CAM implementation for ResNet models.
    Computes class activation maps by using gradients flowing into
    the final convolutional layer.
    """
    
    def __init__(self):
        # Load regular PyTorch model (not TorchScript) for gradient computation
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.model.eval()
        
        # Storage for activations and gradients
        self.activations = None
        self.gradients = None
        
        # Register hooks on the last conv layer (layer4)
        self.model.layer4.register_forward_hook(self._save_activation)
        self.model.layer4.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """Hook to save forward activations."""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward gradients."""
        self.gradients = grad_output[0].detach()
    
    def generate(
        self,
        img: Image.Image,
        target_class: int | None = None,
    ) -> Tuple[np.ndarray, int, float]:
        """
        Generate Grad-CAM heatmap for an image.
        
        Args:
            img: PIL Image to analyze
            target_class: Class index to visualize (None = predicted class)
        
        Returns:
            Tuple of (heatmap array 0-255, predicted class, confidence)
        """
        # Preprocess image
        x = PREPROCESS(img).unsqueeze(0)
        x.requires_grad_(True)
        
        # Forward pass
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)
        
        # Get target class (predicted if not specified)
        if target_class is None:
            target_class = logits.argmax(dim=1).item()
        
        confidence = probs[0, target_class].item()
        
        # Backward pass for target class
        self.model.zero_grad()
        logits[0, target_class].backward()
        
        # Compute Grad-CAM
        # Global average pooling of gradients
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        
        # Weighted combination of activation maps
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        
        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam.squeeze().numpy()
        
        # Normalize to 0-1
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
        
        # Resize to original image size
        cam_resized = np.array(
            Image.fromarray((cam * 255).astype(np.uint8)).resize(
                (224, 224), Image.BILINEAR
            )
        )
        
        return cam_resized, target_class, confidence
    
    def generate_overlay(
        self,
        img: Image.Image,
        target_class: int | None = None,
        alpha: float = 0.5,
    ) -> Tuple[str, int, float]:
        """
        Generate Grad-CAM heatmap overlaid on the original image.
        
        Args:
            img: PIL Image to analyze
            target_class: Class index to visualize
            alpha: Overlay transparency (0-1)
        
        Returns:
            Tuple of (base64 encoded overlay image, predicted class, confidence)
        """
        # Generate heatmap
        heatmap, pred_class, confidence = self.generate(img, target_class)
        
        # Resize original image to 224x224
        img_resized = img.resize((224, 224)).convert("RGB")
        img_array = np.array(img_resized)
        
        # Create colored heatmap (blue -> green -> red)
        heatmap_colored = self._apply_colormap(heatmap)
        
        # Overlay heatmap on image
        overlay = (1 - alpha) * img_array + alpha * heatmap_colored
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        # Convert to base64
        overlay_img = Image.fromarray(overlay)
        buffer = io.BytesIO()
        overlay_img.save(buffer, format="PNG")
        b64_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        
        return b64_str, pred_class, confidence
    
    def _apply_colormap(self, heatmap: np.ndarray) -> np.ndarray:
        """Apply jet-like colormap to grayscale heatmap."""
        # Normalize to 0-1
        heatmap = heatmap.astype(np.float32) / 255.0
        
        # Create RGB channels using jet-like colormap
        r = np.clip(1.5 - np.abs(4.0 * heatmap - 3.0), 0, 1)
        g = np.clip(1.5 - np.abs(4.0 * heatmap - 2.0), 0, 1)
        b = np.clip(1.5 - np.abs(4.0 * heatmap - 1.0), 0, 1)
        
        # Stack and convert to uint8
        colored = np.stack([r, g, b], axis=-1)
        return (colored * 255).astype(np.uint8)


# Singleton instance
_gradcam: GradCAM | None = None

def get_gradcam() -> GradCAM:
    """Get or create singleton GradCAM instance."""
    global _gradcam
    if _gradcam is None:
        _gradcam = GradCAM()
    return _gradcam
