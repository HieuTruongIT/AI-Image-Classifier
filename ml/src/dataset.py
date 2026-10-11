# ml/src/dataset.py
"""
Dataset utilities for VisionScale training pipeline.

Provides consistent data loading, transforms, and validation across
training, evaluation, and inference. Ensures transform parity with
the production model service (apps/backend/app/model_service.py).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Literal, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

# ─────────────────────────────────────────────────────────────
# Constants - MUST MATCH apps/backend/app/model_service.py
# ─────────────────────────────────────────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


# ─────────────────────────────────────────────────────────────
# Transform Factories
# ─────────────────────────────────────────────────────────────
def get_train_transforms(
    crop_size: int = 224,
    crop_scale: Tuple[float, float] = (0.7, 1.0),
    horizontal_flip: bool = True,
    color_jitter: bool = False,
    random_erasing: bool = False,
) -> transforms.Compose:
    """
    Training transforms with augmentation.
    
    Args:
        crop_size: Output image size after cropping
        crop_scale: Scale range for random resized crop
        horizontal_flip: Enable random horizontal flip
        color_jitter: Enable color jitter augmentation
        random_erasing: Enable random erasing (cutout)
    
    Returns:
        Composed transform pipeline for training
    """
    transform_list = [
        transforms.RandomResizedCrop(crop_size, scale=crop_scale),
    ]
    
    if horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    
    if color_jitter:
        transform_list.append(
            transforms.ColorJitter(
                brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
            )
        )
    
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    
    if random_erasing:
        transform_list.append(transforms.RandomErasing(p=0.25))
    
    return transforms.Compose(transform_list)


def get_eval_transforms(
    resize_size: int = 256,
    crop_size: int = 224,
) -> transforms.Compose:
    """
    Evaluation/inference transforms (no augmentation).
    
    IMPORTANT: These transforms MUST match the inference transforms
    in apps/backend/app/model_service.py to prevent accuracy drift.
    
    Args:
        resize_size: Size to resize shorter edge to
        crop_size: Center crop size
    
    Returns:
        Composed transform pipeline for evaluation
    """
    return transforms.Compose([
        transforms.Resize(resize_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ─────────────────────────────────────────────────────────────
# Dataset Loaders
# ─────────────────────────────────────────────────────────────
def get_cifar10_loaders(
    data_dir: str | Path = "./ml/data",
    batch_size: int = 64,
    num_workers: int = 4,
    pin_memory: bool = True,
    val_split: float = 0.0,
    train_transforms: transforms.Compose | None = None,
    eval_transforms: transforms.Compose | None = None,
) -> Tuple[DataLoader, DataLoader, DataLoader | None]:
    """
    Create CIFAR-10 data loaders for training, testing, and optional validation.
    
    Args:
        data_dir: Directory to download/load CIFAR-10 data
        batch_size: Batch size for all loaders
        num_workers: Number of data loading workers
        pin_memory: Pin memory for faster GPU transfer
        val_split: Fraction of training data for validation (0.0 = no validation set)
        train_transforms: Custom training transforms (uses default if None)
        eval_transforms: Custom eval transforms (uses default if None)
    
    Returns:
        Tuple of (train_loader, test_loader, val_loader or None)
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    
    train_tf = train_transforms or get_train_transforms()
    eval_tf = eval_transforms or get_eval_transforms()
    
    # Load datasets
    train_dataset = datasets.CIFAR10(
        root=str(data_dir), train=True, download=True, transform=train_tf
    )
    test_dataset = datasets.CIFAR10(
        root=str(data_dir), train=False, download=True, transform=eval_tf
    )
    
    # Optional validation split
    val_loader = None
    if val_split > 0:
        val_size = int(len(train_dataset) * val_split)
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(
            train_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        # Apply eval transforms to validation set
        val_dataset.dataset = datasets.CIFAR10(
            root=str(data_dir), train=True, download=False, transform=eval_tf
        )
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory
        )
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    
    return train_loader, test_loader, val_loader


# ─────────────────────────────────────────────────────────────
# Custom Dataset for Local Images
# ─────────────────────────────────────────────────────────────
class ImageFolderDataset(Dataset):
    """
    Custom dataset for loading images from a directory structure.
    
    Expected structure:
        data_dir/
            class1/
                img1.jpg
                img2.jpg
            class2/
                img3.jpg
                ...
    
    Args:
        data_dir: Root directory containing class folders
        transform: Transform to apply to images
        extensions: Allowed file extensions
    """
    
    SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
    
    def __init__(
        self,
        data_dir: str | Path,
        transform: Callable | None = None,
        extensions: set[str] | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.transform = transform or get_eval_transforms()
        self.extensions = extensions or self.SUPPORTED_EXTENSIONS
        
        self.classes = sorted([
            d.name for d in self.data_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = self._find_samples()
        
        if len(self.samples) == 0:
            raise ValueError(f"No images found in {data_dir}")
    
    def _find_samples(self) -> list[Tuple[Path, int]]:
        """Find all valid image files and their class indices."""
        samples = []
        for class_name in self.classes:
            class_dir = self.data_dir / class_name
            class_idx = self.class_to_idx[class_name]
            for file_path in class_dir.iterdir():
                if file_path.suffix.lower() in self.extensions:
                    samples.append((file_path, class_idx))
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


# ─────────────────────────────────────────────────────────────
# Validation Utilities
# ─────────────────────────────────────────────────────────────
def validate_image(
    image_path: str | Path,
    max_size_mb: float = 10.0,
    max_dimension: int = 8000,
) -> Tuple[bool, str]:
    """
    Validate an image file for inference.
    
    Args:
        image_path: Path to image file
        max_size_mb: Maximum file size in MB
        max_dimension: Maximum width or height in pixels
    
    Returns:
        Tuple of (is_valid, error_message or "ok")
    """
    path = Path(image_path)
    
    # Check existence
    if not path.exists():
        return False, f"File not found: {path}"
    
    # Check extension
    if path.suffix.lower() not in ImageFolderDataset.SUPPORTED_EXTENSIONS:
        return False, f"Unsupported format: {path.suffix}"
    
    # Check file size
    size_mb = path.stat().st_size / (1024 * 1024)
    if size_mb > max_size_mb:
        return False, f"File too large: {size_mb:.1f}MB > {max_size_mb}MB"
    
    # Check image validity and dimensions
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            if img.width > max_dimension or img.height > max_dimension:
                return False, f"Image too large: {img.width}x{img.height} > {max_dimension}px"
    except Exception as e:
        return False, f"Invalid image: {e}"
    
    return True, "ok"


def get_class_weights(
    labels: list[int],
    num_classes: int,
    method: Literal["inverse", "effective"] = "inverse",
) -> torch.Tensor:
    """
    Compute class weights for imbalanced datasets.
    
    Args:
        labels: List of class labels in dataset
        num_classes: Total number of classes
        method: Weighting method
            - "inverse": Weight = total / (num_classes * class_count)
            - "effective": Effective number of samples weighting
    
    Returns:
        Tensor of shape (num_classes,) with class weights
    """
    counts = torch.bincount(torch.tensor(labels), minlength=num_classes).float()
    
    if method == "inverse":
        weights = len(labels) / (num_classes * counts.clamp(min=1))
    elif method == "effective":
        beta = 0.9999
        effective_num = 1.0 - torch.pow(beta, counts)
        weights = (1.0 - beta) / effective_num.clamp(min=1e-6)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return weights / weights.sum() * num_classes


# ─────────────────────────────────────────────────────────────
# CLI for Testing
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test dataset utilities")
    parser.add_argument("--data-dir", default="./ml/data", help="Data directory")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    args = parser.parse_args()
    
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader, _ = get_cifar10_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=0,
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    print(f"Classes: {CIFAR10_CLASSES}")
    
    # Test a batch
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}")
    print(f"Label range: {labels.min().item()} - {labels.max().item()}")
    print("Dataset utilities OK!")
