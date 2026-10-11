# apps/backend/app/model_service.py
from __future__ import annotations
from typing import List, Dict
from PIL import Image
import os
import torch
from huggingface_hub import hf_hub_download
from torchvision import transforms

# ImageNet normalization — MUST match training transforms in ml/src/train_cifar.py
# to prevent silent accuracy drift between training and production.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

class ModelService:
    def __init__(self, labels: list[str], model: torch.jit.ScriptModule):
        self.categories = labels
        self.model = model.eval()
        self.preprocess = PREPROCESS

    @classmethod
    def create(cls) -> "ModelService":
        """
        Load TorchScript model either from local paths (if present) or from
        a Hugging Face model repo using env vars.
        
        Supports both regular and INT8 quantized models.
        Set USE_QUANTIZED=true to load quantized model for faster inference.
        """
        use_quantized = os.getenv("USE_QUANTIZED", "false").lower() == "true"
        
        # 1) Try local files first
        if use_quantized:
            local_model = os.getenv("MODEL_PATH", "app/models/model.quantized.pt")
        else:
            local_model = os.getenv("MODEL_PATH", "app/models/model.torchscript.pt")
        local_labels = os.getenv("LABELS_PATH", "app/models/labels.txt")
        
        if os.path.exists(local_model) and os.path.exists(local_labels):
            labels = [l.strip() for l in open(local_labels, "r", encoding="utf-8") if l.strip()]
            model = torch.jit.load(local_model, map_location="cpu")
            return cls(labels, model)

        # 2) Otherwise, download from HF Hub
        repo_id    = os.environ["MODEL_REPO"]           # e.g. "prakhyatc/visionscale-cifar10"
        revision   = os.getenv("MODEL_REV", "main")     # or a tag like "v1"
        model_file = os.getenv("MODEL_FILE", "model.torchscript.pt")
        labels_file= os.getenv("LABELS_FILE", "labels.txt")

        mpath = hf_hub_download(repo_id=repo_id, filename=model_file, revision=revision)
        lpath = hf_hub_download(repo_id=repo_id, filename=labels_file, revision=revision)

        labels = [l.strip() for l in open(lpath, "r", encoding="utf-8") if l.strip()]
        model = torch.jit.load(mpath, map_location="cpu")
        return cls(labels, model)

    def predict_image(self, img: Image.Image, k: int = 3) -> List[Dict]:
        x = self.preprocess(img).unsqueeze(0)
        with torch.inference_mode():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0]
        k = min(k, probs.numel())
        confs, idxs = torch.topk(probs, k)
        return [
            {"label": self.categories[i], "confidence": float(c)}
            for c, i in zip(confs.tolist(), idxs.tolist())
        ]