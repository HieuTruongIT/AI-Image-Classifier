# ml/src/export_torchscript.py
import torch
from torchvision import models
import argparse, os

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="ml/out/weights.pt")
    ap.add_argument("--out", default="apps/backend/app/models/model.torchscript.pt")
    args = ap.parse_args()

    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    sd = torch.load(args.weights, map_location="cpu")
    model.load_state_dict(sd, strict=True)
    model.eval()
    scripted = torch.jit.script(model)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    scripted.save(args.out)
    print("Saved TorchScript:", args.out)

if __name__ == "__main__":
    main()