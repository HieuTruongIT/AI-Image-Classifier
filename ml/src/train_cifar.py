# ml/src/train_cifar.py
import argparse, time, os
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# ImageNet normalization — must match inference transforms in model_service.py
# to prevent silent accuracy drift between training and production.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def get_loaders(batch=64):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    test_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train = datasets.CIFAR10(root="./ml/data", train=True, download=True, transform=train_tf)
    test  = datasets.CIFAR10(root="./ml/data", train=False, download=True, transform=test_tf)
    return (
        DataLoader(train, batch_size=batch, shuffle=True, num_workers=0, pin_memory=False),
        DataLoader(test, batch_size=batch, shuffle=False, num_workers=0, pin_memory=False),
    )

def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for x,y in loader:
            x,y = x.to(device), y.to(device)
            logits = model(x)
            preds = logits.argmax(1)
            correct += (preds==y).sum().item()
            total += y.numel()
    return correct/total

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--out", type=str, default="ml/out")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    device = (
    torch.device("mps") if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
    else torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)
    print("Using device:", device)

    train_loader, test_loader = get_loaders(batch=args.batch)

    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(device)

    opt = optim.AdamW(model.parameters(), lr=args.lr)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()

    best_acc = 0.0
    t0 = time.time()
    for epoch in range(1, args.epochs+1):
        model.train()
        for x,y in train_loader:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
        sched.step()
        acc = accuracy(model, test_loader, device)
        print(f"epoch {epoch} acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"{args.out}/weights.pt")
    print(f"best_acc={best_acc:.4f} in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()