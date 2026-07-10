"""
train_smoke.py - Smoke classification: EfficientNetV2-S transfer learning.
Classes: no_smoke, smoke

Usage:
  python train_smoke.py --data_dir smoke_dataset --epochs 20 --batch_size 32 --img_size 300
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt

CLASSES = ["no_smoke", "smoke"]


def get_transforms(img_size):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.75, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, eval_tf


def build_model(num_classes=2):
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def make_weighted_sampler(dataset):
    targets = np.array(dataset.targets)
    class_counts = np.bincount(targets, minlength=len(CLASSES))
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[targets]
    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True
    )
    return sampler, class_weights


def evaluate(model, loader, device, criterion):
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            total_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    return avg_loss, acc, precision, recall, f1, all_labels, all_preds


def plot_confusion_matrix(labels, preds, out_path):
    cm = confusion_matrix(labels, preds, labels=list(range(len(CLASSES))))
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="smoke_dataset")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--img_size", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--out_dir", default="runs/smoke_classifier")
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--log_interval", type=int, default=10,
                    help="Print training progress every N batches")
    ap.add_argument("--debug", action="store_true",
                    help="Print additional debug info for batch failures")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_tf, eval_tf = get_transforms(args.img_size)

    train_ds = datasets.ImageFolder(Path(args.data_dir) / "train", transform=train_tf)
    val_ds = datasets.ImageFolder(Path(args.data_dir) / "val", transform=eval_tf)
    test_ds = datasets.ImageFolder(Path(args.data_dir) / "test", transform=eval_tf)

    assert train_ds.classes == sorted(CLASSES), \
        f"Class folder names must be exactly {sorted(CLASSES)}, got {train_ds.classes}"
    idx_to_class = {v: k for k, v in train_ds.class_to_idx.items()}

    sampler, class_weights = make_weighted_sampler(train_ds)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                               num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True)

    if args.debug:
        print(f"train size={len(train_ds)}, val size={len(val_ds)}, test size={len(test_ds)}")
        print(f"train classes={train_ds.classes}, class_counts={np.bincount(train_ds.targets)}")

    model = build_model(num_classes=len(CLASSES)).to(device)

    weight_tensor = torch.tensor(class_weights / class_weights.sum() * len(CLASSES),
                                  dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.1)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_f1 = -1.0
    epochs_no_improve = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        running_loss = 0.0
        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            try:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                out = model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * x.size(0)
            except Exception as exc:
                print(f"[ERROR] epoch={epoch} batch={batch_idx} size={x.size(0) if 'x' in locals() else 'unknown'}")
                print(f"        exception={type(exc).__name__}: {exc}")
                if args.debug:
                    import traceback
                    traceback.print_exc()
                raise

            if batch_idx % args.log_interval == 0 or batch_idx == len(train_loader):
                avg_batch_loss = running_loss / (batch_idx * args.batch_size)
                print(f"Epoch {epoch}/{args.epochs} | batch {batch_idx}/{len(train_loader)} "
                      f"| batch_loss {loss.item():.4f} | avg_loss {avg_batch_loss:.4f}")

        scheduler.step()
        train_loss = running_loss / len(train_loader.dataset)

        val_loss, val_acc, val_prec, val_rec, val_f1, _, _ = evaluate(model, val_loader, device, criterion)
        dt = time.time() - t0
        print(f"Epoch {epoch:02d}/{args.epochs} | train_loss {train_loss:.4f} | "
              f"val_loss {val_loss:.4f} val_acc {val_acc:.4f} val_f1 {val_f1:.4f} | {dt:.1f}s")

        if args.debug:
            print(f"  debug: lr={optimizer.param_groups[0]['lr']:.6g} "
                  f"| epoch_time={dt:.1f}s | best_val_f1={best_val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_to_idx": train_ds.class_to_idx,
                "img_size": args.img_size,
                "epoch": epoch,
                "val_f1": val_f1,
            }, out_dir / "best_model.pt")
            print(f"  -> new best (val_f1={val_f1:.4f}), checkpoint saved")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {args.patience} epochs)")
                break

        torch.save({
            "model_state_dict": model.state_dict(),
            "class_to_idx": train_ds.class_to_idx,
            "img_size": args.img_size,
            "epoch": epoch,
        }, out_dir / "last_model.pt")

    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    ckpt = torch.load(out_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_loss, test_acc, test_prec, test_rec, test_f1, labels, preds = evaluate(
        model, test_loader, device, criterion
    )
    print("\n=== TEST RESULTS (best checkpoint) ===")
    print(f"Accuracy : {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall   : {test_rec:.4f}")
    print(f"F1       : {test_f1:.4f}")
    print(classification_report(labels, preds, target_names=[idx_to_class[i] for i in range(len(CLASSES))]))

    plot_confusion_matrix(labels, preds, out_dir / "confusion_matrix.png")
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump({"accuracy": test_acc, "precision": test_prec,
                    "recall": test_rec, "f1": test_f1}, f, indent=2)

    print(f"\nArtifacts saved in: {out_dir}")
    print(" - best_model.pt, last_model.pt, history.json, test_metrics.json, confusion_matrix.png")


if __name__ == "__main__":
    main()
