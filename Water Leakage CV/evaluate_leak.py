"""
evaluate_leak.py - Evaluate a saved water-leak classifier checkpoint.

Usage:
  python evaluate_leak.py --data_dir leak_dataset --split test --checkpoint runs/leak_classifier/best_model.pt
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, confusion_matrix, classification_report
)
import matplotlib.pyplot as plt


def build_model(num_classes=2):
    model = models.efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="leak_dataset")
    ap.add_argument("--split", default="test")
    ap.add_argument("--checkpoint", default="runs/leak_classifier/best_model.pt")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out_dir", default="runs/leak_classifier")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device)
    img_size = ckpt.get("img_size", 300)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    eval_tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    ds = datasets.ImageFolder(Path(args.data_dir) / args.split, transform=eval_tf)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = build_model(num_classes=len(class_to_idx)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            out = model(x)
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(y.numpy())

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    print(f"Split: {args.split}")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")
    names = [idx_to_class[i] for i in range(len(idx_to_class))]
    print(classification_report(all_labels, all_preds, target_names=names))

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(names))))
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    out_path = Path(args.out_dir) / f"confusion_matrix_{args.split}.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"Confusion matrix saved to {out_path}")

    with open(Path(args.out_dir) / f"{args.split}_metrics.json", "w") as f:
        json.dump({"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}, f, indent=2)


if __name__ == "__main__":
    main()
