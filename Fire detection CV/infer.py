"""
infer.py - Run inference on a single image.

Usage:
  python infer.py --image image.jpg --checkpoint runs/fire_classifier/best_model.pt

Output (stdout, JSON):
  {"prediction": "No Fire | Controlled Fire | Uncontrolled Fire", "confidence": 0.00}
"""

import argparse
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models

DISPLAY_NAME = {
    "no_fire": "No Fire",
    "controlled_fire": "Controlled Fire",
    "uncontrolled_fire": "Uncontrolled Fire",
}


def build_model(num_classes=3):
    model = models.efficientnet_v2_s(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    return model


def load_model(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device)
    class_to_idx = ckpt["class_to_idx"]
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    img_size = ckpt.get("img_size", 300)
    model = build_model(num_classes=len(class_to_idx)).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, idx_to_class, img_size


def predict(model, idx_to_class, img_size, image_path, device):
    tf = transforms.Compose([
        transforms.Resize(int(img_size * 1.15)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(image_path).convert("RGB")
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]
        conf, pred_idx = torch.max(probs, dim=0)
    pred_class = idx_to_class[pred_idx.item()]
    return {
        "prediction": DISPLAY_NAME[pred_class],
        "confidence": round(conf.item(), 2),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--checkpoint", default="runs/fire_classifier/best_model.pt")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, idx_to_class, img_size = load_model(args.checkpoint, device)
    result = predict(model, idx_to_class, img_size, args.image, device)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
