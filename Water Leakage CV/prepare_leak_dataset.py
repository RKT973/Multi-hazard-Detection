"""
prepare_leak_dataset.py
Merge raw leak/no_leak sources into train/val/test structure.

IMPORTANT: "leak" means an active leakage situation (dripping, pooling from a
failure, cracked/overflowing fitting) — NOT plain presence of water (glass,
lake, bucket, normal tap). Skim folders before running if unsure.

Usage:
  1. Download sources into raw_data_leakage/<name>/ (see README/commands)
  2. Check actual subfolder names with `find raw_data_leakage/<name> -maxdepth 3 -type d`
  3. Edit SOURCES below to match real paths
  4. python prepare_leak_dataset.py
"""

import os
import shutil
import hashlib
import random
import sys
from pathlib import Path
from PIL import Image

random.seed(42)
DEBUG = True

RAW_ROOT = Path("raw_data_leakage")
OUT_ROOT = Path("leak_dataset")
CLASSES = ["no_leak", "leak"]
SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}
MAX_PER_CLASS = 1500

# Adjust right-hand subfolder paths after inspecting actual downloaded structure
SOURCES = {
    "water_network_dataset": "leak",
    "roboflow_water_leakage": "leak",

    # "water_network_dataset/no_leak": "no_leak",
    "pipe_dataset": "no_leak",          # skim manually — must be leak-free pipe images
    "intel_images": "no_leak",
}

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def debug(msg: str, *args):
    if DEBUG:
        print(f"[debug] {msg.format(*args)}")


def collect_images(folder: Path):
    files = []
    debug("collecting images from {0}", folder)
    for root, _, names in os.walk(folder):
        for n in names:
            if Path(n).suffix.lower() in VALID_EXT:
                files.append(Path(root) / n)
    debug("found {0} candidates in {1}", len(files), folder)
    return files


def file_hash(path: Path, chunk=8192):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def is_valid_image(path: Path):
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception as exc:
        debug("invalid image {0}: {1}", path, exc)
        return False


def main():
    print("=== prepare_leak_dataset.py starting ===")
    print(f"RAW_ROOT = {RAW_ROOT}")
    print(f"OUT_ROOT = {OUT_ROOT}")
    print(f"CLASSES = {CLASSES}")
    print(f"SPLIT = {SPLIT}")
    print(f"MAX_PER_CLASS = {MAX_PER_CLASS}")
    print(f"DEBUG = {DEBUG}")
    print("sources:")
    for src, cls in SOURCES.items():
        print(f"  - {src} -> {cls}")
    print()

    per_class_files = {c: [] for c in CLASSES}

    for src, cls in SOURCES.items():
        folder = RAW_ROOT / src
        if not folder.exists():
            print(f"[skip] missing: {folder}")
            continue
        imgs = collect_images(folder)
        print(f"[scan] {src}: {len(imgs)} files -> {cls}")
        per_class_files[cls].extend(imgs)

    for cls in CLASSES:
        seen_hashes = set()
        clean = []
        for p in per_class_files[cls]:
            if not is_valid_image(p):
                continue
            h = file_hash(p)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            clean.append(p)
        random.shuffle(clean)
        if MAX_PER_CLASS:
            clean = clean[:MAX_PER_CLASS]
        per_class_files[cls] = clean
        print(f"[clean] {cls}: {len(clean)} images after dedupe/cap")

    for cls in CLASSES:
        files = per_class_files[cls]
        n = len(files)
        n_train = int(n * SPLIT["train"])
        n_val = int(n * SPLIT["val"])
        splits = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }
        for split_name, split_files in splits.items():
            out_dir = OUT_ROOT / split_name / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            for i, src_path in enumerate(split_files):
                dst = out_dir / f"{cls}_{i:05d}{src_path.suffix.lower()}"
                shutil.copy2(src_path, dst)
            print(f"[write] {split_name}/{cls}: {len(split_files)} images")

    print("Done. Final structure under leak_dataset/")


if __name__ == "__main__":
    main()
