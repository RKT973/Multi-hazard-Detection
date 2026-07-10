"""
prepare_smoke_dataset.py
Merge raw smoke/no_smoke sources into train/val/test structure.

Usage:
  1. Put your existing 800 smoke images in raw_data/my_smoke_images/
  2. Download extra sources into raw_data/<name>/ (see SOURCES below)
  3. Edit SOURCES to map each raw folder -> "smoke" or "no_smoke"
  4. python prepare_smoke_dataset.py
"""

import os
import shutil
import hashlib
import random
from pathlib import Path
from PIL import Image

random.seed(42)

RAW_ROOT = Path("raw_data_smoke")
OUT_ROOT = Path("smoke_dataset")
CLASSES = ["no_smoke", "smoke"]
SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}
MAX_PER_CLASS = 1200  # caps for balance

SOURCES = {
    "my_smoke_images": "smoke",                       # your existing 800
    "smoke_detection_dataset": "smoke",
    "datacluster_fire_smoke/smoke_only": "smoke",

    "intel_images": "no_smoke",
}

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def collect_images(folder: Path):
    files = []
    for root, _, names in os.walk(folder):
        for n in names:
            if Path(n).suffix.lower() in VALID_EXT:
                files.append(Path(root) / n)
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
    except Exception:
        return False


def main():
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
        files_list = per_class_files[cls]
        total = len(files_list)
        print(f"[cleaning] {cls}: validating {total} files")
        seen_hashes = set()
        clean = []
        for idx, p in enumerate(files_list, start=1):
            if idx % 500 == 0:
                print(f"[progress] {cls}: processed {idx}/{total}")
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

    print("Done. Final structure under smoke_dataset/")


if __name__ == "__main__":
    main()
