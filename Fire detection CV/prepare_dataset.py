"""
prepare_dataset.py
Merge raw downloaded dataset folders into the final train/val/test structure.

Usage:
  1. Download all source datasets into raw_data/, e.g.:
       raw_data/intel_images/<scene_folders>/...      -> no_fire
       raw_data/flame_candle/...                       -> controlled_fire
       raw_data/fire_smoke_dataset/flame_only/...       -> controlled_fire (skim manually first)
       raw_data/fire_smoke_dataset/fire_smoke/...       -> uncontrolled_fire
       raw_data/dfire/...                               -> uncontrolled_fire
       raw_data/fire_dataset_kaggle/fire_images/...     -> uncontrolled_fire
       raw_data/fire_dataset_kaggle/non_fire_images/... -> no_fire
       raw_data/forest_fire_c4/Fire/...                 -> uncontrolled_fire
       raw_data/forest_fire_c4/SmokeFire/...             -> uncontrolled_fire
       raw_data/forest_fire_c4/No_Fire/...               -> no_fire

  2. Edit SOURCES below to map each raw folder -> target class.
  3. Run: python prepare_dataset.py
"""

import os
import shutil
import hashlib
import random
from pathlib import Path
from PIL import Image

random.seed(42)

RAW_ROOT = Path("raw_data")
OUT_ROOT = Path("dataset")
CLASSES = ["no_fire", "controlled_fire", "uncontrolled_fire"]
SPLIT = {"train": 0.70, "val": 0.15, "test": 0.15}
MAX_PER_CLASS = 3500  # cap to keep classes roughly balanced; set None to disable

# Map: source folder (relative to RAW_ROOT) -> target class
SOURCES = {
    "forest_fire_c4/Forect Fire/Forest Fire_Dataset/train/nofire": "no_fire",
    "unc_fire_kag/unc_fire_kag/non_fire_images": "no_fire",
    "intel_images": "no_fire",

    "flame_candle/Flame/test/images": "controlled_fire",
    "flame_candle/Flame/train/images": "controlled_fire",
    #"flame_fire_classification": "controlled_fire",
    # "fire_smoke_dataset/flame_only": "controlled_fire",   # skim manually before running
    "datacluster_fire_smoke/controlled_fire": "controlled_fire",

    "datacluster_fire_smoke/uncontrolled_fire": "uncontrolled_fire",
    # "fire_smoke_dataset/fire_smoke": "uncontrolled_fire",
    "dfire": "uncontrolled_fire",
    "unc_fire_kag/unc_fire_kag/fire_images": "uncontrolled_fire",
    "forest_fire_c4/Forect Fire/Forest Fire_Dataset/train/fire": "uncontrolled_fire",
    "forest_fire_c4/Forect Fire/Forest Fire_Dataset/train/smokeFire": "uncontrolled_fire",
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

    # 1. collect + validate + dedupe (by content hash) per class
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
        if MAX_PER_CLASS and len(clean) >= MAX_PER_CLASS:
            print(f"  (capped to {MAX_PER_CLASS})")
            break

    # 2. stratified split + copy
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

    print("Done. Final structure under dataset/")


if __name__ == "__main__":
    main()
