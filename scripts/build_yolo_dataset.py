#!/usr/bin/env python3
"""
Build a YOLOv8-ready dataset from labeled tile rounds (single entrypoint).

Flow alignment (videos -> frames -> tiles_pool -> tiles_man -> rounds -> yolo):
- reads images/labels from data/tiles_pool/tiles_man/tiles_round{1..4}
- expects filenames already standardized (no spaces, hashed prefixes removed)
- matches images to labels (non-empty only)
- splits deterministically into train/val/test
- copies into data/tiles_yolo/{images,labels}/{train,val,test}
- writes configs/yolo_data.yaml and data/tiles_yolo/metadata.json

Note: This script is **not executed here** to preserve existing labeled assets.
"""

from __future__ import annotations

import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

# ---------------- CONFIG ----------------
ROUNDS_ROOT = Path("data/tiles_pool/tiles_man")
ROUND_NAMES = ["tiles_round1", "tiles_round2", "tiles_round3", "tiles_round4"]
DATASET_ROOT = Path("data/tiles_yolo")
SPLIT_RATIOS = (0.7, 0.15, 0.15)  # train, val, test
RANDOM_SEED = 123

# Class order should match the YOLO label files (id -> name)
CLASS_NAMES = ["object", "pine", "worker"]

# Allow empty label files (treat as background examples)
ALLOW_EMPTY_LABELS = True

# Outputs
DATA_YAML = Path("configs/yolo_data.yaml")
META_JSON = DATASET_ROOT / "metadata.json"

IMAGE_EXTS = {".jpg", ".jpeg", ".png"}
# ----------------------------------------


@dataclass
class Pair:
    round_name: str
    image_path: Path
    label_path: Path
    new_name: str  # basename without extension


def gather_round_pairs(round_name: str) -> List[Pair]:
    images_dir = ROUNDS_ROOT / round_name / "images"
    labels_dir = ROUNDS_ROOT / round_name / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        print(f"[WARN] Missing images or labels for {round_name}; skipping.")
        return []

    pairs: List[Pair] = []
    seen_names: Dict[str, int] = {}

    for img_path in images_dir.iterdir():
        if img_path.suffix.lower() not in IMAGE_EXTS:
            continue

        # Direct match expected after standardization
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            print(f"[WARN] No label for image {img_path}")
            continue
        if label_path.stat().st_size == 0 and not ALLOW_EMPTY_LABELS:
            print(f"[WARN] Empty label file {label_path}, skipping.")
            continue

        # Ensure unique new name across rounds
        base = f"{round_name}_{img_path.stem}"
        if base not in seen_names:
            seen_names[base] = 1
            new_name = base
        else:
            seen_names[base] += 1
            new_name = f"{base}_{seen_names[base]-1}"

        pairs.append(
            Pair(
                round_name=round_name,
                image_path=img_path,
                label_path=label_path,
                new_name=new_name,
            )
        )
    return pairs


def split_pairs(pairs: List[Pair]) -> Dict[str, List[Pair]]:
    rng = random.Random(RANDOM_SEED)
    rng.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * SPLIT_RATIOS[0])
    n_val = int(n * SPLIT_RATIOS[1])
    train = pairs[:n_train]
    val = pairs[n_train : n_train + n_val]
    test = pairs[n_train + n_val :]
    return {"train": train, "val": val, "test": test}


def copy_pairs(split_name: str, pairs: List[Pair]):
    img_out_dir = DATASET_ROOT / "images" / split_name
    lbl_out_dir = DATASET_ROOT / "labels" / split_name
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    for pair in pairs:
        new_img = img_out_dir / f"{pair.new_name}.jpg"
        new_lbl = lbl_out_dir / f"{pair.new_name}.txt"
        shutil.copy2(pair.image_path, new_img)
        shutil.copy2(pair.label_path, new_lbl)


def write_data_yaml():
    DATA_YAML.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "path": str(DATASET_ROOT),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(CLASS_NAMES)},
    }
    try:
        import yaml
    except ImportError:
        raise SystemExit("pyyaml not installed. Please run `pip install -r requirements.txt`.")
    with DATA_YAML.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, sort_keys=False)


def write_meta_json(split_map: Dict[str, List[Pair]]):
    DATASET_ROOT.mkdir(parents=True, exist_ok=True)
    files = []
    for split, pairs in split_map.items():
        for pair in pairs:
            files.append(
                {
                    "split": split,
                    "round": pair.round_name,
                    "new_name": pair.new_name,
                    "image": str(pair.image_path),
                    "label": str(pair.label_path),
                }
            )

    meta = {
        "rounds_root": str(ROUNDS_ROOT),
        "dataset_root": str(DATASET_ROOT),
        "splits": {k: len(v) for k, v in split_map.items()},
        "total": sum(len(v) for v in split_map.values()),
        "random_seed": RANDOM_SEED,
        "class_names": CLASS_NAMES,
        "files": files,
    }
    with META_JSON.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)


def main():
    pairs: List[Pair] = []
    for rnd in ROUND_NAMES:
        pairs.extend(gather_round_pairs(rnd))

    if not pairs:
        print("[ERROR] No image/label pairs found; check round paths.")
        return

    split_map = split_pairs(pairs)
    for split_name, split_pairs_list in split_map.items():
        print(f"{split_name}: {len(split_pairs_list)} pairs")
        copy_pairs(split_name, split_pairs_list)

    write_data_yaml()
    write_meta_json(split_map)
    print(f"[DONE] Dataset written to {DATASET_ROOT} and {DATA_YAML}")


if __name__ == "__main__":
    main()
