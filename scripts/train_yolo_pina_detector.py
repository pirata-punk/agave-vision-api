#!/usr/bin/env python3
"""
Train a YOLOv8 detector for pina, worker, object classes using the merged tiles dataset.

Assumptions:
- Dataset already built at data/tiles_yolo with configs/yolo_data.yaml
- Class mapping in data.yaml matches: 0: object, 1: pine, 2: worker
- ultralytics is installed (see requirements.txt)

Outputs:
- Ultralytics run directory (exp) under models/yolov8n_pina/
- best.pt and last.pt inside the exp/weights/ directory

Note: This script is not executed here to preserve your environment.
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


# ---------------- CONFIG ----------------
DATA_YAML = Path("configs/yolo_data.yaml")
MODEL_NAME = "yolov8n"  # base model
OUTPUT_DIR = Path("models/yolov8n_pina")  # where Ultralytics will create exp/
EPOCHS = 100
IMG_SIZE = 640
BATCH = 16
WORKERS = 8
# ----------------------------------------


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(f"data.yaml not found: {DATA_YAML}")

    model = YOLO(f"{MODEL_NAME}.pt")
    model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH,
        workers=WORKERS,
        project=str(OUTPUT_DIR),
        name="exp",
        exist_ok=True,
    )
    print(f"[DONE] Training run stored under {OUTPUT_DIR}/exp/")


if __name__ == "__main__":
    main()
