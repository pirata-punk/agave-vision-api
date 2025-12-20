#!/usr/bin/env python3
"""
Lightweight inference + alerting pipeline for live cameras.

Behavior:
- Loads a YOLOv8 model
- Ingests frames (or images) per camera
- Checks detections against per-camera ROIs:
    * If class == "object" inside forbidden ROI -> raise alert
    * Classes "pine" and "worker" are allowed and ignored for alerts
- Prints JSON-friendly alert dicts; does not modify any on-disk data.

Inputs:
- Model path (weights)
- ROI config (YAML): per-camera polygons and allowed classes

Note: This script is not executed here to preserve existing labeled assets.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import numpy as np
import yaml
from ultralytics import YOLO


ALERT_CLASS_NAME = "object"  # class that triggers alerts in forbidden ROI
PINE_NAME = "pine"
WORKER_NAME = "worker"


@dataclass
class ROIConfig:
    camera_id: str
    forbidden_polygons: List[np.ndarray]  # list of Nx2 polygons


def load_rois(rois_yaml: Path) -> Dict[str, ROIConfig]:
    with rois_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    rois: Dict[str, ROIConfig] = {}
    for cam in data.get("cameras", []):
        polys = []
        for poly in cam.get("forbidden_rois", []):
            polys.append(np.array(poly, dtype=np.int32))
        rois[cam["id"]] = ROIConfig(camera_id=cam["id"], forbidden_polygons=polys)
    return rois


def bbox_center(box: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def point_in_any_roi(pt: Tuple[float, float], rois: List[np.ndarray]) -> bool:
    for poly in rois:
        if cv2.pointPolygonTest(poly, pt, False) >= 0:
            return True
    return False


def run_inference(
    model_path: Path, rois_yaml: Path, frame_paths: List[Path], camera_id: str
):
    model = YOLO(str(model_path))
    rois = load_rois(rois_yaml)
    cfg = rois.get(camera_id)
    if cfg is None:
        raise ValueError(f"No ROI config for camera {camera_id}")

    for frame_path in frame_paths:
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        results = model.predict(img, verbose=False)
        if not results:
            continue

        dets = results[0]
        for box, cls_id, conf in zip(dets.boxes.xyxy, dets.boxes.cls, dets.boxes.conf):
            cls_name = model.names[int(cls_id)]
            if cls_name in (PINE_NAME, WORKER_NAME):
                continue
            if cls_name != ALERT_CLASS_NAME:
                continue

            cx, cy = bbox_center(box.tolist())
            if point_in_any_roi((cx, cy), cfg.forbidden_polygons):
                alert = {
                    "camera_id": camera_id,
                    "frame": str(frame_path),
                    "cls": cls_name,
                    "confidence": float(conf),
                    "bbox": box.tolist(),
                    "roi_hit": True,
                }
                print(json.dumps(alert))


if __name__ == "__main__":
    # Example usage (adjust paths before running):
    # run_inference(Path("models/best.pt"), Path("configs/rois.yaml"), [Path("example.jpg")], "cam_1")
    pass
