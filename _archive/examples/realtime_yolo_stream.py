#!/usr/bin/env python3
"""
Run YOLOv8 in (near) real time on a live source and optionally raise ROI alerts.

Notes:
- Designed for local execution; not run here.
- Source can be a webcam index (0), an RTSP/HTTP URL, or a file path for a "live" replay.
- Detections are drawn; alerts (class == ALERT_CLASS inside forbidden ROI) are logged to stdout.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# ------------- CONFIG -------------
MODEL_PATH = Path("models/yolov8n_pina/exp/weights/best.pt")
SOURCE = 0  # 0 for default webcam; set to "data/videos/your_video.mp4" or rtsp URL
CONF = 0.25
IOU = 0.45
IMGSZ = 640
ALERT_CLASS = "object"  # pina/worker allowed

# Optional ROI polygons: list of polygons, each polygon is list of [x, y] points
FORBIDDEN_ROIS: List[List[Tuple[int, int]]] = [
    # Example: [(100, 100), (500, 100), (500, 400), (100, 400)]
]
# ----------------------------------


def point_in_any_roi(pt: Tuple[float, float], rois: List[List[Tuple[int, int]]]) -> bool:
    if not rois:
        return False
    for poly in rois:
        if cv2.pointPolygonTest(np.array(poly, dtype=np.int32), pt, False) >= 0:
            return True
    return False


def draw_rois(frame, rois):
    for poly in rois:
        cv2.polylines(frame, [np.array(poly, dtype=np.int32)], isClosed=True, color=(0, 0, 255), thickness=1)


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))
    cap = cv2.VideoCapture(SOURCE)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {SOURCE}")

    # Try to sync to source FPS when available
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frame_interval = 1.0 / src_fps if src_fps > 0 else 0

    print(f"[INFO] Streaming from {SOURCE} at ~{src_fps:.2f} FPS (interval {frame_interval:.3f}s)")
    print(f"[INFO] Press 'q' to quit.")

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=CONF, iou=IOU, imgsz=IMGSZ, verbose=False)
        dets = results[0]

        if FORBIDDEN_ROIS:
            draw_rois(frame, FORBIDDEN_ROIS)

        for box, cls_id, conf_score in zip(dets.boxes.xyxy, dets.boxes.cls, dets.boxes.conf):
            cls_name = model.names[int(cls_id)]
            x1, y1, x2, y2 = map(int, box.tolist())
            color = (0, 255, 0)
            alert = False
            if cls_name == ALERT_CLASS and FORBIDDEN_ROIS:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if point_in_any_roi((cx, cy), FORBIDDEN_ROIS):
                    alert = True
                    color = (0, 0, 255)
                    print(json.dumps({
                        "alert": True,
                        "class": cls_name,
                        "confidence": float(conf_score),
                        "bbox": [x1, y1, x2, y2],
                        "center": [cx, cy],
                    }))

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{cls_name} {conf_score:.2f}", (x1, max(0, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        cv2.imshow("YOLOv8 Real-Time", frame)

        # Maintain near real-time pacing if FPS is known
        if frame_interval > 0:
            elapsed = time.time() - t0
            sleep_time = frame_interval - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[DONE] Stream ended.")


if __name__ == "__main__":
    main()
