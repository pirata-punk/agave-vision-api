#!/usr/bin/env python3
"""
Stream a local video through the trained model and save an annotated MP4.
Optional: apply forbidden-ROI alerting (draw red boxes when object in ROI).
"""

from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

# ------------- CONFIG -------------
MODEL_PATH = Path("models/yolov8n_pina/exp/weights/best.pt")
VIDEO_IN = Path(
    "data/videos/Video Volcador B Nave 3 HORNOS  CAM 3 PI„AS/NAVE 3_HORNOS B CAM 3_20250923125056_20250923131506.mp4"
)
VIDEO_OUT = Path("models/yolov8n_pina/demo_output.mp4")
CONF = 0.25
IOU = 0.45
# ROI config (optional): list of polygons per camera; here apply to all frames
FORBIDDEN_ROIS = [
    # Example polygon; replace with your real coords
    # [(100, 100), (500, 100), (500, 400), (100, 400)]
]
ALERT_CLASS = "object"  # pina/worker allowed
# ----------------------------------


def point_in_any_roi(pt, rois):
    for poly in rois:
        if cv2.pointPolygonTest(np.array(poly, dtype=np.int32), pt, False) >= 0:
            return True
    return False


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)
    if not VIDEO_IN.exists():
        raise FileNotFoundError(VIDEO_IN)

    cap = cv2.VideoCapture(str(VIDEO_IN))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(
        str(VIDEO_OUT),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w, h),
    )

    model = YOLO(str(MODEL_PATH))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(frame, conf=CONF, iou=IOU, verbose=False)
        dets = results[0]
        for box, cls_id, conf in zip(dets.boxes.xyxy, dets.boxes.cls, dets.boxes.conf):
            cls_name = model.names[int(cls_id)]
            x1, y1, x2, y2 = map(int, box.tolist())
            color = (0, 255, 0)
            alert = False
            if cls_name == ALERT_CLASS and FORBIDDEN_ROIS:
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if point_in_any_roi((cx, cy), FORBIDDEN_ROIS):
                    alert = True
                    color = (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(
                frame,
                f"{cls_name} {conf:.2f}",
                (x1, max(y1 - 5, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
            if alert:
                cv2.putText(
                    frame,
                    "ALERT",
                    (x1, y2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"[DONE] Annotated video saved to {VIDEO_OUT}")


if __name__ == "__main__":
    main()
