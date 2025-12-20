"""
YOLO Inference Wrapper

Provides a centralized interface for YOLOv8 model loading, inference, and warmup.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from ultralytics import YOLO


@dataclass
class Detection:
    """Structured detection result from YOLO inference."""

    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[float, float, float, float]  # x1, y1, x2, y2
    center: Tuple[float, float]  # cx, cy

    @staticmethod
    def from_yolo_box(box, cls_id: int, conf: float, class_names: Dict[int, str]) -> "Detection":
        """Create Detection from YOLO box tensor."""
        x1, y1, x2, y2 = box.tolist()
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        return Detection(
            class_id=int(cls_id),
            class_name=class_names[int(cls_id)],
            confidence=float(conf),
            bbox=(x1, y1, x2, y2),
            center=(cx, cy),
        )


class YOLOInference:
    """
    Wrapper for YOLOv8 inference with warmup and configuration.

    Args:
        model_path: Path to YOLO model weights (.pt file)
        conf: Minimum confidence threshold (0.0-1.0)
        iou: IoU threshold for NMS (0.0-1.0)
        device: Device for inference ("cuda", "cpu", "mps")
        imgsz: Input image size for YOLO

    Example:
        >>> model = YOLOInference("models/best.pt", conf=0.25, device="cuda")
        >>> model.warmup()
        >>> detections = model.predict(frame)
    """

    def __init__(
        self,
        model_path: str | Path,
        conf: float = 0.25,
        iou: float = 0.45,
        device: str = "cuda",
        imgsz: int = 640,
    ):
        self.model_path = Path(model_path)
        self.conf = conf
        self.iou = iou
        self.device = device
        self.imgsz = imgsz

        # Load model
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        self.model = YOLO(str(self.model_path))
        self._warmup_done = False

    @property
    def class_names(self) -> Dict[int, str]:
        """Return class ID to name mapping."""
        return self.model.names

    def warmup(self, iterations: int = 5) -> None:
        """
        Run warmup inferences to allocate GPU memory and optimize model.

        Args:
            iterations: Number of dummy inferences to run
        """
        if self._warmup_done:
            return

        dummy_img = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        for _ in range(iterations):
            _ = self.model.predict(
                dummy_img, conf=self.conf, iou=self.iou, imgsz=self.imgsz, verbose=False
            )
        self._warmup_done = True

    def predict(
        self, frame: np.ndarray, conf: Optional[float] = None, verbose: bool = False
    ) -> List[Detection]:
        """
        Run inference on a single frame.

        Args:
            frame: Input image as numpy array (BGR format from OpenCV)
            conf: Override confidence threshold (optional)
            verbose: Enable verbose output from YOLO

        Returns:
            List of Detection objects
        """
        if conf is None:
            conf = self.conf

        results = self.model.predict(
            frame, conf=conf, iou=self.iou, imgsz=self.imgsz, verbose=verbose
        )

        if not results or len(results) == 0:
            return []

        dets = results[0]
        if not hasattr(dets, "boxes") or len(dets.boxes) == 0:
            return []

        detections = []
        for box, cls_id, confidence in zip(dets.boxes.xyxy, dets.boxes.cls, dets.boxes.conf):
            detection = Detection.from_yolo_box(box, cls_id, confidence, self.class_names)
            detections.append(detection)

        return detections

    def predict_batch(
        self, frames: List[np.ndarray], conf: Optional[float] = None
    ) -> List[List[Detection]]:
        """
        Run batch inference on multiple frames.

        Args:
            frames: List of input images
            conf: Override confidence threshold (optional)

        Returns:
            List of detection lists (one per frame)
        """
        if conf is None:
            conf = self.conf

        results = self.model.predict(frames, conf=conf, iou=self.iou, imgsz=self.imgsz, verbose=False)

        all_detections = []
        for result in results:
            if not hasattr(result, "boxes") or len(result.boxes) == 0:
                all_detections.append([])
                continue

            frame_detections = []
            for box, cls_id, confidence in zip(
                result.boxes.xyxy, result.boxes.cls, result.boxes.conf
            ):
                detection = Detection.from_yolo_box(box, cls_id, confidence, self.class_names)
                frame_detections.append(detection)

            all_detections.append(frame_detections)

        return all_detections

    def __repr__(self) -> str:
        return (
            f"YOLOInference(model={self.model_path.name}, conf={self.conf}, "
            f"iou={self.iou}, device={self.device})"
        )
