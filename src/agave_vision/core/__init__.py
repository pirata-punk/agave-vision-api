"""Core business logic for YOLO inference, ROI filtering, and alerting."""

from agave_vision.core.inference import YOLOInference
from agave_vision.core.roi import ROIPolygon, CameraROI, ROIManager
from agave_vision.core.alerts import AlertEvent, AlertDebouncer
from agave_vision.core.frames import (
    sliding_window_tiles,
    compute_frame_sharpness,
    compute_frame_brightness,
    is_similar_frame,
    resize_keep_aspect,
    draw_detection_box,
)

__all__ = [
    "YOLOInference",
    "ROIPolygon",
    "CameraROI",
    "ROIManager",
    "AlertEvent",
    "AlertDebouncer",
    "sliding_window_tiles",
    "compute_frame_sharpness",
    "compute_frame_brightness",
    "is_similar_frame",
    "resize_keep_aspect",
    "draw_detection_box",
]
