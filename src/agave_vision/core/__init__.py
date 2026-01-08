"""Core business logic for YOLO inference, ROI filtering, and alerting."""

from agave_vision.core.inference import YOLOInference
from agave_vision.core.roi import ROIPolygon, CameraROI, ROIManager
from agave_vision.core.alerts import AlertEvent
from agave_vision.core.tracking import CentroidTracker
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
    "CentroidTracker",
    "sliding_window_tiles",
    "compute_frame_sharpness",
    "compute_frame_brightness",
    "is_similar_frame",
    "resize_keep_aspect",
    "draw_detection_box",
]
