"""
Agave Vision - YOLOv8 Object Detection for Industrial Cameras

Pure ML API for object detection, ROI filtering, and alert generation.
Designed for easy integration into any server architecture.
"""

__version__ = "2.0.0"

# Main ML API - This is what external teams use
from agave_vision.ml_api import AgaveVisionML

# Core ML components (for advanced usage)
from agave_vision.core.inference import YOLOInference
from agave_vision.core.roi import ROIPolygon, CameraROI, ROIManager
from agave_vision.core.alerts import AlertEvent
from agave_vision.core.tracking import CentroidTracker

# Alert debouncing (moved to services module)
from agave_vision.services.alert_router.debounce import AlertDebouncer

# Storage (for advanced usage)
from agave_vision.storage.alert_store import AlertStore
from agave_vision.storage.detection_logger import DetectionLogger

__all__ = [
    # Version
    "__version__",
    # Main ML API (PRIMARY INTERFACE)
    "AgaveVisionML",
    # Core ML
    "YOLOInference",
    "ROIPolygon",
    "CameraROI",
    "ROIManager",
    "AlertEvent",
    "CentroidTracker",
    # Alert handling
    "AlertDebouncer",
    # Storage
    "AlertStore",
    "DetectionLogger",
]
