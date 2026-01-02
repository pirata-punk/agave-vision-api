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
from agave_vision.core.alerts import AlertEvent, AlertDebouncer

# Storage (for advanced usage)
from agave_vision.storage.alert_store import AlertStore
from agave_vision.storage.detection_logger import DetectionLogger

# Training and data pipeline (for model development)
from agave_vision.models.registry import ModelRegistry, ModelVersion
from agave_vision.training.trainer import YOLOTrainer
from agave_vision.training.evaluator import ModelEvaluator, compare_models
from agave_vision.ingestion.static.video_processor import VideoProcessor
from agave_vision.ingestion.static.tile_generator import TileGenerator
from agave_vision.ingestion.static.dataset_builder import DatasetBuilder

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
    "AlertDebouncer",
    # Storage
    "AlertStore",
    "DetectionLogger",
    # Models
    "ModelRegistry",
    "ModelVersion",
    # Training
    "YOLOTrainer",
    "ModelEvaluator",
    "compare_models",
    # Ingestion
    "VideoProcessor",
    "TileGenerator",
    "DatasetBuilder",
]
