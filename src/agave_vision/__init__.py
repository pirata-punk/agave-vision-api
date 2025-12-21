"""
Agave Vision - YOLOv8 Object Detection for Industrial Cameras

A complete end-to-end system for training and deploying custom object detection
models for industrial video surveillance.
"""

__version__ = "0.1.0"

# Export key classes from modules for easier imports
from agave_vision.core.inference import YOLOInference
from agave_vision.core.roi import ROIPolygon, CameraROI, ROIManager
from agave_vision.core.alerts import AlertEvent, AlertDebouncer
from agave_vision.models.registry import ModelRegistry, ModelVersion
from agave_vision.training.trainer import YOLOTrainer
from agave_vision.training.evaluator import ModelEvaluator, compare_models
from agave_vision.ingestion.static.video_processor import VideoProcessor
from agave_vision.ingestion.static.tile_generator import TileGenerator
from agave_vision.ingestion.static.dataset_builder import DatasetBuilder

__all__ = [
    # Version
    "__version__",
    # Core
    "YOLOInference",
    "ROIPolygon",
    "CameraROI",
    "ROIManager",
    "AlertEvent",
    "AlertDebouncer",
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
