"""
API Request/Response Schemas

Pydantic models for FastAPI request and response validation.
"""

from typing import List, Tuple

from pydantic import BaseModel, Field


class DetectionSchema(BaseModel):
    """Detection result schema."""

    class_id: int = Field(..., description="Class ID")
    class_name: str = Field(..., description="Class name")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence")
    bbox: Tuple[float, float, float, float] = Field(..., description="Bounding box (x1, y1, x2, y2)")
    center: Tuple[float, float] = Field(..., description="Center point (cx, cy)")


class InferResponse(BaseModel):
    """Inference response schema."""

    detections: List[DetectionSchema] = Field(..., description="List of detections")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    image_size: Tuple[int, int] = Field(..., description="Input image size (width, height)")


class HealthResponse(BaseModel):
    """Health check response schema."""

    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether YOLO model is loaded")
    device: str = Field(..., description="Inference device (cuda/cpu/mps)")
    model_path: str = Field(..., description="Model file path")


class CameraROISchema(BaseModel):
    """Camera ROI configuration schema."""

    camera_id: str
    forbidden_zones: List[dict]
    allowed_classes: List[str]
    alert_classes: List[str]


class CameraSchema(BaseModel):
    """Camera configuration schema."""

    id: str
    name: str
    enabled: bool
    fps_target: float
