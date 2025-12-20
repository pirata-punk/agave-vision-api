"""
Configuration Models

Pydantic models for type-safe YAML configuration validation.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from pydantic import BaseModel, Field, field_validator


# ============================================
# ROI Configuration Models
# ============================================


class ROIPolygonConfig(BaseModel):
    """Configuration for a single ROI polygon."""

    points: List[Tuple[int, int]] = Field(..., description="List of (x, y) polygon vertices")
    name: Optional[str] = Field(None, description="Optional name for this ROI")

    @field_validator("points")
    @classmethod
    def validate_points(cls, v):
        if len(v) < 3:
            raise ValueError("ROI polygon must have at least 3 points")
        return v


class CameraROIConfig(BaseModel):
    """ROI configuration for a single camera."""

    camera_id: str = Field(..., description="Camera identifier")
    forbidden_rois: List[ROIPolygonConfig] = Field(
        default_factory=list, description="List of forbidden zone polygons"
    )
    allowed_classes: List[str] = Field(
        default=["pine", "worker"], description="Classes allowed in forbidden zones (no alerts)"
    )
    alert_classes: List[str] = Field(
        default=["object"], description="Classes that trigger alerts in forbidden zones"
    )


class ROIsConfig(BaseModel):
    """Root ROI configuration."""

    cameras: List[CameraROIConfig] = Field(..., description="List of camera ROI configs")


# ============================================
# Camera Configuration Models
# ============================================


class CameraConfig(BaseModel):
    """Configuration for a single camera."""

    id: str = Field(..., description="Unique camera identifier")
    name: str = Field(..., description="Human-readable camera name")
    rtsp_url: str = Field(..., description="RTSP stream URL")
    enabled: bool = Field(default=True, description="Whether camera is active")
    fps_target: float = Field(
        default=5.0, ge=0.1, le=30.0, description="Target frame sampling rate (FPS)"
    )

    @field_validator("rtsp_url")
    @classmethod
    def validate_rtsp_url(cls, v):
        if not v.startswith(("rtsp://", "http://", "https://", "/")):
            raise ValueError("RTSP URL must start with rtsp://, http://, https://, or / (file path)")
        return v


class CamerasConfig(BaseModel):
    """Root camera configuration."""

    cameras: List[CameraConfig] = Field(..., description="List of camera configs")


# ============================================
# Service Configuration Models
# ============================================


class InferenceConfig(BaseModel):
    """YOLO inference configuration."""

    model_path: str = Field(..., description="Path to YOLO model weights (.pt file)")
    confidence: float = Field(
        default=0.25, ge=0.0, le=1.0, description="Minimum confidence threshold"
    )
    iou_threshold: float = Field(default=0.45, ge=0.0, le=1.0, description="IoU threshold for NMS")
    device: str = Field(
        default="cuda", description="Inference device (cuda, cpu, mps)"
    )
    batch_size: int = Field(default=1, ge=1, le=32, description="Batch size for inference")
    warmup_iterations: int = Field(
        default=5, ge=0, description="Number of warmup inferences on startup"
    )
    image_size: int = Field(
        default=640, ge=320, le=1280, description="Input image size for YOLO"
    )

    @field_validator("device")
    @classmethod
    def validate_device(cls, v):
        if v not in ["cuda", "cpu", "mps", "auto"]:
            raise ValueError("Device must be cuda, cpu, mps, or auto")
        return v


class StreamManagerConfig(BaseModel):
    """Stream manager configuration."""

    frame_buffer_size: int = Field(
        default=10, ge=1, le=100, description="Maximum frames buffered per camera"
    )
    reconnect_delay_seconds: float = Field(
        default=5.0, ge=0.1, description="Delay before reconnecting after RTSP failure"
    )
    max_reconnect_attempts: int = Field(
        default=10, ge=-1, description="Max reconnection attempts (-1 for infinite)"
    )
    read_timeout_seconds: float = Field(
        default=30.0, ge=1.0, description="RTSP read timeout"
    )


class AlertingConfig(BaseModel):
    """Alerting configuration."""

    debounce_window_seconds: float = Field(
        default=5.0, ge=0.0, description="Time window for alert deduplication"
    )
    max_alerts_per_window: int = Field(
        default=1, ge=1, description="Max alerts per camera per window"
    )
    protocol: str = Field(
        default="stdout", description="Alert delivery protocol (stdout, webhook, hikvision)"
    )
    webhook_url: Optional[str] = Field(None, description="Webhook URL (if protocol is webhook)")
    webhook_timeout_seconds: float = Field(default=5.0, ge=0.1, description="Webhook timeout")
    webhook_retry_attempts: int = Field(default=3, ge=0, description="Webhook retry attempts")
    hikvision_host: Optional[str] = Field(
        None, description="Hikvision NVR host (if protocol is hikvision)"
    )
    hikvision_port: int = Field(default=8000, ge=1, le=65535, description="Hikvision NVR port")
    hikvision_username: Optional[str] = Field(None, description="Hikvision username")
    hikvision_password: Optional[str] = Field(None, description="Hikvision password")
    redis_stream_name: str = Field(default="alerts", description="Redis stream name")
    redis_consumer_group: str = Field(
        default="alert_router", description="Redis consumer group name"
    )
    redis_max_stream_length: int = Field(
        default=10000, ge=100, description="Max stream length (for XTRIM)"
    )

    @field_validator("protocol")
    @classmethod
    def validate_protocol(cls, v):
        if v not in ["stdout", "webhook", "hikvision"]:
            raise ValueError("Protocol must be stdout, webhook, or hikvision")
        return v


class ServicesConfig(BaseModel):
    """Root services configuration."""

    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    stream_manager: StreamManagerConfig = Field(default_factory=StreamManagerConfig)
    alerting: AlertingConfig = Field(default_factory=AlertingConfig)


# ============================================
# Merged Configuration (All Configs)
# ============================================


class AgaveVisionConfig(BaseModel):
    """Complete application configuration (all services)."""

    cameras: CamerasConfig
    rois: ROIsConfig
    services: ServicesConfig

    class Config:
        """Pydantic config."""

        extra = "forbid"  # Reject unknown fields
