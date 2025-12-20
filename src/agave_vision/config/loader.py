"""
Configuration Loader

Loads and validates YAML configuration files with environment variable overrides.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml

from .models import (
    AlertingConfig,
    CameraConfig,
    CamerasConfig,
    ROIsConfig,
    ServicesConfig,
)


class ConfigLoader:
    """
    Load and merge configuration from YAML files and environment variables.

    Args:
        config_dir: Directory containing YAML config files

    Example:
        >>> loader = ConfigLoader("configs/")
        >>> cameras = loader.load_cameras()
        >>> rois = loader.load_rois()
        >>> services = loader.load_services()
    """

    def __init__(self, config_dir: str | Path = "configs"):
        self.config_dir = Path(config_dir)

    def _load_yaml(self, filename: str) -> dict:
        """Load a YAML file from config directory."""
        filepath = self.config_dir / filename
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")

        with filepath.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        return data or {}

    def load_cameras(self) -> CamerasConfig:
        """
        Load cameras configuration from cameras.yaml.

        Environment overrides:
            CAMERA_{camera_id}_RTSP_URL - Override RTSP URL for specific camera

        Returns:
            Validated CamerasConfig
        """
        data = self._load_yaml("cameras.yaml")
        config = CamerasConfig(**data)

        # Apply environment variable overrides for RTSP URLs
        for camera in config.cameras:
            env_key = f"CAMERA_{camera.id}_RTSP_URL"
            if env_key in os.environ:
                camera.rtsp_url = os.environ[env_key]

        return config

    def load_rois(self) -> ROIsConfig:
        """
        Load ROI configuration from rois.yaml.

        Returns:
            Validated ROIsConfig
        """
        data = self._load_yaml("rois.yaml")
        return ROIsConfig(**data)

    def load_services(self) -> ServicesConfig:
        """
        Load services configuration from services.yaml.

        Environment overrides:
            INFERENCE_MODEL_PATH, INFERENCE_DEVICE, INFERENCE_CONFIDENCE
            STREAM_MANAGER_FRAME_BUFFER_SIZE, STREAM_MANAGER_RECONNECT_DELAY_SECONDS
            ALERTING_PROTOCOL, ALERTING_WEBHOOK_URL

        Returns:
            Validated ServicesConfig
        """
        data = self._load_yaml("services.yaml")
        config = ServicesConfig(**data)

        # Inference overrides
        if "INFERENCE_MODEL_PATH" in os.environ:
            config.inference.model_path = os.environ["INFERENCE_MODEL_PATH"]
        if "INFERENCE_DEVICE" in os.environ:
            config.inference.device = os.environ["INFERENCE_DEVICE"]
        if "INFERENCE_CONFIDENCE" in os.environ:
            config.inference.confidence = float(os.environ["INFERENCE_CONFIDENCE"])
        if "INFERENCE_IOU" in os.environ:
            config.inference.iou_threshold = float(os.environ["INFERENCE_IOU"])
        if "INFERENCE_BATCH_SIZE" in os.environ:
            config.inference.batch_size = int(os.environ["INFERENCE_BATCH_SIZE"])

        # Stream manager overrides
        if "STREAM_MANAGER_FRAME_BUFFER_SIZE" in os.environ:
            config.stream_manager.frame_buffer_size = int(
                os.environ["STREAM_MANAGER_FRAME_BUFFER_SIZE"]
            )
        if "STREAM_MANAGER_RECONNECT_DELAY_SECONDS" in os.environ:
            config.stream_manager.reconnect_delay_seconds = float(
                os.environ["STREAM_MANAGER_RECONNECT_DELAY_SECONDS"]
            )
        if "STREAM_MANAGER_MAX_RECONNECT_ATTEMPTS" in os.environ:
            config.stream_manager.max_reconnect_attempts = int(
                os.environ["STREAM_MANAGER_MAX_RECONNECT_ATTEMPTS"]
            )

        # Alerting overrides
        if "ALERTING_PROTOCOL" in os.environ:
            config.alerting.protocol = os.environ["ALERTING_PROTOCOL"]
        if "ALERTING_WEBHOOK_URL" in os.environ:
            config.alerting.webhook_url = os.environ["ALERTING_WEBHOOK_URL"]
        if "ALERTING_DEBOUNCE_WINDOW_SECONDS" in os.environ:
            config.alerting.debounce_window_seconds = float(
                os.environ["ALERTING_DEBOUNCE_WINDOW_SECONDS"]
            )
        if "ALERTING_MAX_ALERTS_PER_WINDOW" in os.environ:
            config.alerting.max_alerts_per_window = int(
                os.environ["ALERTING_MAX_ALERTS_PER_WINDOW"]
            )

        # Hikvision overrides
        if "ALERTING_HIKVISION_HOST" in os.environ:
            config.alerting.hikvision_host = os.environ["ALERTING_HIKVISION_HOST"]
        if "ALERTING_HIKVISION_PORT" in os.environ:
            config.alerting.hikvision_port = int(os.environ["ALERTING_HIKVISION_PORT"])
        if "ALERTING_HIKVISION_USERNAME" in os.environ:
            config.alerting.hikvision_username = os.environ["ALERTING_HIKVISION_USERNAME"]
        if "ALERTING_HIKVISION_PASSWORD" in os.environ:
            config.alerting.hikvision_password = os.environ["ALERTING_HIKVISION_PASSWORD"]

        return config

    def load_alerting(self) -> AlertingConfig:
        """
        Load alerting configuration from alerting.yaml.

        Returns:
            Validated AlertingConfig
        """
        data = self._load_yaml("alerting.yaml")
        config = AlertingConfig(**data)

        # Apply environment overrides (same as in load_services)
        if "ALERTING_PROTOCOL" in os.environ:
            config.protocol = os.environ["ALERTING_PROTOCOL"]
        if "ALERTING_WEBHOOK_URL" in os.environ:
            config.webhook_url = os.environ["ALERTING_WEBHOOK_URL"]
        if "ALERTING_DEBOUNCE_WINDOW_SECONDS" in os.environ:
            config.debounce_window_seconds = float(os.environ["ALERTING_DEBOUNCE_WINDOW_SECONDS"])
        if "ALERTING_MAX_ALERTS_PER_WINDOW" in os.environ:
            config.max_alerts_per_window = int(os.environ["ALERTING_MAX_ALERTS_PER_WINDOW"])
        if "ALERTING_HIKVISION_HOST" in os.environ:
            config.hikvision_host = os.environ["ALERTING_HIKVISION_HOST"]
        if "ALERTING_HIKVISION_PORT" in os.environ:
            config.hikvision_port = int(os.environ["ALERTING_HIKVISION_PORT"])
        if "ALERTING_HIKVISION_USERNAME" in os.environ:
            config.hikvision_username = os.environ["ALERTING_HIKVISION_USERNAME"]
        if "ALERTING_HIKVISION_PASSWORD" in os.environ:
            config.hikvision_password = os.environ["ALERTING_HIKVISION_PASSWORD"]

        return config

    def load_all(self) -> tuple[CamerasConfig, ROIsConfig, ServicesConfig]:
        """
        Load all configurations.

        Returns:
            Tuple of (cameras_config, rois_config, services_config)
        """
        return (
            self.load_cameras(),
            self.load_rois(),
            self.load_services(),
        )


def get_redis_url() -> str:
    """
    Get Redis URL from environment or use default.

    Returns:
        Redis connection URL
    """
    return os.environ.get("REDIS_URL", "redis://localhost:6379")


def get_log_level() -> str:
    """
    Get log level from environment or use default.

    Returns:
        Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    return os.environ.get("LOG_LEVEL", "INFO").upper()
