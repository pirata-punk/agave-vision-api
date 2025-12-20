"""
Unit Tests for Configuration Module

Tests Pydantic models and configuration loading.
"""

import os
import pytest

from agave_vision.config.loader import ConfigLoader
from agave_vision.config.models import (
    CameraConfig,
    CamerasConfig,
    InferenceConfig,
    ROIsConfig,
    ServicesConfig,
)


class TestConfigModels:
    """Test Pydantic configuration models."""

    def test_camera_config_validation(self):
        """Test camera configuration validation."""
        config = CameraConfig(
            id="test",
            name="Test Camera",
            rtsp_url="rtsp://test:test@localhost:554/stream",
            enabled=True,
            fps_target=5.0,
        )

        assert config.id == "test"
        assert config.fps_target == 5.0

    def test_camera_config_fps_validation(self):
        """Test FPS validation (must be positive)."""
        with pytest.raises(ValueError):
            CameraConfig(
                id="test",
                name="Test",
                rtsp_url="rtsp://localhost/stream",
                fps_target=0.0,  # Invalid: too low
            )

    def test_inference_config_device_validation(self):
        """Test device validation."""
        # Valid devices
        for device in ["cuda", "cpu", "mps", "auto"]:
            config = InferenceConfig(
                model_path="models/test.pt",
                device=device,
            )
            assert config.device == device

        # Invalid device
        with pytest.raises(ValueError):
            InferenceConfig(
                model_path="models/test.pt",
                device="invalid_device",
            )


class TestConfigLoader:
    """Test configuration loader."""

    def test_load_cameras(self, config_dir):
        """Test loading cameras configuration."""
        loader = ConfigLoader(config_dir)
        cameras_config = loader.load_cameras()

        assert isinstance(cameras_config, CamerasConfig)
        assert len(cameras_config.cameras) == 1
        assert cameras_config.cameras[0].id == "test_camera"

    def test_load_rois(self, config_dir):
        """Test loading ROIs configuration."""
        loader = ConfigLoader(config_dir)
        rois_config = loader.load_rois()

        assert isinstance(rois_config, ROIsConfig)
        assert len(rois_config.cameras) == 1
        assert rois_config.cameras[0].camera_id == "test_camera"

    def test_load_services(self, config_dir):
        """Test loading services configuration."""
        loader = ConfigLoader(config_dir)
        services_config = loader.load_services()

        assert isinstance(services_config, ServicesConfig)
        assert services_config.inference.model_path == "models/test.pt"
        assert services_config.inference.device == "cpu"

    def test_env_var_override(self, config_dir, monkeypatch):
        """Test environment variable overrides."""
        loader = ConfigLoader(config_dir)

        # Set environment variable
        monkeypatch.setenv("INFERENCE_DEVICE", "cuda")
        monkeypatch.setenv("INFERENCE_CONFIDENCE", "0.5")

        services_config = loader.load_services()

        assert services_config.inference.device == "cuda"
        assert services_config.inference.confidence == 0.5

    def test_camera_rtsp_url_override(self, config_dir, monkeypatch):
        """Test camera RTSP URL override."""
        loader = ConfigLoader(config_dir)

        # Set environment variable for camera RTSP URL
        override_url = "rtsp://override:pass@192.168.1.10:554/stream"
        monkeypatch.setenv("CAMERA_test_camera_RTSP_URL", override_url)

        cameras_config = loader.load_cameras()

        assert cameras_config.cameras[0].rtsp_url == override_url
