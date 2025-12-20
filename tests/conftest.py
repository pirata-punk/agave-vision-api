"""
Pytest Configuration and Fixtures

Shared test fixtures for unit and integration tests.
"""

import numpy as np
import pytest
from pathlib import Path

from agave_vision.config.loader import ConfigLoader
from agave_vision.config.models import CameraConfig, CameraROIConfig, ROIPolygonConfig
from agave_vision.core.inference import Detection


@pytest.fixture
def test_image():
    """Generate a test image (640x640 RGB)."""
    return np.zeros((640, 640, 3), dtype=np.uint8)


@pytest.fixture
def test_detection():
    """Create a sample detection."""
    return Detection(
        class_id=0,
        class_name="object",
        confidence=0.85,
        bbox=(100.0, 100.0, 200.0, 200.0),
        center=(150.0, 150.0),
    )


@pytest.fixture
def test_roi_polygon():
    """Create a test ROI polygon."""
    return ROIPolygonConfig(
        points=[(100, 100), (300, 100), (300, 300), (100, 300)],
        name="test_zone",
    )


@pytest.fixture
def test_camera_roi_config(test_roi_polygon):
    """Create a test camera ROI configuration."""
    return CameraROIConfig(
        camera_id="test_camera",
        forbidden_rois=[test_roi_polygon],
        allowed_classes=["pine", "worker"],
        alert_classes=["object"],
    )


@pytest.fixture
def test_camera_config():
    """Create a test camera configuration."""
    return CameraConfig(
        id="test_camera",
        name="Test Camera",
        rtsp_url="rtsp://test:test@localhost:554/stream",
        enabled=True,
        fps_target=5.0,
    )


@pytest.fixture
def config_dir(tmp_path):
    """Create a temporary config directory with test YAML files."""
    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    # Create cameras.yaml
    cameras_yaml = config_dir / "cameras.yaml"
    cameras_yaml.write_text("""
cameras:
  - id: test_camera
    name: "Test Camera"
    rtsp_url: "rtsp://test:test@localhost:554/stream"
    enabled: true
    fps_target: 5.0
""")

    # Create rois.yaml
    rois_yaml = config_dir / "rois.yaml"
    rois_yaml.write_text("""
cameras:
  - camera_id: test_camera
    forbidden_rois:
      - name: test_zone
        points:
          - [100, 100]
          - [300, 100]
          - [300, 300]
          - [100, 300]
    allowed_classes: [pine, worker]
    alert_classes: [object]
""")

    # Create services.yaml
    services_yaml = config_dir / "services.yaml"
    services_yaml.write_text("""
inference:
  model_path: "models/test.pt"
  confidence: 0.25
  iou_threshold: 0.45
  device: "cpu"
  batch_size: 1
  warmup_iterations: 0
  image_size: 640

stream_manager:
  frame_buffer_size: 10
  reconnect_delay_seconds: 5.0
  max_reconnect_attempts: 3
  read_timeout_seconds: 30.0

alerting:
  debounce_window_seconds: 5.0
  max_alerts_per_window: 1
  protocol: "stdout"
  webhook_url: null
  hikvision_host: null
  redis_stream_name: "alerts"
  redis_consumer_group: "alert_router"
  redis_max_stream_length: 10000
""")

    return config_dir
