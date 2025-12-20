"""
Unit Tests for Alerts Module

Tests alert data structures and debouncing logic.
"""

from datetime import datetime, timedelta

import pytest

from agave_vision.core.alerts import AlertEvent, AlertDebouncer
from agave_vision.core.inference import Detection


class TestAlertEvent:
    """Test alert event data structure."""

    def test_alert_creation(self, test_detection):
        """Test creating an alert event."""
        alert = AlertEvent(
            camera_id="test_camera",
            timestamp=datetime.utcnow(),
            detection=test_detection,
            roi_hit=True,
            frame_id="frame_001",
        )

        assert alert.camera_id == "test_camera"
        assert alert.detection.class_name == "object"
        assert alert.roi_hit is True
        assert alert.frame_id == "frame_001"

    def test_alert_serialization(self, test_detection):
        """Test alert serialization to dict."""
        now = datetime.utcnow()
        alert = AlertEvent(
            camera_id="test_camera",
            timestamp=now,
            detection=test_detection,
            roi_hit=True,
        )

        alert_dict = alert.to_dict()

        assert alert_dict["camera_id"] == "test_camera"
        assert alert_dict["timestamp"] == now.isoformat()
        assert alert_dict["detection"]["class_name"] == "object"
        assert alert_dict["roi_hit"] is True

    def test_alert_deserialization(self, test_detection):
        """Test alert deserialization from dict."""
        now = datetime.utcnow()
        alert = AlertEvent(
            camera_id="test_camera",
            timestamp=now,
            detection=test_detection,
            roi_hit=True,
        )

        alert_dict = alert.to_dict()
        restored_alert = AlertEvent.from_dict(alert_dict)

        assert restored_alert.camera_id == alert.camera_id
        assert restored_alert.detection.class_name == alert.detection.class_name
        assert restored_alert.roi_hit == alert.roi_hit


class TestAlertDebouncer:
    """Test alert debouncing functionality."""

    def test_first_alert_allowed(self, test_detection):
        """Test that first alert is always allowed."""
        debouncer = AlertDebouncer(window_seconds=5.0, max_per_window=1)

        alert = AlertEvent(
            camera_id="test_camera",
            timestamp=datetime.utcnow(),
            detection=test_detection,
            roi_hit=True,
        )

        assert debouncer.should_emit(alert) is True

    def test_debounce_suppression(self, test_detection):
        """Test that rapid alerts are suppressed."""
        debouncer = AlertDebouncer(window_seconds=5.0, max_per_window=1)

        now = datetime.utcnow()

        # First alert allowed
        alert1 = AlertEvent(
            camera_id="test_camera",
            timestamp=now,
            detection=test_detection,
            roi_hit=True,
        )
        assert debouncer.should_emit(alert1) is True

        # Second alert suppressed (within window)
        alert2 = AlertEvent(
            camera_id="test_camera",
            timestamp=now + timedelta(seconds=1),
            detection=test_detection,
            roi_hit=True,
        )
        assert debouncer.should_emit(alert2) is False

    def test_debounce_window_expiry(self, test_detection):
        """Test that alerts are allowed after window expires."""
        debouncer = AlertDebouncer(window_seconds=5.0, max_per_window=1)

        now = datetime.utcnow()

        # First alert
        alert1 = AlertEvent(
            camera_id="test_camera",
            timestamp=now,
            detection=test_detection,
            roi_hit=True,
        )
        assert debouncer.should_emit(alert1) is True

        # Second alert after window expires
        alert2 = AlertEvent(
            camera_id="test_camera",
            timestamp=now + timedelta(seconds=6),
            detection=test_detection,
            roi_hit=True,
        )
        assert debouncer.should_emit(alert2) is True

    def test_debounce_per_camera(self, test_detection):
        """Test that debouncing is per-camera."""
        debouncer = AlertDebouncer(window_seconds=5.0, max_per_window=1)

        now = datetime.utcnow()

        # Alert from camera 1
        alert1 = AlertEvent(
            camera_id="camera_1",
            timestamp=now,
            detection=test_detection,
            roi_hit=True,
        )
        assert debouncer.should_emit(alert1) is True

        # Alert from camera 2 (different camera, should be allowed)
        alert2 = AlertEvent(
            camera_id="camera_2",
            timestamp=now,
            detection=test_detection,
            roi_hit=True,
        )
        assert debouncer.should_emit(alert2) is True

    def test_reset_debouncer(self, test_detection):
        """Test resetting debouncer state."""
        debouncer = AlertDebouncer(window_seconds=5.0, max_per_window=1)

        now = datetime.utcnow()

        # First alert
        alert1 = AlertEvent(
            camera_id="test_camera",
            timestamp=now,
            detection=test_detection,
            roi_hit=True,
        )
        debouncer.should_emit(alert1)

        # Reset
        debouncer.reset("test_camera")

        # Alert should be allowed after reset
        alert2 = AlertEvent(
            camera_id="test_camera",
            timestamp=now + timedelta(seconds=1),
            detection=test_detection,
            roi_hit=True,
        )
        assert debouncer.should_emit(alert2) is True
