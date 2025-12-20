"""
Unit Tests for ROI Module

Tests ROI polygon geometry and alert filtering logic.
"""

import numpy as np
import pytest

from agave_vision.core.roi import ROIPolygon, CameraROI
from agave_vision.core.inference import Detection


class TestROIPolygon:
    """Test ROI polygon functionality."""

    def test_polygon_creation(self):
        """Test creating an ROI polygon."""
        points = np.array([[0, 0], [100, 0], [100, 100], [0, 100]], dtype=np.int32)
        polygon = ROIPolygon(points=points, name="test")

        assert polygon.name == "test"
        assert isinstance(polygon.points, np.ndarray)
        assert polygon.points.shape == (4, 2)

    def test_point_inside_polygon(self):
        """Test point-in-polygon detection."""
        points = [[0, 0], [100, 0], [100, 100], [0, 100]]
        polygon = ROIPolygon(points=points)

        # Point inside
        assert polygon.contains_point((50, 50)) is True

        # Point outside
        assert polygon.contains_point((150, 150)) is False

        # Point on boundary
        assert polygon.contains_point((0, 0)) is True


class TestCameraROI:
    """Test camera ROI configuration and filtering."""

    def test_should_alert_object_in_zone(self):
        """Test alert for object class in forbidden zone."""
        polygon = ROIPolygon(points=[[0, 0], [100, 0], [100, 100], [0, 100]])
        camera_roi = CameraROI(
            camera_id="test",
            forbidden_zones=[polygon],
            allowed_classes={"pine", "worker"},
            alert_classes={"object"},
        )

        detection = Detection(
            class_id=0,
            class_name="object",
            confidence=0.9,
            bbox=(10, 10, 20, 20),
            center=(50, 50),  # Inside zone
        )

        assert camera_roi.should_alert(detection) is True

    def test_no_alert_allowed_class(self):
        """Test no alert for allowed class in forbidden zone."""
        polygon = ROIPolygon(points=[[0, 0], [100, 0], [100, 100], [0, 100]])
        camera_roi = CameraROI(
            camera_id="test",
            forbidden_zones=[polygon],
            allowed_classes={"pine", "worker"},
            alert_classes={"object"},
        )

        detection = Detection(
            class_id=1,
            class_name="pine",
            confidence=0.9,
            bbox=(10, 10, 20, 20),
            center=(50, 50),  # Inside zone but allowed
        )

        assert camera_roi.should_alert(detection) is False

    def test_no_alert_outside_zone(self):
        """Test no alert for object outside forbidden zone."""
        polygon = ROIPolygon(points=[[0, 0], [100, 0], [100, 100], [0, 100]])
        camera_roi = CameraROI(
            camera_id="test",
            forbidden_zones=[polygon],
            allowed_classes={"pine", "worker"},
            alert_classes={"object"},
        )

        detection = Detection(
            class_id=0,
            class_name="object",
            confidence=0.9,
            bbox=(150, 150, 200, 200),
            center=(175, 175),  # Outside zone
        )

        assert camera_roi.should_alert(detection) is False

    def test_filter_detections(self):
        """Test filtering multiple detections."""
        polygon = ROIPolygon(points=[[0, 0], [100, 0], [100, 100], [0, 100]])
        camera_roi = CameraROI(
            camera_id="test",
            forbidden_zones=[polygon],
            allowed_classes={"pine", "worker"},
            alert_classes={"object"},
        )

        detections = [
            Detection(0, "object", 0.9, (10, 10, 20, 20), (50, 50)),  # Alert
            Detection(1, "pine", 0.9, (10, 10, 20, 20), (50, 50)),  # Allowed
            Detection(0, "object", 0.9, (150, 150, 200, 200), (175, 175)),  # Outside
        ]

        alert_detections = camera_roi.filter_detections(detections)

        assert len(alert_detections) == 1
        assert alert_detections[0].class_name == "object"
        assert alert_detections[0].center == (50, 50)
