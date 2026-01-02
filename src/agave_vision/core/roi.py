"""
ROI (Region of Interest) Management

Handles ROI polygon geometry, point-in-polygon testing, and alert filtering logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import yaml

from .inference import Detection


@dataclass
class ROIPolygon:
    """Represents a single ROI polygon."""

    points: np.ndarray  # Nx2 array of (x, y) coordinates
    name: Optional[str] = None

    def __post_init__(self):
        """Ensure points are a numpy array."""
        if not isinstance(self.points, np.ndarray):
            self.points = np.array(self.points, dtype=np.int32)

    def contains_point(self, point: Tuple[float, float]) -> bool:
        """
        Check if a point is inside this polygon.

        Args:
            point: (x, y) coordinates

        Returns:
            True if point is inside or on the polygon boundary
        """
        result = cv2.pointPolygonTest(self.points, point, measureDist=False)
        return result >= 0  # >= 0 means inside or on boundary


@dataclass
class CameraROI:
    """ROI configuration for a single camera."""

    camera_id: str
    forbidden_zones: List[ROIPolygon]
    allowed_classes: Set[str] = field(default_factory=lambda: {"pine", "worker"})
    alert_classes: Set[str] = field(default_factory=lambda: {"object"})  # Legacy, kept for backward compatibility
    strict_mode: bool = True  # New: Whitelist approach - alert on anything NOT in allowed_classes

    def should_alert(self, detection: Detection) -> bool:
        """
        Determine if a detection should trigger an alert.

        STRICT MODE (default, whitelist approach):
            Alert is triggered if:
            1. Detection class is NOT in allowed_classes
            2. Detection center point is inside any forbidden zone
            3. This includes unknown/low-confidence detections

        LEGACY MODE (strict_mode=False):
            Alert is triggered if:
            1. Detection class is in alert_classes (e.g., "object")
            2. Detection class is NOT in allowed_classes
            3. Detection center point is inside any forbidden zone

        Args:
            detection: Detection object from inference

        Returns:
            True if alert should be triggered
        """
        # Strict mode: Alert on ANYTHING that is NOT in allowed_classes
        if self.strict_mode:
            # Check if class is explicitly allowed
            if detection.class_name in self.allowed_classes:
                return False

            # Check if detection is in any forbidden zone
            for zone in self.forbidden_zones:
                if zone.contains_point(detection.center):
                    return True

            return False

        # Legacy mode: Only alert on specific alert_classes
        else:
            # Check if class triggers alerts
            if detection.class_name not in self.alert_classes:
                return False

            # Check if class is explicitly allowed (override)
            if detection.class_name in self.allowed_classes:
                return False

            # Check if detection is in any forbidden zone
            for zone in self.forbidden_zones:
                if zone.contains_point(detection.center):
                    return True

            return False

    def filter_detections(self, detections: List[Detection]) -> List[Detection]:
        """
        Filter detections to only those that should trigger alerts.

        Args:
            detections: List of detections from inference

        Returns:
            List of detections that trigger alerts
        """
        return [det for det in detections if self.should_alert(det)]


class ROIManager:
    """
    Manages ROI configurations for multiple cameras.

    Args:
        config_path: Path to rois.yaml configuration file

    Example:
        >>> manager = ROIManager("configs/rois.yaml")
        >>> camera_roi = manager.get_camera_rois("cam_nave3_hornos")
        >>> alert_detections = camera_roi.filter_detections(detections)
    """

    def __init__(self, config_path: str | Path):
        self.config_path = Path(config_path)
        self.camera_rois: Dict[str, CameraROI] = {}
        self._load_config()

    def _load_config(self) -> None:
        """Load ROI configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"ROI config file not found: {self.config_path}")

        with self.config_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        cameras = data.get("cameras", [])
        for cam_config in cameras:
            camera_id = cam_config["camera_id"]

            # Parse forbidden zones
            forbidden_zones = []
            for roi in cam_config.get("forbidden_rois", []):
                points = np.array(roi["points"], dtype=np.int32)
                name = roi.get("name")
                forbidden_zones.append(ROIPolygon(points=points, name=name))

            # Parse allowed and alert classes
            allowed_classes = set(cam_config.get("allowed_classes", ["pine", "worker"]))
            alert_classes = set(cam_config.get("alert_classes", ["object"]))
            strict_mode = cam_config.get("strict_mode", True)  # Default to strict mode

            self.camera_rois[camera_id] = CameraROI(
                camera_id=camera_id,
                forbidden_zones=forbidden_zones,
                allowed_classes=allowed_classes,
                alert_classes=alert_classes,
                strict_mode=strict_mode,
            )

    def get_camera_rois(self, camera_id: str) -> Optional[CameraROI]:
        """
        Get ROI configuration for a specific camera.

        Args:
            camera_id: Camera identifier

        Returns:
            CameraROI object or None if camera not configured
        """
        return self.camera_rois.get(camera_id)

    def reload_config(self) -> None:
        """Reload ROI configuration from disk."""
        self.camera_rois.clear()
        self._load_config()

    def __repr__(self) -> str:
        return f"ROIManager(cameras={list(self.camera_rois.keys())})"
