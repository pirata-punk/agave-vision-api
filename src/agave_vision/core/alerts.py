"""
Alert Data Structures and Debouncing

Provides alert event modeling, serialization, and deduplication logic.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from .inference import Detection


@dataclass
class AlertEvent:
    """
    Represents a single alert event triggered by a detection in a forbidden ROI.

    Attributes:
        camera_id: Camera that generated the alert
        timestamp: When the alert was generated (UTC)
        detection: The detection that triggered the alert
        roi_hit: Whether detection was in a forbidden ROI
        frame_id: Optional identifier for the source frame
        roi_name: Name of the ROI that was violated (NEW)
        violation_type: Type of violation (NEW: forbidden_class, unknown_object, low_confidence)
        allowed_classes: List of classes allowed in the ROI (NEW)
        strict_mode: Whether strict mode (whitelist) was enabled (NEW)
    """

    camera_id: str
    timestamp: datetime
    detection: Detection
    roi_hit: bool
    frame_id: Optional[str] = None
    roi_name: Optional[str] = None  # NEW
    violation_type: str = "forbidden_class"  # NEW: forbidden_class, unknown_object, low_confidence
    allowed_classes: Optional[List[str]] = None  # NEW
    strict_mode: bool = True  # NEW

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize alert to dictionary for JSON/Redis.

        Returns:
            Dictionary representation of alert
        """
        return {
            "camera_id": self.camera_id,
            "timestamp": self.timestamp.isoformat(),
            "detection": {
                "class_id": self.detection.class_id,
                "class_name": self.detection.class_name,
                "confidence": self.detection.confidence,
                "bbox": list(self.detection.bbox),
                "center": list(self.detection.center),
                "is_unknown": self.detection.is_unknown,  # NEW
            },
            "roi_hit": self.roi_hit,
            "frame_id": self.frame_id,
            "roi_name": self.roi_name,  # NEW
            "violation_type": self.violation_type,  # NEW
            "allowed_classes": self.allowed_classes,  # NEW
            "strict_mode": self.strict_mode,  # NEW
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AlertEvent":
        """
        Deserialize alert from dictionary.

        Args:
            data: Dictionary from to_dict() or JSON

        Returns:
            AlertEvent object
        """
        det_data = data["detection"]
        detection = Detection(
            class_id=det_data["class_id"],
            class_name=det_data["class_name"],
            confidence=det_data["confidence"],
            bbox=tuple(det_data["bbox"]),
            center=tuple(det_data["center"]),
            is_unknown=det_data.get("is_unknown", False),  # NEW
        )

        return cls(
            camera_id=data["camera_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            detection=detection,
            roi_hit=data["roi_hit"],
            frame_id=data.get("frame_id"),
            roi_name=data.get("roi_name"),  # NEW
            violation_type=data.get("violation_type", "forbidden_class"),  # NEW
            allowed_classes=data.get("allowed_classes"),  # NEW
            strict_mode=data.get("strict_mode", True),  # NEW
        )


class AlertDebouncer:
    """
    Debounce alerts to prevent spam from the same camera.

    Tracks recent alerts per camera and limits emission rate based on
    a sliding time window.

    Args:
        window_seconds: Time window for deduplication (seconds)
        max_per_window: Maximum alerts allowed per camera per window

    Example:
        >>> debouncer = AlertDebouncer(window_seconds=5.0, max_per_window=1)
        >>> if debouncer.should_emit(alert):
        ...     send_alert_to_downstream(alert)
    """

    def __init__(self, window_seconds: float = 5.0, max_per_window: int = 1):
        self.window = timedelta(seconds=window_seconds)
        self.max_per_window = max_per_window
        self.recent_alerts: Dict[str, List[datetime]] = defaultdict(list)

    def should_emit(self, alert: AlertEvent) -> bool:
        """
        Check if alert should be emitted based on recent history.

        Args:
            alert: Alert event to check

        Returns:
            True if alert should be emitted, False if it should be suppressed
        """
        camera_id = alert.camera_id
        now = alert.timestamp

        # Clean old alerts outside the window
        cutoff = now - self.window
        self.recent_alerts[camera_id] = [
            ts for ts in self.recent_alerts[camera_id] if ts > cutoff
        ]

        # Check if under limit
        if len(self.recent_alerts[camera_id]) < self.max_per_window:
            self.recent_alerts[camera_id].append(now)
            return True

        return False

    def reset(self, camera_id: Optional[str] = None) -> None:
        """
        Reset debounce state.

        Args:
            camera_id: Reset for specific camera, or all cameras if None
        """
        if camera_id is None:
            self.recent_alerts.clear()
        else:
            self.recent_alerts.pop(camera_id, None)

    def get_recent_count(self, camera_id: str) -> int:
        """
        Get number of recent alerts for a camera within the window.

        Args:
            camera_id: Camera identifier

        Returns:
            Number of recent alerts
        """
        return len(self.recent_alerts.get(camera_id, []))

    def __repr__(self) -> str:
        return (
            f"AlertDebouncer(window={self.window.total_seconds()}s, "
            f"max_per_window={self.max_per_window})"
        )
