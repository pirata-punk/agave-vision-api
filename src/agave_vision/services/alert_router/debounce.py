"""
Alert Debouncing

Prevents alert spam by rate-limiting alerts per camera, class, and ROI zone.
"""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from agave_vision.core.alerts import AlertEvent
from agave_vision.utils.logging import get_logger

logger = get_logger(__name__)


class AlertDebouncer:
    """
    Debounce alerts to prevent spam.

    Tracks recent alerts per (camera_id, class_name, roi_name) tuple and limits
    emission rate based on a sliding time window. This provides more granular
    deduplication to avoid inflating statistics while still catching different
    types of violations.

    Args:
        window_seconds: Time window for deduplication (seconds)
        max_per_window: Maximum alerts allowed per unique (camera, class, ROI) per window

    Example:
        >>> debouncer = AlertDebouncer(window_seconds=5.0, max_per_window=1)
        >>> if debouncer.should_emit(alert):
        ...     await send_alert(alert)
    """

    def __init__(self, window_seconds: float = 5.0, max_per_window: int = 1):
        self.window = timedelta(seconds=window_seconds)
        self.max_per_window = max_per_window
        # Changed: Track by (camera_id, class_name, roi_name) instead of just camera_id
        self.recent_alerts: Dict[Tuple[str, str, str], List[datetime]] = defaultdict(list)

    def should_emit(self, alert: AlertEvent) -> bool:
        """
        Check if alert should be emitted based on recent history.

        Args:
            alert: Alert event to check

        Returns:
            True if alert should be emitted, False if it should be suppressed
        """
        # Create granular key: (camera_id, class_name, roi_name)
        # This allows different classes or ROI zones to have independent limits
        alert_key = (
            alert.camera_id,
            alert.detection.class_name,
            alert.roi_name or "unknown_roi"
        )
        now = alert.timestamp

        # Clean old alerts outside the window
        cutoff = now - self.window
        self.recent_alerts[alert_key] = [
            ts for ts in self.recent_alerts[alert_key] if ts > cutoff
        ]

        # Check if under limit
        if len(self.recent_alerts[alert_key]) < self.max_per_window:
            self.recent_alerts[alert_key].append(now)
            logger.debug(
                f"Alert allowed for {alert_key} "
                f"({len(self.recent_alerts[alert_key])}/{self.max_per_window})"
            )
            return True

        logger.debug(
            f"Alert suppressed for {alert_key} (debounce limit reached: "
            f"{self.max_per_window} per {self.window.total_seconds()}s)"
        )
        return False

    def reset(self, camera_id: Optional[str] = None) -> None:
        """
        Reset debounce state.

        Args:
            camera_id: Reset for specific camera, or all cameras if None
        """
        if camera_id is None:
            self.recent_alerts.clear()
            logger.info("Reset debounce state for all cameras")
        else:
            self.recent_alerts.pop(camera_id, None)
            logger.info(f"Reset debounce state for camera {camera_id}")

    def get_recent_count(self, camera_id: str, class_name: Optional[str] = None,
                         roi_name: Optional[str] = None) -> int:
        """
        Get number of recent alerts within the window.

        Args:
            camera_id: Camera identifier
            class_name: Optional class name filter
            roi_name: Optional ROI name filter

        Returns:
            Number of recent alerts (total for camera if filters not specified,
            or for specific (camera, class, ROI) tuple if specified)
        """
        if class_name and roi_name:
            alert_key = (camera_id, class_name, roi_name)
            return len(self.recent_alerts.get(alert_key, []))
        else:
            # Return total count across all keys for this camera
            total = 0
            for key in self.recent_alerts:
                if key[0] == camera_id:
                    total += len(self.recent_alerts[key])
            return total

    def __repr__(self) -> str:
        return (
            f"AlertDebouncer(window={self.window.total_seconds()}s, "
            f"max_per_window={self.max_per_window})"
        )
