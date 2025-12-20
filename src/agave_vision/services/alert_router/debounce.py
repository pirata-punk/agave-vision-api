"""
Alert Debouncing

Prevents alert spam by rate-limiting alerts per camera.
"""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from agave_vision.core.alerts import AlertEvent
from agave_vision.utils.logging import get_logger

logger = get_logger(__name__)


class AlertDebouncer:
    """
    Debounce alerts to prevent spam.

    Tracks recent alerts per camera and limits emission rate based on
    a sliding time window.

    Args:
        window_seconds: Time window for deduplication (seconds)
        max_per_window: Maximum alerts allowed per camera per window

    Example:
        >>> debouncer = AlertDebouncer(window_seconds=5.0, max_per_window=1)
        >>> if debouncer.should_emit(alert):
        ...     await send_alert(alert)
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
            logger.debug(
                f"Alert allowed for {camera_id} "
                f"({len(self.recent_alerts[camera_id])}/{self.max_per_window})"
            )
            return True

        logger.debug(
            f"Alert suppressed for {camera_id} (debounce limit reached: "
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
