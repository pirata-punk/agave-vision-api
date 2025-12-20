"""
Stdout Protocol Adapter

Prints alerts to stdout as JSON (for development/testing).
"""

import json

from agave_vision.core.alerts import AlertEvent
from agave_vision.utils.logging import get_logger

from .base import ProtocolAdapter

logger = get_logger(__name__)


class StdoutAdapter(ProtocolAdapter):
    """
    Print alerts to stdout as JSON.

    Useful for development, testing, and piping to other tools.
    """

    async def send_alert(self, alert: AlertEvent) -> None:
        """
        Print alert to stdout.

        Args:
            alert: Alert event to print
        """
        alert_dict = alert.to_dict()
        print(json.dumps(alert_dict, indent=2))
        logger.info(f"Alert sent to stdout: {alert.camera_id} - {alert.detection.class_name}")
