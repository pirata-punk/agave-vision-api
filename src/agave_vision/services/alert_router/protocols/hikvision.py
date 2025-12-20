"""
Hikvision Protocol Adapter

Sends alerts to Hikvision NVR/VMS via proprietary protocol (placeholder).
"""

from agave_vision.core.alerts import AlertEvent
from agave_vision.utils.logging import get_logger

from .base import ProtocolAdapter

logger = get_logger(__name__)


class HikvisionAdapter(ProtocolAdapter):
    """
    Send alerts to Hikvision NVR/VMS via proprietary protocol.

    NOTE: This is a placeholder implementation. The actual implementation
    depends on Hikvision's specific API/SDK requirements.

    Args:
        host: Hikvision NVR/VMS host
        port: Hikvision API port
        username: Authentication username
        password: Authentication password

    TODO: Implement actual Hikvision protocol/SDK integration
    """

    def __init__(
        self,
        host: str,
        port: int = 8000,
        username: str = "",
        password: str = "",
    ):
        self.host = host
        self.port = port
        self.username = username
        self.password = password

        logger.warning(
            "HikvisionAdapter is a placeholder. Implement actual protocol integration."
        )

    async def send_alert(self, alert: AlertEvent) -> None:
        """
        Send alert to Hikvision system.

        Args:
            alert: Alert event to send

        Raises:
            NotImplementedError: Placeholder - needs actual implementation
        """
        # TODO: Implement Hikvision protocol
        # This will depend on Hikvision's API/SDK documentation
        # Possible approaches:
        # 1. HTTP API (if available)
        # 2. ISAPI (Hikvision's XML-based API)
        # 3. SDK integration (if Python bindings exist)

        logger.error(
            f"Hikvision protocol not implemented. Alert NOT sent: "
            f"{alert.camera_id} - {alert.detection.class_name}"
        )

        raise NotImplementedError(
            "Hikvision protocol adapter requires implementation. "
            "See Hikvision API/SDK documentation."
        )
