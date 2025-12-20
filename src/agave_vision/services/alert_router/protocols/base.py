"""
Protocol Adapter Base

Abstract base class for alert delivery protocol adapters.
"""

from abc import ABC, abstractmethod

from agave_vision.core.alerts import AlertEvent


class ProtocolAdapter(ABC):
    """
    Abstract base class for alert protocol adapters.

    Subclasses implement specific delivery mechanisms (stdout, webhook, Hikvision, etc.).
    """

    @abstractmethod
    async def send_alert(self, alert: AlertEvent) -> None:
        """
        Send alert to downstream system.

        Args:
            alert: Alert event to send

        Raises:
            Exception: If alert delivery fails
        """
        pass

    async def close(self) -> None:
        """
        Cleanup resources (optional).

        Override if adapter needs cleanup on shutdown.
        """
        pass
