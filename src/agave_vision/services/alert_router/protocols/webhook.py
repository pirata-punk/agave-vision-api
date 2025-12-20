"""
Webhook Protocol Adapter

Sends alerts via HTTP POST to a webhook URL.
"""

import httpx

from agave_vision.core.alerts import AlertEvent
from agave_vision.utils.logging import get_logger

from .base import ProtocolAdapter

logger = get_logger(__name__)


class WebhookAdapter(ProtocolAdapter):
    """
    Send alerts via HTTP POST to webhook URL.

    Args:
        webhook_url: Destination webhook URL
        timeout: HTTP request timeout in seconds
        retry_attempts: Number of retry attempts on failure

    Example:
        >>> adapter = WebhookAdapter("https://example.com/api/alerts")
        >>> await adapter.send_alert(alert)
    """

    def __init__(
        self,
        webhook_url: str,
        timeout: float = 5.0,
        retry_attempts: int = 3,
    ):
        self.webhook_url = webhook_url
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.client = httpx.AsyncClient(timeout=timeout)

    async def send_alert(self, alert: AlertEvent) -> None:
        """
        Send alert to webhook.

        Args:
            alert: Alert event to send

        Raises:
            httpx.HTTPError: If webhook request fails after retries
        """
        alert_dict = alert.to_dict()

        for attempt in range(self.retry_attempts):
            try:
                response = await self.client.post(
                    self.webhook_url,
                    json=alert_dict,
                )
                response.raise_for_status()

                logger.info(
                    f"Alert sent to webhook: {alert.camera_id} - {alert.detection.class_name}"
                )
                return

            except httpx.HTTPError as e:
                logger.warning(
                    f"Webhook attempt {attempt + 1}/{self.retry_attempts} failed: {e}"
                )
                if attempt == self.retry_attempts - 1:
                    logger.error(f"Webhook failed after {self.retry_attempts} attempts")
                    raise

    async def close(self) -> None:
        """Close HTTP client."""
        await self.client.aclose()
