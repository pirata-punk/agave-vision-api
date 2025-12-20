"""
Redis Alert Publisher

Publishes alerts to Redis Streams for consumption by alert router.
"""

import json
from typing import Optional

try:
    import redis.asyncio as redis
except ImportError:
    import redis  # type: ignore

from agave_vision.core.alerts import AlertEvent
from agave_vision.utils.logging import get_logger

logger = get_logger(__name__)


class RedisPublisher:
    """
    Publish alerts to Redis Streams.

    Args:
        redis_url: Redis connection URL
        stream_name: Redis stream name
        max_stream_length: Maximum stream length (for XTRIM)

    Example:
        >>> publisher = RedisPublisher("redis://localhost:6379")
        >>> await publisher.publish_alert(alert_event)
    """

    def __init__(
        self,
        redis_url: str = "redis://localhost:6379",
        stream_name: str = "alerts",
        max_stream_length: int = 10000,
    ):
        self.redis_url = redis_url
        self.stream_name = stream_name
        self.max_stream_length = max_stream_length
        self.redis_client: Optional[redis.Redis] = None

    async def connect(self):
        """Establish Redis connection."""
        if self.redis_client is None:
            self.redis_client = await redis.from_url(self.redis_url)
            logger.info(f"Connected to Redis: {self.redis_url}")

    async def publish_alert(self, alert: AlertEvent) -> str:
        """
        Publish alert to Redis stream.

        Args:
            alert: Alert event to publish

        Returns:
            Message ID from Redis
        """
        if self.redis_client is None:
            await self.connect()

        # Serialize alert to JSON
        alert_data = json.dumps(alert.to_dict())

        # Add to stream with automatic trimming
        message_id = await self.redis_client.xadd(
            self.stream_name,
            {"data": alert_data},
            maxlen=self.max_stream_length,
            approximate=True,  # Use approximate trimming for performance
        )

        logger.debug(
            f"Published alert for camera {alert.camera_id} (msg_id: {message_id.decode()})"
        )

        return message_id.decode() if isinstance(message_id, bytes) else message_id

    async def close(self):
        """Close Redis connection."""
        if self.redis_client is not None:
            await self.redis_client.close()
            self.redis_client = None
            logger.info("Closed Redis connection")

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
