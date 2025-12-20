"""
Redis Streams Consumer

Consumes alerts from Redis Streams and routes to protocol adapters.
"""

import json
from typing import Optional

try:
    import redis.asyncio as redis
except ImportError:
    import redis  # type: ignore

from agave_vision.core.alerts import AlertEvent
from agave_vision.utils.logging import get_logger

from .debounce import AlertDebouncer
from .protocols.base import ProtocolAdapter

logger = get_logger(__name__)


class RedisConsumer:
    """
    Consume alerts from Redis Streams and route to protocol adapter.

    Args:
        redis_url: Redis connection URL
        stream_name: Redis stream name to consume from
        consumer_group: Consumer group name
        adapter: Protocol adapter for alert delivery
        debouncer: Alert debouncer (optional)

    Example:
        >>> consumer = RedisConsumer(
        ...     redis_url="redis://localhost:6379",
        ...     adapter=StdoutAdapter(),
        ...     debouncer=AlertDebouncer()
        ... )
        >>> await consumer.run()
    """

    def __init__(
        self,
        redis_url: str,
        stream_name: str,
        consumer_group: str,
        adapter: ProtocolAdapter,
        debouncer: Optional[AlertDebouncer] = None,
    ):
        self.redis_url = redis_url
        self.stream_name = stream_name
        self.consumer_group = consumer_group
        self.consumer_name = "router-1"  # Could be hostname or random ID
        self.adapter = adapter
        self.debouncer = debouncer
        self.redis_client: Optional[redis.Redis] = None

    async def connect(self):
        """Establish Redis connection and create consumer group."""
        self.redis_client = await redis.from_url(self.redis_url)
        logger.info(f"Connected to Redis: {self.redis_url}")

        # Create consumer group if not exists
        try:
            await self.redis_client.xgroup_create(
                self.stream_name,
                self.consumer_group,
                id="0",
                mkstream=True,
            )
            logger.info(f"Created consumer group: {self.consumer_group}")
        except redis.ResponseError as e:
            if "BUSYGROUP" in str(e):
                logger.info(f"Consumer group already exists: {self.consumer_group}")
            else:
                raise

    async def run(self):
        """
        Main consumer loop.

        Reads messages from Redis stream, debounces, and sends to protocol adapter.
        """
        if self.redis_client is None:
            await self.connect()

        logger.info(f"Starting alert consumer on stream: {self.stream_name}")

        while True:
            try:
                # Read messages from stream
                messages = await self.redis_client.xreadgroup(
                    self.consumer_group,
                    self.consumer_name,
                    {self.stream_name: ">"},
                    count=10,
                    block=1000,  # Block for 1 second
                )

                if not messages:
                    continue

                # Process messages
                for stream, msg_list in messages:
                    for msg_id, data in msg_list:
                        await self._process_message(msg_id, data)

            except KeyboardInterrupt:
                logger.info("Received interrupt signal")
                break
            except Exception as e:
                logger.error(f"Error in consumer loop: {e}")
                # Continue processing

    async def _process_message(self, msg_id: bytes, data: dict):
        """
        Process a single message from Redis stream.

        Args:
            msg_id: Redis message ID
            data: Message data
        """
        try:
            # Parse alert data
            alert_json = data.get(b"data") or data.get("data")
            if isinstance(alert_json, bytes):
                alert_json = alert_json.decode("utf-8")

            alert_dict = json.loads(alert_json)
            alert = AlertEvent.from_dict(alert_dict)

            # Debounce
            if self.debouncer is not None:
                if not self.debouncer.should_emit(alert):
                    logger.debug(f"Alert suppressed by debouncer: {alert.camera_id}")
                    # Still ACK the message
                    await self.redis_client.xack(self.stream_name, self.consumer_group, msg_id)
                    return

            # Send to protocol adapter
            await self.adapter.send_alert(alert)

            # ACK message
            await self.redis_client.xack(self.stream_name, self.consumer_group, msg_id)

        except Exception as e:
            logger.error(f"Error processing message {msg_id}: {e}")
            # Don't ACK on error - message will be redelivered

    async def close(self):
        """Close Redis connection and protocol adapter."""
        if self.adapter is not None:
            await self.adapter.close()

        if self.redis_client is not None:
            await self.redis_client.close()
            logger.info("Closed Redis connection")
