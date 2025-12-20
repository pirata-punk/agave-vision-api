"""
Alert Router Service

Main entry point for the alert routing service.
Consumes alerts from Redis, debounces, and routes to protocol adapters.
"""

import asyncio
import os
from pathlib import Path

from agave_vision.config.loader import ConfigLoader, get_redis_url
from agave_vision.utils.logging import setup_logging, get_logger

from .consumer import RedisConsumer
from .debounce import AlertDebouncer
from .protocols import get_protocol_adapter


async def main():
    """
    Main entry point for alert router service.

    Loads configuration, initializes protocol adapter, and starts consuming alerts.
    """
    # Setup logging
    logger = setup_logging("alert-router", level="INFO", format="json")
    logger.info("Starting Alert Router service...")

    # Load configurations
    config_loader = ConfigLoader(Path("configs"))
    services_config = config_loader.load_services()
    alerting_config = services_config.alerting

    logger.info(f"Alert protocol: {alerting_config.protocol}")

    # Initialize protocol adapter
    try:
        adapter = get_protocol_adapter(alerting_config)
        logger.info(f"Initialized protocol adapter: {adapter.__class__.__name__}")
    except ValueError as e:
        logger.error(f"Failed to initialize protocol adapter: {e}")
        return

    # Initialize debouncer
    debouncer = AlertDebouncer(
        window_seconds=alerting_config.debounce_window_seconds,
        max_per_window=alerting_config.max_alerts_per_window,
    )
    logger.info(f"Initialized debouncer: {debouncer}")

    # Initialize Redis consumer
    redis_url = get_redis_url()
    logger.info(f"Connecting to Redis at {redis_url}...")

    consumer = RedisConsumer(
        redis_url=redis_url,
        stream_name=alerting_config.redis_stream_name,
        consumer_group=alerting_config.redis_consumer_group,
        adapter=adapter,
        debouncer=debouncer,
    )

    try:
        await consumer.connect()
        logger.info("Alert Router service ready")

        # Start consuming
        await consumer.run()

    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error in alert router: {e}")
        raise
    finally:
        # Cleanup
        await consumer.close()
        logger.info("Alert Router service stopped")


if __name__ == "__main__":
    asyncio.run(main())
