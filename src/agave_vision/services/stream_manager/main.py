"""
Stream Manager Service

Main entry point for the RTSP stream manager service.
Ingests live video streams, runs inference, and publishes alerts.
"""

import asyncio
import os
from pathlib import Path

from agave_vision.config.loader import ConfigLoader, get_redis_url
from agave_vision.core.inference import YOLOInference
from agave_vision.core.roi import ROIManager
from agave_vision.utils.logging import setup_logging, get_logger

from .camera import CameraHandler
from .publisher import RedisPublisher


async def main():
    """
    Main entry point for stream manager service.

    Loads configuration, initializes YOLO model, and starts camera handlers.
    """
    # Setup logging
    logger = setup_logging("stream-manager", level="INFO", format="json")
    logger.info("Starting Stream Manager service...")

    # Load configurations
    config_loader = ConfigLoader(Path("configs"))
    cameras_config = config_loader.load_cameras()
    services_config = config_loader.load_services()

    logger.info(f"Loaded {len(cameras_config.cameras)} camera configurations")

    # Load YOLO model (shared across all cameras)
    logger.info(f"Loading YOLO model from {services_config.inference.model_path}...")
    model = YOLOInference(
        model_path=services_config.inference.model_path,
        conf=services_config.inference.confidence,
        iou=services_config.inference.iou_threshold,
        device=services_config.inference.device,
        imgsz=services_config.inference.image_size,
    )

    # Warmup model
    logger.info(f"Warming up model ({services_config.inference.warmup_iterations} iterations)...")
    model.warmup(iterations=services_config.inference.warmup_iterations)
    logger.info("Model warmup complete")

    # Load ROI manager
    roi_manager = ROIManager(Path("configs/rois.yaml"))
    logger.info(f"Loaded ROIs for {len(roi_manager.camera_rois)} cameras")

    # Initialize Redis publisher
    redis_url = get_redis_url()
    logger.info(f"Connecting to Redis at {redis_url}...")
    publisher = RedisPublisher(
        redis_url=redis_url,
        stream_name=services_config.alerting.redis_stream_name,
        max_stream_length=services_config.alerting.redis_max_stream_length,
    )
    await publisher.connect()
    logger.info("Redis connection established")

    # Create camera handlers
    handlers = []
    for cam in cameras_config.cameras:
        if not cam.enabled:
            logger.info(f"Skipping disabled camera: {cam.id}")
            continue

        handler = CameraHandler(
            camera_config=cam,
            model=model,
            roi_manager=roi_manager,
            publisher=publisher,
            stream_config=services_config.stream_manager,
        )
        handlers.append(handler)

    if not handlers:
        logger.warning("No enabled cameras found, exiting")
        return

    logger.info(f"Starting {len(handlers)} camera handlers...")

    try:
        # Run all handlers concurrently
        await asyncio.gather(*[handler.run() for handler in handlers])
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
    except Exception as e:
        logger.error(f"Fatal error in stream manager: {e}")
        raise
    finally:
        # Cleanup
        await publisher.close()
        logger.info("Stream Manager service stopped")


if __name__ == "__main__":
    asyncio.run(main())
