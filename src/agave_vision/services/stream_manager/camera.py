"""
Camera Handler

Handles RTSP connection, frame sampling, inference, and alert emission for a single camera.
"""

import asyncio
import time
from datetime import datetime
from typing import Optional

import cv2

from agave_vision.config.models import CameraConfig, StreamManagerConfig
from agave_vision.core.alerts import AlertEvent
from agave_vision.core.inference import YOLOInference
from agave_vision.core.roi import ROIManager
from agave_vision.utils.logging import get_logger

from .publisher import RedisPublisher

logger = get_logger(__name__)


class CameraHandler:
    """
    Handles RTSP connection, frame sampling, and inference for a single camera.

    Args:
        camera_config: Camera configuration
        model: YOLO inference model (shared across cameras)
        roi_manager: ROI manager for alert filtering
        publisher: Redis publisher for alerts
        stream_config: Stream manager configuration

    Example:
        >>> handler = CameraHandler(camera_config, model, roi_manager, publisher, stream_config)
        >>> await handler.run()
    """

    def __init__(
        self,
        camera_config: CameraConfig,
        model: YOLOInference,
        roi_manager: ROIManager,
        publisher: RedisPublisher,
        stream_config: StreamManagerConfig,
    ):
        self.config = camera_config
        self.model = model
        self.roi_manager = roi_manager
        self.publisher = publisher
        self.stream_config = stream_config

        # Frame sampling
        self.frame_interval = 1.0 / camera_config.fps_target
        self.last_frame_time = 0.0

        # Reconnection tracking
        self.reconnect_attempts = 0

        # Camera ROI config
        self.camera_roi = roi_manager.get_camera_rois(camera_config.id)
        if self.camera_roi is None:
            logger.warning(f"No ROI config found for camera {camera_config.id}")

    async def run(self):
        """
        Main loop: connect, sample frames, run inference, publish alerts.

        Automatically reconnects on failure.
        """
        logger.info(f"Starting camera handler for {self.config.id}")

        while True:
            try:
                await self._stream_loop()
            except Exception as e:
                logger.error(f"Camera {self.config.id} error: {e}")

                # Check reconnection limit
                if 0 <= self.stream_config.max_reconnect_attempts <= self.reconnect_attempts:
                    logger.error(
                        f"Camera {self.config.id} exceeded max reconnect attempts, stopping"
                    )
                    break

                self.reconnect_attempts += 1
                delay = self.stream_config.reconnect_delay_seconds
                logger.info(
                    f"Reconnecting to {self.config.id} in {delay}s "
                    f"(attempt {self.reconnect_attempts})"
                )
                await asyncio.sleep(delay)

    async def _stream_loop(self):
        """Internal stream processing loop."""
        # Open RTSP connection
        cap = cv2.VideoCapture(self.config.rtsp_url)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open RTSP: {self.config.rtsp_url}")

        logger.info(f"Connected to camera {self.config.id} at {self.config.rtsp_url}")
        self.reconnect_attempts = 0  # Reset on successful connection

        frame_count = 0

        try:
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Failed to read frame from {self.config.id}")
                    break

                # Frame sampling (target FPS)
                now = time.time()
                if now - self.last_frame_time < self.frame_interval:
                    continue

                self.last_frame_time = now
                frame_count += 1

                # Run inference asynchronously (non-blocking)
                await self._process_frame(frame, frame_count)

        finally:
            cap.release()
            logger.info(f"Released camera {self.config.id}")

    async def _process_frame(self, frame, frame_id: int):
        """
        Process a single frame: run inference and check for alerts.

        Args:
            frame: OpenCV frame (BGR)
            frame_id: Frame sequence number
        """
        try:
            # Run inference (blocking, but fast with GPU)
            detections = self.model.predict(frame, verbose=False)

            # Check for alerts
            if self.camera_roi is not None:
                alert_detections = self.camera_roi.filter_detections(detections)

                # Publish alerts
                for det in alert_detections:
                    alert = AlertEvent(
                        camera_id=self.config.id,
                        timestamp=datetime.utcnow(),
                        detection=det,
                        roi_hit=True,
                        frame_id=f"{self.config.id}_{frame_id}",
                    )

                    await self.publisher.publish_alert(alert)
                    logger.info(
                        f"Alert: {self.config.id} detected {det.class_name} "
                        f"(conf={det.confidence:.2f}) in ROI"
                    )

        except Exception as e:
            logger.error(f"Error processing frame from {self.config.id}: {e}")
