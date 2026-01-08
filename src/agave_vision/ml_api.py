"""
Agave Vision ML API

Pure ML interface for object detection, ROI filtering, and alert generation.
This is the main entry point for external teams integrating our ML capabilities.
"""

from __future__ import annotations  # noqa: I001

import time
from datetime import datetime
from pathlib import Path
from typing import Generator, List, Optional, Dict, Any

import cv2
import numpy as np

from agave_vision.config.model_config import get_default_model_path
from agave_vision.core.inference import YOLOInference, Detection
from agave_vision.core.roi import ROIManager
from agave_vision.core.tracking import CentroidTracker
from agave_vision.core.metrics import ModelMetricsTracker


class AgaveVisionML:
    """
    Example:
        >>> from agave_vision.ml_api import AgaveVisionML
        >>> import cv2
        >>>
        >>> # Initialize with default model
        >>> ml = AgaveVisionML(roi_config_path="configs/rois.yaml")
        >>>
        >>> # Or specify custom model
        >>> ml = AgaveVisionML(
        ...     model_path="models/custom-model.pt",
        ...     roi_config_path="configs/rois.yaml"
        ... )
        >>>
        >>> # Single frame inference
        >>> image = cv2.imread("frame.jpg")
        >>> result = ml.predict_frame(image, camera_id="cam1")
        >>> print(f"Alerts: {len(result['alerts'])}")
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        roi_config_path: Optional[str] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        enable_alert_storage: bool = False,
        enable_detection_logging: bool = False,
        enable_tracking: bool = True,
        tracking_max_distance: float = 50.0,
        tracking_max_disappeared: int = 30,
        enable_metrics: bool = True,
        metrics_window_size: int = 1000,
    ):
        """
        Initialize ML engine with model and configuration.

        Args:
            model_path: Path to YOLO model weights (.pt file).
                       If None, uses default from configs/model.yaml
            roi_config_path: Path to ROI configuration YAML (optional)
            conf_threshold: Confidence threshold for detections (0-1)
            iou_threshold: IOU threshold for NMS (0-1)
            enable_alert_storage: Enable persistent alert storage
            enable_detection_logging: Enable detection history logging
            enable_tracking: Enable object tracking across frames (default: True)
            tracking_max_distance: Max distance (pixels) to match objects (default: 50)
            tracking_max_disappeared: Max frames before object deregistration (default: 30)
            enable_metrics: Enable inference metrics tracking (default: True)
            metrics_window_size: Number of recent inferences to track (default: 1000)
        """
        # Use default model path if not provided
        if model_path is None:
            model_path = get_default_model_path()

        self.model_path = Path(model_path)
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

        # Initialize YOLO inference engine
        self.inference_engine = YOLOInference(
            str(self.model_path),
            conf=conf_threshold,  # type: ignore
            iou=iou_threshold,  # type: ignore
        )

        # Initialize ROI manager if config provided
        self.roi_manager: Optional[ROIManager] = None
        if roi_config_path:
            self.roi_manager = ROIManager(roi_config_path)

        # Initialize object tracker
        self.tracker: Optional[CentroidTracker] = None
        if enable_tracking:
            self.tracker = CentroidTracker(
                max_distance=tracking_max_distance,
                max_disappeared=tracking_max_disappeared,
            )

        # Initialize metrics tracker
        self.metrics: Optional[ModelMetricsTracker] = None
        if enable_metrics:
            self.metrics = ModelMetricsTracker(window_size=metrics_window_size)

        # Initialize storage (deferred imports to avoid dependencies if not used)
        self.alert_store = None
        self.detection_logger = None

        if enable_alert_storage:
            from agave_vision.storage.alert_store import AlertStore

            self.alert_store = AlertStore()

        if enable_detection_logging:
            from agave_vision.storage.detection_logger import DetectionLogger

            self.detection_logger = DetectionLogger()

    def predict_frame(
        self,
        image: np.ndarray,
        camera_id: Optional[str] = None,
        store_alerts: bool = True,
        log_detections: bool = False,
    ) -> Dict[str, Any]:
        """
        Run inference on single frame.

        Args:
            image: numpy array (BGR format from cv2.imread)
            camera_id: Camera identifier for ROI lookup (optional)
            store_alerts: Store alerts to persistent storage if enabled
            log_detections: Log detections to history if enabled

        Returns:
            {
                "detections": [
                    {
                        "bbox": [x1, y1, x2, y2],
                        "confidence": 0.85,
                        "class_name": "object",
                        "class_id": 0,
                        "center": [cx, cy],
                        "is_unknown": false
                    },
                    ...
                ],
                "alerts": [
                    {
                        "camera_id": "cam1",
                        "timestamp": "2025-01-02T14:30:00",
                        "detection": {...},
                        "roi_name": "loading_zone",
                        "violation_type": "forbidden_class",
                        "allowed_classes": ["pine", "worker"],
                        "strict_mode": true
                    },
                    ...
                ],
                "inference_time_ms": 45.2,
                "timestamp": "2025-01-02T14:30:00.123456",
                "camera_id": "cam1",
                "num_detections": 3,
                "num_alerts": 1
            }
        """
        start_time = time.time()
        timestamp = datetime.utcnow()

        # Run YOLO inference
        detections = self.inference_engine.predict(image, conf=self.conf_threshold)  # type: ignore

        # Apply object tracking if enabled
        if self.tracker:
            detections = self.tracker.update(detections)

        # Check for ROI violations if camera_id and ROI manager available
        alerts = []
        if camera_id and self.roi_manager:
            camera_roi = self.roi_manager.get_camera_rois(camera_id)
            if camera_roi:
                for detection in detections:
                    if camera_roi.should_alert(detection):
                        # Create alert dictionary
                        alert_dict = {
                            "timestamp": timestamp.isoformat(),
                            "camera_id": camera_id,
                            "detection": self._detection_to_dict(detection),
                            "alert_type": "roi_violation",
                            "reason": f"Detected '{detection.class_name}' in forbidden zone (not in allowed_classes)"
                        }

                        # Store alert if enabled
                        if store_alerts and self.alert_store:
                            self.alert_store.save_alert(alert_dict)

                        alerts.append(alert_dict)

        # Log detections if enabled
        if log_detections and self.detection_logger:
            self.detection_logger.log_detection(
                {
                    "timestamp": timestamp.isoformat(),
                    "camera_id": camera_id,
                    "detections": [self._detection_to_dict(d) for d in detections],
                    "num_alerts": len(alerts),
                }
            )

        inference_time_ms = (time.time() - start_time) * 1000

        # Record metrics if enabled
        if self.metrics:
            self.metrics.record_inference(
                inference_time_ms=inference_time_ms,
                num_detections=len(detections),
                num_alerts=len(alerts),
                camera_id=camera_id,
            )

        return {
            "detections": [self._detection_to_dict(d) for d in detections],
            "alerts": alerts,
            "inference_time_ms": round(inference_time_ms, 2),
            "timestamp": timestamp.isoformat(),
            "camera_id": camera_id,
            "num_detections": len(detections),
            "num_alerts": len(alerts),
        }

    def predict_video_stream(
        self,
        video_source: str,
        camera_id: Optional[str] = None,
        fps_limit: Optional[float] = None,
        max_frames: Optional[int] = None,
        store_alerts: bool = True,
        log_detections: bool = False,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Process video stream and yield results.

        Args:
            video_source: Path to video file or RTSP URL
            camera_id: Camera identifier for ROI lookup
            fps_limit: Optional FPS limiting (processes every Nth frame)
            max_frames: Optional maximum frames to process
            store_alerts: Store alerts to persistent storage
            log_detections: Log detections to history

        Yields:
            Detection results for each frame (same format as predict_frame)

        Example:
            >>> for result in ml.predict_video_stream("rtsp://camera/stream", "cam1", fps_limit=5.0):
            ...     print(f"Frame alerts: {len(result['alerts'])}")
            ...     # Send to your server, database, websocket, etc.
        """  # noqa: E501
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video source: {video_source}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_skip = 1

        if fps_limit and fps_limit < video_fps:
            frame_skip = int(video_fps / fps_limit)

        frame_idx = 0
        processed_frames = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Skip frames if FPS limiting enabled
                if frame_idx % frame_skip != 0:
                    frame_idx += 1
                    continue

                # Run inference
                result = self.predict_frame(
                    frame,
                    camera_id=camera_id,
                    store_alerts=store_alerts,
                    log_detections=log_detections,
                )

                # Add frame metadata
                result["frame_index"] = frame_idx
                result["processed_frame_index"] = processed_frames

                yield result

                frame_idx += 1
                processed_frames += 1

                # Check max frames limit
                if max_frames and processed_frames >= max_frames:
                    break

        finally:
            cap.release()

    def get_alerts(
        self,
        camera_id: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve alerts from storage.

        Args:
            camera_id: Filter by camera ID (optional)
            start_time: Filter by start time ISO format (optional)
            end_time: Filter by end time ISO format (optional)
            limit: Maximum number of alerts to return

        Returns:
            [
                {
                    "alert_id": "uuid",
                    "camera_id": "cam1",
                    "timestamp": "2025-01-02T14:30:00",
                    "detection": {...},
                    "roi_name": "loading_zone",
                    "violation_type": "forbidden_class",
                    ...
                }
            ]

        Raises:
            RuntimeError: If alert storage is not enabled
        """
        if not self.alert_store:
            raise RuntimeError(
                "Alert storage not enabled. Initialize with enable_alert_storage=True"
            )

        # Convert ISO strings to datetime if provided
        start_dt = datetime.fromisoformat(start_time) if start_time else None
        end_dt = datetime.fromisoformat(end_time) if end_time else None

        return self.alert_store.get_alerts(
            camera_id=camera_id,
            start_time=start_dt,
            end_time=end_dt,
            limit=limit,
        )

    def get_detection_logs(
        self,
        camera_id: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve detection history.

        Args:
            camera_id: Filter by camera ID (optional)
            start_time: Filter by start time ISO format (optional)
            end_time: Filter by end time ISO format (optional)
            limit: Maximum number of log entries to return

        Returns:
            [
                {
                    "timestamp": "2025-01-02T14:30:00",
                    "camera_id": "cam1",
                    "detections": [...],
                    "num_alerts": 2
                }
            ]

        Raises:
            RuntimeError: If detection logging is not enabled
        """
        if not self.detection_logger:
            raise RuntimeError(
                "Detection logging not enabled. Initialize with enable_detection_logging=True"
            )

        # Convert ISO strings to datetime if provided
        start_dt = datetime.fromisoformat(start_time) if start_time else None
        end_dt = datetime.fromisoformat(end_time) if end_time else None

        return self.detection_logger.get_logs(
            camera_id=camera_id,
            start_time=start_dt,
            end_time=end_dt,
            limit=limit,
        )

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model metadata.

        Returns:
            {
                "model_path": "models/best.pt",
                "classes": ["object", "pine", "worker"],
                "num_classes": 3,
                "input_size": 640,
                "conf_threshold": 0.25,
                "iou_threshold": 0.45
            }
        """
        return {
            "model_path": str(self.model_path),
            "classes": self.inference_engine.class_names,
            "num_classes": len(self.inference_engine.class_names),
            "input_size": self.inference_engine.model_input_size,  # type: ignore
            "conf_threshold": self.conf_threshold,
            "iou_threshold": self.iou_threshold,
        }

    def get_camera_roi_info(self, camera_id: str) -> Optional[Dict[str, Any]]:
        """
        Get ROI configuration for a camera.

        Args:
            camera_id: Camera identifier

        Returns:
            {
                "camera_id": "cam1",
                "forbidden_zones": [
                    {
                        "name": "loading_zone",
                        "points": [[x1,y1], [x2,y2], ...],
                        "num_points": 4
                    }
                ],
                "allowed_classes": ["pine", "worker"],
                "alert_classes": ["object"],
                "strict_mode": true
            }

            Returns None if camera not found or ROI manager not initialized
        """
        if not self.roi_manager:
            return None

        camera_roi = self.roi_manager.get_camera_rois(camera_id)
        if not camera_roi:
            return None

        return {
            "camera_id": camera_id,
            "forbidden_zones": [
                {
                    "name": zone.name,
                    "points": zone.points.tolist(),
                    "num_points": len(zone.points),
                }
                for zone in camera_roi.forbidden_zones
            ],
            "allowed_classes": list(camera_roi.allowed_classes),
            "alert_classes": list(camera_roi.alert_classes),
            "strict_mode": camera_roi.strict_mode,
        }

    def _detection_to_dict(self, detection: Detection) -> Dict[str, Any]:
        """Convert Detection object to dictionary."""
        result = {
            "bbox": list(detection.bbox),
            "confidence": float(detection.confidence),
            "class_name": detection.class_name,
            "class_id": int(detection.class_id),
            "center": list(detection.center),
            "is_unknown": detection.is_unknown,
        }
        # Include tracking_id if available
        if detection.tracking_id is not None:
            result["tracking_id"] = detection.tracking_id
        return result

    def get_metrics(self) -> Optional[Dict]:
        """
        Get inference metrics statistics.

        Returns:
            Dictionary with metrics statistics or None if metrics disabled
        """
        if self.metrics:
            return self.metrics.get_statistics()
        return None

    def reset_metrics(self) -> None:
        """Reset metrics tracker."""
        if self.metrics:
            self.metrics.reset()

    def __repr__(self) -> str:
        return (
            f"AgaveVisionML("
            f"model={self.model_path.name}, "
            f"conf={self.conf_threshold}, "
            f"roi_enabled={self.roi_manager is not None}, "
            f"tracking_enabled={self.tracker is not None}, "
            f"metrics_enabled={self.metrics is not None})"
        )
