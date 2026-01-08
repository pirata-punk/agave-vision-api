"""
Model Metrics and Tracing

Provides inference metrics tracking for monitoring model performance.
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np


@dataclass
class InferenceMetrics:
    """Metrics for a single inference call."""

    timestamp: datetime
    inference_time_ms: float
    num_detections: int
    num_alerts: int
    camera_id: Optional[str] = None
    frame_id: Optional[str] = None


@dataclass
class ModelMetricsTracker:
    """
    Track and aggregate model inference metrics.

    Provides real-time statistics about model performance including:
    - Inference time (min, max, mean, p50, p95, p99)
    - Detection counts
    - Alert counts
    - Throughput (inferences per second)

    Example:
        >>> tracker = ModelMetricsTracker(window_size=1000)
        >>> tracker.record_inference(
        ...     inference_time_ms=45.2,
        ...     num_detections=3,
        ...     num_alerts=1,
        ...     camera_id="cam1"
        ... )
        >>> stats = tracker.get_statistics()
        >>> print(f"Mean inference time: {stats['inference_time_ms']['mean']:.2f}ms")
    """

    window_size: int = 1000  # Number of recent inferences to track
    recent_metrics: deque = field(default_factory=lambda: deque(maxlen=1000))

    # Aggregated counters
    total_inferences: int = 0
    total_detections: int = 0
    total_alerts: int = 0

    # Per-camera statistics
    camera_stats: Dict[str, Dict] = field(default_factory=lambda: defaultdict(dict))

    # Session tracking
    session_start: datetime = field(default_factory=datetime.utcnow)

    def record_inference(
        self,
        inference_time_ms: float,
        num_detections: int,
        num_alerts: int = 0,
        camera_id: Optional[str] = None,
        frame_id: Optional[str] = None,
    ) -> None:
        """
        Record metrics from a single inference.

        Args:
            inference_time_ms: Inference time in milliseconds
            num_detections: Number of detections returned
            num_alerts: Number of alerts triggered
            camera_id: Optional camera identifier
            frame_id: Optional frame identifier
        """
        metric = InferenceMetrics(
            timestamp=datetime.utcnow(),
            inference_time_ms=inference_time_ms,
            num_detections=num_detections,
            num_alerts=num_alerts,
            camera_id=camera_id,
            frame_id=frame_id,
        )

        self.recent_metrics.append(metric)
        self.total_inferences += 1
        self.total_detections += num_detections
        self.total_alerts += num_alerts

        # Update per-camera stats
        if camera_id:
            if camera_id not in self.camera_stats:
                self.camera_stats[camera_id] = {
                    "inferences": 0,
                    "detections": 0,
                    "alerts": 0,
                }
            self.camera_stats[camera_id]["inferences"] += 1
            self.camera_stats[camera_id]["detections"] += num_detections
            self.camera_stats[camera_id]["alerts"] += num_alerts

    def get_statistics(self) -> Dict:
        """
        Get aggregated statistics from tracked metrics.

        Returns:
            Dictionary with inference time stats, detection stats, and throughput
        """
        if not self.recent_metrics:
            return {
                "total_inferences": 0,
                "total_detections": 0,
                "total_alerts": 0,
                "inference_time_ms": {},
                "detections_per_inference": {},
                "alerts_per_inference": {},
                "throughput": {},
            }

        # Extract recent inference times
        inference_times = [m.inference_time_ms for m in self.recent_metrics]
        detections = [m.num_detections for m in self.recent_metrics]
        alerts = [m.num_alerts for m in self.recent_metrics]

        # Calculate inference time statistics
        inference_stats = {
            "min": float(np.min(inference_times)),
            "max": float(np.max(inference_times)),
            "mean": float(np.mean(inference_times)),
            "median": float(np.median(inference_times)),
            "p50": float(np.percentile(inference_times, 50)),
            "p95": float(np.percentile(inference_times, 95)),
            "p99": float(np.percentile(inference_times, 99)),
            "std": float(np.std(inference_times)),
        }

        # Calculate detection statistics
        detection_stats = {
            "min": int(np.min(detections)),
            "max": int(np.max(detections)),
            "mean": float(np.mean(detections)),
            "median": float(np.median(detections)),
        }

        # Calculate alert statistics
        alert_stats = {
            "min": int(np.min(alerts)),
            "max": int(np.max(alerts)),
            "mean": float(np.mean(alerts)),
            "median": float(np.median(alerts)),
        }

        # Calculate throughput
        elapsed = (datetime.utcnow() - self.session_start).total_seconds()
        throughput = {
            "inferences_per_second": self.total_inferences / elapsed if elapsed > 0 else 0,
            "detections_per_second": self.total_detections / elapsed if elapsed > 0 else 0,
            "alerts_per_second": self.total_alerts / elapsed if elapsed > 0 else 0,
            "session_duration_seconds": elapsed,
        }

        return {
            "total_inferences": self.total_inferences,
            "total_detections": self.total_detections,
            "total_alerts": self.total_alerts,
            "inference_time_ms": inference_stats,
            "detections_per_inference": detection_stats,
            "alerts_per_inference": alert_stats,
            "throughput": throughput,
            "camera_stats": dict(self.camera_stats),
            "window_size": len(self.recent_metrics),
        }

    def get_recent_metrics(self, count: int = 100) -> List[InferenceMetrics]:
        """
        Get most recent metrics.

        Args:
            count: Number of recent metrics to retrieve

        Returns:
            List of InferenceMetrics (most recent first)
        """
        return list(self.recent_metrics)[-count:]

    def reset(self) -> None:
        """Reset all metrics and counters."""
        self.recent_metrics.clear()
        self.total_inferences = 0
        self.total_detections = 0
        self.total_alerts = 0
        self.camera_stats.clear()
        self.session_start = datetime.utcnow()

    def __repr__(self) -> str:
        stats = self.get_statistics()
        return (
            f"ModelMetricsTracker("
            f"inferences={stats['total_inferences']}, "
            f"mean_time={stats['inference_time_ms'].get('mean', 0):.2f}ms, "
            f"throughput={stats['throughput']['inferences_per_second']:.2f}/s)"
        )
