"""
Object Tracking

Provides simple centroid-based tracking for associating detections across frames.
This helps avoid inflating statistics by tracking unique objects through their lifecycle.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np

from .inference import Detection


class CentroidTracker:
    """
    Simple centroid-based object tracker.

    Tracks objects across frames by matching detection centroids using
    Euclidean distance. Assigns persistent tracking IDs to detections
    that represent the same physical object.

    Args:
        max_distance: Maximum distance (pixels) to consider as same object
        max_disappeared: Maximum frames an object can disappear before deregistration

    Example:
        >>> tracker = CentroidTracker(max_distance=50, max_disappeared=30)
        >>> detections = model.predict(frame)
        >>> tracked_detections = tracker.update(detections)
        >>> # tracked_detections now have tracking_id field populated
    """

    def __init__(self, max_distance: float = 50.0, max_disappeared: int = 30):
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared

        # Track next available ID
        self.next_object_id = 0

        # Maps tracking_id -> centroid
        self.objects: OrderedDict[str, Tuple[float, float]] = OrderedDict()

        # Maps tracking_id -> number of consecutive frames object has been missing
        self.disappeared: Dict[str, int] = {}

        # Maps tracking_id -> Detection for metadata persistence
        self.metadata: Dict[str, Detection] = {}

    def register(self, detection: Detection) -> str:
        """
        Register a new object with a unique tracking ID.

        Args:
            detection: Detection to register

        Returns:
            Assigned tracking ID
        """
        tracking_id = f"obj_{self.next_object_id}"
        self.next_object_id += 1

        self.objects[tracking_id] = detection.center
        self.disappeared[tracking_id] = 0
        self.metadata[tracking_id] = detection

        return tracking_id

    def deregister(self, tracking_id: str) -> None:
        """
        Remove an object from tracking.

        Args:
            tracking_id: ID of object to deregister
        """
        del self.objects[tracking_id]
        del self.disappeared[tracking_id]
        del self.metadata[tracking_id]

    def update(self, detections: List[Detection]) -> List[Detection]:
        """
        Update tracker with new detections from current frame.

        Args:
            detections: List of detections from current frame

        Returns:
            Same detections list but with tracking_id field populated
        """
        # If no detections, mark all existing objects as disappeared
        if len(detections) == 0:
            for tracking_id in list(self.disappeared.keys()):
                self.disappeared[tracking_id] += 1

                # Deregister if disappeared too long
                if self.disappeared[tracking_id] > self.max_disappeared:
                    self.deregister(tracking_id)

            return []

        # If no objects are being tracked, register all detections as new
        if len(self.objects) == 0:
            for detection in detections:
                tracking_id = self.register(detection)
                detection.tracking_id = tracking_id

            return detections

        # Build matrix of distances between existing objects and new detections
        object_ids = list(self.objects.keys())
        object_centroids = np.array([self.objects[oid] for oid in object_ids])
        detection_centroids = np.array([det.center for det in detections])

        # Compute distance matrix: rows are existing objects, cols are new detections
        distances = self._compute_distance_matrix(object_centroids, detection_centroids)

        # Match existing objects to new detections using Hungarian algorithm (greedy approximation)
        matched_pairs = self._match_detections(distances, object_ids, detections)

        # Update matched objects
        for obj_id, detection in matched_pairs:
            self.objects[obj_id] = detection.center
            self.disappeared[obj_id] = 0
            self.metadata[obj_id] = detection
            detection.tracking_id = obj_id

        # Handle unmatched existing objects (disappeared)
        matched_object_ids = {obj_id for obj_id, _ in matched_pairs}
        for obj_id in object_ids:
            if obj_id not in matched_object_ids:
                self.disappeared[obj_id] += 1

                # Deregister if disappeared too long
                if self.disappeared[obj_id] > self.max_disappeared:
                    self.deregister(obj_id)

        # Handle unmatched detections (new objects)
        matched_detection_indices = {detections.index(det) for _, det in matched_pairs}
        for i, detection in enumerate(detections):
            if i not in matched_detection_indices:
                tracking_id = self.register(detection)
                detection.tracking_id = tracking_id

        return detections

    def _compute_distance_matrix(
        self, centroids1: np.ndarray, centroids2: np.ndarray
    ) -> np.ndarray:
        """
        Compute pairwise Euclidean distances between two sets of centroids.

        Args:
            centroids1: Nx2 array of (x, y) coordinates
            centroids2: Mx2 array of (x, y) coordinates

        Returns:
            NxM distance matrix
        """
        # Expand dims for broadcasting
        c1 = centroids1[:, np.newaxis, :]  # (N, 1, 2)
        c2 = centroids2[np.newaxis, :, :]  # (1, M, 2)

        # Compute Euclidean distance
        distances = np.sqrt(np.sum((c1 - c2) ** 2, axis=2))

        return distances

    def _match_detections(
        self, distances: np.ndarray, object_ids: List[str], detections: List[Detection]
    ) -> List[Tuple[str, Detection]]:
        """
        Match existing objects to new detections using greedy nearest-neighbor.

        Args:
            distances: NxM distance matrix
            object_ids: List of N object IDs
            detections: List of M detections

        Returns:
            List of (object_id, detection) pairs
        """
        matched_pairs = []

        # Create mutable copies for tracking what's been matched
        remaining_object_indices = set(range(len(object_ids)))
        remaining_detection_indices = set(range(len(detections)))

        # Greedy matching: repeatedly find and match closest pair
        while remaining_object_indices and remaining_detection_indices:
            # Find minimum distance among remaining pairs
            min_dist = float('inf')
            min_obj_idx = None
            min_det_idx = None

            for obj_idx in remaining_object_indices:
                for det_idx in remaining_detection_indices:
                    dist = distances[obj_idx, det_idx]
                    if dist < min_dist:
                        min_dist = dist
                        min_obj_idx = obj_idx
                        min_det_idx = det_idx

            # If minimum distance exceeds threshold, stop matching
            if min_dist > self.max_distance:
                break

            # Record match and remove from remaining sets
            obj_id = object_ids[min_obj_idx]
            detection = detections[min_det_idx]
            matched_pairs.append((obj_id, detection))

            remaining_object_indices.remove(min_obj_idx)
            remaining_detection_indices.remove(min_det_idx)

        return matched_pairs

    def get_active_count(self) -> int:
        """
        Get number of currently tracked objects.

        Returns:
            Number of active tracked objects
        """
        return len(self.objects)

    def get_object_metadata(self, tracking_id: str) -> Detection | None:
        """
        Get last known detection metadata for a tracked object.

        Args:
            tracking_id: Object tracking ID

        Returns:
            Last Detection or None if not found
        """
        return self.metadata.get(tracking_id)

    def reset(self) -> None:
        """Reset tracker state, clearing all tracked objects."""
        self.objects.clear()
        self.disappeared.clear()
        self.metadata.clear()
        self.next_object_id = 0

    def __repr__(self) -> str:
        return (
            f"CentroidTracker(max_distance={self.max_distance}, "
            f"max_disappeared={self.max_disappeared}, "
            f"active_objects={self.get_active_count()})"
        )
