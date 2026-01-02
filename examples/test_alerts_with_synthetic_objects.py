#!/usr/bin/env python3
"""
Alert Testing with Synthetic Object Injection

Injects synthetic objects into video frames to test the ROI alert system.
Uses CV techniques to create realistic test scenarios.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import cv2
import numpy as np
from typing import List, Tuple, Optional

from agave_vision.core.inference import YOLOInference, Detection
from agave_vision.core.roi import ROIManager, CameraROI
from agave_vision.core.frames import draw_detection_box


class SyntheticObjectInjector:
    """Injects synthetic objects into frames for testing."""

    def __init__(self, roi_zones: List[np.ndarray]):
        """
        Initialize injector with ROI zones.

        Args:
            roi_zones: List of ROI polygon coordinates
        """
        self.roi_zones = roi_zones
        self.object_templates = self._create_object_templates()

    def _create_object_templates(self) -> dict:
        """Create synthetic object templates (simple geometric shapes)."""
        templates = {}

        # Template 1: Rectangle (simulating a box/crate)
        rect = np.zeros((80, 60, 3), dtype=np.uint8)
        cv2.rectangle(rect, (5, 5), (55, 75), (139, 69, 19), -1)  # Brown color
        cv2.rectangle(rect, (5, 5), (55, 75), (90, 45, 10), 2)    # Darker outline
        templates['box'] = rect

        # Template 2: Circle (simulating a tire/barrel)
        circle = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.circle(circle, (50, 50), 45, (50, 50, 50), -1)  # Dark gray
        cv2.circle(circle, (50, 50), 45, (30, 30, 30), 3)    # Black outline
        cv2.circle(circle, (50, 50), 25, (70, 70, 70), 2)    # Inner ring
        templates['tire'] = circle

        # Template 3: Irregular blob (simulating debris)
        blob = np.zeros((70, 90, 3), dtype=np.uint8)
        pts = np.array([[10, 30], [40, 10], [70, 25], [80, 50], [60, 65], [20, 60]], np.int32)
        cv2.fillPoly(blob, [pts], (100, 80, 60))  # Grayish-brown
        cv2.polylines(blob, [pts], True, (60, 50, 40), 2)
        templates['debris'] = blob

        return templates

    def get_roi_center(self, roi_idx: int = 0) -> Tuple[int, int]:
        """
        Get the center point of a ROI zone.

        Args:
            roi_idx: Index of ROI zone to use

        Returns:
            (x, y) center coordinates
        """
        if roi_idx >= len(self.roi_zones):
            roi_idx = 0

        roi = self.roi_zones[roi_idx]
        center_x = int(np.mean(roi[:, 0]))
        center_y = int(np.mean(roi[:, 1]))
        return center_x, center_y

    def inject_object(
        self,
        frame: np.ndarray,
        object_type: str = "box",
        position: Optional[Tuple[int, int]] = None,
        scale: float = 1.0,
        alpha: float = 0.8
    ) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """
        Inject a synthetic object into the frame.

        Args:
            frame: Input frame
            object_type: Type of object ('box', 'tire', 'debris')
            position: (x, y) center position, or None to use ROI center
            scale: Scale factor for object size
            alpha: Transparency (0=invisible, 1=opaque)

        Returns:
            Tuple of (modified_frame, bbox)
        """
        if object_type not in self.object_templates:
            object_type = 'box'

        template = self.object_templates[object_type].copy()

        # Scale template
        if scale != 1.0:
            new_size = (int(template.shape[1] * scale), int(template.shape[0] * scale))
            template = cv2.resize(template, new_size)

        h, w = template.shape[:2]

        # Determine position
        if position is None:
            position = self.get_roi_center()

        x, y = position
        x1 = max(0, x - w // 2)
        y1 = max(0, y - h // 2)
        x2 = min(frame.shape[1], x1 + w)
        y2 = min(frame.shape[0], y1 + h)

        # Adjust template if it goes out of bounds
        tw = x2 - x1
        th = y2 - y1
        if tw != w or th != h:
            template = template[:th, :tw]

        # Blend object into frame with alpha transparency
        roi_region = frame[y1:y2, x1:x2]
        blended = cv2.addWeighted(template, alpha, roi_region, 1 - alpha, 0)

        # Copy blended region back
        result = frame.copy()
        result[y1:y2, x1:x2] = blended

        # Return frame and bounding box
        bbox = (x1, y1, x2, y2)
        return result, bbox


def create_synthetic_detection(
    bbox: Tuple[int, int, int, int],
    object_type: str = "object",
    confidence: float = 0.75
) -> Detection:
    """
    Create a synthetic detection for testing.

    Args:
        bbox: Bounding box (x1, y1, x2, y2)
        object_type: Class name
        confidence: Detection confidence

    Returns:
        Detection object
    """
    x1, y1, x2, y2 = bbox
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    return Detection(
        bbox=bbox,
        confidence=confidence,
        class_name=object_type,
        class_id=0,  # Dummy ID
        center=(center_x, center_y),
        is_unknown=False
    )


def main():
    parser = argparse.ArgumentParser(
        description="Test ROI alerts with synthetic object injection"
    )
    parser.add_argument(
        "--video",
        required=True,
        help="Path to input video file"
    )
    parser.add_argument(
        "--roi-config",
        required=True,
        help="Path to ROI configuration YAML"
    )
    parser.add_argument(
        "--camera-id",
        required=True,
        help="Camera ID for ROI lookup"
    )
    parser.add_argument(
        "--object-type",
        choices=["box", "tire", "debris", "all"],
        default="all",
        help="Type of synthetic object to inject"
    )
    parser.add_argument(
        "--position",
        choices=["roi_center", "random", "moving"],
        default="roi_center",
        help="Object placement strategy"
    )
    parser.add_argument(
        "--model",
        help="Path to YOLO model (optional, for real detections too)"
    )

    args = parser.parse_args()

    # Load ROI configuration
    roi_manager = ROIManager(args.roi_config)
    camera_roi = roi_manager.get_camera_rois(args.camera_id)

    if not camera_roi:
        print(f"ERROR: No ROI configuration found for camera '{args.camera_id}'")
        return 1

    print("=" * 60)
    print("ALERT TESTING WITH SYNTHETIC OBJECTS")
    print("=" * 60)
    print(f"Camera:         {args.camera_id}")
    print(f"ROI Zones:      {len(camera_roi.forbidden_zones)}")
    print(f"Strict Mode:    {camera_roi.strict_mode}")
    print(f"Allowed:        {list(camera_roi.allowed_classes)}")
    print(f"Object Type:    {args.object_type}")
    print(f"Position:       {args.position}")
    print("=" * 60)
    print()

    # Initialize injector
    roi_zones = [zone.points for zone in camera_roi.forbidden_zones]
    injector = SyntheticObjectInjector(roi_zones)

    # Track alerts (no debouncer needed for testing)
    seen_alerts = set()  # Track unique alerts to avoid double-counting

    # Optional: Load YOLO model for real detections
    yolo_inference = None
    if args.model:
        yolo_inference = YOLOInference(args.model)
        print(f"Loaded YOLO model: {args.model}")

    # Open video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {args.video}")
        return 1

    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video: {Path(args.video).name}")
    print(f"FPS: {fps}, Total Frames: {total_frames}")
    print()
    print("Controls:")
    print("  SPACE - Pause/Resume")
    print("  Q/ESC - Quit")
    print("  S     - Screenshot")
    print("  1-3   - Change object type")
    print("=" * 60)
    print()

    frame_idx = 0
    paused = False
    total_alerts = 0
    current_object_type = args.object_type
    object_types = ["box", "tire", "debris"]

    # Movement parameters for "moving" position mode
    angle = 0
    radius = 100

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\nEnd of video reached.")
                break

            frame_idx += 1

            # Get real detections if model available
            real_detections = []
            if yolo_inference:
                real_detections = yolo_inference.detect(frame, conf_threshold=0.25)

            # Inject synthetic object(s)
            synthetic_detections = []

            # Determine which object types to inject
            if current_object_type == "all":
                inject_types = object_types
            else:
                inject_types = [current_object_type]

            for i, obj_type in enumerate(inject_types):
                # Determine position
                if args.position == "roi_center":
                    position = injector.get_roi_center(roi_idx=i % len(roi_zones))
                elif args.position == "random":
                    roi_center = injector.get_roi_center(roi_idx=i % len(roi_zones))
                    offset_x = np.random.randint(-100, 100)
                    offset_y = np.random.randint(-100, 100)
                    position = (roi_center[0] + offset_x, roi_center[1] + offset_y)
                else:  # moving
                    roi_center = injector.get_roi_center(roi_idx=i % len(roi_zones))
                    offset_x = int(radius * np.cos(angle + i * 2.0))
                    offset_y = int(radius * np.sin(angle + i * 2.0))
                    position = (roi_center[0] + offset_x, roi_center[1] + offset_y)

                # Inject object
                frame, bbox = injector.inject_object(
                    frame,
                    object_type=obj_type,
                    position=position,
                    scale=1.0 + i * 0.3,  # Vary size
                    alpha=0.7
                )

                # Create synthetic detection
                synthetic_det = create_synthetic_detection(bbox, object_type="object", confidence=0.85)
                synthetic_detections.append(synthetic_det)

            # Update angle for moving objects
            angle += 0.05

            # Combine real and synthetic detections
            all_detections = real_detections + synthetic_detections

            # Check for ROI violations
            for detection in all_detections:
                # Check if this detection would trigger an alert
                would_alert = camera_roi.should_alert(detection)

                if would_alert:
                    # Create unique key to avoid double-counting same position
                    alert_key = (frame_idx, int(detection.center[0]), int(detection.center[1]))
                    if alert_key not in seen_alerts:
                        seen_alerts.add(alert_key)
                        total_alerts += 1

                    # Visual feedback for alert
                    is_alert = True
                    color = (0, 0, 255)  # Red for alerts
                    thickness = 4
                else:
                    is_alert = False
                    # Color by class
                    if detection.class_name == "pine":
                        color = (0, 255, 0)  # Green
                    elif detection.class_name == "worker":
                        color = (0, 165, 255)  # Orange
                    else:
                        color = (255, 255, 0)  # Cyan for objects
                    thickness = 2

                # Draw detection
                label = detection.class_name
                if is_alert:
                    label = f"⚠ ALERT! {label}"

                draw_detection_box(
                    frame,
                    detection.bbox,
                    label,
                    detection.confidence,
                    color,
                    thickness
                )

            # Draw ROI zones
            for zone in camera_roi.forbidden_zones:
                cv2.polylines(
                    frame,
                    [zone.points],
                    isClosed=True,
                    color=(0, 255, 255),  # Yellow
                    thickness=2
                )
                # Label ROI
                center = tuple(zone.points.mean(axis=0).astype(int))
                cv2.putText(
                    frame,
                    f"ROI: {zone.name}",
                    center,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2
                )

            # Overlay stats
            stats_y = 30
            cv2.putText(
                frame,
                f"Frame: {frame_idx}/{total_frames}",
                (10, stats_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )
            cv2.putText(
                frame,
                f"ALERTS: {total_alerts}",
                (10, stats_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255) if total_alerts > 0 else (255, 255, 255),
                2
            )
            cv2.putText(
                frame,
                f"Synthetic Objects: {len(synthetic_detections)}",
                (10, stats_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )
            cv2.putText(
                frame,
                f"Type: {current_object_type}",
                (10, stats_y + 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            # Show frame
            cv2.imshow("Alert Test - Synthetic Objects", frame)

        # Handle keyboard
        key = cv2.waitKey(1 if not paused else 0) & 0xFF

        if key == ord('q') or key == 27:  # Q or ESC
            break
        elif key == ord(' '):  # SPACE
            paused = not paused
            print("PAUSED" if paused else "RESUMED")
        elif key == ord('s'):  # S
            filename = f"alert_test_frame_{frame_idx}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")
        elif key == ord('1'):
            current_object_type = "box"
            print(f"Object type: {current_object_type}")
        elif key == ord('2'):
            current_object_type = "tire"
            print(f"Object type: {current_object_type}")
        elif key == ord('3'):
            current_object_type = "debris"
            print(f"Object type: {current_object_type}")

    cap.release()
    cv2.destroyAllWindows()

    print()
    print("=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Total Alerts: {total_alerts}")
    print(f"Frames Processed: {frame_idx}")
    print(f"Alert Rate: {total_alerts / max(frame_idx, 1) * 100:.2f}%")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
