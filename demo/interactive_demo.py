#!/usr/bin/env python3
"""
Interactive Agave Vision Demo

Real-time object detection demo with:
- Live video display with bounding boxes
- Real-time statistics (detections, alerts)
- GUI controls to toggle synthetic object injection
- ROI visualization
- Mouse click controls for object placement

Usage:
    python demo/interactive_demo.py

Controls:
    SPACE       - Toggle synthetic object injection (force alerts)
    LEFT CLICK  - Place synthetic object at cursor position
    RIGHT CLICK - Remove nearest synthetic object
    Q           - Quit
    P           - Pause/Resume
    R           - Reset statistics
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np
from datetime import datetime
import time
import json
import yaml

from agave_vision.ml_api import AgaveVisionML
from agave_vision.config import get_default_model_path, get_model_info


class SyntheticObjectInjector:
    """Inject synthetic objects into frames to trigger alerts."""

    def __init__(self):
        self.templates = self._create_object_templates()
        self.object_types = list(self.templates.keys())

    def _create_object_templates(self) -> dict:
        """Create synthetic object templates (geometric shapes)."""
        templates = {}

        # Template 1: Box (simulating equipment/crate) - LARGER and more detailed
        box = np.zeros((200, 150, 3), dtype=np.uint8)
        # Main box body (brown/tan)
        cv2.rectangle(box, (10, 10), (140, 190), (101, 67, 33), -1)
        # Shading for 3D effect
        cv2.rectangle(box, (10, 10), (140, 50), (130, 90, 50), -1)  # Top lighter
        cv2.rectangle(box, (10, 150), (140, 190), (70, 45, 20), -1)  # Bottom darker
        # Edges
        cv2.rectangle(box, (10, 10), (140, 190), (50, 30, 10), 3)
        # Straps/bands
        cv2.rectangle(box, (10, 80), (140, 95), (40, 40, 40), -1)
        cv2.rectangle(box, (10, 115), (140, 130), (40, 40, 40), -1)
        templates["box"] = box

        # Template 2: Barrel (circular object) - LARGER with more detail
        barrel = np.zeros((220, 180, 3), dtype=np.uint8)
        # Main barrel body (metallic gray)
        cv2.ellipse(barrel, (90, 110), (80, 100), 0, 0, 360, (120, 120, 120), -1)
        # Barrel bands
        cv2.ellipse(barrel, (90, 50), (70, 20), 0, 0, 360, (60, 60, 60), -1)
        cv2.ellipse(barrel, (90, 110), (75, 25), 0, 0, 360, (60, 60, 60), -1)
        cv2.ellipse(barrel, (90, 170), (70, 20), 0, 0, 360, (60, 60, 60), -1)
        # Highlight for metallic effect
        cv2.ellipse(barrel, (70, 90), (30, 40), 0, 0, 180, (160, 160, 160), -1)
        # Edge
        cv2.ellipse(barrel, (90, 110), (80, 100), 0, 0, 360, (50, 50, 50), 3)
        templates["barrel"] = barrel

        # Template 3: Large Equipment Box - High contrast for detection
        equipment = np.zeros((250, 200, 3), dtype=np.uint8)
        # Bright orange/yellow box (high visibility)
        cv2.rectangle(equipment, (15, 15), (185, 235), (0, 140, 255), -1)
        # Black stripes for pattern
        cv2.rectangle(equipment, (15, 60), (185, 80), (0, 0, 0), -1)
        cv2.rectangle(equipment, (15, 120), (185, 140), (0, 0, 0), -1)
        cv2.rectangle(equipment, (15, 180), (185, 200), (0, 0, 0), -1)
        # Warning symbol (triangle)
        pts_triangle = np.array([[100, 100], [130, 160], [70, 160]], np.int32)
        cv2.fillPoly(equipment, [pts_triangle], (0, 0, 0))
        # Border
        cv2.rectangle(equipment, (15, 15), (185, 235), (0, 0, 0), 4)
        templates["equipment"] = equipment

        return templates

    def inject(
        self,
        frame: np.ndarray,
        position: tuple[int, int],
        object_type: str = "box",
        alpha: float = 0.8,
    ) -> np.ndarray:
        """
        Inject synthetic object into frame.

        Args:
            frame: Input frame
            position: (x, y) position to place object
            object_type: Type of object ('box', 'barrel', 'debris')
            alpha: Transparency (0=transparent, 1=opaque)

        Returns:
            Frame with injected object
        """
        template = self.templates.get(object_type, self.templates["box"])
        h, w = template.shape[:2]
        x, y = position

        # Ensure position is within bounds
        x = max(0, min(x, frame.shape[1] - w))
        y = max(0, min(y, frame.shape[0] - h))

        # Extract ROI
        roi = frame[y : y + h, x : x + w]

        # Blend template with ROI
        blended = cv2.addWeighted(template, alpha, roi, 1 - alpha, 0)
        frame[y : y + h, x : x + w] = blended

        return frame


class InteractiveDemo:
    """Interactive demo application."""

    def __init__(self):
        # Initialize ML API with default model and ROI config
        # enable_tracking=True by default for object persistence
        print("🔧 Initializing Agave Vision ML API...")
        self.ml = AgaveVisionML(roi_config_path="configs/rois.yaml", enable_tracking=True)

        model_info = get_model_info()
        print(f"✓ Model loaded: {model_info['name']} v{model_info['version']}")
        print(f"  Architecture: {model_info['architecture']}")
        print(f"  Classes: {', '.join(model_info['classes'])}")

        # Load display colors configuration
        self.colors = self._load_color_config()

        # Video path
        self.video_path = "demo/NAVE 3_HORNOS B CAM 3_20250923125056_20250923131506.mp4"
        self.camera_id = "cam_nave3_hornos_b_cam3"

        # Synthetic object injector
        self.injector = SyntheticObjectInjector()

        # State
        self.inject_enabled = False
        self.paused = False
        self.injected_objects = []  # List of (x, y, object_type) tuples
        self.all_alerts = []  # Store all alerts for final log

        # Statistics - Track unique objects instead of frame-based counts
        self.stats = {
            "total_frames": 0,
            "unique_objects_seen": set(),  # Track unique tracking IDs
            "detections_by_class": {},
            "unique_by_class": {},  # Track unique IDs per class
            "total_alerts": 0,
            "fps": 0.0,
        }

        # Display settings
        self.display_width = 1280
        self.display_height = 720

        print(f"\n📹 Video: {self.video_path}")
        print(f"🎯 Camera ID: {self.camera_id}\n")

    def _load_color_config(self) -> dict:
        """Load color configuration from YAML file."""
        config_path = Path("configs/display.yaml")
        if config_path.exists():
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        else:
            # Default colors if config not found
            return {
                "class_colors": {
                    "object": [0, 165, 255],  # Orange
                    "pine": [0, 255, 0],       # Green
                    "worker": [255, 255, 0],   # Cyan
                    "unknown": [128, 0, 128],  # Purple
                },
                "roi_colors": {
                    "forbidden_zone": [0, 100, 200],  # Dark red
                    "forbidden_border": [0, 0, 255],   # Red
                },
                "alert_color": [0, 0, 255],  # Red
            }

    def _get_detection_color(self, class_name: str, is_alert: bool) -> tuple:
        """Get color for detection based on class and alert status."""
        if is_alert:
            # Red reserved ONLY for alerts
            return tuple(self.colors["alert_color"])
        else:
            # Use class-specific color
            class_colors = self.colors.get("class_colors", {})
            color = class_colors.get(class_name, [255, 255, 255])  # Default white
            return tuple(color)

    def reset_stats(self):
        """Reset statistics counters."""
        self.stats = {
            "total_frames": 0,
            "unique_objects_seen": set(),
            "detections_by_class": {},
            "unique_by_class": {},
            "total_alerts": 0,
            "fps": 0.0,
        }
        self.all_alerts = []
        # Also reset the tracker
        if self.ml.tracker:
            self.ml.tracker.reset()
        print("📊 Statistics reset")

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for object placement."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Left click - add object at position
            # Scale coordinates back to original frame size
            scale_x = param['original_width'] / self.display_width
            scale_y = param['original_height'] / self.display_height
            orig_x = int(x * scale_x)
            orig_y = int(y * scale_y)

            object_type = np.random.choice(self.injector.object_types)
            self.injected_objects.append((orig_x, orig_y, object_type))
            print(f"➕ Added {object_type} at ({orig_x}, {orig_y})")

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Right click - remove nearest object
            if self.injected_objects:
                # Scale coordinates back to original frame size
                scale_x = param['original_width'] / self.display_width
                scale_y = param['original_height'] / self.display_height
                orig_x = int(x * scale_x)
                orig_y = int(y * scale_y)

                # Find nearest object
                min_dist = float('inf')
                nearest_idx = -1
                for idx, (ox, oy, _) in enumerate(self.injected_objects):
                    dist = np.sqrt((ox - orig_x)**2 + (oy - orig_y)**2)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_idx = idx

                if nearest_idx >= 0:
                    removed = self.injected_objects.pop(nearest_idx)
                    print(f"➖ Removed {removed[2]} at ({removed[0]}, {removed[1]})")

    def draw_detections(
        self, frame: np.ndarray, detections: list, alerts: list
    ) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame.

        Args:
            frame: Input frame
            detections: List of detection dictionaries
            alerts: List of alert dictionaries

        Returns:
            Annotated frame
        """
        # Create set of alerted detection centers for highlighting
        alerted_centers = set()
        for alert in alerts:
            det = alert.get("detection", {})
            center = tuple(det.get("center", []))
            if center:
                alerted_centers.add(center)

        for det in detections:
            bbox = det["bbox"]
            x1, y1, x2, y2 = map(int, bbox)
            confidence = det["confidence"]
            class_name = det["class_name"]
            center = tuple(map(int, det["center"]))

            # Determine color based on alert status and class
            is_alert = center in alerted_centers
            color = self._get_detection_color(class_name, is_alert)
            thickness = 3 if is_alert else 2

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

            # Draw label background
            label = f"{class_name}: {confidence:.2f}"
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            )
            cv2.rectangle(
                frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1
            )

            # Draw label text
            cv2.putText(
                frame,
                label,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            # Draw center point
            cv2.circle(frame, center, 4, color, -1)

        return frame

    def draw_roi_zones(self, frame: np.ndarray) -> np.ndarray:
        """Draw ROI zones on frame."""
        if not self.ml.roi_manager:
            return frame

        camera_roi = self.ml.roi_manager.get_camera_rois(self.camera_id)
        if not camera_roi:
            return frame

        # Draw forbidden ROI zones with transparency
        overlay = frame.copy()
        zone_color = tuple(self.colors["roi_colors"]["forbidden_zone"])
        border_color = tuple(self.colors["roi_colors"]["forbidden_border"])

        for roi in camera_roi.forbidden_zones:
            points = np.array(roi.points, dtype=np.int32)
            cv2.fillPoly(overlay, [points], zone_color)  # Dark red overlay
            cv2.polylines(overlay, [points], True, border_color, 4)  # Thicker border

            # Draw ROI name with background for better visibility
            centroid = points.mean(axis=0).astype(int)
            label = roi.name
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2
            )
            # Draw label background
            cv2.rectangle(
                overlay,
                (centroid[0] - label_w // 2 - 5, centroid[1] - label_h - 5),
                (centroid[0] + label_w // 2 + 5, centroid[1] + 5),
                (0, 0, 0),
                -1,
            )
            # Draw label text
            cv2.putText(
                overlay,
                label,
                (centroid[0] - label_w // 2, centroid[1]),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

        # Blend overlay with original frame (more visible)
        frame = cv2.addWeighted(frame, 0.5, overlay, 0.5, 0)

        return frame

    def draw_stats_panel(self, frame: np.ndarray) -> np.ndarray:
        """Draw statistics panel on frame."""
        h, w = frame.shape[:2]
        panel_height = 220
        panel_width = 380

        # Create semi-transparent panel (more opaque)
        overlay = frame.copy()
        cv2.rectangle(
            overlay, (10, 10), (panel_width, panel_height), (0, 0, 0), -1
        )
        frame = cv2.addWeighted(frame, 0.4, overlay, 0.6, 0)

        # Title
        y = 40
        cv2.putText(
            frame,
            "STATISTICS",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
        )

        # Stats
        y += 40
        cv2.putText(
            frame,
            f"Frames: {self.stats['total_frames']}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2,
        )

        y += 30
        cv2.putText(
            frame,
            f"FPS: {self.stats['fps']:.1f}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (200, 200, 200),
            2,
        )

        y += 30
        cv2.putText(
            frame,
            f"Alerts: {self.stats['total_alerts']}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255) if self.stats["total_alerts"] > 0 else (200, 200, 200),
            2,
        )

        y += 35
        cv2.putText(
            frame,
            "Unique objects tracked:",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        y += 25
        cv2.putText(
            frame,
            f"  Total: {len(self.stats['unique_objects_seen'])}",
            (25, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            2,
        )

        # Show unique counts per class
        for class_name in sorted(self.stats["unique_by_class"].keys()):
            unique_count = len(self.stats["unique_by_class"][class_name])
            frame_count = self.stats["detections_by_class"].get(class_name, 0)
            y += 25
            cv2.putText(
                frame,
                f"  {class_name}: {unique_count} unique ({frame_count} frames)",
                (25, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (200, 200, 200),
                2,
            )

        return frame

    def draw_controls_panel(self, frame: np.ndarray) -> np.ndarray:
        """Draw controls panel on frame."""
        h, w = frame.shape[:2]
        panel_height = 200
        panel_width = 450

        # Position at bottom-left
        y_start = h - panel_height - 10

        # Create semi-transparent panel (more opaque)
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (10, y_start),
            (panel_width, h - 10),
            (0, 0, 0),
            -1,
        )
        frame = cv2.addWeighted(frame, 0.4, overlay, 0.6, 0)

        # Title
        y = y_start + 30
        cv2.putText(
            frame,
            "CONTROLS",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # Control instructions
        y += 35
        cv2.putText(
            frame,
            "SPACE       - Toggle Auto-Inject",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            2,
        )

        y += 25
        cv2.putText(
            frame,
            "LEFT CLICK  - Place Object",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            2,
        )

        y += 25
        cv2.putText(
            frame,
            "RIGHT CLICK - Remove Object",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            2,
        )

        y += 25
        cv2.putText(
            frame,
            "P - Pause | R - Reset | Q - Quit",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            2,
        )

        # Injection status
        y += 35
        status_text = "AUTO-INJECT" if self.inject_enabled else "MANUAL"
        status_color = (0, 255, 255) if self.inject_enabled else (0, 255, 0)
        cv2.putText(
            frame,
            f"Mode: {status_text} | Objects: {len(self.injected_objects)}",
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2,
        )

        return frame

    def run(self):
        """Run the interactive demo."""
        # Open video
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"❌ Error: Could not open video: {self.video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"📹 Video: {fps:.1f} FPS, {total_frames} frames ({original_width}x{original_height})")
        print("\n" + "=" * 70)
        print("CONTROLS:")
        print("  SPACE       - Toggle auto-inject (places objects in ROI automatically)")
        print("  LEFT CLICK  - Place synthetic object at cursor position")
        print("  RIGHT CLICK - Remove nearest synthetic object")
        print("  P           - Pause/Resume")
        print("  R           - Reset statistics")
        print("  Q           - Quit")
        print("=" * 70 + "\n")

        # Create window and set mouse callback
        cv2.namedWindow("Agave Vision - Interactive Demo")
        mouse_param = {
            'original_width': original_width,
            'original_height': original_height
        }
        cv2.setMouseCallback("Agave Vision - Interactive Demo", self.mouse_callback, mouse_param)

        frame_count = 0
        fps_start = time.time()
        current_frame = None

        while True:
            if not self.paused:
                ret, frame = cap.read()
                if not ret:
                    # Loop video
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                frame_count += 1
                self.stats["total_frames"] += 1

                # Inject synthetic objects if auto-inject is enabled
                if self.inject_enabled:
                    # Get ROI center to inject object there
                    camera_roi = self.ml.roi_manager.get_camera_rois(self.camera_id)
                    if camera_roi and camera_roi.forbidden_zones:
                        roi = camera_roi.forbidden_zones[0]
                        points = np.array(roi.points, dtype=np.int32)
                        centroid = points.mean(axis=0).astype(int)
                        frame = self.injector.inject(
                            frame,
                            tuple(centroid - 50),
                            object_type=np.random.choice(self.injector.object_types),
                        )

                # Inject manually placed objects
                for obj_x, obj_y, obj_type in self.injected_objects:
                    frame = self.injector.inject(
                        frame,
                        (obj_x, obj_y),
                        object_type=obj_type,
                    )

                # Run inference
                result = self.ml.predict_frame(
                    frame, camera_id=self.camera_id, store_alerts=False
                )

                detections = result["detections"]
                alerts = result["alerts"]

                # Inject synthetic detections for manually placed objects
                # These are marked as "unknown" (low confidence) to trigger unknown_object alerts
                if self.injected_objects:
                    for obj_x, obj_y, obj_type in self.injected_objects:
                        # Get object size based on template
                        template = self.injector.templates[obj_type]
                        h, w = template.shape[:2]

                        # Create synthetic detection as UNKNOWN (below threshold)
                        # This represents an unrecognized object that should trigger alerts in ROI
                        synthetic_det = {
                            "bbox": [obj_x, obj_y, obj_x + w, obj_y + h],
                            "confidence": 0.12,  # Below typical unknown threshold (0.15)
                            "class_name": "unknown",  # Mark as unknown/unrecognized
                            "center": [obj_x + w//2, obj_y + h//2],
                            "is_unknown": True  # Explicitly mark as unknown
                        }

                        # Add to detections list
                        detections.append(synthetic_det)

                        # Check if it should alert (unknown object in ROI = alert)
                        if self.ml.roi_manager:
                            camera_roi = self.ml.roi_manager.get_camera_rois(self.camera_id)
                            if camera_roi:
                                # Create a Detection object for ROI check
                                from agave_vision.core.inference import Detection
                                det_obj = Detection(
                                    bbox=synthetic_det["bbox"],
                                    confidence=synthetic_det["confidence"],
                                    class_id=-1,  # Unknown class ID
                                    class_name="unknown",
                                    center=tuple(synthetic_det["center"]),
                                    is_unknown=True
                                )

                                # Unknown objects in strict mode ROIs should always alert
                                if camera_roi.should_alert(det_obj):
                                    alert_dict = {
                                        "timestamp": datetime.now().isoformat(),
                                        "camera_id": self.camera_id,
                                        "detection": synthetic_det,
                                        "alert_type": "roi_violation",
                                        "violation_type": "unknown_object",
                                        "reason": f"Unknown/unrecognized object (synthetic) detected in forbidden zone"
                                    }
                                    alerts.append(alert_dict)

                # Debug: Print when objects are injected and if alerts are triggered
                if self.inject_enabled or self.injected_objects:
                    if frame_count % 30 == 0:  # Print every 30 frames to avoid spam
                        print(f"🔍 Frame {self.stats['total_frames']}: "
                              f"Injected {len(self.injected_objects) + (1 if self.inject_enabled else 0)} objects, "
                              f"Detected {len(detections)} items, "
                              f"Triggered {len(alerts)} alerts")
                        if detections:
                            for det in detections:
                                print(f"   └─ Detected: {det['class_name']} (conf: {det['confidence']:.2f})")
                        if alerts:
                            for alert in alerts:
                                print(f"   ⚠️ Alert: {alert['detection']['class_name']} in ROI")

                # Store alerts for final log
                if alerts:
                    for alert in alerts:
                        alert_record = {
                            "timestamp": datetime.now().isoformat(),
                            "frame_number": self.stats["total_frames"],
                            "alert": alert
                        }
                        self.all_alerts.append(alert_record)

                # Update statistics - Track unique objects instead of per-frame counts
                for det in detections:
                    class_name = det["class_name"]

                    # Track per-frame detection count (for historical purposes)
                    self.stats["detections_by_class"][class_name] = (
                        self.stats["detections_by_class"].get(class_name, 0) + 1
                    )

                    # Track unique objects using tracking_id
                    if "tracking_id" in det and det["tracking_id"]:
                        tracking_id = det["tracking_id"]
                        self.stats["unique_objects_seen"].add(tracking_id)

                        # Track unique objects per class
                        if class_name not in self.stats["unique_by_class"]:
                            self.stats["unique_by_class"][class_name] = set()
                        self.stats["unique_by_class"][class_name].add(tracking_id)

                self.stats["total_alerts"] += len(alerts)

                # Calculate FPS
                if frame_count % 30 == 0:
                    elapsed = time.time() - fps_start
                    self.stats["fps"] = 30 / elapsed if elapsed > 0 else 0
                    fps_start = time.time()

                # Draw visualizations
                frame = self.draw_roi_zones(frame)
                frame = self.draw_detections(frame, detections, alerts)
                frame = self.draw_stats_panel(frame)
                frame = self.draw_controls_panel(frame)

                # Resize for display
                current_frame = cv2.resize(frame, (self.display_width, self.display_height))

            else:
                # Show paused message on current frame
                if current_frame is not None:
                    overlay = current_frame.copy()
                    h, w = current_frame.shape[:2]
                    cv2.rectangle(overlay, (w // 2 - 100, h // 2 - 30), (w // 2 + 100, h // 2 + 30), (0, 0, 0), -1)
                    current_frame = cv2.addWeighted(current_frame, 0.7, overlay, 0.3, 0)
                    cv2.putText(
                        current_frame,
                        "PAUSED",
                        (w // 2 - 60, h // 2 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 255),
                        2,
                    )

            # Display frame
            if current_frame is not None:
                cv2.imshow("Agave Vision - Interactive Demo", current_frame)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("\n👋 Exiting...")
                break
            elif key == ord(" "):  # Space
                self.inject_enabled = not self.inject_enabled
                status = "ENABLED" if self.inject_enabled else "DISABLED"
                print(f"🎛️  Synthetic object injection: {status}")
            elif key == ord("p"):
                self.paused = not self.paused
                status = "PAUSED" if self.paused else "RESUMED"
                print(f"⏯️  Video {status}")
            elif key == ord("r"):
                self.reset_stats()

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()

        # Save final log with alert data
        log_filename = f"demo/alert_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Convert sets to lists for JSON serialization
        serializable_stats = {
            "total_frames": self.stats["total_frames"],
            "unique_objects_total": len(self.stats["unique_objects_seen"]),
            "unique_objects_ids": list(self.stats["unique_objects_seen"]),
            "detections_by_class": self.stats["detections_by_class"],
            "unique_by_class": {
                class_name: len(ids)
                for class_name, ids in self.stats["unique_by_class"].items()
            },
            "unique_ids_by_class": {
                class_name: list(ids)
                for class_name, ids in self.stats["unique_by_class"].items()
            },
            "total_alerts": self.stats["total_alerts"],
            "fps": self.stats["fps"],
        }

        log_data = {
            "session_info": {
                "video_file": self.video_path,
                "camera_id": self.camera_id,
                "model": get_model_info(),
                "session_end": datetime.now().isoformat()
            },
            "statistics": serializable_stats,
            "alerts": self.all_alerts,
            "injected_objects_final": [
                {"x": x, "y": y, "type": obj_type}
                for x, y, obj_type in self.injected_objects
            ]
        }

        try:
            with open(log_filename, "w") as f:
                json.dump(log_data, f, indent=2)
            print(f"\n💾 Alert log saved: {log_filename}")
        except Exception as e:
            print(f"\n⚠️  Failed to save alert log: {e}")

        # Print final statistics
        print("\n" + "=" * 70)
        print("FINAL STATISTICS")
        print("=" * 70)
        print(f"Total frames processed: {self.stats['total_frames']}")
        print(f"Total alerts: {self.stats['total_alerts']}")
        print(f"Manual objects placed: {len(self.injected_objects)}")
        print("\nDetections by class:")
        for class_name, count in sorted(self.stats["detections_by_class"].items()):
            print(f"  {class_name}: {count}")
        print("=" * 70)


def main():
    """Main entry point."""
    print("\n" + "=" * 60)
    print("  AGAVE VISION - INTERACTIVE DEMO")
    print("=" * 60 + "\n")

    demo = InteractiveDemo()
    demo.run()


if __name__ == "__main__":
    main()
