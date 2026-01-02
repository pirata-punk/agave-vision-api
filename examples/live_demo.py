#!/usr/bin/env python3
"""
Live Demo - Real-time Object Detection Showcase

Professional demonstration of the trained YOLO model on archived videos.
Shows real-time classification with bounding boxes, labels, and statistics.

Usage:
    python examples/live_demo.py --video path/to/video.mp4
    python examples/live_demo.py --video path/to/video.mp4 --model path/to/model.pt
    python examples/live_demo.py --webcam  # Use webcam instead

Controls:
    SPACE - Pause/Resume
    Q/ESC - Quit
    S     - Save screenshot
    R     - Reset statistics
"""

import argparse
import sys
import time
from pathlib import Path
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

try:
    from agave_vision.core.roi import ROIManager
    from agave_vision.core.inference import YOLOInference, Detection
    ROI_SUPPORT = True
except ImportError:
    ROI_SUPPORT = False


class LiveDemo:
    """Professional live demo for object detection."""

    def __init__(
        self,
        model_path: str,
        video_path: str = None,
        webcam: bool = False,
        confidence: float = 0.25,
        display_width: int = 1280,
        display_height: int = 720,
        roi_config: str = None,
        camera_id: str = None,
    ):
        self.model_path = Path(model_path)
        self.video_path = Path(video_path) if video_path else None
        self.webcam = webcam
        self.confidence = confidence
        self.display_width = display_width
        self.display_height = display_height
        self.camera_id = camera_id

        # Load model
        print(f"Loading model from {self.model_path}...")
        self.model = YOLO(str(self.model_path))
        self.class_names = self.model.names
        print(f"✓ Model loaded. Classes: {list(self.class_names.values())}")

        # Load ROI configuration if provided
        self.roi_manager = None
        self.camera_roi = None
        if roi_config and camera_id and ROI_SUPPORT:
            roi_path = Path(roi_config)
            if roi_path.exists():
                try:
                    self.roi_manager = ROIManager(roi_path)
                    self.camera_roi = self.roi_manager.get_camera_rois(camera_id)
                    if self.camera_roi:
                        print(f"✓ ROI config loaded for {camera_id}")
                        print(f"  Strict mode: {self.camera_roi.strict_mode}")
                        print(f"  Allowed classes: {list(self.camera_roi.allowed_classes)}")
                        print(f"  Forbidden zones: {len(self.camera_roi.forbidden_zones)}")
                    else:
                        print(f"⚠ No ROI config found for camera {camera_id}")
                except Exception as e:
                    print(f"⚠ Failed to load ROI config: {e}")
            else:
                print(f"⚠ ROI config file not found: {roi_config}")
        elif roi_config and not ROI_SUPPORT:
            print("⚠ ROI support not available (install package: pip install -e .)")

        # Class colors (BGR format for OpenCV)
        self.colors = {
            'object': (0, 0, 255),      # Red
            'pine': (0, 255, 0),        # Green
            'worker': (255, 165, 0),    # Orange
        }

        # Statistics
        self.frame_count = 0
        self.total_detections = defaultdict(int)
        self.total_alerts = 0  # NEW: Track alert-triggering detections
        self.fps_history = deque(maxlen=30)
        self.paused = False

        # Screenshot counter
        self.screenshot_count = 0

    def run(self):
        """Run the live demo."""
        # Open video source
        if self.webcam:
            cap = cv2.VideoCapture(0)
            source_name = "Webcam"
        else:
            cap = cv2.VideoCapture(str(self.video_path))
            source_name = self.video_path.name

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video source: {source_name}")

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"\n{'='*60}")
        print(f"🎥 LIVE DEMO - Agave Vision Object Detection")
        print(f"{'='*60}")
        print(f"Source: {source_name}")
        print(f"FPS: {fps:.2f}")
        if not self.webcam:
            print(f"Total Frames: {total_frames:,}")
            print(f"Duration: {total_frames/fps:.1f}s")
        print(f"Model: {self.model_path.name}")
        print(f"Classes: {', '.join(self.class_names.values())}")
        print(f"\nControls: SPACE=Pause | Q/ESC=Quit | S=Screenshot | R=Reset Stats")
        print(f"{'='*60}\n")

        # Create window
        window_name = 'Agave Vision - Live Demo'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.display_width, self.display_height)

        print("📺 Opening video player window...")
        print("   If window doesn't appear, check if you're running in a GUI environment")
        print()

        try:
            while True:
                if not self.paused:
                    ret, frame = cap.read()
                    if not ret:
                        if self.webcam:
                            print("Lost webcam connection")
                            break
                        else:
                            # Loop video
                            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            self.reset_stats()
                            continue

                    self.frame_count += 1

                    # Run inference
                    start_time = time.time()
                    results = self.model(frame, conf=self.confidence, verbose=False)[0]
                    inference_time = time.time() - start_time

                    # Calculate FPS
                    current_fps = 1.0 / inference_time if inference_time > 0 else 0
                    self.fps_history.append(current_fps)
                    avg_fps = np.mean(self.fps_history)

                    # Process detections
                    annotated_frame = self.draw_detections(frame, results)

                    # Add overlay with statistics
                    display_frame = self.add_overlay(annotated_frame, avg_fps, results)
                else:
                    # Show paused frame
                    display_frame = self.add_pause_overlay(display_frame)

                # Display frame
                cv2.imshow(window_name, display_frame)

                # Handle keyboard input (30ms delay for smoother display)
                key = cv2.waitKey(30) & 0xFF
                if key == ord('q') or key == 27:  # Q or ESC
                    break
                elif key == ord(' '):  # SPACE
                    self.paused = not self.paused
                    print(f"{'⏸ Paused' if self.paused else '▶ Resumed'}")
                elif key == ord('s'):  # S
                    self.save_screenshot(display_frame)
                elif key == ord('r'):  # R
                    self.reset_stats()
                    print("📊 Statistics reset")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.print_final_stats()

    def draw_detections(self, frame, results):
        """Draw bounding boxes and labels on frame."""
        annotated = frame.copy()

        # Draw ROI polygons if configured
        if self.camera_roi:
            for zone in self.camera_roi.forbidden_zones:
                cv2.polylines(
                    annotated,
                    [zone.points],
                    isClosed=True,
                    color=(0, 255, 255),  # Yellow for forbidden zones
                    thickness=2
                )
                # Add zone label
                if zone.name:
                    cv2.putText(
                        annotated,
                        f"ROI: {zone.name}",
                        tuple(zone.points[0]),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        1
                    )

        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                class_name = self.class_names[cls]

                # Update statistics
                self.total_detections[class_name] += 1

                # Check if this detection would trigger an alert
                would_alert = False
                if self.camera_roi:
                    # Create Detection object for ROI checking
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    detection = Detection(
                        class_id=cls,
                        class_name=class_name,
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        center=(cx, cy),
                        is_unknown=conf < 0.15  # Use default unknown threshold
                    )
                    would_alert = self.camera_roi.should_alert(detection)
                    if would_alert:
                        self.total_alerts += 1

                # Get color for this class (red if alert, normal color otherwise)
                if would_alert:
                    color = (0, 0, 255)  # Red for alerts
                    thickness = 4  # Thicker for alerts
                else:
                    color = self.colors.get(class_name, (255, 255, 255))
                    thickness = 3

                # Draw bounding box
                cv2.rectangle(
                    annotated,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    color,
                    thickness
                )

                # Draw label background
                label = f"{'⚠ ALERT! ' if would_alert else ''}{class_name} {conf:.2f}"
                (label_w, label_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(
                    annotated,
                    (int(x1), int(y1) - label_h - 10),
                    (int(x1) + label_w, int(y1)),
                    color,
                    -1
                )

                # Draw label text
                cv2.putText(
                    annotated,
                    label,
                    (int(x1), int(y1) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

        return annotated

    def add_overlay(self, frame, fps, results):
        """Add professional overlay with statistics."""
        overlay = frame.copy()
        h, w = overlay.shape[:2]

        # Create semi-transparent panel for stats
        panel_height = 180
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        cv2.rectangle(panel, (0, 0), (w, panel_height), (0, 0, 0), -1)

        # Add transparency
        alpha = 0.7
        overlay[0:panel_height, 0:w] = cv2.addWeighted(
            overlay[0:panel_height, 0:w], 1 - alpha,
            panel, alpha, 0
        )

        # Add title
        cv2.putText(
            overlay,
            "AGAVE VISION - LIVE OBJECT DETECTION",
            (20, 35),
            cv2.FONT_HERSHEY_DUPLEX,
            0.9,
            (255, 255, 255),
            2
        )

        # Add FPS
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(
            overlay,
            fps_text,
            (w - 150, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        # Add frame counter
        frame_text = f"Frame: {self.frame_count:,}"
        cv2.putText(
            overlay,
            frame_text,
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        # Add current detections
        num_detections = len(results.boxes) if results.boxes is not None else 0
        det_text = f"Detections: {num_detections}"
        cv2.putText(
            overlay,
            det_text,
            (20, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        # Add detection breakdown by class
        y_offset = 130
        for class_name, color in self.colors.items():
            count = sum(1 for box in (results.boxes or [])
                       if self.class_names[int(box.cls[0])] == class_name)
            total = self.total_detections.get(class_name, 0)

            text = f"{class_name.upper()}: {count} (Total: {total:,})"
            cv2.putText(
                overlay,
                text,
                (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2
            )
            y_offset += 25

        # Add alert statistics if ROI is configured
        if self.camera_roi:
            alert_text = f"ALERTS: {self.total_alerts:,}"
            cv2.putText(
                overlay,
                alert_text,
                (w - 200, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),  # Red
                2
            )

        return overlay

    def add_pause_overlay(self, frame):
        """Add pause indicator."""
        overlay = frame.copy()
        h, w = overlay.shape[:2]

        # Add pause text
        text = "PAUSED"
        (text_w, text_h), _ = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_DUPLEX, 2.0, 3
        )
        x = (w - text_w) // 2
        y = (h + text_h) // 2

        # Background
        cv2.rectangle(
            overlay,
            (x - 20, y - text_h - 20),
            (x + text_w + 20, y + 20),
            (0, 0, 0),
            -1
        )

        # Text
        cv2.putText(
            overlay,
            text,
            (x, y),
            cv2.FONT_HERSHEY_DUPLEX,
            2.0,
            (255, 255, 0),
            3
        )

        return overlay

    def save_screenshot(self, frame):
        """Save current frame as screenshot."""
        self.screenshot_count += 1
        filename = f"demo_screenshot_{self.screenshot_count:03d}.jpg"
        cv2.imwrite(filename, frame)
        print(f"📸 Screenshot saved: {filename}")

    def reset_stats(self):
        """Reset statistics."""
        self.frame_count = 0
        self.total_detections.clear()
        self.total_alerts = 0
        self.fps_history.clear()

    def print_final_stats(self):
        """Print final statistics."""
        print(f"\n{'='*60}")
        print(f"📊 DEMO STATISTICS")
        print(f"{'='*60}")
        print(f"Total Frames Processed: {self.frame_count:,}")
        print(f"Average FPS: {np.mean(self.fps_history):.2f}" if self.fps_history else "N/A")
        print(f"\nDetections by Class:")
        for class_name in sorted(self.total_detections.keys()):
            count = self.total_detections[class_name]
            print(f"  {class_name.upper()}: {count:,}")
        if self.camera_roi:
            print(f"\nAlert Statistics:")
            print(f"  Total Alerts: {self.total_alerts:,}")
            print(f"  Strict Mode: {'Enabled' if self.camera_roi.strict_mode else 'Disabled'}")
            print(f"  Allowed Classes: {', '.join(self.camera_roi.allowed_classes)}")
        print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Live demo for Agave Vision object detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run demo on video file
  python examples/live_demo.py --video data/videos/video.mp4

  # Use custom model
  python examples/live_demo.py --video data/videos/video.mp4 --model models/custom/best.pt

  # Use webcam
  python examples/live_demo.py --webcam

  # Adjust confidence threshold
  python examples/live_demo.py --video data/videos/video.mp4 --confidence 0.5

  # Enable ROI alerting (NEW - Phase 1 enhancement)
  python examples/live_demo.py --video data/videos/video.mp4 \\
      --roi-config configs/rois.yaml --camera-id cam_nave3_hornos
        """
    )

    parser.add_argument(
        "--video",
        type=str,
        help="Path to video file"
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Use webcam instead of video file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/yolov8n_pina/exp/weights/best.pt",
        help="Path to trained model (default: models/yolov8n_pina/exp/weights/best.pt)"
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.25,
        help="Confidence threshold for detections (default: 0.25)"
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="Display width (default: 1280)"
    )
    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="Display height (default: 720)"
    )
    parser.add_argument(
        "--roi-config",
        type=str,
        help="Path to ROI configuration file (e.g., configs/rois.yaml)"
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        help="Camera ID for ROI lookup (required if --roi-config is specified)"
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.webcam and not args.video:
        parser.error("Either --video or --webcam must be specified")

    if args.video and not Path(args.video).exists():
        parser.error(f"Video file not found: {args.video}")

    if not Path(args.model).exists():
        parser.error(f"Model file not found: {args.model}")

    # Run demo
    demo = LiveDemo(
        model_path=args.model,
        video_path=args.video,
        webcam=args.webcam,
        confidence=args.confidence,
        display_width=args.width,
        display_height=args.height,
        roi_config=args.roi_config,
        camera_id=args.camera_id,
    )

    demo.run()


if __name__ == "__main__":
    main()
