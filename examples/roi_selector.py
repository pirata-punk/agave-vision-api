#!/usr/bin/env python3
"""
Interactive ROI Selector

GUI tool for manually selecting Region of Interest (ROI) polygons for cameras.
Click points on a video frame to define forbidden zones where objects should trigger alerts.

Usage:
    # Select ROI from video file
    python examples/roi_selector.py --video path/to/video.mp4 --camera-id cam_nave3_hornos

    # Select ROI from specific frame
    python examples/roi_selector.py --video path/to/video.mp4 --camera-id cam_id --frame 100

Controls:
    LEFT CLICK  - Add point to polygon
    RIGHT CLICK - Remove last point
    C           - Clear all points
    N           - Finish current ROI and start new one
    S           - Save ROIs to config file
    Q/ESC       - Quit without saving
    ENTER       - Save and quit
"""

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import yaml


class ROISelector:
    """Interactive ROI polygon selector."""

    def __init__(
        self,
        video_path: str,
        camera_id: str,
        frame_number: int = 0,
        config_path: str = "configs/rois.yaml"
    ):
        self.video_path = Path(video_path)
        self.camera_id = camera_id
        self.frame_number = frame_number
        self.config_path = Path(config_path)

        # ROI data
        self.rois = []  # List of completed ROIs
        self.current_points = []  # Points for current ROI being drawn
        self.roi_names = []  # Names of ROIs

        # Display settings
        self.window_name = f"ROI Selector - {camera_id}"
        self.display_frame = None
        self.original_frame = None

        # Colors
        self.current_color = (0, 255, 255)  # Yellow for current polygon
        self.completed_color = (0, 0, 255)  # Red for completed polygons
        self.point_color = (255, 0, 0)  # Blue for points

    def load_frame(self):
        """Load specific frame from video."""
        cap = cv2.VideoCapture(str(self.video_path))

        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        # Seek to specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, self.frame_number)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            raise RuntimeError(f"Failed to read frame {self.frame_number}")

        self.original_frame = frame.copy()
        self.display_frame = frame.copy()

        print(f"✓ Loaded frame {self.frame_number} from {self.video_path.name}")
        print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for polygon drawing."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add point
            self.current_points.append((x, y))
            self.update_display()
            print(f"  Point {len(self.current_points)}: ({x}, {y})")

        elif event == cv2.EVENT_RBUTTONDOWN:
            # Remove last point
            if self.current_points:
                removed = self.current_points.pop()
                self.update_display()
                print(f"  Removed point: {removed}")

    def update_display(self):
        """Update display with current polygons."""
        self.display_frame = self.original_frame.copy()

        # Draw completed ROIs
        for i, roi_points in enumerate(self.rois):
            if len(roi_points) >= 3:
                # Draw filled polygon with transparency
                overlay = self.display_frame.copy()
                pts = np.array(roi_points, np.int32)
                cv2.fillPoly(overlay, [pts], self.completed_color)
                cv2.addWeighted(overlay, 0.3, self.display_frame, 0.7, 0, self.display_frame)

                # Draw polygon outline
                cv2.polylines(
                    self.display_frame,
                    [pts],
                    isClosed=True,
                    color=self.completed_color,
                    thickness=2
                )

                # Draw ROI label
                roi_name = self.roi_names[i] if i < len(self.roi_names) else f"ROI {i+1}"
                cv2.putText(
                    self.display_frame,
                    roi_name,
                    (roi_points[0][0], roi_points[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    self.completed_color,
                    2
                )

        # Draw current polygon being edited
        if len(self.current_points) >= 2:
            # Draw lines between points
            for i in range(len(self.current_points) - 1):
                cv2.line(
                    self.display_frame,
                    self.current_points[i],
                    self.current_points[i + 1],
                    self.current_color,
                    2
                )

            # Draw closing line if we have 3+ points
            if len(self.current_points) >= 3:
                cv2.line(
                    self.display_frame,
                    self.current_points[-1],
                    self.current_points[0],
                    self.current_color,
                    2
                )

        # Draw points
        for point in self.current_points:
            cv2.circle(self.display_frame, point, 5, self.point_color, -1)

        # Add instructions overlay
        self.add_instructions()

        cv2.imshow(self.window_name, self.display_frame)

    def add_instructions(self):
        """Add instructions overlay to display."""
        h, w = self.display_frame.shape[:2]

        # Create semi-transparent panel
        panel_height = 160
        overlay = self.display_frame.copy()
        cv2.rectangle(
            overlay,
            (0, h - panel_height),
            (w, h),
            (0, 0, 0),
            -1
        )
        cv2.addWeighted(overlay, 0.7, self.display_frame, 0.3, 0, self.display_frame)

        # Add instructions text
        y = h - panel_height + 25
        instructions = [
            f"Camera: {self.camera_id}  |  Current Points: {len(self.current_points)}  |  ROIs: {len(self.rois)}",
            "",
            "LEFT CLICK: Add point  |  RIGHT CLICK: Remove point  |  C: Clear",
            "N: New ROI  |  S: Save  |  ENTER: Save & Quit  |  Q/ESC: Quit"
        ]

        for text in instructions:
            cv2.putText(
                self.display_frame,
                text,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            y += 30

    def finish_current_roi(self):
        """Finish current ROI and start new one."""
        if len(self.current_points) < 3:
            print("⚠ Need at least 3 points to create an ROI")
            return False

        # Get ROI name from user
        print(f"\nFinishing ROI with {len(self.current_points)} points")
        roi_name = input("Enter ROI name (e.g., 'loading_zone'): ").strip()
        if not roi_name:
            roi_name = f"roi_{len(self.rois) + 1}"

        self.rois.append(self.current_points.copy())
        self.roi_names.append(roi_name)
        self.current_points = []

        print(f"✓ ROI '{roi_name}' created")
        print(f"  Total ROIs: {len(self.rois)}")
        self.update_display()
        return True

    def save_to_config(self):
        """Save ROIs to YAML configuration file."""
        # Finish current ROI if it has points
        if len(self.current_points) >= 3:
            print("\n⚠ You have unsaved points. Finishing current ROI...")
            if not self.finish_current_roi():
                return False

        if not self.rois:
            print("⚠ No ROIs to save")
            return False

        # Load existing config or create new
        if self.config_path.exists():
            with open(self.config_path) as f:
                config = yaml.safe_load(f) or {}
        else:
            config = {}

        if 'cameras' not in config:
            config['cameras'] = []

        # Find or create camera entry
        camera_config = None
        for cam in config['cameras']:
            if cam.get('camera_id') == self.camera_id:
                camera_config = cam
                break

        if camera_config is None:
            camera_config = {'camera_id': self.camera_id}
            config['cameras'].append(camera_config)

        # Update forbidden ROIs
        camera_config['forbidden_rois'] = []
        for roi_points, roi_name in zip(self.rois, self.roi_names):
            # Convert tuples to lists for YAML compatibility
            points_as_lists = [list(point) if isinstance(point, tuple) else point for point in roi_points]
            camera_config['forbidden_rois'].append({
                'name': roi_name,
                'points': points_as_lists
            })

        # Set default classes if not present
        if 'allowed_classes' not in camera_config:
            camera_config['allowed_classes'] = ['pine', 'worker']
        if 'alert_classes' not in camera_config:
            camera_config['alert_classes'] = ['object']
        if 'strict_mode' not in camera_config:
            camera_config['strict_mode'] = True  # Default to strict mode (Phase 1 enhancement)

        # Save to file
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        print(f"\n{'='*60}")
        print(f"✓ ROI configuration saved to {self.config_path}")
        print(f"{'='*60}")
        print(f"Camera ID: {self.camera_id}")
        print(f"ROIs defined: {len(self.rois)}")
        for i, (roi_name, roi_points) in enumerate(zip(self.roi_names, self.rois)):
            print(f"  {i+1}. {roi_name}: {len(roi_points)} points")
        print(f"{'='*60}\n")

        return True

    def run(self):
        """Run the interactive ROI selector."""
        self.load_frame()

        # Create window and set mouse callback
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)

        # Resize window to fit screen
        h, w = self.original_frame.shape[:2]
        max_width = 1600
        max_height = 900
        if w > max_width or h > max_height:
            scale = min(max_width / w, max_height / h)
            cv2.resizeWindow(self.window_name, int(w * scale), int(h * scale))

        print(f"\n{'='*60}")
        print(f"📐 ROI Selector - Interactive Mode")
        print(f"{'='*60}")
        print(f"Instructions:")
        print(f"  1. Click on the image to add points for your ROI polygon")
        print(f"  2. Right-click to remove the last point")
        print(f"  3. Press 'N' when done with current ROI to start a new one")
        print(f"  4. Press 'S' to save all ROIs to config file")
        print(f"  5. Press ENTER to save and quit")
        print(f"{'='*60}\n")

        self.update_display()

        while True:
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == 27:  # Q or ESC
                print("\n⚠ Quitting without saving")
                break

            elif key == ord('c'):  # Clear current points
                self.current_points = []
                self.update_display()
                print("✓ Cleared current points")

            elif key == ord('n'):  # New ROI
                if self.finish_current_roi():
                    print("\n→ Start drawing next ROI...")

            elif key == ord('s'):  # Save
                if self.save_to_config():
                    print("✓ ROIs saved. Continue editing or press ENTER to quit.")

            elif key == 13:  # ENTER - Save and quit
                if self.save_to_config():
                    break
                else:
                    print("⚠ Nothing to save or save failed")

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(
        description="Interactive ROI selector for camera configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Select ROI from video
  python examples/roi_selector.py --video data/videos/video.mp4 --camera-id cam_nave3_hornos

  # Use specific frame
  python examples/roi_selector.py --video data/videos/video.mp4 --camera-id cam1 --frame 500

  # Specify custom config file
  python examples/roi_selector.py --video data/videos/video.mp4 --camera-id cam1 --config configs/custom_rois.yaml
        """
    )

    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to video file"
    )
    parser.add_argument(
        "--camera-id",
        type=str,
        required=True,
        help="Camera ID (e.g., 'cam_nave3_hornos')"
    )
    parser.add_argument(
        "--frame",
        type=int,
        default=0,
        help="Frame number to use for ROI selection (default: 0)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/rois.yaml",
        help="Path to ROI config file (default: configs/rois.yaml)"
    )

    args = parser.parse_args()

    # Validate video file
    if not Path(args.video).exists():
        print(f"❌ Error: Video file not found: {args.video}")
        sys.exit(1)

    # Run selector
    selector = ROISelector(
        video_path=args.video,
        camera_id=args.camera_id,
        frame_number=args.frame,
        config_path=args.config
    )

    selector.run()


if __name__ == "__main__":
    main()
