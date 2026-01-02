#!/usr/bin/env python3
"""
Integration Example: How to use Agave Vision ML API

This demonstrates how external teams would integrate our ML capabilities
into their server architecture. No HTTP, no Docker - just Python function calls.

The external team would:
1. Import and initialize the AgaveVisionML class
2. Call predict_frame() or predict_video_stream()
3. Handle results (send to frontend, store in database, trigger notifications, etc.)
4. Query alert history using get_alerts()
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
from agave_vision.ml_api import AgaveVisionML


def example_1_single_frame_inference():
    """
    Example 1: Single Frame Inference

    Most basic usage - process a single image and get detections + alerts.
    """
    print("=" * 60)
    print("Example 1: Single Frame Inference")
    print("=" * 60)
    print()

    # Initialize ML engine
    ml = AgaveVisionML(
        model_path="models/yolov8n_pina/exp/weights/best.pt",
        roi_config_path="configs/rois.yaml",
        conf_threshold=0.25,
    )

    print(f"Initialized: {ml}")
    print(f"Model info: {ml.get_model_info()}")
    print()

    # Load an image
    image_path = "data/videos/Video Volcador B Nave 3 HORNOS  CAM 3 PI„AS/NAVE 3_HORNOS B CAM 3_20250923125056_20250923131506.jpg"

    if not Path(image_path).exists():
        print(f"Image not found: {image_path}")
        print("Please provide a valid image path")
        return

    image = cv2.imread(image_path)
    print(f"Loaded image: {image.shape}")

    # Run inference
    result = ml.predict_frame(
        image,
        camera_id="cam_nave3_hornos_b_cam3"
    )

    # Display results
    print(f"Inference time: {result['inference_time_ms']:.2f}ms")
    print(f"Detections: {result['num_detections']}")
    print(f"Alerts: {result['num_alerts']}")
    print()

    # Print detection details
    for i, detection in enumerate(result['detections'], 1):
        print(f"  Detection {i}:")
        print(f"    Class: {detection['class_name']}")
        print(f"    Confidence: {detection['confidence']:.2f}")
        print(f"    BBox: {detection['bbox']}")
        print(f"    Unknown: {detection['is_unknown']}")

    print()

    # Print alert details
    if result['alerts']:
        print("ALERTS TRIGGERED:")
        for i, alert in enumerate(result['alerts'], 1):
            print(f"  Alert {i}:")
            print(f"    ROI: {alert.get('roi_name')}")
            print(f"    Violation: {alert.get('violation_type')}")
            print(f"    Class: {alert['detection']['class_name']}")
    else:
        print("No alerts triggered")

    print()


def example_2_video_stream_processing():
    """
    Example 2: Video Stream Processing

    Process video stream (file or RTSP) frame by frame.
    This is how you'd integrate with live camera feeds.
    """
    print("=" * 60)
    print("Example 2: Video Stream Processing")
    print("=" * 60)
    print()

    # Initialize ML engine
    ml = AgaveVisionML(
        model_path="models/yolov8n_pina/exp/weights/best.pt",
        roi_config_path="configs/rois.yaml",
        conf_threshold=0.25,
    )

    # Video source (could be RTSP URL for live camera)
    video_source = "data/videos/Video Volcador B Nave 3 HORNOS  CAM 3 PI„AS/NAVE 3_HORNOS B CAM 3_20250923125056_20250923131506.mp4"

    if not Path(video_source).exists():
        print(f"Video not found: {video_source}")
        return

    print(f"Processing video: {Path(video_source).name}")
    print(f"FPS limit: 5 (processes every 6th frame)")
    print(f"Max frames: 50 (for demo)")
    print()

    # Process video stream
    frame_count = 0
    alert_count = 0

    for result in ml.predict_video_stream(
        video_source=video_source,
        camera_id="cam_nave3_hornos_b_cam3",
        fps_limit=5.0,  # Process at 5 FPS
        max_frames=50,  # Limit for demo
    ):
        frame_count += 1
        alert_count += result['num_alerts']

        # Your server would handle this result:
        # - Send to frontend via WebSocket
        # - Store in your database
        # - Trigger notifications if alerts
        # - Log to your monitoring system
        # etc.

        if result['num_alerts'] > 0:
            print(f"  Frame {frame_count}: {result['num_detections']} detections, "
                  f"{result['num_alerts']} ALERTS ({result['inference_time_ms']:.1f}ms)")

    print()
    print(f"Processed {frame_count} frames")
    print(f"Total alerts: {alert_count}")
    print()


def example_3_with_persistent_storage():
    """
    Example 3: With Persistent Alert Storage

    Enable alert storage to query alert history later.
    """
    print("=" * 60)
    print("Example 3: Persistent Alert Storage")
    print("=" * 60)
    print()

    # Initialize with storage enabled
    ml = AgaveVisionML(
        model_path="models/yolov8n_pina/exp/weights/best.pt",
        roi_config_path="configs/rois.yaml",
        conf_threshold=0.25,
        enable_alert_storage=True,  # Enable persistent storage
        enable_detection_logging=True,  # Enable detection logging
    )

    print("Alert storage: ENABLED")
    print("Detection logging: ENABLED")
    print()

    # Process some frames (same as before)
    video_source = "data/videos/Video Volcador B Nave 3 HORNOS  CAM 3 PI„AS/NAVE 3_HORNOS B CAM 3_20250923125056_20250923131506.mp4"

    if not Path(video_source).exists():
        print(f"Video not found: {video_source}")
        return

    print("Processing video and storing alerts...")

    for result in ml.predict_video_stream(
        video_source=video_source,
        camera_id="cam_nave3_hornos_b_cam3",
        fps_limit=5.0,
        max_frames=20,
        store_alerts=True,  # Alerts are automatically stored
        log_detections=True,  # Detections are logged
    ):
        pass  # Results are automatically stored

    print("Done processing")
    print()

    # Query alert history
    print("Querying alert history...")
    alerts = ml.get_alerts(
        camera_id="cam_nave3_hornos_b_cam3",
        limit=10
    )

    print(f"Found {len(alerts)} alerts in storage")
    for i, alert in enumerate(alerts[:3], 1):  # Show first 3
        print(f"  Alert {i}:")
        print(f"    Time: {alert.get('timestamp')}")
        print(f"    ROI: {alert.get('roi_name')}")
        print(f"    Class: {alert['detection']['class_name']}")

    print()

    # Query detection logs
    print("Querying detection logs...")
    logs = ml.get_detection_logs(
        camera_id="cam_nave3_hornos_b_cam3",
        limit=5
    )

    print(f"Found {len(logs)} log entries")
    for i, log in enumerate(logs[:2], 1):  # Show first 2
        print(f"  Log {i}:")
        print(f"    Time: {log.get('timestamp')}")
        print(f"    Detections: {len(log.get('detections', []))}")
        print(f"    Alerts: {log.get('num_alerts', 0)}")

    print()


def example_4_roi_configuration():
    """
    Example 4: Getting ROI Configuration

    Query ROI configuration for cameras.
    """
    print("=" * 60)
    print("Example 4: ROI Configuration Query")
    print("=" * 60)
    print()

    ml = AgaveVisionML(
        model_path="models/yolov8n_pina/exp/weights/best.pt",
        roi_config_path="configs/rois.yaml",
    )

    # Get ROI info for a camera
    camera_id = "cam_nave3_hornos_b_cam3"
    roi_info = ml.get_camera_roi_info(camera_id)

    if roi_info:
        print(f"Camera: {roi_info['camera_id']}")
        print(f"Forbidden zones: {len(roi_info['forbidden_zones'])}")
        print(f"Allowed classes: {roi_info['allowed_classes']}")
        print(f"Alert classes: {roi_info['alert_classes']}")
        print(f"Strict mode: {roi_info['strict_mode']}")
        print()

        for i, zone in enumerate(roi_info['forbidden_zones'], 1):
            print(f"  Zone {i}: {zone['name']}")
            print(f"    Points: {zone['num_points']} vertices")
    else:
        print(f"No ROI configuration found for {camera_id}")

    print()


def example_5_integration_pattern():
    """
    Example 5: Typical Integration Pattern

    Shows how external team would integrate into their Flask/FastAPI server.
    """
    print("=" * 60)
    print("Example 5: Server Integration Pattern")
    print("=" * 60)
    print()

    print("Pseudo-code for Flask/FastAPI integration:")
    print()

    integration_code = '''
from flask import Flask, jsonify, request
from agave_vision.ml_api import AgaveVisionML
import cv2
import base64

app = Flask(__name__)

# Initialize ML engine once at startup
ml = AgaveVisionML(
    model_path="models/best.pt",
    roi_config_path="configs/rois.yaml",
    enable_alert_storage=True,
    enable_detection_logging=True,
)

@app.route("/api/detect", methods=["POST"])
def detect():
    """Process image and return detections + alerts."""

    # Get image from request
    image_b64 = request.json["image"]
    camera_id = request.json["camera_id"]

    # Decode image
    image_bytes = base64.b64decode(image_b64)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run ML inference
    result = ml.predict_frame(image, camera_id=camera_id)

    # Return JSON response
    return jsonify(result)

@app.route("/api/alerts", methods=["GET"])
def get_alerts():
    """Get alert history."""

    camera_id = request.args.get("camera_id")
    limit = int(request.args.get("limit", 100))

    # Query alerts from storage
    alerts = ml.get_alerts(camera_id=camera_id, limit=limit)

    return jsonify({"alerts": alerts, "count": len(alerts)})

@app.route("/api/model/info", methods=["GET"])
def model_info():
    """Get model information."""
    return jsonify(ml.get_model_info())

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
'''

    print(integration_code)
    print()
    print("Key Points:")
    print("  1. Initialize AgaveVisionML once at server startup")
    print("  2. Call predict_frame() in your API endpoint")
    print("  3. Return ML results to your frontend/clients")
    print("  4. Use get_alerts() for alert history queries")
    print("  5. ML API is thread-safe for concurrent requests")
    print()


def main():
    """Run all examples."""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "AGAVE VISION ML API - INTEGRATION EXAMPLES" + " " * 6 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    print("This demonstrates how external teams integrate our ML API")
    print("into their server architecture (Flask, FastAPI, etc.)")
    print()
    print("NO HTTP, NO DOCKER - Just pure Python function calls")
    print()

    examples = [
        ("1", "Single Frame Inference", example_1_single_frame_inference),
        ("2", "Video Stream Processing", example_2_video_stream_processing),
        ("3", "Persistent Alert Storage", example_3_with_persistent_storage),
        ("4", "ROI Configuration Query", example_4_roi_configuration),
        ("5", "Server Integration Pattern", example_5_integration_pattern),
    ]

    print("Available examples:")
    for num, name, _ in examples:
        print(f"  {num}. {name}")
    print(f"  A. Run all examples")
    print()

    choice = input("Select example to run (1-5, A for all, Q to quit): ").strip().upper()

    if choice == "Q":
        return

    if choice == "A":
        for num, name, func in examples:
            try:
                func()
            except Exception as e:
                print(f"Error in example {num}: {e}")
                import traceback
                traceback.print_exc()
            print()
    else:
        for num, name, func in examples:
            if choice == num:
                try:
                    func()
                except Exception as e:
                    print(f"Error: {e}")
                    import traceback
                    traceback.print_exc()
                break
        else:
            print(f"Invalid choice: {choice}")

    print()
    print("=" * 60)
    print("For more information, see: docs/ml_api_reference.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
