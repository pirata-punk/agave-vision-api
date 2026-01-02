# Agave Vision ML API Reference

Complete reference for the Agave Vision ML API - the clean Python interface for object detection, ROI filtering, and alert generation.

## Table of Contents

- [Overview](#overview)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [API Reference](#api-reference)
  - [AgaveVisionML Class](#agavevisionml-class)
  - [Methods](#methods)
  - [Return Types](#return-types)
- [Storage Systems](#storage-systems)
- [Configuration](#configuration)
- [Examples](#examples)
- [Error Handling](#error-handling)
- [Performance](#performance)

---

## Overview

The Agave Vision ML API provides a simple Python interface for:

- **Object Detection**: YOLOv8-based detection for pines, workers, and objects
- **ROI Filtering**: Region-of-Interest based alert generation
- **Alert Management**: Whitelist approach (alert on anything NOT allowed)
- **Storage**: Optional persistent storage for alerts and logs
- **Video Processing**: Frame-by-frame or stream processing

**Key Design Principles:**
- Pure Python - No HTTP, no Docker, no server concerns
- Simple to integrate - Just import and call
- Thread-safe - Safe for concurrent requests
- Storage-agnostic - Works with or without persistence

---

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from agave_vision.ml_api import AgaveVisionML; print('OK')"
```

---

## Quick Start

```python
from agave_vision.ml_api import AgaveVisionML
import cv2

# Initialize
ml = AgaveVisionML(
    model_path="models/yolov8n_pina/exp/weights/best.pt",
    roi_config_path="configs/rois.yaml"
)

# Single frame inference
image = cv2.imread("frame.jpg")
result = ml.predict_frame(image, camera_id="cam1")

print(f"Detections: {result['num_detections']}")
print(f"Alerts: {result['num_alerts']}")
```

---

## API Reference

### AgaveVisionML Class

Main entry point for all ML operations.

#### Constructor

```python
AgaveVisionML(
    model_path: str,
    roi_config_path: Optional[str] = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    enable_alert_storage: bool = False,
    enable_detection_logging: bool = False,
)
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `str` | Required | Path to YOLO model weights (.pt file) |
| `roi_config_path` | `str \| None` | `None` | Path to ROI configuration YAML |
| `conf_threshold` | `float` | `0.25` | Confidence threshold for detections (0-1) |
| `iou_threshold` | `float` | `0.45` | IOU threshold for NMS (0-1) |
| `enable_alert_storage` | `bool` | `False` | Enable persistent alert storage (SQLite) |
| `enable_detection_logging` | `bool` | `False` | Enable detection history logging |

**Example:**

```python
ml = AgaveVisionML(
    model_path="models/best.pt",
    roi_config_path="configs/rois.yaml",
    conf_threshold=0.3,
    enable_alert_storage=True
)
```

---

### Methods

#### predict_frame()

Run inference on a single frame.

```python
def predict_frame(
    image: np.ndarray,
    camera_id: Optional[str] = None,
    store_alerts: bool = True,
    log_detections: bool = False,
) -> Dict[str, Any]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image` | `np.ndarray` | Required | Input image (BGR format from cv2) |
| `camera_id` | `str \| None` | `None` | Camera ID for ROI lookup |
| `store_alerts` | `bool` | `True` | Store alerts to persistent storage |
| `log_detections` | `bool` | `False` | Log detections to history |

**Returns:**

```python
{
    "detections": [
        {
            "bbox": [x1, y1, x2, y2],      # Bounding box coordinates
            "confidence": 0.85,             # Detection confidence (0-1)
            "class_name": "object",         # Class name
            "class_id": 0,                  # Class ID
            "center": [cx, cy],             # Center point
            "is_unknown": false             # Unknown object flag
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
```

**Example:**

```python
image = cv2.imread("frame.jpg")
result = ml.predict_frame(image, camera_id="cam_nave3_hornos_a")

for detection in result['detections']:
    print(f"{detection['class_name']}: {detection['confidence']:.2f}")

if result['num_alerts'] > 0:
    print(f"⚠️ {result['num_alerts']} alerts triggered!")
```

---

#### predict_video_stream()

Process video stream frame by frame.

```python
def predict_video_stream(
    video_source: str,
    camera_id: Optional[str] = None,
    fps_limit: Optional[float] = None,
    max_frames: Optional[int] = None,
    store_alerts: bool = True,
    log_detections: bool = False,
) -> Generator[Dict[str, Any], None, None]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `video_source` | `str` | Required | Path to video file or RTSP URL |
| `camera_id` | `str \| None` | `None` | Camera ID for ROI lookup |
| `fps_limit` | `float \| None` | `None` | Limit processing FPS (skips frames) |
| `max_frames` | `int \| None` | `None` | Maximum frames to process |
| `store_alerts` | `bool` | `True` | Store alerts to persistent storage |
| `log_detections` | `bool` | `False` | Log detections to history |

**Yields:**

Same format as `predict_frame()`, plus:
- `frame_index`: Original frame number in video
- `processed_frame_index`: Index of processed frames (after skipping)

**Example:**

```python
# Process RTSP stream at 5 FPS
for result in ml.predict_video_stream(
    video_source="rtsp://camera.local/stream",
    camera_id="cam1",
    fps_limit=5.0
):
    # Your server handles this result
    if result['num_alerts'] > 0:
        send_notification(result['alerts'])

    send_to_frontend_websocket(result)
```

---

#### get_alerts()

Retrieve alerts from storage.

```python
def get_alerts(
    camera_id: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    limit: int = 100,
) -> List[Dict[str, Any]]
```

**Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `camera_id` | `str \| None` | `None` | Filter by camera ID |
| `start_time` | `str \| None` | `None` | Filter by start time (ISO format) |
| `end_time` | `str \| None` | `None` | Filter by end time (ISO format) |
| `limit` | `int` | `100` | Maximum alerts to return |

**Returns:**

List of alert dictionaries (same format as `alerts` in `predict_frame()`)

**Raises:**

- `RuntimeError`: If alert storage not enabled

**Example:**

```python
# Get recent alerts for camera
alerts = ml.get_alerts(
    camera_id="cam_nave3_hornos_a",
    start_time="2025-01-02T00:00:00",
    limit=50
)

print(f"Found {len(alerts)} alerts")
for alert in alerts:
    print(f"  {alert['timestamp']}: {alert['roi_name']}")
```

---

#### get_detection_logs()

Retrieve detection history.

```python
def get_detection_logs(
    camera_id: Optional[str] = None,
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    limit: int = 1000,
) -> List[Dict[str, Any]]
```

**Parameters:**

Same as `get_alerts()`

**Returns:**

```python
[
    {
        "timestamp": "2025-01-02T14:30:00",
        "camera_id": "cam1",
        "detections": [...],
        "num_alerts": 2
    },
    ...
]
```

**Raises:**

- `RuntimeError`: If detection logging not enabled

**Example:**

```python
logs = ml.get_detection_logs(camera_id="cam1", limit=100)

for log in logs:
    print(f"{log['timestamp']}: {len(log['detections'])} detections")
```

---

#### get_model_info()

Get model metadata.

```python
def get_model_info() -> Dict[str, Any]
```

**Returns:**

```python
{
    "model_path": "models/best.pt",
    "classes": ["object", "pine", "worker"],
    "num_classes": 3,
    "input_size": 640,
    "conf_threshold": 0.25,
    "iou_threshold": 0.45
}
```

**Example:**

```python
info = ml.get_model_info()
print(f"Model classes: {info['classes']}")
print(f"Input size: {info['input_size']}")
```

---

#### get_camera_roi_info()

Get ROI configuration for a camera.

```python
def get_camera_roi_info(camera_id: str) -> Optional[Dict[str, Any]]
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `camera_id` | `str` | Camera identifier |

**Returns:**

```python
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
```

Returns `None` if camera not found or ROI manager not initialized.

**Example:**

```python
roi_info = ml.get_camera_roi_info("cam_nave3_hornos_a")
if roi_info:
    print(f"Allowed classes: {roi_info['allowed_classes']}")
    print(f"ROI zones: {len(roi_info['forbidden_zones'])}")
```

---

## Storage Systems

### Alert Storage

Persistent storage for alert history using SQLite or JSON.

**Initialization:**

```python
ml = AgaveVisionML(
    model_path="models/best.pt",
    enable_alert_storage=True  # Enable storage
)
```

**Storage Location:**

- SQLite: `data/alerts.db`
- JSON: `data/alerts.json`

**Query Examples:**

```python
# All alerts for camera
alerts = ml.get_alerts(camera_id="cam1")

# Alerts in time range
alerts = ml.get_alerts(
    start_time="2025-01-02T00:00:00",
    end_time="2025-01-02T23:59:59"
)

# Recent 10 alerts
alerts = ml.get_alerts(limit=10)
```

---

### Detection Logging

Logs all detection results to rotating JSON files.

**Initialization:**

```python
ml = AgaveVisionML(
    model_path="models/best.pt",
    enable_detection_logging=True  # Enable logging
)
```

**Storage Location:**

- Log files: `data/detection_logs/detections_YYYYMMDD_HHMMSS.jsonl`
- Format: JSON Lines (one JSON object per line)
- Rotation: 10,000 entries per file
- Retention: 7 days (configurable)

**Query Examples:**

```python
# Recent logs
logs = ml.get_detection_logs(limit=100)

# Logs for camera
logs = ml.get_detection_logs(camera_id="cam1")

# Export to CSV
from agave_vision.storage.detection_logger import DetectionLogger
logger = DetectionLogger()
logger.export_logs("export.csv", format="csv")
```

---

## Configuration

### ROI Configuration

ROI configuration defines forbidden zones and allowed classes.

**Format:** `configs/rois.yaml`

```yaml
cameras:
  - camera_id: cam_nave3_hornos_a
    forbidden_rois:
      - name: loading_zone
        points:
          - [245, 180]
          - [580, 190]
          - [620, 450]
          - [210, 440]

    allowed_classes: [pine, worker]
    alert_classes: [object]
    strict_mode: true
```

**Fields:**

- `camera_id`: Unique camera identifier
- `forbidden_rois`: List of ROI zones (polygons)
  - `name`: Zone name
  - `points`: List of [x, y] coordinates
- `allowed_classes`: Classes allowed in ROI (strict mode)
- `alert_classes`: Classes that always trigger alerts
- `strict_mode`: If true, alert on anything NOT in allowed_classes

**Strict Mode Logic:**

```python
if strict_mode:
    if class_name NOT IN allowed_classes:
        trigger_alert()  # Whitelist approach
else:
    if class_name IN alert_classes:
        trigger_alert()  # Blacklist approach
```

---

## Examples

### Example 1: Basic Integration

```python
from agave_vision.ml_api import AgaveVisionML
import cv2

ml = AgaveVisionML(
    model_path="models/best.pt",
    roi_config_path="configs/rois.yaml"
)

image = cv2.imread("frame.jpg")
result = ml.predict_frame(image, camera_id="cam1")

print(f"Detections: {result['num_detections']}")
print(f"Alerts: {result['num_alerts']}")
```

### Example 2: Flask API

```python
from flask import Flask, jsonify, request
from agave_vision.ml_api import AgaveVisionML
import cv2
import base64
import numpy as np

app = Flask(__name__)
ml = AgaveVisionML(
    model_path="models/best.pt",
    roi_config_path="configs/rois.yaml",
    enable_alert_storage=True
)

@app.route("/api/detect", methods=["POST"])
def detect():
    # Decode image
    image_b64 = request.json["image"]
    camera_id = request.json["camera_id"]

    image_bytes = base64.b64decode(image_b64)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Run inference
    result = ml.predict_frame(image, camera_id=camera_id)

    return jsonify(result)

@app.route("/api/alerts", methods=["GET"])
def get_alerts():
    camera_id = request.args.get("camera_id")
    limit = int(request.args.get("limit", 100))

    alerts = ml.get_alerts(camera_id=camera_id, limit=limit)

    return jsonify({"alerts": alerts, "count": len(alerts)})
```

### Example 3: Video Stream Processing

```python
from agave_vision.ml_api import AgaveVisionML

ml = AgaveVisionML(
    model_path="models/best.pt",
    roi_config_path="configs/rois.yaml"
)

for result in ml.predict_video_stream(
    video_source="rtsp://camera.local/stream",
    camera_id="cam1",
    fps_limit=5.0  # Process at 5 FPS
):
    # Handle result
    if result['num_alerts'] > 0:
        print(f"Alert! Frame {result['frame_index']}")

    # Send to your system
    send_to_websocket(result)
```

---

## Error Handling

### Common Exceptions

```python
try:
    ml = AgaveVisionML(model_path="invalid.pt")
except FileNotFoundError:
    print("Model file not found")

try:
    alerts = ml.get_alerts()
except RuntimeError as e:
    print(f"Alert storage not enabled: {e}")

try:
    result = ml.predict_frame(invalid_image)
except ValueError as e:
    print(f"Invalid image: {e}")
```

### Best Practices

1. **Initialize once**: Create AgaveVisionML instance at startup, reuse for all requests
2. **Handle storage errors**: Check if storage is enabled before querying
3. **Validate inputs**: Ensure images are valid BGR numpy arrays
4. **Monitor performance**: Track inference_time_ms for optimization

---

## Performance

### Benchmarks

| Operation | Avg Time | Notes |
|-----------|----------|-------|
| Single frame inference | 30-50ms | YOLOv8n on CPU |
| ROI filtering | <1ms | Point-in-polygon |
| Alert storage (SQLite) | <5ms | Per alert |
| Detection logging | <1ms | Async write |

### Optimization Tips

1. **Use GPU**: Install CUDA-enabled PyTorch for 5-10x speedup
2. **Batch processing**: Process multiple frames together
3. **FPS limiting**: Use `fps_limit` to reduce load
4. **Disable storage**: Set `enable_alert_storage=False` if not needed

### Thread Safety

AgaveVisionML is thread-safe for concurrent requests:

```python
from concurrent.futures import ThreadPoolExecutor

ml = AgaveVisionML(model_path="models/best.pt")

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [
        executor.submit(ml.predict_frame, image, "cam1")
        for image in images
    ]
    results = [f.result() for f in futures]
```

---

## See Also

- [Integration Example](../examples/integration_example.py) - Complete working examples
- [ROI Setup Guide](roi_setup_guide.md) - Configure ROI zones
- [Phase 1 Validation](phase1_validation_checklist.md) - Alert system testing
- [Synthetic Object Testing](synthetic_object_testing.md) - Test alert logic

---

**Questions?** See [examples/integration_example.py](../examples/integration_example.py) for complete working code.
