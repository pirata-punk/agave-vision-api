# ML Enhancements Roadmap

**Project Focus:** Pure ML capabilities - Model inference, alerting logic, and API exposure
**Out of Scope:** Server architecture, deployment, Docker services (handled by separate team)

---

## ✅ Phase 0: Customer Demo (COMPLETED)

- [x] Live video demo with real-time detection
- [x] Interactive ROI selector for forbidden zones
- [x] Visual bounding boxes with class labels
- [x] Basic alert logic (object detection in ROI)

**Customer Feedback:** ✅ Demo approved, proceeding with enhancements

---

## 🎯 Phase 1: Enhanced Alert Logic - "Anything But Pines/Workers"

### Objective
Change alert logic to trigger on ANY detection that is NOT a pine or worker in forbidden ROI zones. This eliminates the need to pre-train on specific objects (tires, logs, debris, etc.) and ensures strict zone control.

### Current Behavior
```python
# Alert triggers on:
if class_name == "object" and inside_roi:
    trigger_alert()
```

### New Behavior
```python
# Alert triggers on:
if class_name NOT IN ["pine", "worker"] and inside_roi:
    trigger_alert()

# This includes:
# - "object" class
# - Unknown/unclassified detections
# - Any future classes added to model
# - Low-confidence detections (configurable threshold)
```

### TODO List - Phase 1

#### 1.1 Update Core ROI Logic
- [ ] **File:** `src/agave_vision/core/roi.py`
- [ ] **Task:** Modify `should_alert()` method to use "whitelist" approach
- [ ] **Change:** Instead of checking `alert_classes`, check if detection is NOT in `allowed_classes`
- [ ] **Add:** Configuration option for `strict_mode` (default: True)
- [ ] **Test:** Unit tests for new alert logic

**Implementation:**
```python
# Current (roi.py)
def should_alert(self, class_name: str, bbox: list) -> bool:
    if class_name not in self.alert_classes:
        return False
    # ... check if in ROI ...

# New (roi.py)
def should_alert(self, class_name: str, bbox: list, strict_mode: bool = True) -> bool:
    # Whitelist approach: alert on anything NOT allowed
    if strict_mode and class_name not in self.allowed_classes:
        # Check if in forbidden ROI
        # ...
        return True
    return False
```

#### 1.2 Update Configuration Schema
- [ ] **File:** `src/agave_vision/config/models.py`
- [ ] **Task:** Update ROI config Pydantic models
- [ ] **Add:** `strict_mode` field to camera ROI configuration
- [ ] **Add:** `unknown_object_threshold` for handling low-confidence detections
- [ ] **Document:** Update config examples with new fields

**Configuration Example:**
```yaml
# configs/rois.yaml
cameras:
  - camera_id: cam_nave3_hornos
    forbidden_rois:
      - name: loading_zone
        points: [[245, 180], [580, 190], [620, 450], [210, 440]]

    # NEW FIELDS
    strict_mode: true  # Alert on anything NOT in allowed_classes
    allowed_classes: [pine, worker]  # Only these are allowed in ROI
    unknown_object_threshold: 0.15  # Min confidence to consider as "unknown object"
```

#### 1.3 Handle Unknown/Unclassified Detections
- [ ] **File:** `src/agave_vision/core/inference.py`
- [ ] **Task:** Add detection of unknown objects (below class confidence threshold)
- [ ] **Add:** Flag for detections with low class confidence as "unknown"
- [ ] **Return:** Enhanced detection results with confidence scores
- [ ] **Test:** Verify unknown object detection works

**Implementation:**
```python
# src/agave_vision/core/inference.py
class Detection:
    bbox: list
    class_id: int
    class_name: str
    confidence: float
    is_unknown: bool = False  # NEW: Flag for unknown objects

def classify_detection(self, detection, unknown_threshold=0.15):
    """Classify detection as known class or unknown object."""
    if detection.confidence < unknown_threshold:
        detection.class_name = "unknown"
        detection.is_unknown = True
    return detection
```

#### 1.4 Update Alert Events
- [ ] **File:** `src/agave_vision/core/alerts.py`
- [ ] **Task:** Enhance `AlertEvent` dataclass with new fields
- [ ] **Add:** `violation_type` (unknown_object, forbidden_class, etc.)
- [ ] **Add:** `allowed_classes` to alert context
- [ ] **Add:** `strict_mode_enabled` flag
- [ ] **Document:** Alert event schema

**Enhanced Alert Event:**
```python
@dataclass
class AlertEvent:
    camera_id: str
    timestamp: str
    roi_name: str
    detected_class: str
    confidence: float
    bbox: list

    # NEW FIELDS
    violation_type: str  # "forbidden_class", "unknown_object", "low_confidence"
    allowed_classes: list[str]
    strict_mode: bool
    is_unknown: bool
```

#### 1.5 Update Demo Scripts
- [ ] **File:** `examples/live_demo.py`
- [ ] **Task:** Show alert logic in action during demo
- [ ] **Add:** Visual indicator when detection would trigger alert
- [ ] **Add:** Display "ALERT" overlay for forbidden detections
- [ ] **Color:** Red flash/border when alert would trigger

#### 1.6 Testing & Validation
- [ ] **Create:** Test cases for new alert logic
- [ ] **Test:** Pine in ROI → No alert
- [ ] **Test:** Worker in ROI → No alert
- [ ] **Test:** Object in ROI → Alert
- [ ] **Test:** Unknown detection in ROI → Alert
- [ ] **Test:** Low confidence detection → Configurable alert
- [ ] **Document:** Test results and edge cases

---

## 🔧 Phase 2: ML-Focused API Architecture

### Objective
Restructure project as pure ML module with clean API for external integration. Separate ML logic from production infrastructure concerns.

### Architecture Principle
```
┌─────────────────────────────────────────────────────┐
│  EXTERNAL SERVER (Other Team)                      │
│  - HTTP/gRPC endpoints                             │
│  - Authentication                                  │
│  - Rate limiting                                   │
│  - Load balancing                                  │
└────────────────┬────────────────────────────────────┘
                 │ Calls
                 ▼
┌─────────────────────────────────────────────────────┐
│  AGAVE VISION ML MODULE (Our Team)                 │
│                                                     │
│  ┌─────────────────────────────────────┐           │
│  │  ML API (Simple Interface)          │           │
│  │  - predict(image, camera_id)        │           │
│  │  - get_alerts(camera_id, timeframe) │           │
│  │  - get_logs()                        │           │
│  └─────────────────────────────────────┘           │
│                                                     │
│  ┌─────────────────────────────────────┐           │
│  │  Core ML Logic                      │           │
│  │  - Model inference                  │           │
│  │  - ROI filtering                    │           │
│  │  - Alert generation                 │           │
│  └─────────────────────────────────────┘           │
│                                                     │
│  ┌─────────────────────────────────────┐           │
│  │  Data Layer                         │           │
│  │  - Alert storage (JSON/SQLite)      │           │
│  │  - Log storage                      │           │
│  │  - Model registry                   │           │
│  └─────────────────────────────────────┘           │
└─────────────────────────────────────────────────────┘
```

### TODO List - Phase 2

#### 2.1 Create Clean ML API Interface
- [ ] **Create:** `src/agave_vision/ml_api.py`
- [ ] **Task:** Simple Python API for external integration
- [ ] **Methods:**
  - `predict_frame()` - Run inference on single frame
  - `predict_video_stream()` - Process video stream (generator)
  - `get_alerts()` - Retrieve alerts for timeframe
  - `get_detection_logs()` - Retrieve detection history
  - `get_model_info()` - Get model metadata
- [ ] **Return:** Plain Python dictionaries (JSON-serializable)
- [ ] **No:** HTTP, REST, gRPC, Docker - just pure Python

**API Interface:**
```python
# src/agave_vision/ml_api.py

class AgaveVisionML:
    """
    Pure ML API for Agave Vision object detection.

    This is the ONLY interface external teams need to use.
    No HTTP, no Docker, no server concerns - just ML.
    """

    def __init__(self, model_path: str, config_path: str):
        """Initialize ML engine with model and configuration."""
        pass

    def predict_frame(
        self,
        image: np.ndarray,
        camera_id: str
    ) -> dict:
        """
        Run inference on single frame.

        Args:
            image: numpy array (BGR format from cv2.imread)
            camera_id: Camera identifier for ROI lookup

        Returns:
            {
                "detections": [...],
                "alerts": [...],
                "inference_time_ms": 45.2,
                "timestamp": "2025-01-15T14:30:00"
            }
        """
        pass

    def predict_video_stream(
        self,
        video_source: str,
        camera_id: str,
        fps_limit: float = None
    ) -> Generator[dict, None, None]:
        """
        Process video stream and yield results.

        Args:
            video_source: Path to video or RTSP URL
            camera_id: Camera identifier
            fps_limit: Optional FPS limiting

        Yields:
            Detection results for each frame
        """
        pass

    def get_alerts(
        self,
        camera_id: str = None,
        start_time: str = None,
        end_time: str = None,
        limit: int = 100
    ) -> list[dict]:
        """
        Retrieve alerts from storage.

        Returns:
            [
                {
                    "alert_id": "uuid",
                    "camera_id": "cam1",
                    "timestamp": "2025-01-15T14:30:00",
                    "detected_class": "object",
                    "roi_name": "loading_zone",
                    "violation_type": "forbidden_class",
                    ...
                }
            ]
        """
        pass

    def get_detection_logs(
        self,
        camera_id: str = None,
        limit: int = 1000
    ) -> list[dict]:
        """
        Retrieve detection history.

        Returns:
            [
                {
                    "timestamp": "...",
                    "camera_id": "...",
                    "detections": [...]
                }
            ]
        """
        pass

    def get_model_info(self) -> dict:
        """
        Get model metadata.

        Returns:
            {
                "model_path": "...",
                "classes": ["object", "pine", "worker"],
                "input_size": 640,
                "confidence_threshold": 0.25
            }
        """
        pass
```

#### 2.2 Implement Alert Storage
- [ ] **Create:** `src/agave_vision/storage/alert_store.py`
- [ ] **Task:** Simple alert persistence (SQLite or JSON)
- [ ] **Methods:**
  - `save_alert()` - Store alert
  - `get_alerts()` - Query alerts
  - `clear_old_alerts()` - Cleanup
- [ ] **Format:** JSON-serializable dictionaries
- [ ] **Storage:** Local SQLite database or JSON files (configurable)

**Alert Storage:**
```python
# src/agave_vision/storage/alert_store.py

class AlertStore:
    """Simple alert storage using SQLite or JSON."""

    def __init__(self, storage_type: str = "sqlite", path: str = "data/alerts.db"):
        """Initialize alert storage."""
        pass

    def save_alert(self, alert: dict) -> str:
        """Save alert and return alert_id."""
        pass

    def get_alerts(
        self,
        camera_id: str = None,
        start_time: datetime = None,
        end_time: datetime = None,
        limit: int = 100
    ) -> list[dict]:
        """Query alerts with filters."""
        pass

    def clear_old_alerts(self, days: int = 30):
        """Remove alerts older than N days."""
        pass
```

#### 2.3 Implement Detection Logging
- [ ] **Create:** `src/agave_vision/storage/detection_logger.py`
- [ ] **Task:** Log all detections for analysis
- [ ] **Methods:**
  - `log_detection()` - Store detection result
  - `get_logs()` - Retrieve logs
  - `export_logs()` - Export to CSV/JSON
- [ ] **Storage:** Rotating JSON files or SQLite
- [ ] **Retention:** Configurable (default 7 days)

#### 2.4 Restructure Project for ML Focus
- [ ] **Move:** Production services to `_archive/production/` (reference only)
- [ ] **Keep:** Only ML core modules in main project
- [ ] **Create:** `examples/integration_example.py` showing how other team uses ML API
- [ ] **Update:** README to focus on ML API usage
- [ ] **Remove:** Docker, server deployment docs from main README

**New Project Structure:**
```
agave-vision-api/
├── src/agave_vision/
│   ├── ml_api.py              # 🎯 MAIN API - External teams use this
│   ├── core/                  # ML logic
│   ├── storage/               # Alert/log storage
│   ├── models/                # Model registry
│   ├── training/              # Training tools
│   └── ingestion/             # Data pipeline
│
├── examples/
│   ├── integration_example.py # How to use ML API
│   ├── live_demo.py
│   └── roi_selector.py
│
├── configs/                   # ML configs only
│   ├── model.yaml
│   ├── rois.yaml
│   └── cameras.yaml
│
├── _archive/
│   └── production/            # Old server code (reference)
│
└── README.md                  # ML-focused documentation
```

#### 2.5 Create Integration Examples
- [ ] **Create:** `examples/integration_example.py`
- [ ] **Task:** Show how external team would use ML API
- [ ] **Examples:**
  - Basic inference
  - Video stream processing
  - Alert retrieval
  - Log querying
- [ ] **Document:** Clear usage patterns

**Integration Example:**
```python
# examples/integration_example.py

"""
Example: How to integrate Agave Vision ML into your server.

This shows how another team would use our ML API.
No HTTP, no Docker - just Python function calls.
"""

from agave_vision.ml_api import AgaveVisionML
import cv2

# Initialize ML engine
ml = AgaveVisionML(
    model_path="models/yolov8n_pina/exp/weights/best.pt",
    config_path="configs"
)

# Example 1: Single frame inference
image = cv2.imread("frame.jpg")
result = ml.predict_frame(image, camera_id="cam_nave3_hornos")

print(f"Detections: {len(result['detections'])}")
print(f"Alerts: {len(result['alerts'])}")

# Example 2: Process video stream
for frame_result in ml.predict_video_stream(
    video_source="rtsp://camera/stream",
    camera_id="cam_nave3_hornos",
    fps_limit=5.0
):
    # Your server handles this result
    # - Send to frontend via websocket
    # - Store in your database
    # - Trigger notifications
    # etc.
    pass

# Example 3: Get recent alerts
alerts = ml.get_alerts(
    camera_id="cam_nave3_hornos",
    start_time="2025-01-15T00:00:00",
    limit=50
)

# Example 4: Get detection logs
logs = ml.get_detection_logs(camera_id="cam_nave3_hornos", limit=1000)
```

#### 2.6 Update Documentation
- [ ] **File:** `README.md`
- [ ] **Task:** Rewrite as ML-focused documentation
- [ ] **Sections:**
  - Quick start (Python API usage)
  - ML API reference
  - Alert system
  - Training pipeline
  - ROI configuration
- [ ] **Remove:** Docker, server deployment, production architecture
- [ ] **Add:** "Integration Guide" for external teams

#### 2.7 Create API Documentation
- [ ] **Create:** `docs/ml_api_reference.md`
- [ ] **Task:** Complete API documentation
- [ ] **Include:**
  - Method signatures
  - Parameters
  - Return types
  - Examples
  - Error handling
- [ ] **Format:** Clear, copy-paste ready examples

#### 2.8 Performance Optimization
- [ ] **Task:** Optimize for ML performance only
- [ ] **Focus:**
  - Model inference speed
  - Memory efficiency
  - Batch processing
  - GPU utilization
- [ ] **Ignore:** HTTP latency, network, containers (not our concern)
- [ ] **Benchmark:** Measure FPS and inference time

---

## 📋 Implementation Order

### Week 1: Alert Logic Enhancement
1. Update ROI logic to "whitelist" approach
2. Handle unknown object detection
3. Update configuration schema
4. Test and validate new alert logic
5. Update demo to show new alerts

### Week 2: ML API Foundation
1. Create `ml_api.py` with clean interface
2. Implement alert storage
3. Implement detection logging
4. Create integration examples
5. Test API with mock external usage

### Week 3: Restructure & Documentation
1. Reorganize project structure
2. Archive production code
3. Update README (ML-focused)
4. Write ML API reference docs
5. Create integration guide

---

## 🎯 Success Criteria

### Phase 1 Complete When:
- [x] Alerts trigger on ANY non-pine/worker in ROI
- [x] Unknown objects are detected and flagged
- [x] Configuration supports strict mode
- [x] Live demo shows new alert behavior
- [x] All tests pass

### Phase 2 Complete When:
- [x] External team can use ML API with simple Python imports
- [x] Alerts are stored and queryable
- [x] Detection logs are available
- [x] Integration examples work
- [x] Documentation is ML-focused
- [x] Production code is archived (not deleted, just separated)

---

## 🚫 Out of Scope (Other Team Handles)

- HTTP/REST API endpoints
- gRPC services
- Docker containers
- Kubernetes deployment
- Load balancing
- Authentication/Authorization
- Rate limiting
- API gateway
- Database setup (beyond simple SQLite)
- Server monitoring
- Logging infrastructure
- CI/CD pipelines

**Our Deliverable:** Clean Python ML API that they can integrate however they want.

---

## 📞 Handoff to Server Team

### What We Provide:
1. **ML API** (`AgaveVisionML` class) - Pure Python interface
2. **Model files** - Trained YOLO weights
3. **Alert storage** - SQLite/JSON with query interface
4. **Detection logs** - Queryable detection history
5. **Configuration** - ROI definitions, camera configs
6. **Integration examples** - Working code samples
7. **Documentation** - Complete API reference

### What They Build:
1. HTTP/gRPC wrapper around `AgaveVisionML`
2. Production database (if they want to replace SQLite)
3. Authentication & authorization
4. Server deployment & scaling
5. Frontend integration
6. Real-time streaming (WebSocket, SSE, etc.)

---

## 📝 Notes

- Focus: **Pure ML and data**, nothing else
- Keep it simple: Python functions, not microservices
- Make it callable: Easy to integrate into any server architecture
- Store results: Alerts and logs must be queryable
- Document everything: Clear examples for integration

---

**Let's build the ML core right, and let the server team handle the infrastructure!** 🚀
