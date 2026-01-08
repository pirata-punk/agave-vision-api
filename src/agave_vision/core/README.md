# agave-vision-api / src/agave_vision/core

## Overview
- Core ML primitives: inference, ROI logic, alert modeling, tracking, and frame utilities.
- Used by both the ML API and services.
- Primary users: developers extending detection and alert logic.

## Quickstart
### Prerequisites
- Model weights and OpenCV-compatible frames.

### Install
- `pip install -e .`.

### Configure
- Pass model paths and ROI config explicitly to consuming APIs.

### Run (development)
- Use `YOLOInference` and `ROIManager` directly in a script.

### Run (production-like)
- Consumed by services and `AgaveVisionML`.

### Common commands
- `python -c "from agave_vision.core.inference import YOLOInference; print(YOLOInference)"`.

### Troubleshooting
1. Model not found -> check path.
2. ROI file missing -> verify `configs/rois.yaml`.
3. Low-confidence detections -> adjust thresholds.
4. Tracking IDs missing -> enable tracking.
5. SSIM comparison fails -> install `scikit-image` if using `frames.is_similar_frame` with `ssim`.

## Architecture
### High-level diagram
```
YOLOInference -> Detection -> ROIManager/CameraROI -> AlertEvent
```

### Key concepts
- `Detection`: normalized YOLO output with class, confidence, bbox, and center.
- `YOLOInference`: model wrapper with warmup and prediction.
- `ROIManager` and `CameraROI`: polygon-based filtering and alert rules.
- `AlertEvent`: structured alert serialization.
- `CentroidTracker`: centroid-based tracking across frames.
- Frame utilities: tiling, sharpness, brightness, similarity, drawing.

### Runtime flow
- `YOLOInference` loads the model and predicts detections.
- `ROIManager` filters detections for forbidden zones.
- `AlertEvent` serializes detections into alert payloads.
- Optional tracking updates detections with IDs.

### Data flow
- Inputs: frames, polygons, inference thresholds.
- Transformations: model inference -> detection normalization -> ROI filtering.
- Outputs: detection lists and alert structures.

### Component map
| Component | Location (path) | Responsibility | Inputs | Outputs | Dependencies | Failure modes or notes |
| --- | --- | --- | --- | --- | --- | --- |
| Inference | `src/agave_vision/core/inference.py` | YOLO model loading and prediction | frames | Detection list | ultralytics | Missing model file |
| ROI | `src/agave_vision/core/roi.py` | Point-in-polygon and alert rules | detections, polygons | alerts/detections | opencv, pyyaml | Missing ROI config |
| Alerts | `src/agave_vision/core/alerts.py` | AlertEvent modeling/serialization | Detection | dict payloads | dataclasses | None |
| Tracking | `src/agave_vision/core/tracking.py` | Centroid-based tracking | detections | detections with IDs | numpy | Reset method not present |
| Frames | `src/agave_vision/core/frames.py` | Frame utilities and drawing | frames | metrics/tiles/annotated frames | opencv, scipy | `scikit-image` optional for SSIM |

## Entry points
- **CLI commands or scripts:** none.
- **APIs or routes:** consumed by inference API and stream manager.
- **Workers, schedulers, or jobs:** none.
- **Notebooks:** none.

## Configuration and secrets
- **Configuration sources:** model paths and ROI YAMLs passed in by callers.
- **Precedence:** caller-provided values > defaults in configs.
- **Secrets handling patterns:** none.
- **Safe defaults and what must never be committed:** avoid committing proprietary model paths.

## Observability
- **Logging strategy:** minimal logging in core; services handle logging.
- **Metrics or tracing:** none.
- **Debug workflow:** call functions directly and inspect returned structures.

## Testing
- **How to run tests:** `pytest`.
- **Summary of the test pyramid:** no tests in repo.
- **Gaps and recommended minimal regression coverage:** tests for ROI and tracking logic.

## Deployment (if applicable)
- **Build artifacts:** Python package modules.
- **Deployment targets:** services or embedded library.
- **CI/CD overview:** none found.
- **Rollback strategy:** revert code changes.

## Contributing
- **Repository conventions:** keep core APIs stable and documented.
- **Branching and release practices:** not defined.
- **How to safely add a new feature and where to start:** extend core modules and update `AgaveVisionML`.

## Known gaps and assumptions
- `scikit-image` is optional for SSIM but not declared in `pyproject.toml`.
