# agave-vision-api / src/agave_vision

## Overview
- Python package exposing the main ML API (`AgaveVisionML`) and core primitives.
- Houses core inference, ROI logic, alert modeling, and tracking.
- Exports services for inference API, stream manager, and alert router.
- Primary users: developers integrating detection into other systems.

## Quickstart
### Prerequisites
- Python 3.11+.
- Model weights.
- ROI config if using alerts.

### Install
- `pip install -e .`.

### Configure
- Model path from `configs/model.yaml` or `INFERENCE_MODEL_PATH`.
- ROI config path passed to `AgaveVisionML`.

### Run (development)
- Import and use `AgaveVisionML` in a script.

### Run (production-like)
- Embed `AgaveVisionML` or run services under `src/agave_vision/services/`.

### Common commands
- `python -c "from agave_vision.ml_api import AgaveVisionML; print(AgaveVisionML())"`.

### Troubleshooting
1. Model not found -> set model path.
2. ROI config missing -> pass `roi_config_path`.
3. Missing optional deps -> install extras.
4. Low-confidence detections -> adjust thresholds.
5. Tracking not present -> ensure `enable_tracking=True`.

## Architecture
### High-level diagram
```
AgaveVisionML -> YOLOInference -> ROIManager -> Alerts/Detections
```

### Key concepts
- `AgaveVisionML`: primary interface in `src/agave_vision/ml_api.py`.
- `YOLOInference`: model wrapper in `src/agave_vision/core/inference.py`.
- `ROIManager`: polygon rules in `src/agave_vision/core/roi.py`.
- `AlertEvent`: alert serialization in `src/agave_vision/core/alerts.py`.
- `CentroidTracker`: object tracking in `src/agave_vision/core/tracking.py`.

### Runtime flow
- Initialize `AgaveVisionML` with model and ROI config.
- Call `predict_frame` or `predict_video_stream`.
- Detections optionally pass through tracking and ROI filtering.
- Alerts and detections returned as dictionaries.

### Data flow
- Inputs: numpy frames and configuration paths.
- Transformations: YOLO inference -> tracking -> ROI filtering.
- Outputs: detection and alert dictionaries.

### Component map
| Component | Location (path) | Responsibility | Inputs | Outputs | Dependencies | Failure modes or notes |
| --- | --- | --- | --- | --- | --- | --- |
| ML API | `src/agave_vision/ml_api.py` | Public API for inference and alerts | frames, config | dict results | opencv, numpy | Missing model or ROI config |
| Core primitives | `src/agave_vision/core/` | Inference, ROI, alerts, tracking | frames, polygons | detections/alerts | ultralytics | Missing optional deps |
| Services | `src/agave_vision/services/` | Runtime services | configs, Redis | alerts, API responses | fastapi, redis | Missing optional deps |
| Storage | `src/agave_vision/storage/` | Alert/detection persistence | alert/detection dicts | SQLite/JSON files | sqlite3 | Disk permissions |

## Entry points
- **CLI commands or scripts:** none defined within package beyond services.
- **APIs or routes:** inference API under `src/agave_vision/services/inference_api/`.
- **Workers, schedulers, or jobs:** stream manager and alert router under `src/agave_vision/services/`.
- **Notebooks:** none found.

## Configuration and secrets
- **Configuration sources:** `configs/` and env overrides used by services; `AgaveVisionML` accepts explicit paths.
- **Precedence:** env overrides > YAML > defaults (service usage).
- **Secrets handling patterns:** inject credentials via env vars.
- **Safe defaults and what must never be committed:** avoid committing RTSP credentials.

## Observability
- **Logging strategy:** structured JSON logging via `src/agave_vision/utils/logging.py`.
- **Metrics or tracing:** none found.
- **Debug workflow:** inspect logs and returned dictionaries.

## Testing
- **How to run tests:** `pytest`.
- **Summary of the test pyramid:** no tests in repo.
- **Gaps and recommended minimal regression coverage:** unit tests for `AgaveVisionML` and ROI logic.

## Deployment (if applicable)
- **Build artifacts:** Python package.
- **Deployment targets:** embedded library or service processes.
- **CI/CD overview:** none found.
- **Rollback strategy:** revert package version.

## Contributing
- **Repository conventions:** format with Black and lint with Ruff.
- **Branching and release practices:** not defined.
- **How to safely add a new feature and where to start:** extend core modules and update `__all__` exports.

## Known gaps and assumptions
- Version mismatch between `pyproject.toml` and `src/agave_vision/__init__.py`.
