# agave-vision-api

## Overview

- Industrial vision codebase for YOLOv8 object detection, ROI filtering, and alert generation (`src/agave_vision/`, `configs/`).
- Includes a reusable Python ML API (`AgaveVisionML`) and three services: inference API, stream manager, and alert router (`src/agave_vision/services/`).
- Targets RTSP camera streams and image uploads; emits detections and ROI-based alerts.
- Provides an interactive demo for visualization and alert testing (`demo/`).
- Primary users: ML/vision engineers and ops teams integrating detection into production.

## Quickstart

### Prerequisites

- Python 3.11+ (`pyproject.toml`).
- Model weights at `models/agave-industrial-vision-v1.0.0.pt`.
- OpenCV-compatible video/RTSP sources.
- Redis required for stream/alert services.

### Install

- `python -m venv .venv && source .venv/bin/activate && pip install -e .`
- Add extras as needed: `pip install -e .[api,stream,alerts]`.

### Configure

- Configs in `configs/` are primary; environment overrides are applied by `src/agave_vision/config/loader.py` in this order: env > YAML > defaults.
- Secrets to inject via env: RTSP credentials (`CAMERA_{camera_id}_RTSP_URL`), webhook URL (`ALERTING_WEBHOOK_URL`), Hikvision credentials (`ALERTING_HIKVISION_*`), and `REDIS_URL`.

### Run (development)

- Inference API: `uvicorn agave_vision.services.inference_api.app:app --host 0.0.0.0 --port 8000`.
- Stream Manager: `python -m agave_vision.services.stream_manager.main`.
- Alert Router: `python -m agave_vision.services.alert_router.main`.
- Demo: `./demo/run_demo.sh`.

### Run (production)

- Run the three services as separate processes with external Redis and `configs/services.yaml`.
- Container orchestration is not present in this repo (assumption).

### Common commands

- `uvicorn agave_vision.services.inference_api.app:app`.
- `python -m agave_vision.services.stream_manager.main`.
- `python -m agave_vision.services.alert_router.main`.
- `./demo/run_demo.sh`.

### Troubleshooting

1. Model file missing at `models/agave-industrial-vision-v1.0.0.pt` -> update `configs/model.yaml` or set `INFERENCE_MODEL_PATH`.
2. RTSP connection failures -> verify `configs/cameras.yaml` or override `CAMERA_{id}_RTSP_URL`.
3. Redis connection errors -> set `REDIS_URL` and ensure Redis is reachable.
4. Missing optional deps -> install `.[api,stream,alerts]`.
5. ROI alerts not firing -> ensure camera IDs in `configs/cameras.yaml` match `configs/rois.yaml`.

## Architecture

### High-level diagram

```
RTSP/Video Sources
      |
      v
Stream Manager (`src/agave_vision/services/stream_manager/`)
      |
      v
YOLO Inference (`src/agave_vision/core/inference.py`)
      |
      v
ROI Filter (`src/agave_vision/core/roi.py`)
      |
      v
Redis Stream (`alerts`)
      |
      v
Alert Router (`src/agave_vision/services/alert_router/`)
      |
      v
Protocol Adapters (stdout/webhook/hikvision)

Parallel entry: Inference API (`src/agave_vision/services/inference_api/`)
Library entry: `AgaveVisionML` (`src/agave_vision/ml_api.py`)
```

### Key concepts

- Camera configuration: RTSP endpoints and sampling rates in `configs/cameras.yaml`.
- ROI (Region of Interest): forbidden polygons and class allowlists in `configs/rois.yaml`.
- Detection: normalized YOLO outputs in `src/agave_vision/core/inference.py`.
- Alert: ROI violations serialized in `src/agave_vision/core/alerts.py`.
- Debouncing: per-camera/class/ROI suppression in `src/agave_vision/services/alert_router/debounce.py`.
- Redis stream: alert transport between services (`configs/services.yaml`).

### Runtime flow

- Stream Manager loads configs, warms the YOLO model, and opens RTSP streams (`src/agave_vision/services/stream_manager/main.py`).
- Each camera handler samples frames, runs inference, filters ROI violations, and publishes alerts to Redis (`src/agave_vision/services/stream_manager/camera.py`).
- Alert Router consumes Redis alerts, debounces, and delivers via protocol adapters (`src/agave_vision/services/alert_router/consumer.py`).
- Inference API serves ad-hoc image inference and config read endpoints (`src/agave_vision/services/inference_api/routes.py`).
- Interactive demo drives `AgaveVisionML` directly for visualization (`demo/interactive_demo.py`).

### Data flow

- Inputs: RTSP streams, image uploads, and local demo video.
- Transformations: YOLO inference -> detection normalization -> ROI filtering -> optional tracking.
- Outputs: JSON detection responses, Redis stream entries, optional SQLite/JSON alert logs, optional detection logs.

### Component map

| Component      | Location (path)                                        | Responsibility                                     | Inputs                               | Outputs                               | Dependencies               | Failure modes or notes                                                                   |
| -------------- | ------------------------------------------------------ | -------------------------------------------------- | ------------------------------------ | ------------------------------------- | -------------------------- | ---------------------------------------------------------------------------------------- |
| Core ML API    | `src/agave_vision/ml_api.py`, `src/agave_vision/core/` | Inference, tracking, ROI filtering, alert creation | Numpy frames, model path, ROI config | Detections and alert dicts            | ultralytics, opencv, numpy | Missing model file; ROI config missing                                                   |
| Configuration  | `src/agave_vision/config/`, `configs/*.yaml`           | Load/validate YAML configs with env overrides      | YAML files, env vars                 | Pydantic config objects               | pydantic, pyyaml           | Missing files; invalid schema                                                            |
| Inference API  | `src/agave_vision/services/inference_api/`             | FastAPI image inference + config endpoints         | Image uploads, configs               | JSON detection payloads               | fastapi, uvicorn, opencv   | Missing model or invalid image                                                           |
| Stream Manager | `src/agave_vision/services/stream_manager/`            | RTSP ingest, inference, alert publish              | RTSP streams, configs                | Redis alerts                          | opencv, redis, ultralytics | RTSP failures; Redis unavailable; camera/ROI mismatch                                    |
| Alert Router   | `src/agave_vision/services/alert_router/`              | Consume Redis alerts, debounce, route              | Redis stream entries                 | stdout/webhook/hikvision side effects | redis, httpx               | Protocol adapter factory missing (see gaps); webhook failures; Hikvision not implemented |
| Storage        | `src/agave_vision/storage/`                            | Persist alerts and detection logs                  | Alert/detection dicts                | SQLite DB, JSON/JSONL files           | sqlite3, filesystem        | Disk permissions; file growth                                                            |
| Demo           | `demo/`                                                | Interactive visualization and synthetic alerts     | MP4 video, model, ROIs               | UI display, JSON alert log            | opencv, numpy              | Missing demo video file; missing model                                                   |
| Models         | `models/`                                              | YOLO weights                                       | Training artifacts (not in repo)     | .pt weights                           | ultralytics                | Weight incompatibility or missing files                                                  |

## Entry points

- **CLI commands or scripts:** `./demo/run_demo.sh` for the interactive demo; `agave-ingest` and `agave-train` are declared but modules are missing (see gaps).
- **APIs or routes:** `GET /health`, `POST /infer`, `GET /config/rois`, `GET /config/cameras` in `src/agave_vision/services/inference_api/routes.py`.
- **Workers, schedulers, or jobs:** Stream Manager (`python -m agave_vision.services.stream_manager.main`) and Alert Router (`python -m agave_vision.services.alert_router.main`).

## Configuration and secrets

- **Configuration sources:** YAML files in `configs/` and environment variables set at runtime.
- **Precedence:** environment variables > YAML values > Pydantic defaults (`src/agave_vision/config/loader.py`).
- **Secrets handling patterns:** inject RTSP credentials and webhook/Hikvision creds via env vars; avoid committing them in configs.
- **Safe defaults and what must never be committed:** default thresholds and paths are safe; credentials and internal URLs must not be committed.

## Observability

- **Logging strategy:** JSON logs via `src/agave_vision/utils/logging.py` for services; demo uses stdout.
- **Debug workflow:** run the service locally, increase `LOG_LEVEL`, inspect Redis stream and alert router logs.

## Testing

- **How to run tests:** `pytest` (configured in `pyproject.toml`).

## Deployment

- **Build artifacts:** Python package and model weights.
- **Deployment targets:** service processes; no container or serverless definitions found.
- **Rollback strategy:** not defined.

## Contributing

- **Repository conventions:** Black and Ruff (`pyproject.toml`), mypy settings present.
- **How to safely add a new feature and where to start:** add or extend configs, update core logic in `src/agave_vision/core/`, and add services/adapters in `src/agave_vision/services/`
