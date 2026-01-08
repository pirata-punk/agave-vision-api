# agave-vision-api / src/agave_vision/services/inference_api

## Overview
- FastAPI service providing image inference and config read endpoints.
- Loads YOLO model and ROI config on startup.
- Primary users: external clients needing ad-hoc inference.

## Quickstart
### Prerequisites
- Python 3.11+.
- Model weights.
- `configs/services.yaml`, `configs/rois.yaml`.

### Install
- `pip install -e .[api]`.

### Configure
- Set `INFERENCE_*` env vars to override `configs/services.yaml`.

### Run (development)
- `uvicorn agave_vision.services.inference_api.app:app --host 0.0.0.0 --port 8000`.

### Run (production-like)
- Run with a process manager; configure CORS and network policies as needed.

### Common commands
- `curl http://localhost:8000/health`.

### Troubleshooting
1. Model load fails -> update `configs/services.yaml`.
2. Invalid image -> ensure JPEG/PNG.
3. CORS issues -> adjust CORS middleware.
4. Missing ROIs -> check `configs/rois.yaml`.
5. Slow inference -> adjust device and image size.

## Architecture
### High-level diagram
```
HTTP client -> FastAPI -> YOLOInference -> JSON response
```

### Key concepts
- Lifespan startup loads model and ROI config.
- `/infer` accepts image uploads and returns detections.
- `/config/rois` and `/config/cameras` expose read-only config.

### Runtime flow
- Startup loads configs and warms model (`app.py`).
- `/infer` decodes image, runs model, returns detections (`routes.py`).
- `/config/*` returns sanitized configs (`routes.py`).

### Data flow
- Inputs: HTTP multipart image upload.
- Transformations: image decode -> inference -> schema serialization.
- Outputs: JSON response with detections and timing.

### Component map
| Component | Location (path) | Responsibility | Inputs | Outputs | Dependencies | Failure modes or notes |
| --- | --- | --- | --- | --- | --- | --- |
| App factory | `src/agave_vision/services/inference_api/app.py` | App startup, model load | configs | app state | fastapi | Model not found |
| Routes | `src/agave_vision/services/inference_api/routes.py` | REST endpoints | images | JSON responses | opencv, numpy | Invalid image |
| Schemas | `src/agave_vision/services/inference_api/schemas.py` | Response models | detection data | JSON schema | pydantic | None |
| Dependencies | `src/agave_vision/services/inference_api/dependencies.py` | Access app state | request | model/ROI/config | fastapi | None |

## Entry points
- **CLI commands or scripts:** `uvicorn agave_vision.services.inference_api.app:app`.
- **APIs or routes:** `GET /health`, `POST /infer`, `GET /config/rois`, `GET /config/cameras`.
- **Workers, schedulers, or jobs:** none.
- **Notebooks:** none.

## Configuration and secrets
- **Configuration sources:** `configs/services.yaml`, `configs/rois.yaml`, `configs/cameras.yaml`, env overrides.
- **Precedence:** env overrides > YAML > defaults.
- **Secrets handling patterns:** avoid embedding credentials in configs.
- **Safe defaults and what must never be committed:** do not commit RTSP credentials.

## Observability
- **Logging strategy:** JSON logs via `setup_logging` in `app.py`.
- **Metrics or tracing:** none.
- **Debug workflow:** inspect `/health` and service logs.

## Testing
- **How to run tests:** `pytest`.
- **Summary of the test pyramid:** no tests in repo.
- **Gaps and recommended minimal regression coverage:** add tests for `/infer` and `/health`.

## Deployment (if applicable)
- **Build artifacts:** Python package.
- **Deployment targets:** HTTP service.
- **CI/CD overview:** none found.
- **Rollback strategy:** revert service version.

## Contributing
- **Repository conventions:** keep API schemas stable.
- **Branching and release practices:** not defined.
- **How to safely add a new feature and where to start:** add routes and schemas in this module.

## Known gaps and assumptions
- No authentication or rate limiting is implemented.
