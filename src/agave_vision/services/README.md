# agave-vision-api / src/agave_vision/services

## Overview
- Service implementations for inference API, stream manager, and alert router.
- Designed to run as separate processes coordinated via Redis streams.
- Primary users: operators deploying real-time inference pipelines.

## Quickstart
### Prerequisites
- Python 3.11+.
- Model weights.
- Configs under `configs/`.
- Redis for stream/alert services.

### Install
- `pip install -e .[api,stream,alerts]`.

### Configure
- Edit `configs/services.yaml`, `configs/cameras.yaml`, `configs/rois.yaml`.
- Set env overrides as needed (`REDIS_URL`, `INFERENCE_*`, `ALERTING_*`, `CAMERA_{id}_RTSP_URL`).

### Run (development)
- `uvicorn agave_vision.services.inference_api.app:app --host 0.0.0.0 --port 8000`.
- `python -m agave_vision.services.stream_manager.main`.
- `python -m agave_vision.services.alert_router.main`.

### Run (production-like)
- Run the three services as independent processes with a shared Redis.

### Common commands
- See commands above.

### Troubleshooting
1. Redis unavailable -> set `REDIS_URL`.
2. Model path invalid -> set `INFERENCE_MODEL_PATH`.
3. RTSP failures -> verify `CAMERA_{id}_RTSP_URL`.
4. Alert router fails -> protocol adapter factory missing (see gaps).
5. Missing optional deps -> install `.[api,stream,alerts]`.

## Architecture
### High-level diagram
```
Stream Manager -> Redis Stream -> Alert Router -> Protocol Adapter
Inference API runs independently for image uploads
```

### Key concepts
- Stream Manager: RTSP ingestion + inference + Redis publishing.
- Alert Router: Redis consumer + debouncer + protocol delivery.
- Inference API: FastAPI service for image inference.
- Protocol adapters: stdout/webhook/hikvision.

### Runtime flow
- Stream Manager loads configs, warms model, reads RTSP streams, and publishes alerts.
- Alert Router consumes alerts, debounces, and forwards to configured protocol.
- Inference API provides REST endpoints for single-image inference.

### Data flow
- Inputs: RTSP streams, image uploads, configs.
- Transformations: inference and ROI filtering.
- Outputs: Redis alerts and HTTP responses.

### Component map
| Component | Location (path) | Responsibility | Inputs | Outputs | Dependencies | Failure modes or notes |
| --- | --- | --- | --- | --- | --- | --- |
| Inference API | `src/agave_vision/services/inference_api/` | REST API for inference | image uploads | JSON detections | fastapi, uvicorn | Invalid image or model missing |
| Stream Manager | `src/agave_vision/services/stream_manager/` | RTSP ingest + alert publish | RTSP streams | Redis alerts | redis, opencv | RTSP disconnects |
| Alert Router | `src/agave_vision/services/alert_router/` | Redis consume + route | Redis stream | stdout/webhook/hikvision | redis, httpx | Protocol adapter missing |

## Entry points
- **CLI commands or scripts:** see run commands above.
- **APIs or routes:** `GET /health`, `POST /infer`, `GET /config/rois`, `GET /config/cameras`.
- **Workers, schedulers, or jobs:** stream manager and alert router loops.
- **Notebooks:** none.

## Configuration and secrets
- **Configuration sources:** `configs/services.yaml`, `configs/cameras.yaml`, `configs/rois.yaml`, env overrides.
- **Precedence:** env overrides > YAML > defaults.
- **Secrets handling patterns:** RTSP credentials and webhook/Hikvision credentials via env vars.
- **Safe defaults and what must never be committed:** avoid committing credentials.

## Observability
- **Logging strategy:** JSON logs via `src/agave_vision/utils/logging.py`.
- **Metrics or tracing:** none.
- **Debug workflow:** check service logs and Redis stream contents.

## Testing
- **How to run tests:** `pytest`.
- **Summary of the test pyramid:** no tests in repo.
- **Gaps and recommended minimal regression coverage:** add integration tests for Redis alert flow.

## Deployment (if applicable)
- **Build artifacts:** Python package.
- **Deployment targets:** service processes.
- **CI/CD overview:** none found.
- **Rollback strategy:** revert configs or service version.

## Contributing
- **Repository conventions:** keep service configs backward compatible.
- **Branching and release practices:** not defined.
- **How to safely add a new feature and where to start:** add a service module and update configs.

## Known gaps and assumptions
- `get_protocol_adapter` is referenced but not implemented in `src/agave_vision/services/alert_router/protocols/`.
