# agave-vision-api / src/agave_vision/services/stream_manager

## Overview
- RTSP ingestion service that samples frames, runs inference, and publishes alerts to Redis.
- Runs one handler per enabled camera.
- Primary users: operators running real-time alerting pipelines.

## Quickstart
### Prerequisites
- Python 3.11+.
- Redis.
- RTSP camera URLs.
- Model weights.
- `configs/cameras.yaml`, `configs/rois.yaml`, `configs/services.yaml`.

### Install
- `pip install -e .[stream,alerts]`.

### Configure
- Set camera RTSP URLs in `configs/cameras.yaml` or via `CAMERA_{id}_RTSP_URL`.
- Set Redis URL via `REDIS_URL`.

### Run (development)
- `python -m agave_vision.services.stream_manager.main`.

### Run (production-like)
- Run as a long-lived process with stable Redis and RTSP connectivity.

### Common commands
- `python -m agave_vision.services.stream_manager.main`.

### Troubleshooting
1. RTSP connection fails -> validate URL and network.
2. Redis publish errors -> set `REDIS_URL`.
3. No alerts -> check ROI camera ID match.
4. High CPU -> lower `fps_target`.
5. Model load error -> check `INFERENCE_MODEL_PATH`.

## Architecture
### High-level diagram
```
RTSP stream -> CameraHandler -> YOLOInference -> ROI filter -> Redis Stream
```

### Key concepts
- CameraHandler: per-camera loop for sampling and inference.
- RedisPublisher: publishes alerts to a Redis stream.
- ROIManager: filters detections for forbidden zones.

### Runtime flow
- Loads camera and service configs.
- Initializes YOLO model and ROI manager.
- Creates a camera handler for each enabled camera.
- Publishes alert events to Redis.

### Data flow
- Inputs: RTSP frames, configs.
- Transformations: inference -> ROI filtering -> alert serialization.
- Outputs: Redis stream entries.

### Component map
| Component | Location (path) | Responsibility | Inputs | Outputs | Dependencies | Failure modes or notes |
| --- | --- | --- | --- | --- | --- | --- |
| Main | `src/agave_vision/services/stream_manager/main.py` | Service startup and orchestration | configs | running handlers | redis, ultralytics | No enabled cameras |
| CameraHandler | `src/agave_vision/services/stream_manager/camera.py` | RTSP loop and alerting | frames | AlertEvent -> Redis | opencv | RTSP disconnects |
| RedisPublisher | `src/agave_vision/services/stream_manager/publisher.py` | Publish alerts | AlertEvent | Redis stream message | redis | Redis unavailable |

## Entry points
- **CLI commands or scripts:** `python -m agave_vision.services.stream_manager.main`.
- **APIs or routes:** none.
- **Workers, schedulers, or jobs:** camera handler loops.
- **Notebooks:** none.

## Configuration and secrets
- **Configuration sources:** `configs/cameras.yaml`, `configs/rois.yaml`, `configs/services.yaml`, env overrides.
- **Precedence:** env overrides > YAML > defaults.
- **Secrets handling patterns:** RTSP credentials via env vars.
- **Safe defaults and what must never be committed:** do not commit RTSP passwords.

## Observability
- **Logging strategy:** JSON logs via `setup_logging` in `main.py`.
- **Metrics or tracing:** none.
- **Debug workflow:** inspect logs for connection and publish errors.

## Testing
- **How to run tests:** `pytest`.
- **Summary of the test pyramid:** no tests in repo.
- **Gaps and recommended minimal regression coverage:** integration test for Redis publishing.

## Deployment (if applicable)
- **Build artifacts:** Python package.
- **Deployment targets:** service process.
- **CI/CD overview:** none found.
- **Rollback strategy:** revert configs or service version.

## Contributing
- **Repository conventions:** keep camera handling resilient and configurable.
- **Branching and release practices:** not defined.
- **How to safely add a new feature and where to start:** extend `CameraHandler` or publisher.

## Known gaps and assumptions
- Assumption: Redis stream exists or can be created by publisher.
