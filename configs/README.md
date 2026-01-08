# agave-vision-api / configs

## Overview

- YAML configuration files consumed by services and the ML API.
- Defines camera registry, ROI polygons, service runtime settings, and model metadata.
- Includes demo display colors and YOLO dataset config for training artifacts.
- Primary users: operators and developers deploying or tuning the system.

## Quickstart

### Prerequisites

- None beyond editing YAML; services load these at startup.

### Install

- Not applicable.

### Configure

- Edit `configs/*.yaml`; overrides applied by `src/agave_vision/config/loader.py` in order env > YAML > defaults.
- Inject secrets via env for RTSP and webhook/Hikvision credentials.

### Run (development)

- Start services; they read `configs/` on startup.

### Run (production-like)

- Same as development; ensure configs are mounted/readable by services.

### Common commands

- `cat configs/cameras.yaml`.
- `cat configs/rois.yaml`.
- `cat configs/services.yaml`.

### Troubleshooting

1. YAML parse errors -> validate indentation and list syntax.
2. ROI alerts missing -> ensure `camera_id` matches `configs/cameras.yaml`.
3. Model path invalid -> update `configs/model.yaml` or `INFERENCE_MODEL_PATH`.
4. Wrong device -> set `services.inference.device` or `INFERENCE_DEVICE`.
5. RTSP auth failures -> override `CAMERA_{camera_id}_RTSP_URL` via env.

## Architecture

### High-level diagram

```
configs/*.yaml -> ConfigLoader (`src/agave_vision/config/loader.py`) -> services and ML API
```

### Key concepts

- Camera registry: `configs/cameras.yaml` defines RTSP URLs and sampling rates.
- ROI definitions: `configs/rois.yaml` defines forbidden polygons and class allowlists.
- Service runtime: `configs/services.yaml` defines inference, stream manager, and alerting settings.
- Model registry: `configs/model.yaml` defines default model path and metadata.
- Demo display: `configs/display.yaml` defines class and ROI colors for UI.
- YOLO dataset: `configs/yolo_data.yaml` defines train/val/test paths.

### Runtime flow

- Services load `configs/services.yaml` and `configs/rois.yaml` on startup.
- Stream manager and inference API use model path and thresholds from `configs/services.yaml`.
- Demo loads `configs/display.yaml` and `configs/rois.yaml`.

### Data flow

- Inputs: YAML files and env var overrides.
- Transformations: YAML -> Pydantic models (`src/agave_vision/config/models.py`).
- Outputs: validated config objects used by services and ML API.

### Component map

| Component        | Location (path)          | Responsibility                           | Inputs                             | Outputs              | Dependencies     | Failure modes or notes             |
| ---------------- | ------------------------ | ---------------------------------------- | ---------------------------------- | -------------------- | ---------------- | ---------------------------------- |
| Camera registry  | `configs/cameras.yaml`   | Define RTSP endpoints and sampling rates | RTSP URLs                          | CameraConfig objects | pydantic         | Invalid RTSP URL format            |
| ROI registry     | `configs/rois.yaml`      | Define forbidden zones and alert rules   | Polygon points                     | CameraROI objects    | pydantic, opencv | Camera ID mismatch                 |
| Service settings | `configs/services.yaml`  | Inference/stream/alert config            | thresholds, device, Redis settings | ServicesConfig       | pydantic         | Device unsupported                 |
| Model metadata   | `configs/model.yaml`     | Default model path and metadata          | model paths                        | ModelConfig          | yaml             | Missing file uses defaults         |
| Display config   | `configs/display.yaml`   | Demo UI colors                           | class/ROI colors                   | dict of colors       | yaml             | Missing file uses defaults in demo |
| YOLO dataset     | `configs/yolo_data.yaml` | Training dataset paths                   | data paths                         | training config      | yaml             | Paths may not exist                |

## Configuration and secrets

- **Configuration sources:** YAML files under `configs/` with environment variable overrides.
- **Precedence:** environment variables > YAML values > Pydantic defaults.
- **Secrets handling patterns:** set RTSP credentials and alerting credentials via env vars.
- **Safe defaults and what must never be committed:** do not commit RTSP passwords or webhook/Hikvision credentials.

## Observability

- **Debug workflow:** verify values via `ConfigLoader` usage and service logs.

## Deployment

- **Build artifacts:** YAML files copied/mounted with services.
- **Deployment targets:** same as services.
- **Rollback strategy:** revert YAML changes.

## Contributing

- **Repository conventions:** keep YAML valid and consistent with `src/agave_vision/config/models.py`.
- **Branching and release practices:** not defined.
- **How to safely add a new feature and where to start:** update config models first, then extend YAML files.
