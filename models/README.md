# agave-vision-api / models

## Overview

- Model weight storage for YOLOv8 inference.
- Default model path is referenced in `configs/model.yaml`.
- Primary users: developers deploying inference or demos.

## Quickstart

### Prerequisites

- A YOLOv8 `.pt` weights file.

### Install

- Place model weights at the path defined in `configs/model.yaml`.

### Configure

- Set `INFERENCE_MODEL_PATH` to override; otherwise `configs/model.yaml` is used by services.

### Run (development)

- Start inference API or stream manager after the model file exists.

### Run (production-like)

- Same as development; ensure the model file is mounted.

### Common commands

- `ls -lh models/`.

### Troubleshooting

1. File not found -> update `configs/model.yaml`.
2. Incompatible weights -> ensure YOLOv8-compatible `.pt`.
3. Slow load -> use local disk.
4. Wrong classes -> update `configs/model.yaml` metadata.
5. Permissions -> ensure read access.

## Architecture

### High-level diagram

```
models/*.pt -> YOLOInference (`src/agave_vision/core/inference.py`) -> detections
```

### Key concepts

- Default model path: `configs/model.yaml` `default_model`.
- Model metadata: `configs/model.yaml` `model_info`.
- Inference defaults: `configs/model.yaml` `inference`.

### Runtime flow

- Services read model path from `configs/services.yaml` or env override.
- `YOLOInference` loads the model on startup.
- Model warmup runs for configured iterations.

### Data flow

- Inputs: `.pt` weight file and inference config.
- Transformations: weights loaded into Ultralytics YOLO runtime.
- Outputs: in-memory model used for inference.

### Component map

| Component    | Location (path)      | Responsibility              | Inputs             | Outputs      | Dependencies | Failure modes or notes       |
| ------------ | -------------------- | --------------------------- | ------------------ | ------------ | ------------ | ---------------------------- |
| Weights      | `models/*.pt`        | Model parameters for YOLOv8 | training artifacts | loaded model | ultralytics  | Missing or incompatible file |
| Model config | `configs/model.yaml` | Default path and metadata   | model registry     | model path   | yaml         | Missing file uses defaults   |

## Configuration and secrets

- **Configuration sources:** `configs/model.yaml` and `INFERENCE_MODEL_PATH`.
- **Precedence:** env override > YAML > default fallback in `src/agave_vision/config/model_config.py`.
