# agave-vision-api / demo

## Overview

- Interactive object detection demo with ROI visualization and synthetic object injection.
- Uses the production ML API (`AgaveVisionML`) and configs in `configs/`.
- Designed for local validation, demos, and alert logic testing.
- Primary users: developers and stakeholders validating detection and ROI behavior.

## Quickstart

### Prerequisites

- Python 3.11+.
- Model weights at `models/agave-industrial-vision-v1.0.0.pt`.
- The demo video file referenced in `demo/interactive_demo.py`.

### Install

- `python -m venv .venv && source .venv/bin/activate && pip install -e .`.

### Configure

- Update `demo/interactive_demo.py` for video path and camera ID.
- ROIs from `configs/rois.yaml`.
- Model path from `configs/model.yaml`.

### Run (development)

- `./demo/run_demo.sh` or `python demo/interactive_demo.py`.

### Common commands

- `./demo/run_demo.sh`.

### Troubleshooting

1. Video file missing -> update `self.video_path`.
2. Model file missing -> update `configs/model.yaml`.
3. No ROI overlay -> ensure `configs/rois.yaml` has matching `camera_id`.
4. Low FPS -> reduce display size or disable heavy overlays in code.
5. OpenCV window issues -> ensure GUI support on your OS.

## Architecture

### High-level diagram

```
Video file -> `AgaveVisionML` -> Detections + ROI alerts -> Visualization + JSON alert log
```

### Key concepts

- Synthetic object injection: overlays shapes to force alert triggers.
- ROI visualization: forbidden zones drawn from `configs/rois.yaml`.
- Tracking: uses `CentroidTracker` when enabled in `AgaveVisionML`.
- Alert log: JSON output written to `demo/alert_log_*.json`.

### Runtime flow

- Loads model and ROI config (`demo/interactive_demo.py`).
- Reads frames from the demo video file.
- Optionally injects synthetic objects into frames.
- Runs inference, applies ROI rules, and renders overlays.
- Writes alert log on completion.

### Data flow

- Inputs: demo video, model weights, ROI config.
- Transformations: frame -> detection -> ROI alert -> visualization.
- Outputs: on-screen UI and JSON alert log file.

### Component map

| Component        | Location (path)            | Responsibility                        | Inputs       | Outputs                    | Dependencies  | Failure modes or notes |
| ---------------- | -------------------------- | ------------------------------------- | ------------ | -------------------------- | ------------- | ---------------------- |
| Interactive demo | `demo/interactive_demo.py` | Main UI, controls, and inference loop | Video frames | Display output, alert logs | opencv, numpy | Missing video or model |
| Demo launcher    | `demo/run_demo.sh`         | Ensures venv and files exist          | venv, files  | Runs demo                  | bash          | Hard-coded paths       |

## Entry points

- **CLI commands or scripts:** `./demo/run_demo.sh`, `python demo/interactive_demo.py`.

## Configuration and secrets

- **Configuration sources:** `configs/model.yaml`, `configs/rois.yaml`, `configs/display.yaml`.
- **Precedence:** local demo settings override defaults in code.
- **Safe defaults and what must never be committed:** avoid committing real RTSP credentials if added for demo.

## Observability

- **Logging strategy:** stdout prints and JSON alert log file.
- **Metrics or tracing:** FPS and counts displayed on screen.
- **Debug workflow:** run with console visible, inspect alert log.

## Contributing

- **Repository conventions:** keep demo paths configurable and avoid hardcoding secrets.
- **Branching and release practices:** not defined.
- **How to safely add a new feature and where to start:** extend `InteractiveDemo` for new overlays or controls.
