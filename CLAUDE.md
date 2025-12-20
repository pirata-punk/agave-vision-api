# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Agave Vision is a YOLOv8-based computer vision system for detecting objects (pina, worker, object) in industrial camera feeds. The system uses Region of Interest (ROI) filtering to generate alerts when forbidden objects enter restricted zones.

**Core detection classes** (YOLO IDs):
- 0: object (triggers alerts in forbidden ROIs)
- 1: pine (allowed, no alerts)
- 2: worker (allowed, no alerts)

## Environment Setup

```bash
# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

**Key dependencies**: ultralytics (YOLOv8), opencv-python, numpy, pandas, pyyaml, shapely

## Data Pipeline Flow

The pipeline follows a strict sequential flow (scripts are run manually, not auto-triggered):

```
videos → frames (deduped) → tiles_pool → tiles_man → tiles_round{1..4} → tiles_yolo → train/infer
```

### Pipeline Scripts (in order)

1. **Extract frames**: `python scripts/extract_frames.py`
   - Input: `data/videos/`
   - Output: `data/frames/` + `frames_manifest.json`
   - Deduplicates frames using MAD/SSIM

2. **Scan frame metadata**: `python scripts/scan_frames_metadata.py`
   - Output: `frames_metadata.json`

3. **Generate tiles**: `python scripts/generate_tiles.py`
   - Input: `data/frames/`
   - Output: `data/tiles_pool/` + metadata.json

4. **Clean tiles**: `python scripts/clean_tiles.py`
   - Input: `data/tiles_pool/`
   - Output: `data/tiles_pool/tiles_man/` + metadata_man.json
   - Applies edge filters and quotas

5. **Standardize filenames**: `python scripts/standardize_round_filenames.py`
   - Normalizes filenames in tiles_round{1..4}
   - Decodes percent-encodings, strips hashes, removes spaces, collapses underscores
   - Maintains 1:1 image/label pairing

6. **Build YOLO dataset**: `python scripts/build_yolo_dataset.py`
   - Input: `data/tiles_pool/tiles_man/tiles_round{1..4}/`
   - Output: `data/tiles_yolo/{images,labels}/{train,val,test}` + `configs/yolo_data.yaml` + metadata.json
   - Deterministic split (seed=123): 70% train, 15% val, 15% test

7. **Train model**: `python scripts/train_yolo_pina_detector.py`
   - Trains YOLOv8n on the merged dataset
   - Output: `models/yolov8n_pina/exp/weights/best.pt`

## Training

```bash
# After building the dataset with build_yolo_dataset.py
python scripts/train_yolo_pina_detector.py

# Or manually with ultralytics CLI:
yolo detect train model=yolov8n.pt data=configs/yolo_data.yaml epochs=100 imgsz=640 batch=16
```

Training config is in `scripts/train_yolo_pina_detector.py`:
- Base model: yolov8n.pt
- Epochs: 100
- Image size: 640
- Batch: 16
- Workers: 8

## Inference & Alerting

### Real-time streaming

```bash
python scripts/realtime_yolo_stream.py
```

- Supports webcam (SOURCE=0), RTSP URLs, or video files
- Displays annotated frames with detections
- Logs JSON alerts to stdout when "object" class enters forbidden ROI
- Edit `FORBIDDEN_ROIS` in the script to define restricted zones

### Alert pipeline (programmatic)

```bash
python scripts/infer_alert.py
```

- Loads ROI config from YAML (see `configs/rois.example.yaml`)
- Processes frames/images per camera
- Emits JSON alerts when "object" class detected in forbidden ROI
- "pine" and "worker" classes are allowed and do not trigger alerts

### Demo video inference

```bash
python scripts/demo_video_infer.py
```

- File-to-file annotated video output for offline review

## ROI Configuration

ROIs are defined per-camera in YAML format:

```yaml
cameras:
  - id: cam_1
    forbidden_rois:
      - [[100, 100], [500, 100], [500, 400], [100, 400]]  # polygon as [x, y] points
```

See `configs/rois.example.yaml` for the template.

## Architecture Patterns

### Filename Standardization

When working with labeled datasets in tiles_round{1..4}, filenames must be normalized to maintain image/label pairing:

1. Decode percent-encodings (`%20` → space)
2. Strip hashed prefixes (`0a8f977d-` or `0a5b0597__`)
3. Remove `images/` or `images\` artifacts
4. Replace spaces and slashes with `_`
5. Collapse multiple underscores (`__` → `_`)
6. Apply to both images and labels identically

### Detection & Alert Logic

- ROI filtering is applied **post-inference** (after NMS)
- Alert trigger: bbox center point inside forbidden ROI polygon
- Use `cv2.pointPolygonTest()` for point-in-polygon checks
- Only "object" class triggers alerts; "pine" and "worker" are explicitly allowed

### Model Versioning

- Trained weights stored under `models/yolov8n_pina/<version>/`
- Keep `configs/yolo_data.yaml`, training command, and git commit hash with each model version
- Maintain `data/tiles_yolo/metadata.json` for data lineage and split tracking

## Production System Architecture

The codebase includes design docs for a production-ready deployment (see `docs/system_architecture.md`):

- **Inference Service**: FastAPI/gRPC with YOLOv8, model versioning, health checks
- **Stream Manager**: RTSP ingestion from Hikvision cameras, frame sampling, ROI caching
- **Alert Router**: Kafka/Redis message bus, protocol adapter for Hikvision integration
- **Config Store**: PostgreSQL for camera registry, ROI polygons, model metadata
- **Observability**: Prometheus metrics, structured logs, Grafana dashboards

Deployment targets: Docker Compose (single-node GPU) or Kubernetes (multi-node cluster).

## Directory Structure

```
data/
  frames/          # Extracted & deduplicated frames
  tiles_pool/      # Generated tiles and cleaned subsets
    tiles_man/     # Manually selected tiles
      tiles_round{1..4}/  # Labeled rounds (images/ and labels/)
  tiles_yolo/      # Final YOLO dataset (train/val/test splits)
  videos/          # Source videos

configs/
  yolo_data.yaml   # YOLO dataset config (paths, class names)
  rois.example.yaml  # ROI polygon definitions per camera

scripts/          # All pipeline and inference scripts (see Pipeline Flow)

models/
  yolov8n_pina/   # Training output directory (exp/weights/best.pt)

docs/             # Architecture notes, labeling instructions, planning docs
```

## Important Conventions

- **No auto-execution**: All pipeline scripts are manual-run only; they do not auto-trigger
- **Deterministic splits**: Always use seed=123 for reproducible train/val/test splits
- **Class order matters**: YOLO IDs must match the order in `yolo_data.yaml` (0: object, 1: pine, 2: worker)
- **Non-empty labels only**: By default, only keep image/label pairs where label file is non-empty (configurable)
- **Python 3.12**: Virtual environment uses Python 3.12; ensure compatibility

## Development Notes

- Frame sampling targets 5–10 FPS to balance detection responsiveness and compute load
- Use `half` precision on GPU for faster inference with minimal accuracy loss
- ROI filtering is cheap; apply after NMS to avoid unnecessary overhead
- For large-scale inference, consider ONNX/TensorRT export for optimization
