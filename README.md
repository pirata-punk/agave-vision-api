# Agave Vision API

Production-ready YOLOv8 object detection system for industrial cameras with real-time RTSP streaming and ROI-based alerting.

## Overview

Agave Vision is a complete end-to-end system for training and deploying custom object detection models for industrial video surveillance. The system is organized into **three functional modules** plus production deployment:

1. **[Ingestion](ingestion/)** - Data preparation pipeline (videos → frames → tiles → datasets)
2. **[Training](training/)** - Model training workbench with versioning and evaluation
3. **[Models](models/)** - Model registry for version management and deployment
4. **[Production](production/)** - Containerized microservices for live RTSP streaming and alerting

## Features

### Data Pipeline
- **Video Processing**: Extract frames with deduplication and sharpness filtering
- **Tile Generation**: Sliding window tiling for large images
- **Dataset Builder**: YOLO-ready datasets with train/val/test splits

### Training Workbench
- **Model Training**: YOLOv8 training with hyperparameter management
- **Evaluation**: Metrics computation (mAP, precision, recall)
- **Model Comparison**: Side-by-side comparison of multiple versions

### Model Registry
- **Version Management**: Semantic versioning with metadata tracking
- **Metrics Tracking**: Store evaluation metrics with each version
- **Tagging System**: Organize models by stage (baseline, production, etc.)

### Production Services
- **Real-time Detection**: YOLOv8-based detection for custom classes (object, pine, worker)
- **Multi-Camera Support**: Concurrent RTSP stream processing from multiple cameras
- **ROI-Based Alerting**: Flexible region-of-interest filtering with customizable alert rules
- **Microservice Architecture**: Three containerized services (Inference API, Stream Manager, Alert Router)
- **Alert Delivery**: Pluggable protocols (stdout, webhook, Hikvision integration)
- **Production-Ready**: Docker Compose deployment with GPU support, health checks, and auto-restart

## System Architecture

### Complete Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                       DATA PREPARATION                              │
│                                                                     │
│  Videos → Frames → Tiles → Labels → YOLO Dataset                   │
│  (ingestion module)                                                │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       TRAINING WORKBENCH                            │
│                                                                     │
│  Train → Evaluate → Compare → Experiment                           │
│  (training module)                                                 │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       MODEL REGISTRY                                │
│                                                                     │
│  Register → Tag → Version → Deploy                                 │
│  (models module)                                                   │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       PRODUCTION SERVICES                           │
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │
│  │ Inference    │    │ Stream       │    │ Alert        │         │
│  │ API          │    │ Manager      │    │ Router       │         │
│  │ (FastAPI)    │    │ (RTSP        │    │ (Redis       │         │
│  │              │    │  Ingestion)  │    │  Consumer)   │         │
│  │ Port: 8000   │    │              │    │              │         │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘         │
│         │                   │                    │                 │
│         └───────────────────┼────────────────────┘                 │
│                             │                                      │
│                      ┌──────▼───────┐                              │
│                      │ Redis        │                              │
│                      │ (Message Bus)│                              │
│                      └──────────────┘                              │
│  (production module)                                               │
└─────────────────────────────────────────────────────────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker & Docker Compose (for production deployment)
- NVIDIA GPU with drivers (for GPU training and inference)
- NVIDIA Container Toolkit (for Docker GPU support)

### Installation

```bash
git clone <repository-url>
cd agave-vision-api
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .[all]
```

### Complete Workflow Example

#### 1. Prepare Training Data (Ingestion Module)

```bash
# Extract frames from videos
python ingestion/cli.py extract-frames \
    --video-dir data/videos \
    --output-dir data/frames \
    --sample-rate 30

# Generate tiles for labeling
python ingestion/cli.py generate-tiles \
    --frames-dir data/frames \
    --output-dir data/tiles_pool \
    --tile-size 640 \
    --overlap 128

# Manually label tiles using your preferred tool (Roboflow, LabelImg, etc.)
# Then build YOLO dataset

python ingestion/cli.py build-dataset \
    --rounds-dir data/tiles_pool/tiles_man \
    --output-dir data/tiles_yolo \
    --classes object pine worker \
    --train-ratio 0.7 \
    --val-ratio 0.15 \
    --test-ratio 0.15
```

See [ingestion/README.md](ingestion/README.md) for details.

#### 2. Train Model (Training Workbench)

```bash
# Train a new model
python training/cli.py train \
    --data-yaml configs/yolo_data.yaml \
    --model yolov8n \
    --epochs 100 \
    --batch 16 \
    --device cuda \
    --version v1_baseline

# Evaluate on test set
python training/cli.py evaluate \
    --model training/runs/v1_baseline/weights/best.pt \
    --split test
```

See [training/README.md](training/README.md) for details.

#### 3. Register Model (Model Registry)

```python
from agave_vision.models.registry import ModelRegistry

registry = ModelRegistry("models")
registry.register_model(
    version_name="v1_baseline",
    weights_path="training/runs/v1_baseline/weights/best.pt",
    metrics={"map50": 0.842, "map50_95": 0.678},
    config={"model": "yolov8n", "epochs": 100},
    tags=["production"]
)
```

See [models/README.md](models/README.md) for details.

#### 4. Deploy to Production

**Configure cameras and ROIs**:
Edit `configs/cameras.yaml` and `configs/rois.yaml` with your camera URLs and ROI polygons.

**Update service configuration** to use registered model:
```yaml
# configs/services.yaml
inference:
  model_path: "models/v1_baseline/weights/best.pt"
```

**Deploy with Docker Compose**:

```bash
# Configure environment
cp .env.example .env
# Edit .env with your camera RTSP URLs and settings

# Build and start services
cd production
docker compose build
docker compose up -d

# Check service health
curl http://localhost:8000/health

# Monitor alerts
docker compose logs -f alert-router
```

See production deployment section below for details.

## Configuration

### Camera Configuration (`configs/cameras.yaml`)

```yaml
cameras:
  - id: cam_nave3_hornos
    name: "Nave 3 Hornos B CAM 3"
    rtsp_url: "rtsp://admin:password@192.168.1.10:554/Streaming/Channels/101"
    enabled: true
    fps_target: 5.0
```

### ROI Configuration (`configs/rois.yaml`)

```yaml
cameras:
  - camera_id: cam_nave3_hornos
    forbidden_rois:
      - name: loading_zone
        points:
          - [100, 100]
          - [500, 100]
          - [500, 400]
          - [100, 400]
    allowed_classes: [pine, worker]
    alert_classes: [object]
```

### Alert Configuration (`configs/alerting.yaml`)

```yaml
debounce_window_seconds: 5.0
max_alerts_per_window: 1
protocol: "stdout"  # Options: stdout, webhook, hikvision
webhook_url: null   # Set if using webhook protocol
```

## API Documentation

### Inference API Endpoints

**Health Check**
```bash
GET /health
```

**Run Inference**
```bash
POST /infer
Content-Type: multipart/form-data

Parameters:
- file: Image file (JPEG/PNG)
- conf: Confidence threshold (default: 0.25)

Response:
{
  "detections": [...],
  "inference_time_ms": 45.2,
  "image_size": [640, 480]
}
```

**Get ROI Configurations**
```bash
GET /config/rois
```

**Get Camera Configurations**
```bash
GET /config/cameras
```

## Modules

### 1. Ingestion Module

Handles data preparation from raw videos to YOLO-ready datasets.

**Features:**
- Frame extraction with deduplication and sharpness filtering
- Tile generation using sliding window
- Dataset builder with train/val/test splits
- Support for both static data and live streaming (RTSP)

**CLI:**
```bash
python ingestion/cli.py extract-frames --help
python ingestion/cli.py generate-tiles --help
python ingestion/cli.py build-dataset --help
```

See [ingestion/README.md](ingestion/README.md) for complete documentation.

### 2. Training Workbench

Dedicated workspace for training, evaluating, and comparing YOLO models.

**Features:**
- YOLOv8 training with versioning
- Model evaluation (mAP, precision, recall)
- Multi-model comparison
- Training config tracking

**CLI:**
```bash
python training/cli.py train --help
python training/cli.py evaluate --help
python training/cli.py compare --help
```

See [training/README.md](training/README.md) for complete documentation.

### 3. Model Registry

Centralized model versioning and metadata management.

**Features:**
- Version tracking with semantic names
- Metrics and config storage
- Tag-based organization (baseline, production, etc.)
- Deployment artifact management

**Usage:**
```python
from agave_vision.models.registry import ModelRegistry

registry = ModelRegistry("models")
model = registry.get_latest(tag="production")
```

See [models/README.md](models/README.md) for complete documentation.

### 4. Production Services

Containerized microservices for real-time RTSP streaming and alerting.

**Services:**
- **Inference API** (FastAPI) - REST API for inference requests
- **Stream Manager** - RTSP stream ingestion and processing
- **Alert Router** - Alert delivery with configurable protocols

See production deployment section below for details.

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src/agave_vision --cov-report=html

# Run specific test file
pytest tests/unit/test_roi.py

# Run integration tests (requires model)
pytest -m integration
```

## Environment Variables

Key environment variables (see `.env.example`):

```bash
# Redis
REDIS_URL=redis://localhost:6379

# Inference
INFERENCE_DEVICE=cuda
INFERENCE_MODEL_PATH=/app/models/yolov8n_pina/exp/weights/best.pt
INFERENCE_CONFIDENCE=0.25

# Camera RTSP URLs (override cameras.yaml)
CAMERA_cam_nave3_hornos_RTSP_URL=rtsp://...

# Alerting
ALERTING_PROTOCOL=stdout
ALERTING_WEBHOOK_URL=https://...
```

## Project Structure

```
agave-vision-api/
├── pyproject.toml                   # Package configuration
├── README.md                        # This file
│
├── ingestion/                       # 📦 MODULE 1: Data Ingestion
│   ├── cli.py                       # Ingestion CLI
│   ├── configs/                     # Ingestion configs
│   └── README.md                    # Ingestion documentation
│
├── training/                        # 📦 MODULE 2: Training Workbench
│   ├── cli.py                       # Training CLI
│   ├── configs/                     # Training hyperparameters
│   ├── runs/                        # Training outputs (gitignored)
│   └── README.md                    # Training documentation
│
├── models/                          # 📦 MODULE 3: Model Registry
│   ├── registry.yaml                # Model registry index
│   ├── v1_baseline/                 # Versioned models
│   │   ├── weights/best.pt
│   │   └── metadata.json
│   └── README.md                    # Registry documentation
│
├── production/                      # 📦 MODULE 4: Production Services
│   ├── docker-compose.yml           # Service orchestration
│   ├── Dockerfile.inference-api     # API service
│   ├── Dockerfile.stream-manager    # Stream service
│   └── Dockerfile.alert-router      # Alert service
│
├── src/agave_vision/                # Core library (shared)
│   ├── ingestion/                   # Ingestion implementation
│   │   └── static/                  # Static data pipeline
│   │       ├── video_processor.py
│   │       ├── tile_generator.py
│   │       └── dataset_builder.py
│   ├── training/                    # Training implementation
│   │   ├── trainer.py
│   │   └── evaluator.py
│   ├── models/                      # Model registry implementation
│   │   └── registry.py
│   ├── core/                        # Production core logic
│   │   ├── inference.py             # YOLO wrapper
│   │   ├── roi.py                   # ROI filtering
│   │   ├── alerts.py                # Alert structures
│   │   └── frames.py                # Frame utilities
│   ├── config/                      # Configuration management
│   │   ├── models.py                # Pydantic configs
│   │   └── loader.py                # Config loader
│   ├── services/                    # Production services
│   │   ├── inference_api/           # FastAPI service
│   │   ├── stream_manager/          # RTSP ingestion
│   │   └── alert_router/            # Alert routing
│   └── utils/                       # Shared utilities
│       ├── logging.py
│       └── video.py
│
├── configs/                         # YAML configurations
│   ├── yolo_data.yaml               # YOLO dataset config
│   ├── cameras.yaml                 # Camera registry
│   ├── rois.yaml                    # ROI definitions
│   ├── services.yaml                # Service runtime config
│   └── alerting.yaml                # Alert rules
│
├── tests/                           # Test suite
│   ├── unit/
│   └── integration/
│
├── data/                            # Training data (gitignored)
│   ├── videos/                      # Raw videos
│   ├── frames/                      # Extracted frames
│   ├── tiles_pool/                  # Generated tiles
│   └── tiles_yolo/                  # YOLO dataset
│
├── scripts/                         # Legacy scripts (deprecated)
└── docs/                            # Documentation
```

## Development

### Code Quality

```bash
# Format code
black src/ tests/

# Lint code
ruff src/ tests/

# Type check
mypy src/
```

### Adding New Protocol Adapter

1. Create new adapter in `src/agave_vision/services/alert_router/protocols/`
2. Inherit from `ProtocolAdapter` base class
3. Implement `send_alert()` method
4. Register in `protocols/__init__.py`
5. Add configuration to `config/models.py`

## Production Deployment

### Prerequisites

- Docker & Docker Compose installed
- NVIDIA GPU with drivers (for GPU inference)
- NVIDIA Container Toolkit configured
- Trained model registered in model registry
- Camera RTSP URLs available

### Deployment Steps

**1. Prepare Configuration Files**

```bash
# Copy environment template
cp .env.example .env

# Edit camera configuration
vim configs/cameras.yaml

# Edit ROI configuration
vim configs/rois.yaml

# Update service configuration to use registered model
vim configs/services.yaml
```

**2. Verify Model Exists**

Ensure your production model is registered and available:

```bash
ls -lh models/v1_baseline/weights/best.pt
```

**3. Build Docker Images**

```bash
cd production
docker compose build
```

**4. Start Services**

```bash
# Start all services in detached mode
docker compose up -d

# Check service status
docker compose ps
```

**5. Verify Deployment**

```bash
# Check API health
curl http://localhost:8000/health

# Expected response:
# {"status": "healthy", "model_loaded": true}

# View service logs
docker compose logs -f inference-api
docker compose logs -f stream-manager
docker compose logs -f alert-router

# Check GPU usage
nvidia-smi
```

**6. Monitor Alerts**

```bash
# Watch alert output (stdout protocol)
docker compose logs -f alert-router

# Example alert:
# [2025-01-15 14:30:00] ALERT: Camera cam_nave3_hornos - object detected in loading_zone
```

### Service Management

```bash
# Stop all services
docker compose down

# Restart a specific service
docker compose restart stream-manager

# View resource usage
docker stats

# Update service after config change
docker compose up -d --force-recreate stream-manager

# View logs for specific service
docker compose logs --tail=100 -f inference-api
```

### Scaling Considerations

For production deployment with multiple cameras:

1. **GPU Memory**: Monitor VRAM usage with `nvidia-smi`. Each model instance uses ~500MB
2. **CPU Cores**: Stream manager is CPU-intensive for frame decoding
3. **Network Bandwidth**: RTSP streams consume ~2-4 Mbps per camera
4. **Redis Memory**: Alerts are ephemeral, but monitor with `docker stats redis`

### Health Checks

All services include health checks:

```bash
# Check which services are healthy
docker compose ps

# Services show as "healthy" when:
# - inference-api: Model loaded and API responding
# - stream-manager: Connected to at least one camera
# - alert-router: Connected to Redis
```

### Security Considerations

1. **RTSP Credentials**: Use environment variables, not hardcoded in `cameras.yaml`
2. **Network Isolation**: Use Docker networks to isolate services
3. **API Authentication**: Add authentication middleware for production API
4. **Secrets Management**: Use Docker secrets or external vault

### Updating Models in Production

To deploy a new model version:

```bash
# 1. Register new model in registry
python -c "
from agave_vision.models.registry import ModelRegistry
registry = ModelRegistry('models')
registry.tag_model('v2_optimized', tags=['production'])
"

# 2. Update service configuration
vim configs/services.yaml
# Change model_path to: models/v2_optimized/weights/best.pt

# 3. Restart services
cd production
docker compose restart inference-api stream-manager

# 4. Verify new model loaded
curl http://localhost:8000/health
docker compose logs inference-api | grep "Model loaded"
```

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA drivers
nvidia-smi

# Check Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### RTSP Connection Issues

```bash
# Test RTSP URL with ffplay
ffplay rtsp://admin:password@192.168.1.10:554/stream

# Check Docker network connectivity
docker compose exec stream-manager ping <camera-ip>
```

### Service Won't Start

```bash
# Check logs
docker compose logs inference-api
docker compose logs stream-manager
docker compose logs alert-router

# Check model file exists
ls -lh models/yolov8n_pina/exp/weights/best.pt

# Verify configurations
python -c "from agave_vision.config.loader import ConfigLoader; ConfigLoader().load_all()"
```

## Performance Tuning

- **Frame sampling**: Adjust `fps_target` in camera config (5-10 FPS recommended)
- **Inference batch size**: Set in `configs/services.yaml` (use 1 for lowest latency)
- **GPU memory**: Use FP16 inference for 2x speedup (`model.half()`)
- **Alert debouncing**: Adjust `debounce_window_seconds` to reduce spam

## License

MIT

## Support

For issues and questions, see `docs/` directory or raise an issue in the repository.

## Acknowledgments

- Built with [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- Powered by [FastAPI](https://fastapi.tiangolo.com/)
- Deployed with [Docker](https://www.docker.com/)
