# Production Services

Docker-based microservices for real-time RTSP streaming, inference, and ROI-based alerting.

## Services

### 1. Inference API (FastAPI)
REST API for running inference on uploaded images.

**Port:** 8000

**Endpoints:**
- `GET /health` - Health check
- `POST /infer` - Run inference on image
- `GET /config/rois` - Get ROI configurations
- `GET /config/cameras` - Get camera configurations

### 2. Stream Manager
RTSP stream ingestion and frame processing service.

**Features:**
- Multi-camera RTSP stream handling
- Frame sampling at configurable FPS
- Inference orchestration
- Alert emission to Redis

### 3. Alert Router
Redis consumer that routes alerts to configured protocols.

**Supported Protocols:**
- `stdout` - Print alerts to console (default)
- `webhook` - HTTP webhook delivery
- `hikvision` - Hikvision NVR integration (planned)

### 4. Redis
Message bus for alert delivery between services.

## Deployment

### Quick Start

```bash
# From production directory
cd production

# Build images
docker compose build

# Start services
docker compose up -d

# Check health
curl http://localhost:8000/health

# View logs
docker compose logs -f
```

### Configuration

**Environment Variables** (`.env`):
```bash
REDIS_URL=redis://redis:6379
INFERENCE_DEVICE=cuda
INFERENCE_MODEL_PATH=/app/models/v1_baseline/weights/best.pt
INFERENCE_CONFIDENCE=0.25
ALERTING_PROTOCOL=stdout
```

**Camera Configuration** (`../configs/cameras.yaml`):
```yaml
cameras:
  - id: cam_nave3_hornos
    name: "Nave 3 Hornos B CAM 3"
    rtsp_url: "rtsp://admin:password@192.168.1.10:554/Streaming/Channels/101"
    enabled: true
    fps_target: 5.0
```

**ROI Configuration** (`../configs/rois.yaml`):
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

### Service Management

```bash
# Stop all services
docker compose down

# Restart a service
docker compose restart stream-manager

# View service logs
docker compose logs -f inference-api

# Check resource usage
docker stats

# Update after config change
docker compose up -d --force-recreate
```

## Demo Scripts

The production module includes standalone demo scripts for testing and development:

### `demo_video_infer.py`
Run inference on a local video file and save annotated output.

```bash
python production/demo_video_infer.py
```

**What it does:**
- Loads trained YOLOv8 model
- Processes video frame by frame
- Draws bounding boxes on detections
- Saves annotated video as MP4

### `infer_alert.py`
Lightweight inference with ROI-based alerting.

```bash
python production/infer_alert.py
```

**What it does:**
- Loads YOLOv8 model
- Ingests frames from camera or video
- Checks detections against forbidden ROIs
- Raises alerts when `object` class detected in ROI
- Ignores `pine` and `worker` classes

### `realtime_yolo_stream.py`
Real-time inference on live RTSP stream.

```bash
python production/realtime_yolo_stream.py
```

**What it does:**
- Connects to RTSP stream or webcam
- Runs inference in real-time
- Displays annotated video with OpenCV
- Logs ROI alerts to stdout

**Note:** These demo scripts are for testing and development. For production deployment, use the Docker Compose services.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Docker Compose Network                        │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │ Inference    │    │ Stream       │    │ Alert        │      │
│  │ API          │    │ Manager      │    │ Router       │      │
│  │ (FastAPI)    │    │ (RTSP        │    │ (Redis       │      │
│  │              │    │  Ingestion)  │    │  Consumer)   │      │
│  │ Port: 8000   │    │              │    │              │      │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘      │
│         │                   │                    │              │
│         └───────────────────┼────────────────────┘              │
│                             │                                   │
│                      ┌──────▼───────┐                           │
│                      │ Redis        │                           │
│                      │ (Message Bus)│                           │
│                      └──────────────┘                           │
│                                                                  │
│  Shared Volumes: ../models/, ../configs/                        │
│  GPU: Shared across inference-api and stream-manager            │
└─────────────────────────────────────────────────────────────────┘
```

## Health Checks

All services include health checks that run every 30 seconds:

- **inference-api**: Model loaded and API responding
- **stream-manager**: Connected to at least one camera
- **alert-router**: Connected to Redis and consuming messages
- **redis**: Accepting connections

Check service health:
```bash
docker compose ps
```

Services show as `healthy` when all checks pass.

## Troubleshooting

### GPU Not Detected

```bash
# Check NVIDIA drivers
nvidia-smi

# Test Docker GPU support
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

### RTSP Connection Issues

```bash
# Test RTSP URL with ffplay
ffplay rtsp://admin:password@192.168.1.10:554/stream

# Check network connectivity from container
docker compose exec stream-manager ping <camera-ip>
```

### Service Won't Start

```bash
# Check logs for errors
docker compose logs inference-api
docker compose logs stream-manager
docker compose logs alert-router

# Verify model exists
ls -lh ../models/v1_baseline/weights/best.pt

# Verify configurations
cat ../configs/cameras.yaml
cat ../configs/rois.yaml
```

### High Memory Usage

```bash
# Check resource usage
docker stats

# Limit memory per service (docker-compose.yml):
services:
  stream-manager:
    deploy:
      resources:
        limits:
          memory: 2G
```

## Performance Tuning

- **Frame sampling**: Lower `fps_target` (3-5 FPS) reduces CPU load
- **Batch size**: Use `batch_size: 1` for lowest latency
- **GPU memory**: Enable FP16 for 2x speedup with half memory
- **Alert debouncing**: Increase `debounce_window_seconds` to reduce spam

## Security

**Production Checklist:**

1. ✅ Use environment variables for RTSP credentials
2. ✅ Enable Docker network isolation
3. ✅ Add API authentication middleware
4. ✅ Use secrets management (Docker secrets, Vault)
5. ✅ Restrict Redis access to internal network
6. ✅ Enable HTTPS for API endpoints
7. ✅ Regularly update base images for security patches

## Scaling

For multi-camera deployments:

**Single Node (Current):**
- Up to 8-10 cameras per GPU (depends on model size and FPS)
- Redis handles ~10k messages/sec

**Multi-Node (Future):**
- Deploy stream-manager per GPU node
- Centralized Redis cluster
- Load-balanced inference-api instances

## Monitoring

Add Prometheus metrics and Grafana dashboards:

```yaml
# docker-compose.yml additions
services:
  prometheus:
    image: prom/prometheus
    # ... configuration

  grafana:
    image: grafana/grafana
    # ... configuration
```

**Key Metrics to Monitor:**
- Inference latency (ms)
- Frame processing rate (FPS)
- GPU utilization (%)
- Alert rate (alerts/min)
- Service uptime

## Updating Models

To deploy a new model version:

1. Register model in model registry
2. Update `configs/services.yaml` with new model path
3. Restart services: `docker compose restart inference-api stream-manager`
4. Verify: `curl http://localhost:8000/health`

See [models/README.md](../models/README.md) for model registry details.
