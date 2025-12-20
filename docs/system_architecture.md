# Agave Vision System Architecture (Draft)

High-level blueprint for serving the YOLOv8 pina/worker/object detector in production with live video ingestion and alert signaling.

## Goals
- Ingest live video from Hikvision (or RTSP) cameras.
- Run real-time detection on frames, focusing on forbidden-object alerts within ROIs.
- Emit alerts/signals consumable by downstream systems (Hikvision protocol or adapter).
- Support model lifecycle: versioning, deployment, monitoring, and retraining.

## Core Components

### 1) Inference Service
- **Tech**: Python 3, Ultralytics YOLOv8, ONNX/TensorRT (optional), FastAPI/gRPC for control APIs.
- **Container**: Docker image with CUDA (if GPU) or optimized CPU build (OpenVINO/onnxruntime).
- **Model Store**: `models/` directory or object storage (S3/minio) with versioned `best.pt` / ONNX.
- **Runtime**:
  - Load model once per worker process.
  - Expose endpoints:
    - `/health` (liveness/readiness)
    - `/infer` (optional sync API for single images)
    - `/config/rois` (CRUD for per-camera ROI polygons)
  - Supports batch/frame inference loop for live streams (see Video Ingest).

### 2) Video Ingest / Stream Manager
- **Tech**: Python/Go service or FFmpeg-based workers; use OpenCV/FFmpeg for RTSP pulling from Hikvision.
- **Function**:
  - Maintain persistent RTSP connections to cameras (reconnect logic).
  - Sample frames at target FPS (e.g., 5–10 FPS) for inference; drop if backlog grows.
  - Push frames to Inference Service via in-process call (co-located) or gRPC/REST if remote.
  - Optional: run inference inline (same process) when camera count is small; otherwise separate microservice.
- **ROI Management**:
  - Fetch per-camera ROI config from a config service or static YAML; cache in memory.
  - Apply ROI filtering post-inference.

### 3) Alert Router / Signal Adapter
- **Tech**: Python/Node/Go microservice; message bus (Kafka/Redis Streams) for decoupling.
- **Function**:
  - Consume detection/alert events from the Inference Service.
  - Transform into target protocol for Hikvision or intermediary (e.g., HTTP callback, MQTT, TCP socket).
  - Enforce debounce/rate limiting to avoid alert storms.
  - Persist alerts to a database (PostgreSQL) for audit.

### 4) Config & Metadata Store
- **Tech**: PostgreSQL (ROIs, camera registry, model versions), Redis for cache.
- **Data**:
  - Cameras (id, RTSP URL, status).
  - ROI polygons per camera, allowed classes.
  - Model registry (version, path, metrics).
  - Alert delivery settings/thresholds.

### 5) Monitoring & Ops
- **Metrics**: Prometheus (export FPS, latency, queue depth, alert counts, per-class confidence stats).
- **Logs**: Structured JSON to ELK/CloudWatch.
- **Tracing**: Optional OpenTelemetry for multi-service traces.
- **Dashboard**: Grafana for real-time stream health and alert rates.

## Data/Model Flow
1. Camera RTSP → Stream Manager pulls frames.
2. Frames → Inference (YOLOv8) → detections.
3. ROI filter → allow `pine`/`worker`, alert on `object` inside forbidden ROI.
4. Alerts → Alert Router → Hikvision protocol adapter / downstream consumer.
5. Storage (optional): sampled frames and detections to object storage / DB for QA.

## Deployment Options
- **Single-node GPU**: Docker Compose with services (stream manager + inference + alert router + Prometheus + Grafana + Postgres).
- **Cluster**: Kubernetes with:
  - Deployments for inference (GPU nodes, HPA on CPU/GPU utilization).
  - Stateful set for Postgres; Redis cache.
  - Daemonset or separate deployment for stream manager (node affinity for camera locality).
  - ConfigMaps/Secrets for ROIs and credentials.
- **Model artifacts**: stored in S3/minio and pulled on startup; bake default `best.pt` into image for fallback.

## Performance Considerations
- Frame sampling: tune to 5–10 FPS per camera to balance load vs. responsiveness.
- Batch size: small batches (1–4) to minimize latency; use `half` precision on GPU.
- Warmup: run a few dummy inferences on startup.
- Backpressure: drop frames if inference queue grows; log/sketch alert when dropping too many.
- ROI filtering: apply after NMS to minimize overhead.

## MLOps Practices
- **Versioning**: keep trained weights under `models/yolov8n_pina/<version>/best.pt` with `results.csv`.
- **Repro**: store `configs/yolo_data.yaml`, training command, and git commit hash per model version.
- **Data lineage**: maintain `data/tiles_yolo/metadata.json` with splits and source rounds.
- **Evaluation**: run `yolo val` on held-out set for each model; record metrics in registry.
- **Canary**: support side-by-side inference of new model on a subset of streams, compare alert stats before rollout.
- **CI checks**: lint/test scripts; optional minimal smoke test for inference load.

## Real-Time Demo Hooks
- `scripts/realtime_yolo_stream.py`: reads webcam/RTSP/file, overlays detections, optional ROI alerts, JSON alerts to stdout.
- `scripts/demo_video_infer.py`: file-to-file annotated video for offline checks.

## Alert Protocol Adapter (Hikvision TBD)
- Define an adapter interface: accepts alert payload `{camera_id, ts, bbox, class, confidence, roi_hit}` and translates to:
  - If Hikvision supports HTTP/MQTT events: send JSON payload.
  - If proprietary: build a thin TCP/SDK shim; keep this isolated in its own service.
- Ensure idempotency and rate limits; queue via Kafka/Redis Streams to decouple from camera load.

## Security & Secrets
- Store RTSP creds and model store keys in secrets (K8s Secrets / Vault).
- TLS for inter-service comms when off-box; restrict ingress to control/API.

## Next Steps
- Validate RTSP connectivity and ROI definitions per camera.
- Decide deployment target (single GPU host vs. K8s).
- Implement alert adapter once Hikvision protocol is confirmed.
- Add Prometheus exporters to inference/stream services and a Grafana dashboard.

## ASCII Architecture & Data Flow

```
                       ┌─────────────────────────────┐
                       │         Model Store          │
                       │   (S3/minio, versioned PT)   │
                       └──────────────┬───────────────┘
                                      │ (pull on boot)
                                      ▼
┌─────────────────────┐     ┌─────────────────────────┐        ┌────────────────────┐
│  Cameras (RTSP/Hik) │────▶│ Stream Manager          │────────▶│ Inference Service │
│ (Hikvision RTSP)    │     │ - RTSP pull/reconnect   │frames   │ (YOLOv8)          │
└─────────────────────┘     │ - FPS sampling/backoff  │        └────────────────────┘
                             │ - Per-camera ROI cache  │               │detections
                             └──────────┬──────────────┘               │
                                        │ alerts/detections            │
                                        ▼                              ▼
                             ┌────────────────────┐         ┌────────────────────┐
                             │  Alert Router      │────────▶│  Protocol Adapter  │
                             │ (Kafka/Redis bus)  │ alerts  │ (Hikvision TBD)    │
                             └──────────┬─────────┘         └────────────────────┘
                                        │
                                        ▼
                              ┌──────────────────┐
                              │ Alert Sink / DB  │
                              │ (Postgres/HTTP)  │
                              └──────────────────┘

Observability:
  - Metrics/logs from Stream Manager, Inference, Alert Router → Prometheus/ELK

Config/Metadata:
  - Postgres/Redis: camera registry, ROIs, model versions, thresholds

Local Persistence (optional/minimal):
  - Target a streaming-first footprint; do not persist frames/videos in normal ops.
  - Optional short-lived caches for troubleshooting:
      /tmp/frames_cache/ (bounded, auto-pruned) — sampled frames for QA
      /tmp/alerts_snapshots/ (bounded) — alert snapshots for audit
  - Models pulled from object store into memory or a small on-disk cache (e.g., /opt/models/)
  - Configs (ROIs, camera registry) from DB/ConfigMap; keep read-only copies in memory.

Runtime Flow:
  1) Stream Manager pulls RTSP, samples frames (no disk writes).
  2) Inference Service runs YOLO, filters by ROI (allow pine/worker, alert on object).
  3) Alert Router debounces/queues alerts.
  4) Protocol Adapter emits alerts to Hikvision/consumer.
  5) Metrics/logs collected for monitoring; artifacts/versioning tracked in model store/DB (not local disk).
```
