# Project Lineage Map - Agave Vision API

**Branch:** `stage/handoff`
**Date:** 2026-01-06
**Purpose:** Track all components and their usage status for ML API handoff

---

## Executive Summary

This document maps every component in the project to determine what's actively used by the **ML API** versus what's leftover from development/deployment infrastructure.

### Current State
- **ML API Module:** ✅ Active (`src/agave_vision/ml_api.py`)
- **Core ML Components:** ✅ Active (inference, ROI, alerts)
- **Storage Systems:** ✅ Active (optional features)
- **Production Services:** ⚠️ **UNUSED** by ML API (Docker/microservices)
- **Tests:** ⚠️ **EMPTY** directory
- **Development Modules:** ✅ Removed (training, ingestion, model registry)

---

## Component Lineage Analysis

### 1. ML API Core (ACTIVE - KEEP)

**Location:** `src/agave_vision/`

| Component | Status | Used By | Purpose |
|-----------|--------|---------|---------|
| `ml_api.py` | ✅ ACTIVE | External teams | Main ML API interface |
| `__init__.py` | ✅ ACTIVE | Package exports | Package entry point (10 exports) |

**Dependencies:**
- `core/inference.py` - YOLO model wrapper
- `core/roi.py` - ROI filtering
- `storage/` - Optional alert/detection logging

**Verdict:** **KEEP** - This is the primary deliverable

---

### 2. Core ML Components (ACTIVE - KEEP)

**Location:** `src/agave_vision/core/`

| File | Status | Used By | Purpose |
|------|--------|---------|---------|
| `inference.py` | ✅ ACTIVE | `ml_api.py`, `services/*` | YOLO inference wrapper |
| `roi.py` | ✅ ACTIVE | `ml_api.py`, `services/*` | ROI polygon filtering |
| `alerts.py` | ✅ ACTIVE | `ml_api.py`, `services/*` | Alert event structures |
| `frames.py` | ✅ ACTIVE | Core utilities | Frame processing utilities |

**Import Chain:**
```
ml_api.py
  ├─> core/inference.py (YOLOInference, Detection)
  └─> core/roi.py (ROIManager)
      └─> core/alerts.py (AlertEvent)
```

**Verdict:** **KEEP** - Essential for ML API functionality

---

### 3. Storage Systems (ACTIVE - KEEP)

**Location:** `src/agave_vision/storage/`

| File | Status | Used By | Purpose |
|------|--------|---------|---------|
| `alert_store.py` | ✅ ACTIVE | `ml_api.py` (optional) | Persistent alert storage (SQLite/JSON) |
| `detection_logger.py` | ✅ ACTIVE | `ml_api.py` (optional) | Detection history logging (JSONL) |

**Usage Pattern:**
```python
# Optional feature - enabled via flags
ml = AgaveVisionML(
    model_path="models/best.pt",
    enable_alert_storage=True,      # Uses alert_store.py
    enable_detection_logging=True   # Uses detection_logger.py
)
```

**Verdict:** **KEEP** - Optional ML API features, minimal footprint (60KB)

---

### 4. Configuration System (ACTIVE - KEEP)

**Location:** `src/agave_vision/config/`

| File | Status | Used By | Purpose |
|------|--------|---------|---------|
| `models.py` | ✅ ACTIVE | ML API, Services | Pydantic config models |
| `loader.py` | ✅ ACTIVE | ML API, Services | YAML config loader |

**Loaded Configs:**
- `configs/rois.yaml` - ROI definitions for cameras
- `configs/cameras.yaml` - Camera registry (used by services)

**Verdict:** **KEEP** - Required for ROI configuration

---

### 5. Production Services (UNUSED BY ML API - CONSIDER REMOVAL)

**Location:** `src/agave_vision/services/`

#### Service Breakdown

**Inference API** (`services/inference_api/`)
- **Purpose:** FastAPI REST server for inference
- **Used by:** External HTTP clients (NOT the ML API itself)
- **Components:**
  - `app.py` - FastAPI application
  - `routes.py` - HTTP endpoints
  - `dependencies.py` - Dependency injection
  - `schemas.py` - Pydantic request/response models

**Stream Manager** (`services/stream_manager/`)
- **Purpose:** RTSP camera stream processing
- **Used by:** Production deployment (NOT the ML API)
- **Components:**
  - `main.py` - Stream manager entry point
  - `camera.py` - Camera handler
  - `publisher.py` - Redis publisher

**Alert Router** (`services/alert_router/`)
- **Purpose:** Alert delivery system
- **Used by:** Production deployment (NOT the ML API)
- **Components:**
  - `main.py` - Alert router entry point
  - `consumer.py` - Redis consumer
  - `debounce.py` - Alert debouncing
  - `protocols/` - Alert delivery protocols (stdout, webhook, Hikvision)

#### Dependency Analysis

```
ML API (ml_api.py)
  ├─> core/* ✅
  ├─> storage/* ✅
  └─> config/* ✅

Services (services/*)
  ├─> core/* (same as ML API)
  ├─> config/* (same as ML API)
  └─> External: FastAPI, Redis, httpx

❌ ML API does NOT import from services/
✅ Services import from core/ (shared components)
```

**Verdict:** **SERVICES ARE INDEPENDENT**

The `services/` directory is a **deployment option**, not a requirement for the ML API. These are microservices that **use** the core ML components but are not **used by** the ML API.

#### Decision Options

**Option A: REMOVE services/** (Pure ML API handoff)
- Project becomes pure Python library
- External teams integrate directly via `AgaveVisionML`
- Smallest footprint (80KB removed)

**Option B: KEEP services/** (Include deployment option)
- Provides reference FastAPI implementation
- Teams can deploy as-is or customize
- Requires Docker/Redis dependencies

**Recommendation:** **REMOVE** for pure ML API handoff (you have copy in another branch)

---

### 6. Utilities (ACTIVE - KEEP)

**Location:** `src/agave_vision/utils/`

| File | Status | Used By | Purpose |
|------|--------|---------|---------|
| `logging.py` | ✅ ACTIVE | All modules | Logging configuration |
| `video.py` | ✅ ACTIVE | Core, Services | Video utilities |

**Verdict:** **KEEP** - Shared utilities (24KB)

---

### 7. Tests Directory (EMPTY - REMOVE)

**Location:** `tests/`

**Status:** ⚠️ Empty directory (0 bytes)

**Analysis:**
- No test files present
- Directory exists but contains nothing
- Tests were likely removed in previous cleanup

**Verdict:** **REMOVE** - Empty directory serves no purpose

---

### 8. Configuration Files (ACTIVE - KEEP)

**Location:** `configs/`

| File | Status | Used By | Purpose |
|------|--------|---------|---------|
| `rois.yaml` | ✅ ACTIVE | ML API | ROI definitions for cameras |
| `cameras.yaml` | ⚠️ SERVICES ONLY | Services | Camera RTSP URLs |

**rois.yaml:**
```yaml
cameras:
  - camera_id: cam_nave3_hornos
    forbidden_rois:
      - name: loading_zone
        points: [[x,y], ...]
    allowed_classes: [pine, worker]
```

**cameras.yaml:**
```yaml
cameras:
  - id: cam_nave3_hornos
    rtsp_url: "rtsp://..."
```

**Verdict:**
- `rois.yaml` - **KEEP** (used by ML API)
- `cameras.yaml` - **REMOVE** if removing services/

---

### 9. Model Artifacts (ACTIVE - KEEP)

**Location:** `models/`

| File | Status | Size | Purpose |
|------|--------|------|---------|
| `best.pt` | ✅ ACTIVE | 6.3MB | Trained YOLOv8 model weights |

**Usage:**
```python
ml = AgaveVisionML(model_path="models/best.pt")
```

**Verdict:** **KEEP** - Essential model file

---

### 10. Documentation (UPDATE NEEDED)

**Location:** Root and `docs/` (if exists)

| File | Status | Accuracy | Action |
|------|--------|----------|--------|
| `README.md` | ⚠️ OUTDATED | References removed modules | UPDATE |
| `requirements.txt` | ❓ UNKNOWN | May include unused deps | REVIEW |
| `pyproject.toml` | ❓ UNKNOWN | May include unused deps | REVIEW |

**README Issues:**
- References ingestion module (removed)
- References training module (removed)
- References model registry (removed)
- References production services (consider removing)
- References data/ directory (removed)

**Verdict:** **UPDATE** - Documentation needs to match current ML API-only state

---

## Removed Modules (Previous Cleanup)

### Already Deleted ✅

| Module | Removed Date | Reason |
|--------|--------------|--------|
| `src/agave_vision/ingestion/` | 2026-01-06 | Data pipeline (development only) |
| `src/agave_vision/training/` | 2026-01-06 | Training workbench (development only) |
| `src/agave_vision/models/` | 2026-01-06 | Model registry (unused) |
| `models/yolov8n_pina/exp/` | 2026-01-06 | Training artifacts (7.9GB) |
| `training/runs/` | 2026-01-06 | Training output directory |
| `data/` | Before handoff | Videos, frames, tiles (development) |

---

## Dependency Graph

```
┌─────────────────────────────────────────────┐
│         EXTERNAL INTEGRATION                │
│                                             │
│   from agave_vision.ml_api import          │
│   AgaveVisionML                             │
│                                             │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│         ML API (ml_api.py)                  │
│                                             │
│  - predict_frame()                          │
│  - predict_video_stream()                   │
│  - get_alerts()                             │
│  - get_model_info()                         │
│                                             │
└──────────────────┬──────────────────────────┘
                   │
         ┌─────────┴─────────┬─────────────┐
         │                   │             │
         ▼                   ▼             ▼
┌─────────────┐    ┌──────────────┐  ┌──────────┐
│   core/     │    │  storage/    │  │ config/  │
│             │    │  (optional)  │  │          │
│ - inference │    │              │  │ - loader │
│ - roi       │    │ - alert_store│  │ - models │
│ - alerts    │    │ - det_logger │  │          │
│ - frames    │    │              │  │          │
└─────────────┘    └──────────────┘  └──────────┘
         │                   │             │
         └─────────┬─────────┴─────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│         utils/                              │
│  - logging                                  │
│  - video                                    │
└─────────────────────────────────────────────┘

┌─────────────────────────────────────────────┐
│    SEPARATE: services/ (microservices)      │
│                                             │
│  Uses core/*, config/* but NOT used by      │
│  ML API. Independent deployment option.     │
│                                             │
│  - inference_api/ (FastAPI)                 │
│  - stream_manager/ (RTSP)                   │
│  - alert_router/ (Redis)                    │
└─────────────────────────────────────────────┘
```

---

## Import Analysis

### ML API Imports (src/agave_vision/ml_api.py)

```python
# External dependencies
import cv2
import numpy as np

# Internal dependencies (agave_vision)
from agave_vision.core.inference import YOLOInference, Detection
from agave_vision.core.roi import ROIManager

# Optional dependencies (when enabled)
# from agave_vision.storage.alert_store import AlertStore
# from agave_vision.storage.detection_logger import DetectionLogger
```

**Result:** ML API does NOT import from `services/`

### Services Import Pattern

All service modules import FROM core components:

```python
# services/inference_api/routes.py
from agave_vision.core.inference import YOLOInference
from agave_vision.core.roi import ROIManager

# services/stream_manager/main.py
from agave_vision.core.inference import YOLOInference
from agave_vision.core.roi import ROIManager

# services/alert_router/consumer.py
from agave_vision.core.alerts import AlertEvent
```

**Result:** Services USE core components, but ML API doesn't USE services

---

## Recommendations

### For Pure ML API Handoff

**KEEP (Essential):**
- ✅ `src/agave_vision/ml_api.py` - Main API
- ✅ `src/agave_vision/core/` - ML engine (84KB)
- ✅ `src/agave_vision/storage/` - Optional features (60KB)
- ✅ `src/agave_vision/config/` - Configuration (16KB)
- ✅ `src/agave_vision/utils/` - Utilities (24KB)
- ✅ `models/best.pt` - Trained model (6.3MB)
- ✅ `configs/rois.yaml` - ROI definitions

**REMOVE (Unused):**
- ❌ `src/agave_vision/services/` - Microservices (80KB) - NOT used by ML API
- ❌ `tests/` - Empty directory
- ❌ `configs/cameras.yaml` - Only used by services

**UPDATE:**
- 📝 `README.md` - Remove references to deleted modules
- 📝 `requirements.txt` - Remove service dependencies (FastAPI, Redis, httpx)
- 📝 `pyproject.toml` - Update dependencies

### For ML API + Deployment Reference

If you want to include deployment examples:

**KEEP:**
- Everything above PLUS
- ✅ `src/agave_vision/services/` - Reference FastAPI implementation
- ✅ `configs/cameras.yaml` - Camera configuration example
- 📝 Add `DEPLOYMENT.md` explaining services are optional

---

## File Inventory

### Current State (Post-Cleanup)

```
agave-vision-api/
├── models/
│   └── best.pt                    # 6.3MB ✅ KEEP
│
├── src/agave_vision/
│   ├── __init__.py                # ✅ KEEP
│   ├── ml_api.py                  # ✅ KEEP (main API)
│   │
│   ├── core/                      # ✅ KEEP (84KB)
│   │   ├── __init__.py
│   │   ├── inference.py
│   │   ├── roi.py
│   │   ├── alerts.py
│   │   └── frames.py
│   │
│   ├── storage/                   # ✅ KEEP (60KB)
│   │   ├── __init__.py
│   │   ├── alert_store.py
│   │   └── detection_logger.py
│   │
│   ├── config/                    # ✅ KEEP (16KB)
│   │   ├── __init__.py
│   │   ├── models.py
│   │   └── loader.py
│   │
│   ├── services/                  # ⚠️ REMOVE? (80KB)
│   │   ├── __init__.py
│   │   ├── inference_api/         # FastAPI server
│   │   ├── stream_manager/        # RTSP processor
│   │   └── alert_router/          # Alert delivery
│   │
│   └── utils/                     # ✅ KEEP (24KB)
│       ├── __init__.py
│       ├── logging.py
│       └── video.py
│
├── configs/
│   ├── rois.yaml                  # ✅ KEEP (used by ML API)
│   └── cameras.yaml               # ⚠️ REMOVE? (only for services)
│
├── tests/                         # ❌ REMOVE (empty)
│
├── README.md                      # 📝 UPDATE
├── requirements.txt               # 📝 REVIEW
└── pyproject.toml                 # 📝 REVIEW
```

### Size Analysis

| Component | Size | Status |
|-----------|------|--------|
| `models/best.pt` | 6.3MB | Essential |
| `src/agave_vision/core/` | 84KB | Essential |
| `src/agave_vision/services/` | 80KB | **Optional** |
| `src/agave_vision/storage/` | 60KB | Essential |
| `src/agave_vision/config/` | 16KB | Essential |
| `src/agave_vision/utils/` | 24KB | Essential |
| **Total (without services)** | **~6.5MB** | Minimal |
| **Total (with services)** | **~6.6MB** | +Deployment |

---

## Action Items

### Immediate (for pure ML API handoff)

1. ❌ **Remove** `src/agave_vision/services/` directory
2. ❌ **Remove** `tests/` empty directory
3. ❌ **Remove** `configs/cameras.yaml` (services only)
4. 📝 **Update** `README.md` to reflect ML API-only focus
5. 📝 **Review** `requirements.txt` for unused dependencies
6. 📝 **Review** `pyproject.toml` for unused dependencies

### Optional (for reference implementation)

1. ✅ **Keep** `src/agave_vision/services/` as deployment example
2. 📝 **Add** `DEPLOYMENT.md` explaining services are optional
3. 📝 **Add** Docker examples if services are kept

---

## Conclusion

**Current State:**
- ML API: ✅ Clean and functional
- Core Dependencies: ✅ Minimal and essential
- Services: ⚠️ Independent microservices (not used by ML API)
- Tests: ❌ Empty directory

**Recommended Action:**
Remove `services/` for pure ML API handoff (80KB savings, reduced complexity)

**Final Structure:**
```
ML API (ml_api.py)
  └─> Core (inference, ROI, alerts)
      └─> Storage (optional)
          └─> Config + Utils
```

**Final Size:** ~6.5MB (model + code)

This is the **minimum viable ML API** for handoff, with all development artifacts removed.
