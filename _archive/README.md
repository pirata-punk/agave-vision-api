# Archive

This directory contains deprecated or superseded artifacts preserved for historical reference.

## Purpose

These files were part of earlier project phases but have been replaced by the Phase 2 ML-focused API architecture. They are archived here to preserve project history and provide context for future maintainers.

## Contents

### `production/`
**Archived:** 2025-01-02
**Reason:** Production Docker infrastructure removed - project is now pure ML API
**Replacement:** External teams wrap `AgaveVisionML` in their own server architecture

Contains:
- `docker-compose.yml` - Multi-service Docker composition
- `Dockerfile.alert-router` - Alert routing service
- `Dockerfile.inference-api` - HTTP inference API wrapper
- `Dockerfile.stream-manager` - Stream management service

**Note:** These were server infrastructure components. Phase 2 architecture deliberately excludes HTTP/Docker concerns to focus purely on ML. External teams now import `AgaveVisionML` directly into their own servers.

---

### `docs/`
**Archived:** 2025-01-02
**Reason:** Early development documentation superseded by formal Phase 1 & 2 docs

Contains:
- `plan.md` - Early project planning notes
- `project_notes.md` - Ad-hoc development notes
- `smart_tile_selection.md` - Training pipeline documentation
- `system_architecture.md` - Old system architecture (server-focused)
- `labeling_instructions.md` - Data labeling guidelines

**Replacement:**
- `docs/ml_api_reference.md` - Complete ML API documentation
- `docs/phase1_validation_checklist.md` - Alert system validation
- `docs/phase2_complete.md` - ML-focused architecture overview
- `docs/roi_setup_guide.md` - ROI configuration guide
- `docs/synthetic_object_testing.md` - Alert testing guide
- `docs/ml_enhancements_roadmap.md` - Project roadmap

---

### `examples/`
**Archived:** 2025-01-02
**Reason:** Old example scripts superseded by comprehensive integration example

Contains:
- `demo_video_infer.py` - Basic video inference demo
- `infer_alert.py` - Simple alert detection example
- `realtime_yolo_stream.py` - Real-time RTSP streaming example

**Replacement:**
- `examples/integration_example.py` - Comprehensive integration guide with 5 examples
- `examples/live_demo.py` - Enhanced Phase 1 demo with ROI alerts
- `examples/roi_selector.py` - Interactive ROI configuration tool
- `examples/test_alerts_with_synthetic_objects.py` - Alert testing with CV injection

---

## Recovery Instructions

If you need to reference or recover any archived file:

```bash
# The archive is in git history and preserved in this directory
cd _archive/

# To see the original file locations and git history:
git log --all --full-history -- path/to/file
```

---

## What Changed in Phase 2

**Before (Phase 0-1):**
- Server-focused architecture with Docker services
- HTTP API for inference
- Microservices for streaming and alerts
- Production deployment infrastructure

**After (Phase 2):**
- Pure ML API (`AgaveVisionML` class)
- No HTTP, Docker, or server code in this repository
- Simple Python imports for integration
- External teams wrap ML API in their own infrastructure

**Rationale:**
Focus on ML excellence, let infrastructure experts handle deployment. This separation makes the ML code cleaner, more maintainable, and easier to integrate into any server architecture.

---

## Do Not Reintroduce

❌ **HTTP/REST endpoints** - Keep ML-only
❌ **Docker Compose services** - External team responsibility
❌ **Server deployment code** - Out of scope
❌ **Production infrastructure** - Separation of concerns

✅ **ML API** - `AgaveVisionML` class
✅ **Core ML logic** - Inference, ROI, alerts
✅ **Storage layers** - Alert/log persistence (optional)
✅ **Training pipeline** - Model development tools

---

**Archive Created:** 2025-01-02
**Phase:** Post-Phase 2 Cleanup
**Version:** 2.0.0
