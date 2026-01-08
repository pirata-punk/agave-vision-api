# agave-vision-api / src/agave_vision/storage

## Overview
- Optional persistence for alerts and detection logs.
- Supports SQLite or JSON for alerts and JSONL rotating logs for detections.
- Primary users: developers needing auditability or debugging.

## Quickstart
### Prerequisites
- Filesystem access for `data/` directory.

### Install
- `pip install -e .`.

### Configure
- Pass `storage_type` and `path` to `AlertStore`.
- Configure `DetectionLogger` path and retention.

### Run (development)
- Enable in `AgaveVisionML` with `enable_alert_storage=True` or `enable_detection_logging=True`.

### Run (production-like)
- Ensure data directories are durable and monitored.

### Common commands
- None.

### Troubleshooting
1. Permission denied -> ensure `data/` is writable.
2. Disk growth -> adjust retention.
3. SQLite locked -> avoid concurrent writers.
4. JSON corruption -> validate file.
5. Missing alerts -> ensure storage enabled.

## Architecture
### High-level diagram
```
AlertEvent/Detection -> AlertStore/DetectionLogger -> SQLite/JSONL files
```

### Key concepts
- `AlertStore`: saves and queries alerts (SQLite or JSON).
- `DetectionLogger`: writes JSONL logs with rotation and retention.
- Retention: automatic cleanup of old detection logs.

### Runtime flow
- Storage classes are initialized on demand.
- Alerts and detections are persisted when enabled.
- Logs can be queried or exported.

### Data flow
- Inputs: alert and detection dictionaries.
- Transformations: serialization to JSON or SQLite schema.
- Outputs: files under `data/`.

### Component map
| Component | Location (path) | Responsibility | Inputs | Outputs | Dependencies | Failure modes or notes |
| --- | --- | --- | --- | --- | --- | --- |
| AlertStore | `src/agave_vision/storage/alert_store.py` | Persist and query alerts | alert dicts | SQLite or JSON | sqlite3 | Disk permissions |
| DetectionLogger | `src/agave_vision/storage/detection_logger.py` | Log detection history | detection dicts | JSONL files | filesystem | Disk growth |

## Entry points
- **CLI commands or scripts:** none.
- **APIs or routes:** none.
- **Workers, schedulers, or jobs:** none.
- **Notebooks:** none.

## Configuration and secrets
- **Configuration sources:** constructor parameters only.
- **Precedence:** constructor args > defaults.
- **Secrets handling patterns:** none.
- **Safe defaults and what must never be committed:** avoid committing large data files.

## Observability
- **Logging strategy:** no internal logging.
- **Metrics or tracing:** none.
- **Debug workflow:** inspect SQLite or JSONL files directly.

## Testing
- **How to run tests:** `pytest`.
- **Summary of the test pyramid:** no tests in repo.
- **Gaps and recommended minimal regression coverage:** tests for save/query paths.

## Deployment (if applicable)
- **Build artifacts:** data files in `data/`.
- **Deployment targets:** local disk or mounted volume.
- **CI/CD overview:** none found.
- **Rollback strategy:** restore from backups.

## Contributing
- **Repository conventions:** keep schema backwards compatible.
- **Branching and release practices:** not defined.
- **How to safely add a new feature and where to start:** extend storage classes with new backends.

## Known gaps and assumptions
- Assumption: single-process writers to SQLite.
