# agave-vision-api / src/agave_vision/utils

## Overview
- Utility modules for logging and video I/O.
- Provides JSON logging helpers and OpenCV wrappers.
- Primary users: services and developers using shared helpers.

## Quickstart
### Prerequisites
- OpenCV for video utilities.

### Install
- `pip install -e .`.

### Configure
- Set `LOG_LEVEL` for services using `setup_logging`.

### Run (development)
- Import utilities directly.

### Run (production-like)
- Used by services at runtime.

### Common commands
- `python -c "from agave_vision.utils.logging import setup_logging; setup_logging('test').info('ok')"`.

### Troubleshooting
1. No logs -> ensure logger configured.
2. JSON format not desired -> use text format.
3. Video read failures -> verify source URL/path.
4. FPS zero -> stream metadata missing.
5. Codec issues -> install proper OpenCV build.

## Architecture
### High-level diagram
```
setup_logging -> JSON logs
VideoCapture/VideoWriter -> OpenCV I/O
```

### Key concepts
- `setup_logging`: JSON or text logging for services.
- `VideoCapture`: wrapper with reconnection logic.
- `VideoWriter`: convenience wrapper for writing video.
- `get_video_info`: metadata utility.

### Runtime flow
- Services call `setup_logging` at startup.
- Video utilities wrap OpenCV for read/write.

### Data flow
- Inputs: log messages, video streams.
- Transformations: formatting to JSON, frame encoding/decoding.
- Outputs: stdout logs, video files.

### Component map
| Component | Location (path) | Responsibility | Inputs | Outputs | Dependencies | Failure modes or notes |
| --- | --- | --- | --- | --- | --- | --- |
| Logging | `src/agave_vision/utils/logging.py` | Structured logging | log records | JSON/text logs | logging | None |
| Video I/O | `src/agave_vision/utils/video.py` | Video capture/writer utilities | streams/files | frames/files | opencv | Stream not open |

## Entry points
- **CLI commands or scripts:** none.
- **APIs or routes:** none.
- **Workers, schedulers, or jobs:** none.
- **Notebooks:** none.

## Configuration and secrets
- **Configuration sources:** env var `LOG_LEVEL`.
- **Precedence:** env override > defaults.
- **Secrets handling patterns:** none.
- **Safe defaults and what must never be committed:** none.

## Observability
- **Logging strategy:** JSON formatted logs by default.
- **Metrics or tracing:** none.
- **Debug workflow:** switch to text logging for readability.

## Testing
- **How to run tests:** `pytest`.
- **Summary of the test pyramid:** no tests in repo.
- **Gaps and recommended minimal regression coverage:** add tests for JSON formatter.

## Deployment (if applicable)
- **Build artifacts:** Python package modules.
- **Deployment targets:** services.
- **CI/CD overview:** none found.
- **Rollback strategy:** revert code changes.

## Contributing
- **Repository conventions:** keep helpers small and dependency-light.
- **Branching and release practices:** not defined.
- **How to safely add a new feature and where to start:** add new utility modules.

## Known gaps and assumptions
- Assumption: OpenCV has access to system codecs for RTSP.
