# agave-vision-api / src/agave_vision/config

## Overview
- Configuration loader and Pydantic models for YAML validation.
- Defines schemas for cameras, ROIs, inference settings, and alerting.
- Applies environment variable overrides for runtime settings.
- Primary users: services and developers integrating config.

## Quickstart
### Prerequisites
- YAML files in `configs/`.

### Install
- `pip install -e .`.

### Configure
- Use `ConfigLoader` to load YAML and apply env overrides in `src/agave_vision/config/loader.py`.

### Run (development)
- `python -c "from agave_vision.config.loader import ConfigLoader; print(ConfigLoader().load_services())"`.

### Run (production-like)
- Services load configs on startup.

### Common commands
- `python -c "from agave_vision.config.model_config import get_default_model_path; print(get_default_model_path())"`.

### Troubleshooting
1. Missing config files -> ensure `configs/` exists.
2. Validation errors -> fix YAML schema.
3. Wrong RTSP URLs -> verify `CAMERA_{id}_RTSP_URL` overrides.
4. Wrong device -> set `INFERENCE_DEVICE`.
5. Alerting config missing -> `configs/alerting.yaml` not present (see gaps).

## Architecture
### High-level diagram
```
configs/*.yaml + env vars -> ConfigLoader -> Pydantic config models
```

### Key concepts
- `ConfigLoader`: loads YAML and applies env overrides.
- `CamerasConfig`, `ROIsConfig`, `ServicesConfig`: Pydantic schemas.
- `ModelConfig`: reads default model metadata from `configs/model.yaml`.

### Runtime flow
- Services instantiate `ConfigLoader` with `configs/`.
- Loader reads YAML and merges environment overrides.
- Validated config objects are passed to services.

### Data flow
- Inputs: YAML files and environment variables.
- Transformations: YAML -> dict -> Pydantic models.
- Outputs: typed config objects.

### Component map
| Component | Location (path) | Responsibility | Inputs | Outputs | Dependencies | Failure modes or notes |
| --- | --- | --- | --- | --- | --- | --- |
| Config loader | `src/agave_vision/config/loader.py` | Load YAML + env overrides | configs, env | config objects | pyyaml, pydantic | Missing file or invalid schema |
| Pydantic models | `src/agave_vision/config/models.py` | Validate schemas | dicts | typed config | pydantic | Validation errors |
| Model config | `src/agave_vision/config/model_config.py` | Default model metadata | `configs/model.yaml` | model paths | pyyaml | Missing file uses defaults |

## Entry points
- **CLI commands or scripts:** none.
- **APIs or routes:** none.
- **Workers, schedulers, or jobs:** none.
- **Notebooks:** none.

## Configuration and secrets
- **Configuration sources:** YAML in `configs/` plus env overrides.
- **Precedence:** env overrides > YAML > defaults.
- **Secrets handling patterns:** store credentials in env and inject at runtime.
- **Safe defaults and what must never be committed:** do not commit RTSP/webhook/Hikvision credentials.

## Observability
- **Logging strategy:** `model_config.py` logs model config loading.
- **Metrics or tracing:** none.
- **Debug workflow:** print loaded configs via `ConfigLoader`.

## Testing
- **How to run tests:** `pytest`.
- **Summary of the test pyramid:** no tests in repo.
- **Gaps and recommended minimal regression coverage:** add tests validating YAML -> Pydantic conversion.

## Deployment (if applicable)
- **Build artifacts:** none.
- **Deployment targets:** runtime services.
- **CI/CD overview:** none found.
- **Rollback strategy:** revert config changes.

## Contributing
- **Repository conventions:** keep model and YAML schemas in sync.
- **Branching and release practices:** not defined.
- **How to safely add a new feature and where to start:** extend `models.py`, then update loader and YAML.

## Known gaps and assumptions
- `ConfigLoader.load_alerting()` expects `configs/alerting.yaml`, which is missing.
