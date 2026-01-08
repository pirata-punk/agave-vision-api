# agave-vision-api / src/agave_vision/services/alert_router/protocols

## Overview
- Protocol adapters for delivering alerts downstream.
- Includes stdout and webhook adapters; Hikvision adapter is a placeholder.
- Primary users: developers integrating alert delivery mechanisms.

## Quickstart
### Prerequisites
- None beyond optional adapter dependencies.

### Install
- `pip install -e .[alerts]` for webhook support.

### Configure
- Set `ALERTING_PROTOCOL` and adapter settings in `configs/services.yaml` or env vars.

### Run (development)
- Run alert router and observe adapter output.

### Run (production-like)
- Ensure outbound network access for webhooks.

### Common commands
- None.

### Troubleshooting
1. Webhook failures -> validate URL.
2. Stdout output missing -> check alert router logs.
3. Hikvision errors -> not implemented.
4. Timeouts -> increase webhook timeout.
5. Missing adapter factory -> implement `get_protocol_adapter`.

## Architecture
### High-level diagram
```
AlertEvent -> ProtocolAdapter -> stdout/webhook/hikvision
```

### Key concepts
- `ProtocolAdapter`: abstract base class.
- `StdoutAdapter`: prints JSON to stdout.
- `WebhookAdapter`: POSTs alert JSON.
- `HikvisionAdapter`: placeholder.

### Runtime flow
- Alert router constructs adapter based on config.
- Adapter sends alert and reports errors to logs.

### Data flow
- Inputs: AlertEvent.
- Transformations: serialize to JSON.
- Outputs: stdout or HTTP request.

### Component map
| Component | Location (path) | Responsibility | Inputs | Outputs | Dependencies | Failure modes or notes |
| --- | --- | --- | --- | --- | --- | --- |
| Base adapter | `src/agave_vision/services/alert_router/protocols/base.py` | Adapter interface | AlertEvent | None | none | None |
| Stdout adapter | `src/agave_vision/services/alert_router/protocols/stdout.py` | Print alerts | AlertEvent | stdout | none | None |
| Webhook adapter | `src/agave_vision/services/alert_router/protocols/webhook.py` | POST alerts | AlertEvent | HTTP requests | httpx | Network errors |
| Hikvision adapter | `src/agave_vision/services/alert_router/protocols/hikvision.py` | Placeholder | AlertEvent | NotImplementedError | none | Not implemented |

## Entry points
- **CLI commands or scripts:** none.
- **APIs or routes:** none.
- **Workers, schedulers, or jobs:** none.
- **Notebooks:** none.

## Configuration and secrets
- **Configuration sources:** `configs/services.yaml`, env overrides.
- **Precedence:** env overrides > YAML > defaults.
- **Secrets handling patterns:** webhook/Hikvision creds via env vars.
- **Safe defaults and what must never be committed:** do not commit webhook URLs or passwords.

## Observability
- **Logging strategy:** adapter logs use `get_logger`.
- **Metrics or tracing:** none.
- **Debug workflow:** verify adapter output and alert router logs.

## Testing
- **How to run tests:** `pytest`.
- **Summary of the test pyramid:** no tests in repo.
- **Gaps and recommended minimal regression coverage:** mock HTTP and stdout delivery.

## Deployment (if applicable)
- **Build artifacts:** Python modules.
- **Deployment targets:** alert router service.
- **CI/CD overview:** none found.
- **Rollback strategy:** revert adapter changes.

## Contributing
- **Repository conventions:** add new adapters under this directory.
- **Branching and release practices:** not defined.
- **How to safely add a new feature and where to start:** implement adapter and update factory.

## Known gaps and assumptions
- `get_protocol_adapter` factory is missing in this package.
