# agave-vision-api / src/agave_vision/services/alert_router

## Overview
- Redis consumer that debounces alerts and routes them to protocol adapters.
- Supports stdout and webhook adapters; Hikvision is a placeholder.
- Primary users: operators integrating alerts with downstream systems.

## Quickstart
### Prerequisites
- Python 3.11+.
- Redis.
- Configured alerting settings in `configs/services.yaml`.

### Install
- `pip install -e .[alerts]`.

### Configure
- Set `ALERTING_PROTOCOL` and adapter-specific settings in `configs/services.yaml` or env overrides.

### Run (development)
- `python -m agave_vision.services.alert_router.main`.

### Run (production-like)
- Run as a long-lived process with stable Redis.

### Common commands
- `python -m agave_vision.services.alert_router.main`.

### Troubleshooting
1. Redis connection errors -> set `REDIS_URL`.
2. Alerts not delivered -> verify protocol adapter config.
3. Webhook failures -> check URL and network.
4. Debounce suppressing alerts -> adjust `ALERTING_DEBOUNCE_WINDOW_SECONDS`.
5. Hikvision adapter errors -> not implemented.

## Architecture
### High-level diagram
```
Redis stream -> RedisConsumer -> AlertDebouncer -> ProtocolAdapter
```

### Key concepts
- `RedisConsumer`: reads from Redis Streams and ACKs messages.
- `AlertDebouncer`: rate limits alerts by camera/class/ROI.
- Protocol adapters: stdout, webhook, Hikvision placeholder.

### Runtime flow
- Loads alerting config and initializes protocol adapter.
- Creates Redis consumer group and reads stream entries.
- Applies debouncing rules before sending alerts.
- Sends alerts via adapter and ACKs messages.

### Data flow
- Inputs: Redis stream messages containing alert JSON.
- Transformations: JSON -> AlertEvent -> debounce -> adapter send.
- Outputs: stdout logs or webhook requests.

### Component map
| Component | Location (path) | Responsibility | Inputs | Outputs | Dependencies | Failure modes or notes |
| --- | --- | --- | --- | --- | --- | --- |
| Main | `src/agave_vision/services/alert_router/main.py` | Service startup and orchestration | configs | running consumer | redis, httpx | Adapter factory missing |
| Consumer | `src/agave_vision/services/alert_router/consumer.py` | Redis stream consumption | stream messages | adapter calls | redis | Redis unavailable |
| Debouncer | `src/agave_vision/services/alert_router/debounce.py` | Alert rate limiting | AlertEvent | allow/deny | datetime | Over-suppression |
| Stdout adapter | `src/agave_vision/services/alert_router/protocols/stdout.py` | Print alerts to stdout | AlertEvent | console output | none | None |
| Webhook adapter | `src/agave_vision/services/alert_router/protocols/webhook.py` | HTTP POST alerts | AlertEvent | HTTP requests | httpx | Network errors |
| Hikvision adapter | `src/agave_vision/services/alert_router/protocols/hikvision.py` | Placeholder | AlertEvent | NotImplementedError | none | Not implemented |

## Entry points
- **CLI commands or scripts:** `python -m agave_vision.services.alert_router.main`.
- **APIs or routes:** none.
- **Workers, schedulers, or jobs:** Redis consumer loop.
- **Notebooks:** none.

## Configuration and secrets
- **Configuration sources:** `configs/services.yaml`, env overrides.
- **Precedence:** env overrides > YAML > defaults.
- **Secrets handling patterns:** webhook and Hikvision credentials via env vars.
- **Safe defaults and what must never be committed:** do not commit webhook URLs or passwords.

## Observability
- **Logging strategy:** JSON logs via `setup_logging` in `main.py`.
- **Metrics or tracing:** none.
- **Debug workflow:** inspect logs and Redis stream entries.

## Testing
- **How to run tests:** `pytest`.
- **Summary of the test pyramid:** no tests in repo.
- **Gaps and recommended minimal regression coverage:** add tests for debouncing and adapter delivery.

## Deployment (if applicable)
- **Build artifacts:** Python package.
- **Deployment targets:** service process.
- **CI/CD overview:** none found.
- **Rollback strategy:** revert configs or service version.

## Contributing
- **Repository conventions:** add new adapters in `protocols/` and update the adapter factory.
- **Branching and release practices:** not defined.
- **How to safely add a new feature and where to start:** implement `get_protocol_adapter` and add adapters.

## Known gaps and assumptions
- `get_protocol_adapter` is referenced but not implemented in `src/agave_vision/services/alert_router/protocols/`.
