# metrale-telemetry

**Path:** `crates/telemetry/`

Run metrics, kernel audit, launch trace and progress hooks, and the metal-up instrument set. Levels are `Off` (the default, where every instrument entry point is one relaxed load and a return), `Basic` and `Kernel`. The layers run from device (NVML via dlopen at 10 Hz) through kernel, scheduler, cache and speculative-decoding counters to per-request TTFT/TPOT, plus live J/token from NVML energy-counter deltas. Everything is exported from one `TelemetrySnapshot` as Prometheus text, JSONL events or OTLP.
