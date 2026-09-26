# metrale-bench

**Path:** `crates/bench/`

Plugins and the benchmark suite that the `met serve` TUI drives. `Plugin` is the general abstraction: on load it receives a `PluginHandle`, the seam onto the host terminal (status, log, progress), the `~/.metrale` artifact store, the endpoint it is pointed at and a cancellation flag. `Benchmark` specialises it into a drivable state machine that does one step per `Benchmark::next`; `Benchmark::run` drives that loop and streams `BenchmarkResult` frames, and `executor::drive` is the single driver loop. The benchmark implementations live under `src/benchmarks/`; the certification gate lives under `src/gate/`.
