# metrale-scheduler

**Path:** `crates/scheduler/`

The scheduler's I/O contract (SBIO). The core, in [metrale-server](./metrale-server.md), decides; everything it decides about a device is a `StepPlan` or an `Effect`, handed to a `DeviceIo`. The clock is a `ClockIo` and swap files a `SpillIo`; `driver::run` is the one loop every router runs under. The crate names no model type — the sequence state and the device logits handle are type parameters, which is the compile-time proof that the contract is model-free. `ScriptedDeviceIo` answers the decode lane from a script, so a test can drive the real core with no model of its own.
