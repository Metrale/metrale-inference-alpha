// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The scheduler shutdown sequence: pipeline drain, session eviction,
//! finishing the actives, failing the in-flight work, EP shutdown, stream
//! quiesce and the model teardown.
//!
//! Owner: scheduler.
//! Invariants:
//! - A synchronise of the default and prefill streams is attempted before
//!   `DeviceIo::teardown`; a failure is logged and teardown proceeds.

use super::*;

impl SchedulerCore {
    pub(super) fn finish_run(mut self) {
        self.drain_pipeline();
        let Self {
            ctx: sched,
            active,
            prefilling,
            swapped,
            preempted,
            mut session_manager,
            prefill_stream,
            ..
        } = self;
        // 2026-09-25: Evict expired sessions, freeing their SSM snapshot slots.
        {
            let freed_slots = session_manager.evict_expired();
            if !freed_slots.is_empty() {
                tracing::info!(
                    "Session eviction: freed {} SSM snapshot slot(s), {} sessions active",
                    freed_slots.len(),
                    session_manager.session_count()
                );
            }
        }

        for mut a in active {
            finish_sequence(&sched.io, &mut a, sched.limits.max_seq_len);
        }
        shutdown_drain::abort_in_flight_on_shutdown(
            sched.io.dev.model(),
            &sched.io,
            prefilling,
            swapped,
            preempted,
            sched.io.spill.as_deref(),
        );
        let _ = sched.io.dev.apply(io::Effect::EpShutdown);

        // 2026-09-25: Release the model's device memory explicitly, in order and able
        // to report a failure; `Drop` can do neither (see `metrale_core::scope`).
        // `Model::teardown` is documented to run only after the stream is
        // synchronised, so synchronise both streams the scheduler submits to (the
        // model's default stream and `prefill_stream`) first. A failure is logged
        // and teardown proceeds: refusing to free would leak the whole model.
        let streams = [
            ("default", sched.io.dev.model().default_stream()),
            ("prefill", prefill_stream),
        ];
        let unsynced = teardown::quiesce_streams(&streams, |stream| {
            sched
                .io
                .dev
                .apply(io::Effect::Quiesce { stream })
                .map(|_| ())
                .map_err(io::DeviceError::into_inner)
        });
        for name in unsynced {
            tracing::error!(
                "could not synchronise the {name} stream before teardown — freeing \
                 anyway, but device memory may still be in use"
            );
        }
        // 2026-09-25: Nothing may free device memory between the synchronise above
        // and this teardown.
        let dev = sched.io.dev;
        if let Err(e) = dev.teardown() {
            tracing::error!("model teardown reported a failure: {e:#}");
        }
        tracing::info!("Scheduler stopped");
    }
}
