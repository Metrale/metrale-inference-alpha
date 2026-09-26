// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of the readiness line: logged with the real address
//! after a bind, never after a failed bind, and rendered as a usable address.
//!
//! Owner: server (HTTP layer).
//! Invariants: none beyond the types.

use std::sync::mpsc::{Sender, channel};

use tracing::field::{Field, Visit};
use tracing_subscriber::layer::SubscriberExt as _;

use super::{bind_and_announce, ready_line};
use crate::main_modules::model_host::ModelHost;

/// 2026-09-26: Collects each event's `message`, through a subscriber local to
/// the test rather than a process-global one, so other tests' lines never
/// reach it.
struct MessageSink(Sender<String>);

impl<S: tracing::Subscriber> tracing_subscriber::Layer<S> for MessageSink {
    fn on_event(
        &self,
        event: &tracing::Event<'_>,
        _ctx: tracing_subscriber::layer::Context<'_, S>,
    ) {
        struct V(Option<String>);
        impl Visit for V {
            fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
                if field.name() == "message" {
                    self.0 = Some(format!("{value:?}"));
                }
            }
        }
        let mut v = V(None);
        event.record(&mut v);
        if let Some(msg) = v.0 {
            let _ = self.0.send(msg);
        }
    }
}

/// 2026-09-26: Run `f` with the sink installed. `with_default` is
/// thread-scoped, so callers use a current-thread runtime and every event
/// reaches the sink.
fn logged<T>(f: impl FnOnce() -> T) -> (T, Vec<String>) {
    let (tx, rx) = channel();
    let sub = tracing_subscriber::registry().with(MessageSink(tx));
    let out = tracing::subscriber::with_default(sub, f);
    (out, rx.try_iter().collect())
}

fn current_thread_rt() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("runtime")
}

/// 2026-09-26: A loopback port that was free when asked.
fn free_port() -> u16 {
    let sock = std::net::TcpListener::bind("127.0.0.1:0").expect("ephemeral bind");
    sock.local_addr().expect("addr").port()
}

#[test]
fn the_ready_line_is_emitted_after_a_successful_bind_with_the_real_address() {
    // 2026-09-26: `bind_and_announce`, not `build_and_serve`, whose accept
    // loop does not return and disarms the startup escape for the process.
    let port = free_port();
    let (result, lines) = logged(|| {
        let rt = current_thread_rt();
        rt.block_on(bind_and_announce(&ModelHost::empty(), "127.0.0.1", port))
    });
    assert!(result.is_ok(), "the bind succeeds: {result:?}");
    // 2026-09-26: No model is loaded, so the line is the live-only variant.
    let expected = format!("Server live at 127.0.0.1:{port}");
    assert!(
        lines.iter().any(|l| l.contains(&expected)),
        "the line and its real address are in the log: {lines:#?}"
    );
    assert!(
        !lines.iter().any(|l| l.contains("live and ready")),
        "a modelless boot must not claim a model: {lines:#?}"
    );
}

#[test]
fn a_failed_bind_never_announces_readiness() {
    let taken = std::net::TcpListener::bind("127.0.0.1:0").expect("occupy");
    let port = taken.local_addr().expect("addr").port();
    let (result, lines) = logged(|| {
        let rt = current_thread_rt();
        rt.block_on(bind_and_announce(&ModelHost::empty(), "127.0.0.1", port))
    });
    assert!(result.is_err(), "the bind must fail: port is held");
    assert!(
        !lines.iter().any(|l| l.contains("Server live")),
        "no readiness claim on the failure path: {lines:#?}"
    );
}

#[test]
fn the_ready_line_names_the_model_the_way_the_dashboard_does() {
    // 2026-09-26: The caller passes `ModelHost::live_model()`, the current
    // `AppState::model_name`; building an `AppState` needs a loaded model, so
    // the with-model text is checked on `ready_line` directly.
    assert_eq!(
        ready_line("127.0.0.1", 8888, Some("Qwen/Qwen3.6-35B-A3B-FP8")),
        "Server live and ready at 127.0.0.1:8888 running Qwen/Qwen3.6-35B-A3B-FP8"
    );
}

#[test]
fn a_wildcard_bind_is_rendered_as_an_address_a_user_can_paste() {
    assert_eq!(
        ready_line("0.0.0.0", 8888, Some("m")),
        "Server live and ready at 127.0.0.1:8888 running m"
    );
    assert_eq!(
        ready_line("::", 8000, Some("m")),
        "Server live and ready at 127.0.0.1:8000 running m"
    );
    assert_eq!(
        ready_line("10.10.10.1", 8000, Some("m")),
        "Server live and ready at 10.10.10.1:8000 running m"
    );
    assert_eq!(
        ready_line("::1", 8000, Some("m")),
        "Server live and ready at [::1]:8000 running m"
    );
}
