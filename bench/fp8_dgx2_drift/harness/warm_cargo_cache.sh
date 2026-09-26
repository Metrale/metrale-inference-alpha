#!/usr/bin/env bash

# 2026-09-26: Pre-warms the shared cargo target dir that the agent's builds and
# score_run.py's webserver build use.
#
# Owner: bench, FP8 drift harness.
# Invariants: none beyond the types.
#
# Builds a template Axum project, with the dependencies and features in the
# Cargo.toml below, into METRALE_WARM_TARGET_DIR, so a generated project built
# there compiles little more than its own crate. run_tier.sh and score_run.py
# point CARGO_TARGET_DIR at the same path, with the same default; warm_tests.rs
# in the bench crate checks that the three defaults agree.
#
# On a re-run cargo reuses the dependencies it has already built.












set -euo pipefail

WARM_TARGET_DIR="${METRALE_WARM_TARGET_DIR:-${HOME}/.cargo/metrale-warm-target}"
TEMPLATE_DIR="${METRALE_WARM_TEMPLATE_DIR:-${HOME}/.cargo/metrale-warm-template}"

echo "[warm] warm target dir : ${WARM_TARGET_DIR}" >&2
echo "[warm] template project: ${TEMPLATE_DIR}" >&2

mkdir -p "${TEMPLATE_DIR}/src"

# 2026-09-26: Version requirements are caret ranges, so cargo resolves the newest
# compatible release of each; a generation that resolves to the same versions and
# features reuses the warm rlibs.
cat > "${TEMPLATE_DIR}/Cargo.toml" <<'TOML'
[package]
name = "metrale-warm-template"
version = "0.1.0"
edition = "2021"

[dependencies]
# 2026-09-26: The template's dependencies and features. `TEMPLATE_MANIFEST`
# in the bench crate's agentic/warm.rs must parse to the same TOML
# (warm_tests.rs asserts it), so a change here goes there too.



axum = { version = "0.8", features = ["json", "macros", "ws", "multipart"] }
tokio = { version = "1", features = ["full"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
tower = { version = "0.5", features = ["full"] }
tower-http = { version = "0.6", features = ["full"] }
hyper = { version = "1", features = ["full"] }
reqwest = { version = "0.12", features = ["json"] }
anyhow = "1"
thiserror = "2"
tracing = "0.1"
tracing-subscriber = { version = "0.3", features = ["env-filter"] }

[dev-dependencies]
reqwest = { version = "0.12", features = ["json"] }
TOML

cat > "${TEMPLATE_DIR}/src/main.rs" <<'RUST'
// Touches each dependency so its rlib is compiled into the warm target dir.
use axum::{routing::get, Router};

async fn ping() -> &'static str {
    "pong"
}

#[tokio::main]
async fn main() {
    let _ = serde_json::json!({"ok": true});
    let _v: tower::ServiceBuilder<tower::layer::util::Identity> = tower::ServiceBuilder::new();
    let app = Router::new().route("/ping", get(ping));
    let port: u16 = std::env::var("METRALE_HARNESS_PORT")
        .unwrap_or_else(|_| "3001".to_string())
        .parse()
        .unwrap();
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
RUST

mkdir -p "${WARM_TARGET_DIR}"

# 2026-09-26: Warm both profiles, which cargo builds separately: `cargo test` and
# `cargo run` use debug, `cargo build --release` uses release.
#   cargo test --no-run    -> debug profile rlibs and the test harness link
#   cargo build --release  -> release profile rlibs
echo "[warm] compiling DEBUG profile (cargo test) into shared target dir..." >&2
CARGO_TARGET_DIR="${WARM_TARGET_DIR}" cargo test --no-run \
    --manifest-path "${TEMPLATE_DIR}/Cargo.toml" >&2

echo "[warm] compiling RELEASE profile (cargo build --release) into shared target dir..." >&2
CARGO_TARGET_DIR="${WARM_TARGET_DIR}" cargo build --release \
    --manifest-path "${TEMPLATE_DIR}/Cargo.toml" >&2

echo "[warm] warm cache ready (debug + release profiles)." >&2
du -sh "${WARM_TARGET_DIR}" 2>/dev/null | sed 's/^/[warm] target dir size: /' >&2 || true
