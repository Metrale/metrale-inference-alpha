// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests that `Carried` holds the process-scoped stores and rate
//! limiter by shared pointer.
//!
//! Owner: server startup (`met serve`).
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-26: A clone of `Carried` points at the same stores and limiter.
/// Checked by pointer identity (`Arc::ptr_eq`), which a swap relies on; an
/// equal but separate store would have lost its contents.
#[test]
fn carried_state_is_the_same_allocation_not_an_equal_one() {
    let first = Carried::from_env().expect("no METRALE_* config is set in the test environment");
    let cloned = first.clone();

    assert!(
        std::sync::Arc::ptr_eq(&first.response_store, &cloned.response_store),
        "responses must survive a swap"
    );
    assert!(
        std::sync::Arc::ptr_eq(&first.rate_limiter, &cloned.rate_limiter),
        "rate-limit buckets must survive a swap"
    );
    assert!(
        std::sync::Arc::ptr_eq(&first.conversation_store, &cloned.conversation_store),
        "stored conversations must survive a swap"
    );
}

/// 2026-09-26: Two `from_env()` calls build different allocations, which is
/// why `load_model` takes `Carried` instead of building its own.
#[test]
fn building_from_env_twice_would_lose_the_stores() {
    let first = Carried::from_env().expect("no METRALE_* config is set in the test environment");
    let second = Carried::from_env().expect("no METRALE_* config is set in the test environment");
    assert!(
        !std::sync::Arc::ptr_eq(&first.conversation_store, &second.conversation_store),
        "if this ever passes, from_env has become a singleton and the carried \
         parameter is no longer what protects the stores — re-check the swap"
    );
}

#[test]
fn carried_uses_the_process_limiter_rather_than_minting_its_own() {
    // 2026-09-26: Handlers refund through `AppState.rate_limiter` and the
    // middleware debits through the host's limiter, so they must be one
    // instance.
    let host = crate::main_modules::model_host::ModelHost::empty();
    let carried = Carried::from_env().expect("no METRALE_* config is set in the test environment");
    let process = carried.rate_limiter.clone();
    host.set_process(carried);
    assert!(
        std::sync::Arc::ptr_eq(&host.rate_limiter().expect("installed"), &process),
        "the host's limiter IS the one the model's AppState will hold"
    );
}
