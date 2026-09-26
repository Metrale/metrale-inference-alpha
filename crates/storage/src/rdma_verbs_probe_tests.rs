// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests that `cfg(metrale_rdma_verbs)` reaches this crate.
//!
//! Owner: storage.
//! Invariants: none beyond the types.
//!
//! `rustc-cfg` does not cross crates: build.rs re-emits the cfg when metrale-gpu-sys
//! publishes `DEP_METRALE_RDMA_SHIM_HAS_VERBS`. If that broke, every
//! `#[cfg(metrale_rdma_verbs)]` module here would compile out without a build error.

/// 2026-09-25: Compiled only when the cfg is on for this crate; metrale-gpu-sys must
/// then report the shim built too.
#[cfg(metrale_rdma_verbs)]
#[test]
fn verbs_cfg_is_on_for_storage() {
    assert!(crate::rdma_verbs_enabled());
    assert!(metrale_gpu_sys::verbs_enabled());
}

/// 2026-09-25: The cfg matches metrale-gpu-sys's whether it is on or off; that
/// crate's build.rs makes the decision.
#[test]
fn verbs_cfg_agrees_with_metrale_rdma() {
    assert_eq!(
        crate::rdma_verbs_enabled(),
        metrale_gpu_sys::verbs_enabled()
    );
}
