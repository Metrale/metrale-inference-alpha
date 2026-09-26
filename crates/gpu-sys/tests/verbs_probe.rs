// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Public-API tests of the verbs layer that need no RDMA hardware
//! and no network: the cfg witness, and `Verbs::create` failing for a device
//! name that cannot exist.
//!
//! Owner: metrale-gpu-sys.
//! Invariants: none beyond the types.

/// 2026-09-26: The always-compiled witness agrees with the cfg this test target
/// sees. It runs in both cfg states: `true` when build.rs compiled the shim,
/// `false` when it left it out (a target other than Linux, `METRALE_NO_RDMA=1`,
/// or a skip-build variable).
#[test]
fn verbs_enabled_matches_cfg() {
    assert_eq!(metrale_gpu_sys::verbs_enabled(), cfg!(metrale_rdma_verbs));
}

#[cfg(metrale_rdma_verbs)]
mod with_shim {
    use metrale_gpu_sys::{Gid, MrKeys, Verbs};

    /// 2026-09-26: A nonexistent device fails `rs_create`, and the error names
    /// the device. This runs the real C shim and libibverbs linkage.
    #[test]
    fn create_unknown_device_errors() {
        let err = match Verbs::create("metrale-rdma-no-such-dev", 3, 0x0012_3456) {
            Ok(_) => panic!("create() succeeded for a device that cannot exist"),
            Err(e) => e,
        };
        let msg = format!("{err:#}");
        assert!(msg.contains("rs_create failed"), "unexpected error: {msg}");
        assert!(
            msg.contains("metrale-rdma-no-such-dev"),
            "device name missing: {msg}"
        );
    }

    /// 2026-09-26: `MrKeys` is `Copy`, and `Gid` is 16 bytes, the size of the
    /// `gid` field on the wire.
    #[test]
    fn mr_keys_copy_and_gid_layout() {
        let k = MrKeys { lkey: 1, rkey: 2 };
        let k2 = k;
        assert_eq!((k.lkey, k.rkey), (k2.lkey, k2.rkey));
        assert_eq!(std::mem::size_of::<Gid>(), 16);
    }
}
