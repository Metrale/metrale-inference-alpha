// SPDX-License-Identifier: AGPL-3.0-only

use super::ordered_builder_eligible;

#[test]
fn admitted_shapes_match_fixed_shared_storage_and_packing() {
    for experts in [1, 128, 256] {
        for n_tiles in [1, 8, 32, 64] {
            assert!(ordered_builder_eligible(experts, n_tiles, 128, true));
        }
    }
}

#[test]
fn unsupported_shapes_and_missing_kernel_keep_reference() {
    for experts in [0, 257, u32::MAX] {
        assert!(!ordered_builder_eligible(experts, 32, 128, true));
    }
    for n_tiles in [0, 65, u32::MAX] {
        assert!(!ordered_builder_eligible(256, n_tiles, 128, true));
    }
    for m_tile in [0, 16, 64, 129, u32::MAX] {
        assert!(!ordered_builder_eligible(256, 32, m_tile, true));
    }
    assert!(!ordered_builder_eligible(256, 32, 128, false));
}
