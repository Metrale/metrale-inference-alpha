// SPDX-License-Identifier: AGPL-3.0-only
//! Dispatch boundary for the measured exact-order Hopper router specialization.

pub(super) fn eligible(m: u32, n: u32, k: u32, decode: bool, kernel: bool) -> bool {
    m == 128 && n == 256 && k == 2048 && !decode && kernel
}

#[cfg(test)]
mod tests {
    use super::eligible;

    #[test]
    fn only_measured_prefill_shape_with_kernel() {
        assert!(eligible(128, 256, 2048, false, true));
        for m in [0, 1, 7, 16, 32, 64, 127, 129, 256] {
            assert!(!eligible(m, 256, 2048, false, true));
        }
        for n in [0, 128, 255, 257, 512] {
            assert!(!eligible(128, n, 2048, false, true));
        }
        for k in [0, 1024, 2047, 2049, 4096] {
            assert!(!eligible(128, 256, k, false, true));
        }
        assert!(!eligible(128, 256, 2048, true, true));
        assert!(!eligible(128, 256, 2048, false, false));
    }
}
