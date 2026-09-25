// SPDX-License-Identifier: AGPL-3.0-only
//! Boundary for measured Hopper shared-expert FP8 projection tiles.

pub(super) fn eligible(
    m: u32,
    n: u32,
    k: u32,
    decode: bool,
    kernel: bool,
    experts: usize,
    topk: usize,
) -> bool {
    m == 128
        && matches!((n, k), (512, 2048) | (2048, 512))
        && !decode
        && kernel
        && experts == 256
        && topk == 8
}

#[cfg(test)]
mod tests {
    use super::eligible;
    #[test]
    fn measured_shapes_only() {
        for (n, k) in [(512, 2048), (2048, 512)] {
            assert!(eligible(128, n, k, false, true, 256, 8));
            for m in [0, 1, 16, 64, 127, 129, 256] {
                assert!(!eligible(m, n, k, false, true, 256, 8));
            }
            assert!(!eligible(128, n, k, true, true, 256, 8));
            assert!(!eligible(128, n, k, false, false, 256, 8));
            assert!(!eligible(128, n, k, false, true, 128, 8));
            assert!(!eligible(128, n, k, false, true, 256, 6));
        }
        for (n, k) in [(512, 512), (2048, 2048), (511, 2048), (512, 2047)] {
            assert!(!eligible(128, n, k, false, true, 256, 8));
        }
    }
}
