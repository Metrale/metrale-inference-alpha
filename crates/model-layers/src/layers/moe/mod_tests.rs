// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for `Bf16SharedExpert::new`: it keeps the three weights and
//! refuses a null one.
//!
//! Owner: model-layers (MoE).
//! Invariants: none beyond the types.

use super::*;
#[test]
fn bf16_shared_expert_requires_three_non_null_weights() {
    let gate = DenseWeight {
        weight: DevicePtr(11),
    };
    let up = DenseWeight {
        weight: DevicePtr(22),
    };
    let down = DenseWeight {
        weight: DevicePtr(33),
    };

    let shared = Bf16SharedExpert::new(gate, up, down).expect("valid BF16 shared expert");
    assert_eq!(shared.gate_proj.weight, gate.weight);
    assert_eq!(shared.up_proj.weight, up.weight);
    assert_eq!(shared.down_proj.weight, down.weight);

    let null = DenseWeight {
        weight: DevicePtr::NULL,
    };
    assert!(Bf16SharedExpert::new(null, up, down).is_err());
    assert!(Bf16SharedExpert::new(gate, null, down).is_err());
    assert!(Bf16SharedExpert::new(gate, up, null).is_err());
}
