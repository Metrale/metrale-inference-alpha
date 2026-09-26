// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests of `scale_is_per_row` and `proj_is_fp8_per_row`: which
//! scale layouts count as per-row.
//!
//! Owner: model-arch weight loader (Qwen3.5 dense).
//! Invariants: none beyond the types.

use std::collections::HashMap;

use metrale_gpu_runtime::gpu::DevicePtr;
use metrale_model_weights::weights::{WeightDtype, WeightStore, WeightTensor};

use super::{proj_is_fp8_per_row, scale_is_per_row};

fn projection_store(
    weight_dtype: WeightDtype,
    weight_shape: Vec<usize>,
    scale_shape: Option<Vec<usize>>,
) -> WeightStore {
    let mut tensors = HashMap::from([(
        "proj.weight".to_string(),
        WeightTensor {
            ptr: DevicePtr::NULL,
            shape: weight_shape,
            dtype: weight_dtype,
        },
    )]);
    if let Some(shape) = scale_shape {
        tensors.insert(
            "proj.weight_scale".to_string(),
            WeightTensor {
                ptr: DevicePtr::NULL,
                shape,
                dtype: WeightDtype::BF16,
            },
        );
    }
    WeightStore::from_map(tensors)
}

#[test]
fn per_row_shapes_are_accepted() {
    assert!(scale_is_per_row(4096, &[4096], 4096), "[N]");
    assert!(scale_is_per_row(4096, &[4096, 1], 4096), "[N,1]");
}

#[test]
fn a_scalar_scale_is_not_per_row() {
    assert!(!scale_is_per_row(4096, &[1], 1));
    assert!(!scale_is_per_row(4096, &[], 1));
}

/// 2026-09-25: A `[N/128, K/128]` block grid is not per-row.
#[test]
fn a_block_grid_is_not_per_row() {
    assert!(!scale_is_per_row(4096, &[32, 40], 1280));
}

/// 2026-09-25: `[1, N]` has `N` elements but is rejected: its first axis is
/// not `N`.
#[test]
fn a_column_vector_is_rejected_even_with_n_elements() {
    assert!(!scale_is_per_row(4096, &[1, 4096], 4096));
}

#[test]
fn projection_metadata_gates_the_rowwise_loader_path() {
    assert!(proj_is_fp8_per_row(
        &projection_store(WeightDtype::FP8E4M3, vec![4, 8], Some(vec![4, 1])),
        "proj"
    ));
    assert!(!proj_is_fp8_per_row(
        &projection_store(WeightDtype::BF16, vec![4, 8], Some(vec![4, 1])),
        "proj"
    ));
    assert!(!proj_is_fp8_per_row(
        &projection_store(WeightDtype::FP8E4M3, vec![32], Some(vec![4, 1])),
        "proj"
    ));
    assert!(!proj_is_fp8_per_row(
        &projection_store(WeightDtype::FP8E4M3, vec![4, 8], None),
        "proj"
    ));
}
