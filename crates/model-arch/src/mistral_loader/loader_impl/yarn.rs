// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The YaRN inv_freq table for the MLA rope dims, computed on the host and uploaded as f32.
//!
//! Owner: model-arch (Mistral loader).
//! Invariants: none beyond the types.

use anyhow::Result;
use metrale_config::ModelConfig;
use metrale_gpu_runtime::gpu::{DevicePtr, GpuBackend};

use super::super::gpu_alloc_or_managed;

pub(super) fn compute_yarn_inv_freq(
    config: &ModelConfig,
    rope: usize,
    gpu: &dyn GpuBackend,
) -> Result<DevicePtr> {
    // 2026-09-25: YaRN in dimension-index space: `low` and `high` are the
    // correction dims of `beta_fast` and `beta_slow`, and pair `j` blends the
    // interpolated and extrapolated frequency with a linear ramp between
    // them. The Mistral config parser reads `params.json` `yarn.alpha` into
    // `yarn_beta_slow` and `yarn.beta` into `yarn_beta_fast`.
    let factor = if config.yarn_factor > 0.0 {
        config.yarn_factor
    } else {
        128.0
    };
    let beta_fast = if config.yarn_beta_fast > 0.0 {
        config.yarn_beta_fast
    } else {
        32.0
    };
    let beta_slow = if config.yarn_beta_slow > 0.0 {
        config.yarn_beta_slow
    } else {
        1.0
    };
    let original_max_pos = if config.yarn_original_max_position_embeddings > 0 {
        config.yarn_original_max_position_embeddings as f32
    } else {
        8192.0
    };
    let dim_f = rope as f32;
    let theta_f = config.rope_theta as f32;
    let n_pairs = rope / 2;

    let find_correction_dim = |num_rot: f32| -> f32 {
        (dim_f * (original_max_pos / (num_rot * 2.0 * std::f32::consts::PI)).ln())
            / (2.0 * theta_f.ln())
    };
    let low = find_correction_dim(beta_fast).floor().max(0.0);
    let high = find_correction_dim(beta_slow).ceil().min((rope - 1) as f32);
    let ramp_denom = if (high - low).abs() < 1e-6 {
        high - low + 0.001
    } else {
        high - low
    };

    let mut inv_freq_table = vec![0.0f32; n_pairs];
    for j in 0..n_pairs {
        let pos_freq = theta_f.powf((2 * j) as f32 / dim_f);
        let inv_freq_extrap = 1.0 / pos_freq;
        let inv_freq_interp = 1.0 / (factor * pos_freq);

        let ramp = ((j as f32 - low) / ramp_denom).clamp(0.0, 1.0);
        let extrap_factor = 1.0 - ramp;

        inv_freq_table[j] =
            inv_freq_interp * (1.0 - extrap_factor) + inv_freq_extrap * extrap_factor;
    }
    let bytes: Vec<u8> = inv_freq_table
        .iter()
        .flat_map(|v| v.to_le_bytes())
        .collect();
    let ptr = gpu_alloc_or_managed(gpu, bytes.len())?;
    gpu.copy_h2d(&bytes, ptr)?;
    tracing::info!(
        "YaRN inv_freq: {} pairs, factor={factor}, beta_fast={beta_fast}, \
         beta_slow={beta_slow}, max_pos={original_max_pos}, low_dim={low:.1}, high_dim={high:.1}",
        n_pairs,
    );
    tracing::info!(
        "YaRN inv_freq sample: [0]={:.6e} [12]={:.6e} [25]={:.6e} [31]={:.6e}",
        inv_freq_table.first().copied().unwrap_or(0.0),
        inv_freq_table.get(12).copied().unwrap_or(0.0),
        inv_freq_table.get(25).copied().unwrap_or(0.0),
        inv_freq_table.get(31).copied().unwrap_or(0.0),
    );
    Ok(ptr)
}
