// SPDX-License-Identifier: AGPL-3.0-only

//! Optional GPU certificate; unsupported/tied/nonfinite rows use host sampling.

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kernel_args::KernelLaunch;

use super::TransformerModel;

impl TransformerModel {
    pub(super) fn argmax_unique_dispatch(
        &self,
        logits: DevicePtr,
        rows: usize,
    ) -> Result<Option<Vec<u32>>> {
        if self.argmax_unique_kernel.0 == 0 {
            return Ok(None);
        }
        let vocab = self.config.vocab_size;
        ensure!(
            vocab > 0 && vocab < u32::MAX as usize,
            "invalid argmax vocabulary"
        );
        let rows_u32 = u32::try_from(rows)?;
        if rows == 0 {
            return Ok(Some(Vec::new()));
        }
        // Same scratch and default-stream ordering as legacy argmax_batch.
        // Kernel overwrites every output row, including ambiguous rows.
        let output = self.buffers.scratch();
        KernelLaunch::new(self.gpu.as_ref(), self.argmax_unique_kernel)
            .grid([rows_u32, 1, 1])
            .block([1024, 1, 1])
            .arg_ptr(logits)
            .arg_ptr(output)
            .arg_u32(vocab as u32)
            .arg_u32(vocab as u32)
            .launch(self.gpu.default_stream())?;
        let mut bytes = vec![
            0u8;
            rows.checked_mul(4)
                .ok_or_else(|| anyhow::anyhow!("argmax output overflow"))?
        ];
        self.gpu.copy_d2h(output, &mut bytes)?;
        decode_unique_indices(&bytes, vocab)
    }
}

fn decode_unique_indices(bytes: &[u8], vocab: usize) -> Result<Option<Vec<u32>>> {
    ensure!(bytes.len().is_multiple_of(4), "invalid argmax output size");
    let mut tokens = Vec::with_capacity(bytes.len() / 4);
    for row in bytes.chunks_exact(4) {
        let token = u32::from_le_bytes(row.try_into()?);
        if token == u32::MAX {
            return Ok(None);
        }
        ensure!(
            (token as usize) < vocab,
            "argmax returned out-of-vocabulary token"
        );
        tokens.push(token);
    }
    Ok(Some(tokens))
}

#[cfg(test)]
mod tests {
    use super::decode_unique_indices;

    #[test]
    fn sentinel_declines_entire_batch() {
        let bytes: Vec<u8> = [2u32, u32::MAX, 1]
            .into_iter()
            .flat_map(u32::to_le_bytes)
            .collect();
        assert_eq!(decode_unique_indices(&bytes, 10).unwrap(), None);
        assert_eq!(
            decode_unique_indices(&2u32.to_le_bytes(), 10).unwrap(),
            Some(vec![2])
        );
    }

    #[test]
    fn malformed_or_non_sentinel_oov_is_an_error() {
        assert!(decode_unique_indices(&[0, 1], 10).is_err());
        assert!(decode_unique_indices(&10u32.to_le_bytes(), 10).is_err());
    }
}
