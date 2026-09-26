// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The model's canonical quantization label, and whether a kernel
//! target's label can serve it.
//!
//! Owner: server startup (`met serve`).
//! Invariants: none beyond the types.

/// 2026-09-26: The canonical quantization label from `quantization_config`:
/// `"bf16"` without one; else `"mxfp4"`, `"nvfp4"` or `"fp8"` by
/// `quant_algo`, `quant_method` and `format`; else the lowercased
/// `quant_algo`, then `quant_method`, when nonempty; else `"unknown"`.
pub(crate) fn canonicalize_model_quant(config: &metrale_config::ModelConfig) -> String {
    let Some(qc) = config.quantization_config.as_ref() else {
        return "bf16".to_string();
    };
    let method = qc.quant_method.to_ascii_lowercase();
    let algo = qc.quant_algo.to_ascii_lowercase();
    let fmt = qc.format.to_ascii_lowercase();
    if algo == "mxfp4" || fmt == "mxfp4-pack-quantized" {
        return "mxfp4".into();
    }
    if algo == "nvfp4" || algo == "mixed_precision" || fmt.contains("nvfp4") {
        return "nvfp4".into();
    }
    // 2026-09-26: `float-quantized` contains no `fp8`, so it is matched by
    // name.
    if algo == "fp8" || method.contains("fp8") || fmt.contains("fp8") || fmt.contains("float-quant")
    {
        return "fp8".into();
    }
    if !algo.is_empty() {
        return algo;
    }
    if !method.is_empty() {
        return method;
    }
    "unknown".into()
}

/// 2026-09-26: The quantization fields as declared, for the mismatch error.
pub(crate) fn describe_quant_source(config: &metrale_config::ModelConfig) -> String {
    match config.quantization_config.as_ref() {
        Some(qc) => format!(
            "quant_method={:?}, quant_algo={:?}, format={:?}",
            qc.quant_method, qc.quant_algo, qc.format
        ),
        None => "no quantization_config in config.json".into(),
    }
}

/// 2026-09-26: Whether a kernel target labelled `kernel_quant` accepts a model
/// labelled `model_quant`.
pub(crate) fn quant_pair_compatible(kernel_quant: &str, model_quant: &str) -> bool {
    if kernel_quant == model_quant {
        return true;
    }
    matches!(
        (kernel_quant, model_quant),
        ("nvfp4", "fp8") |
        // 2026-09-26: A `bf16` model on an `nvfp4` target.
        ("nvfp4", "bf16") |
        ("bf16", "fp8") |
        ("bf16", "nvfp4")
    )
}
