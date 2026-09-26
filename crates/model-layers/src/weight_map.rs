// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Maps HuggingFace safetensors tensor names onto the typed per-layer
//! weight structures, through the `weight_map/` sub-modules re-exported here.
//!
//! Owner: model-layers (weight loading).
//! Invariants: none beyond the types.

#[path = "weight_map/expert.rs"]
mod expert;
#[path = "weight_map/fp8_dequant.rs"]
pub mod fp8_dequant;
#[path = "weight_map/fp8_lut.rs"]
pub mod fp8_lut;
pub(crate) use fp8_lut::dequant_nvfp4_to_bf16;
#[path = "weight_map/loaders_fp8.rs"]
pub mod loaders_fp8;
#[path = "weight_map/loaders_moe.rs"]
pub mod loaders_moe;
#[path = "weight_map/loaders_mtp.rs"]
mod loaders_mtp;
#[path = "weight_map/model_a.rs"]
pub mod model_a;
#[path = "weight_map/model_b.rs"]
mod model_b;
#[path = "weight_map/moe.rs"]
mod moe;
#[path = "weight_map/nemotron.rs"]
pub mod nemotron;
#[path = "weight_map/nvfp4_detect.rs"]
pub mod nvfp4_detect;
#[path = "weight_map/quant_helpers.rs"]
pub mod quant_helpers;
#[path = "weight_map/quantize_fns.rs"]
mod quantize_fns;
#[path = "weight_map/quantize_fp8_bs.rs"]
mod quantize_fp8_bs;
#[path = "weight_map/quantized.rs"]
pub mod quantized;
#[path = "weight_map/ssm_qwen35.rs"]
pub mod ssm_qwen35;
#[path = "weight_map/ssm_qwen35_more.rs"]
pub mod ssm_qwen35_more;

pub use expert::*;
pub(crate) use fp8_dequant::*;
pub use loaders_fp8::*;
pub use loaders_mtp::*;
pub use model_a::*;
pub use moe::*;
pub use nemotron::*;
pub use nvfp4_detect::*;
pub use quantize_fns::*;
pub use quantize_fp8_bs::*;
pub use quantized::*;
pub use ssm_qwen35::*;

#[allow(unused_imports)]
pub(crate) use {fp8_lut::*, loaders_moe::*, model_b::*, quant_helpers::*, ssm_qwen35_more::*};
