// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: A build-time kernel compilation target: the [`Vendor`] parsed from `HARDWARE.toml`, the [`ComputeTarget`] trait, and its nvcc implementation.
//!
//! No code outside this file uses the module. The kernel build script declares
//! its own `ComputeTarget` trait and vendor implementations in
//! `crates/kernels/build_target.rs`. The runtime side is the `GpuBackend` trait
//! in metrale-gpu-runtime.
//!
//! Owner: core.
//! Invariants: none beyond the types.

use std::path::{Path, PathBuf};

/// 2026-09-25: Vendor identifier parsed from a `HARDWARE.toml` `vendor` string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Vendor {
    /// 2026-09-25: `nvidia` or `cuda`; [`NvidiaTarget`] compiles for it.
    Nvidia,
    /// 2026-09-25: `amd`, `rocm` or `hip`; no target in this module.
    Amd,
    /// 2026-09-25: `apple` or `metal`; no target in this module.
    Apple,
    /// 2026-09-25: `intel`, `oneapi` or `sycl`; no target in this module.
    Intel,
}

impl Vendor {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "nvidia" | "cuda" => Some(Self::Nvidia),
            "amd" | "rocm" | "hip" => Some(Self::Amd),
            "apple" | "metal" => Some(Self::Apple),
            "intel" | "oneapi" | "sycl" => Some(Self::Intel),
            _ => None,
        }
    }
}

impl std::fmt::Display for Vendor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nvidia => write!(f, "nvidia"),
            Self::Amd => write!(f, "amd"),
            Self::Apple => write!(f, "apple"),
            Self::Intel => write!(f, "intel"),
        }
    }
}

/// 2026-09-25: How one vendor compiles kernel source files into loadable
/// modules. [`NvidiaTarget`] is the only implementation.
///
/// The diagram shows the flow of the kernel build, which uses its own trait of
/// the same name in `crates/kernels/build_target.rs`, not this one.
///
/// ```text
/// [build.rs]                                [runtime]
///
/// .cu / .metal / .cl                        GpuBackend::new(modules)
///        │                                         │
///        ▼                                         ▼
///  ComputeTarget::compile()              GpuBackend::kernel(name, fn)
///        │                                         │
///        ▼                                         ▼
///   .ptx / .metallib / .spv              GpuBackend::launch(handle, ...)
///        │
///        ▼
///  include_str!() / include_bytes!()
///        │
///        ▼
///  Embedded in binary as &str / &[u8]
/// ```
pub trait ComputeTarget {
    /// 2026-09-25: File extension of kernel source files, without the dot
    /// (`"cu"` for [`NvidiaTarget`]).
    fn source_extension(&self) -> &str;

    /// 2026-09-25: File extension of compiled modules, without the dot
    /// (`"ptx"` for [`NvidiaTarget`]).
    fn output_extension(&self) -> &str;

    /// 2026-09-25: Whether the compiled output is text (`include_str!` can
    /// embed it) or binary (`include_bytes!` only).
    fn output_is_text(&self) -> bool;

    /// 2026-09-25: The compiler executable, or `None` if it is not installed.
    fn find_compiler(&self) -> Option<PathBuf>;

    /// 2026-09-25: Compile one kernel source file.
    ///
    /// - `source`: the kernel source file;
    /// - `output`: where to write the compiled module;
    /// - `arch`: the target architecture string (e.g. `"sm_121f"`);
    /// - `extra_flags`: appended to the compiler arguments.
    ///
    /// Returns `Err` with a message if the compiler cannot be started or
    /// fails; for [`NvidiaTarget`] the message carries nvcc's stderr, and a
    /// path that is not UTF-8 panics.
    fn compile(
        &self,
        source: &Path,
        output: &Path,
        arch: &str,
        extra_flags: &[String],
    ) -> Result<(), String>;

    fn vendor(&self) -> Vendor;
}

/// 2026-09-25: NVIDIA compilation target: `.cu` to PTX with
/// `nvcc --ptx -arch=<arch> -O3`.
pub struct NvidiaTarget {
    nvcc_path: PathBuf,
}

impl NvidiaTarget {
    /// 2026-09-25: Locate nvcc (see `find_nvcc`); `None` if it is not found.
    pub fn new() -> Option<Self> {
        let nvcc = find_nvcc()?;
        Some(Self { nvcc_path: nvcc })
    }

    pub fn with_compiler(nvcc_path: PathBuf) -> Self {
        Self { nvcc_path }
    }
}

impl ComputeTarget for NvidiaTarget {
    fn source_extension(&self) -> &str {
        "cu"
    }

    fn output_extension(&self) -> &str {
        "ptx"
    }

    fn output_is_text(&self) -> bool {
        true
    }

    fn find_compiler(&self) -> Option<PathBuf> {
        Some(self.nvcc_path.clone())
    }

    fn compile(
        &self,
        source: &Path,
        output: &Path,
        arch: &str,
        extra_flags: &[String],
    ) -> Result<(), String> {
        let arch_flag = format!("-arch={arch}");
        let mut args = vec!["--ptx".to_string(), arch_flag, "-O3".to_string()];
        args.extend(extra_flags.iter().cloned());
        args.push(source.to_str().unwrap().to_string());
        args.push("-o".to_string());
        args.push(output.to_str().unwrap().to_string());

        let result = std::process::Command::new(&self.nvcc_path)
            .args(&args)
            .output()
            .map_err(|e| format!("Failed to run nvcc: {e}"))?;

        if result.status.success() {
            Ok(())
        } else {
            let stderr = String::from_utf8_lossy(&result.stderr);
            Err(format!(
                "nvcc --ptx failed for {}: {}",
                source.display(),
                stderr
            ))
        }
    }

    fn vendor(&self) -> Vendor {
        Vendor::Nvidia
    }
}

/// 2026-09-25: Locate nvcc: `bin/nvcc` under `CUDA_HOME`, `CUDA_PATH` or
/// `CUDA_ROOT`, then four fixed install paths, then `PATH`.
fn find_nvcc() -> Option<PathBuf> {
    for var in ["CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"] {
        if let Ok(dir) = std::env::var(var) {
            let nvcc = PathBuf::from(dir).join("bin/nvcc");
            if nvcc.exists() {
                return Some(nvcc);
            }
        }
    }
    for path in [
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-13.0/bin/nvcc",
        "/usr/local/cuda-12.0/bin/nvcc",
        "/opt/cuda/bin/nvcc",
    ] {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }
    which_in_path("nvcc")
}

fn which_in_path(name: &str) -> Option<PathBuf> {
    std::env::var_os("PATH").and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|dir| dir.join(name))
            .find(|p| p.exists())
    })
}

/// 2026-09-25: Resolve the [`ComputeTarget`] for a `HARDWARE.toml` vendor
/// string. A missing or unrecognised vendor resolves to [`NvidiaTarget`].
///
/// Panics if nvcc is not found, or if the vendor is AMD, Apple or Intel.
pub fn target_for_vendor(vendor: Option<&str>) -> Box<dyn ComputeTarget> {
    match vendor.and_then(Vendor::parse) {
        Some(Vendor::Nvidia) | None => {
            Box::new(NvidiaTarget::new().expect("nvcc not found — install CUDA toolkit"))
        }
        Some(v) => {
            panic!("Compute target '{v}' is not yet implemented. Only 'nvidia' is supported.")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vendor_parse_accepts_supported_names() {
        assert_eq!(Vendor::parse("nvidia"), Some(Vendor::Nvidia));
        assert_eq!(Vendor::parse("CUDA"), Some(Vendor::Nvidia));
        assert_eq!(Vendor::parse("amd"), Some(Vendor::Amd));
        assert_eq!(Vendor::parse("rocm"), Some(Vendor::Amd));
        assert_eq!(Vendor::parse("hip"), Some(Vendor::Amd));
        assert_eq!(Vendor::parse("apple"), Some(Vendor::Apple));
        assert_eq!(Vendor::parse("metal"), Some(Vendor::Apple));
        assert_eq!(Vendor::parse("intel"), Some(Vendor::Intel));
        assert_eq!(Vendor::parse("oneapi"), Some(Vendor::Intel));
        assert_eq!(Vendor::parse("sycl"), Some(Vendor::Intel));
        assert_eq!(Vendor::parse("unknown"), None);
    }

    #[test]
    fn nvidia_target_metadata() {
        let target = NvidiaTarget::with_compiler(PathBuf::from("nvcc"));
        assert_eq!(target.source_extension(), "cu");
        assert_eq!(target.output_extension(), "ptx");
        assert!(target.output_is_text());
        assert_eq!(target.vendor(), Vendor::Nvidia);
    }
}
