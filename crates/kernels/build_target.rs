// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The kernel compilers `build.rs` can drive, behind the
//! `ComputeTarget` trait, and `resolve_compute_target`, which picks one from
//! the HARDWARE.toml vendor.
//!
//! Owner: kernels build script.
//! Invariants: none beyond the types.

use std::path::PathBuf;
use std::process::Command;

use super::build_codegen::find_cuda_dir;

/// 2026-09-25: A build-time kernel compiler and its output format:
/// NVIDIA (nvcc to PTX text), Apple (xcrun to metallib), SCALE (SCALE's nvcc
/// to an AMD GPU ELF relocatable) and native HIP (hipcc to a code object).
pub(super) trait ComputeTarget: Send + Sync {
    fn source_extension(&self) -> &str;
    fn output_extension(&self) -> &str;
    /// 2026-09-25: Whether the runtime loads this backend's kernels through
    /// the CUDA module API (`cuModuleLoadData`), so the codegen emits the real
    /// `all_ptx_sets()` registry. True for NVIDIA, SCALE and HIP; false for
    /// Metal, which gets a stub.
    fn uses_cuda_module_api(&self) -> bool;
    fn compile(
        &self,
        source: &std::path::Path,
        output: &std::path::Path,
        arch: &str,
        extra_flags: &[String],
    ) -> Result<(), String>;

    /// 2026-09-25: Identity of the compiler that will emit the device code,
    /// hashed into the per-target closure attestation.
    ///
    /// `None` when it cannot be determined; `closure_attestation` in
    /// `build.rs` then attests to nothing (`{}`) rather than hash a
    /// placeholder that two toolchains would share.
    fn compiler_id(&self) -> Option<String>;
}

/// 2026-09-25: The last non-empty stdout line of `<binary> <args>`, with
/// whitespace runs collapsed to one space. `None` when the command cannot run
/// or exits non-zero.
fn compiler_version(bin: &std::path::Path, args: &[&str]) -> Option<String> {
    let out = Command::new(bin).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text.lines().rev().find(|l| !l.trim().is_empty())?;
    Some(line.split_whitespace().collect::<Vec<_>>().join(" "))
}

/// 2026-09-25: cudafe diagnostic numbers the strict nvcc compile passes to
/// `--diag_suppress`. Each entry needs a comment giving its reason.
const KERNEL_STRICT_DIAG_SUPPRESS: &[u32] = &[
    // 2026-09-25: Empty: no diagnostic is suppressed.
];

struct NvidiaTarget {
    nvcc: PathBuf,
}

impl ComputeTarget for NvidiaTarget {
    fn compiler_id(&self) -> Option<String> {
        compiler_version(&self.nvcc, &["--version"])
    }

    fn source_extension(&self) -> &str {
        "cu"
    }
    fn output_extension(&self) -> &str {
        "ptx"
    }
    fn uses_cuda_module_api(&self) -> bool {
        true
    }

    fn compile(
        &self,
        source: &std::path::Path,
        output: &std::path::Path,
        arch: &str,
        extra_flags: &[String],
    ) -> Result<(), String> {
        let mut args = vec!["--ptx".into(), format!("-arch={arch}"), "-O3".into()];
        // 2026-09-25: A Windows target gets `-std=c++17` explicitly; other
        // targets get no `-std` flag.
        if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
            args.push("-std=c++17".into());
        }
        args.extend(extra_flags.iter().cloned());
        // 2026-09-25: Strict compile: every nvcc warning is an error
        // (`--Werror all-warnings`), less the `KERNEL_STRICT_DIAG_SUPPRESS`
        // entries. Only `METRALE_KERNEL_NO_STRICT=1` turns it off; any other
        // value, `0` included, keeps it on.
        if std::env::var("METRALE_KERNEL_NO_STRICT").as_deref() != Ok("1") {
            args.push("--Werror".into());
            args.push("all-warnings".into());
            for num in KERNEL_STRICT_DIAG_SUPPRESS {
                args.push("-Xcudafe".into());
                args.push(format!("--diag_suppress={num}"));
            }
        }
        // 2026-09-25: `METRALE_EXTRA_NVCC_FLAGS`: whitespace-separated nvcc
        // arguments appended to every NVIDIA kernel compile, after the
        // KERNEL.toml flags (for example `-D<MACRO>=1` to flip an `#ifdef` path).
        if let Ok(s) = std::env::var("METRALE_EXTRA_NVCC_FLAGS") {
            for tok in s.split_whitespace() {
                args.push(tok.to_string());
            }
        }
        args.push(source.to_str().unwrap().into());
        args.push("-o".into());
        args.push(output.to_str().unwrap().into());

        let status = Command::new(&self.nvcc)
            .args(&args)
            .status()
            .map_err(|e| format!("Failed to run nvcc: {e}"))?;
        if status.success() {
            Ok(())
        } else {
            Err(format!("nvcc --ptx failed for {}", source.display()))
        }
    }
}

/// 2026-09-25: Apple Metal: `.metal` to AIR with `xcrun -sdk macosx metal -c`,
/// then AIR to `.metallib` with `xcrun -sdk macosx metallib`.
struct AppleTarget {
    xcrun: PathBuf,
}

impl ComputeTarget for AppleTarget {
    fn compiler_id(&self) -> Option<String> {
        compiler_version(&self.xcrun, &["metal", "--version"])
    }

    fn source_extension(&self) -> &str {
        "metal"
    }
    fn output_extension(&self) -> &str {
        "metallib"
    }
    fn uses_cuda_module_api(&self) -> bool {
        false
    }

    fn compile(
        &self,
        source: &std::path::Path,
        output: &std::path::Path,
        arch: &str,
        extra_flags: &[String],
    ) -> Result<(), String> {
        // 2026-09-25: The intermediate AIR file is written next to the output,
        // with the extension changed to `air`.
        let air_path = output.with_extension("air");

        let mut metal_args: Vec<String> = vec!["-sdk".into(), "macosx".into(), "metal".into()];
        // 2026-09-25: A non-empty `arch` is passed as `-std=<arch>`.
        if !arch.is_empty() {
            metal_args.push(format!("-std={arch}"));
        }
        metal_args.push("-c".into());
        metal_args.push("-O3".into());
        metal_args.extend(extra_flags.iter().cloned());
        metal_args.push(source.to_str().unwrap().into());
        metal_args.push("-o".into());
        metal_args.push(air_path.to_str().unwrap().into());

        let status = Command::new(&self.xcrun)
            .args(&metal_args)
            .status()
            .map_err(|e| format!("Failed to run xcrun metal: {e}"))?;
        if !status.success() {
            return Err(format!(
                "xcrun metal compile failed for {}",
                source.display()
            ));
        }

        let metallib_args: Vec<&str> = vec![
            "-sdk",
            "macosx",
            "metallib",
            air_path.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
        ];
        let status = Command::new(&self.xcrun)
            .args(&metallib_args)
            .status()
            .map_err(|e| format!("Failed to run xcrun metallib: {e}"))?;
        if !status.success() {
            return Err(format!(
                "xcrun metallib link failed for {}",
                source.display()
            ));
        }
        Ok(())
    }
}

/// 2026-09-25: SCALE: compiles the CUDA `.cu` sources for an AMD GPU with the
/// per-arch SCALE compiler `<scale_root>/targets/<arch>/bin/nvcc`, emitting
/// an AMD GPU ELF relocatable (`--cuda-device-only -c`) rather than PTX.
struct ScaleTarget {
    /// 2026-09-25: SCALE install root, the directory holding `targets/`
    /// (`build_codegen::find_scale_dir`).
    scale_root: PathBuf,
}

impl ComputeTarget for ScaleTarget {
    /// 2026-09-25: Always `None`: SCALE's compiler lives under a per-arch
    /// directory this method is not given, and the install root is a path,
    /// not a version. SCALE targets therefore carry no attestation.
    fn compiler_id(&self) -> Option<String> {
        None
    }

    fn source_extension(&self) -> &str {
        "cu"
    }
    fn output_extension(&self) -> &str {
        "o"
    }
    fn uses_cuda_module_api(&self) -> bool {
        true
    }

    fn compile(
        &self,
        source: &std::path::Path,
        output: &std::path::Path,
        arch: &str,
        extra_flags: &[String],
    ) -> Result<(), String> {
        let nvcc = self.scale_root.join("targets").join(arch).join("bin/nvcc");
        if !nvcc.exists() {
            return Err(format!(
                "SCALE arch toolchain not found: {} — `{}` is not a SCALE \
                 target (check kernels/<hw>/HARDWARE.toml `arch` and the \
                 installed SCALE `targets/` dir).",
                nvcc.display(),
                arch
            ));
        }

        let mut args: Vec<String> = vec!["--cuda-device-only".into(), "-c".into(), "-O3".into()];
        args.extend(extra_flags.iter().cloned());
        args.push(source.to_str().unwrap().into());
        args.push("-o".into());
        args.push(output.to_str().unwrap().into());

        let status = Command::new(&nvcc)
            .args(&args)
            .status()
            .map_err(|e| format!("Failed to run SCALE nvcc ({}): {e}", nvcc.display()))?;
        if status.success() {
            Ok(())
        } else {
            Err(format!(
                "SCALE `--cuda-device-only -c` failed for {} (arch {arch})",
                source.display()
            ))
        }
    }
}

/// 2026-09-25: Native HIP: compiles the CUDA `.cu` sources with `hipcc` for an
/// AMD `gfx*` arch into a HIP code object (`--genco`). The runtime still calls
/// the CUDA driver API; `hip/libcuda_hip_shim.cpp` maps `cuModuleLoadData` to
/// `hipModuleLoadData`. `build.rs` exports the `hip/compat` header directory as
/// `METRALE_HIP_COMPAT_INCLUDE` and compiles a mask-widened mirror of each
/// source.
struct HipTarget {
    hipcc: PathBuf,
}

impl ComputeTarget for HipTarget {
    fn compiler_id(&self) -> Option<String> {
        compiler_version(&self.hipcc, &["--version"])
    }

    fn source_extension(&self) -> &str {
        "cu"
    }
    fn output_extension(&self) -> &str {
        "co"
    }
    fn uses_cuda_module_api(&self) -> bool {
        true
    }

    fn compile(
        &self,
        source: &std::path::Path,
        output: &std::path::Path,
        arch: &str,
        extra_flags: &[String],
    ) -> Result<(), String> {
        // 2026-09-25: The CUDA-to-HIP compat headers (`hip/compat`), passed as
        // the first `-I`.
        let compat = std::env::var("METRALE_HIP_COMPAT_INCLUDE").map_err(|_| {
            "METRALE_HIP_COMPAT_INCLUDE not set — build.rs must stage the CUDA→HIP \
             compat-header dir before compiling HIP kernels."
                .to_string()
        })?;
        let mut args: Vec<String> = vec![
            "-x".into(),
            "hip".into(),
            "--genco".into(),
            format!("--offload-arch={arch}"),
            "-O3".into(),
            format!("-I{compat}"),
            "-include".into(),
            "hip/hip_runtime.h".into(),
        ];
        // 2026-09-25: On a Windows host, also force-include
        // `metrale_hip_win_shims.h` (found through `-I{compat}`) after
        // `hip/hip_runtime.h`, whose `__shfl*`/`__ballot` it wraps.
        if cfg!(windows) {
            args.push("-include".into());
            args.push("metrale_hip_win_shims.h".into());
        }
        // 2026-09-25: KERNEL.toml flags are written for nvcc: `--fmad=false`
        // and `--fmad=true` become `-ffp-contract=off` and `-ffp-contract=fast`,
        // any other `--fmad=` is dropped, and the rest pass through.
        for f in extra_flags {
            match f.as_str() {
                "--fmad=false" => args.push("-ffp-contract=off".into()),
                "--fmad=true" => args.push("-ffp-contract=fast".into()),
                s if s.starts_with("--fmad=") => {}
                other => args.push(other.into()),
            }
        }
        args.push(source.to_str().unwrap().into());
        args.push("-o".into());
        args.push(output.to_str().unwrap().into());

        let status = Command::new(&self.hipcc)
            .args(&args)
            .status()
            .map_err(|e| format!("Failed to run hipcc ({}): {e}", self.hipcc.display()))?;
        if status.success() {
            Ok(())
        } else {
            Err(format!(
                "hipcc --genco failed for {} (arch {arch})",
                source.display()
            ))
        }
    }
}

fn find_hipcc() -> PathBuf {
    if let Ok(p) = std::env::var("METRALE_HIPCC") {
        return PathBuf::from(p);
    }
    let canonical = PathBuf::from("/opt/rocm/bin/hipcc");
    if canonical.exists() {
        return canonical;
    }
    if let Some(path_var) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path_var) {
            let p = dir.join("hipcc");
            if p.exists() {
                return p;
            }
        }
    }
    panic!("hipcc not found — install ROCm or set METRALE_HIPCC to its path.");
}

fn find_xcrun() -> PathBuf {
    let canonical = PathBuf::from("/usr/bin/xcrun");
    if canonical.exists() {
        return canonical;
    }
    if let Some(path_var) = std::env::var_os("PATH") {
        for dir in std::env::split_paths(&path_var) {
            let p = dir.join("xcrun");
            if p.exists() {
                return p;
            }
        }
    }
    panic!(
        "xcrun not found — install Xcode Command Line Tools \
         (xcode-select --install) or set PATH to a directory containing xcrun."
    );
}

/// 2026-09-25: The compiler for a HARDWARE.toml `vendor`; no vendor means
/// NVIDIA, and an unknown vendor panics.
pub(super) fn resolve_compute_target(vendor: Option<&str>) -> Box<dyn ComputeTarget> {
    match vendor.unwrap_or("nvidia") {
        "nvidia" | "cuda" => {
            let nvcc = find_cuda_dir().join("bin/nvcc");
            Box::new(NvidiaTarget { nvcc })
        }
        "apple" | "metal" => {
            let xcrun = find_xcrun();
            Box::new(AppleTarget { xcrun })
        }
        "amd" | "rocm" | "scale" => {
            let scale_root = super::build_codegen::find_scale_dir();
            Box::new(ScaleTarget { scale_root })
        }
        "hip" => Box::new(HipTarget {
            hipcc: find_hipcc(),
        }),
        other => panic!(
            "Unsupported compute vendor '{other}'. Supported: nvidia, apple, amd, hip.\n\
             To add support for a new vendor, implement the ComputeTarget trait \n\
             in metrale-kernels/build_target.rs and metrale-core/src/compute.rs."
        ),
    }
}
