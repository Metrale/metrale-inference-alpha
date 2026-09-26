// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of `serve` command-line parsing and of the per-layer KV
//! dtype helpers in `kv_dtypes.rs`.
//!
//! Owner: server startup (`met serve`).
//! Invariants: none beyond the types.

use clap::Parser;

use crate::cli::{Cli, Command};
use crate::main_modules::build_layer_kv_dtypes;

#[test]
fn test_cli_parse_positional_model() {
    let cli = Cli::try_parse_from([
        "met",
        "serve",
        "nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4",
        "--port",
        "9999",
        "--max-seq-len",
        "8192",
    ]);
    assert!(cli.is_ok());
    match cli.unwrap().command {
        Command::Benchmark(_)
        | Command::DumpServeOptions
        | Command::SyncRecipes
        | Command::Doctor => {
            unreachable!("this test parses a serve command")
        }
        Command::Serve(args) => {
            assert_eq!(
                args.model.as_deref(),
                Some("nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4"),
            );
            assert!(args.model_from_path.is_none());
            assert_eq!(args.port, 9999);
            assert_eq!(args.max_seq_len, 8192);
            assert_eq!(args.gpu_memory_utilization, 0.90);
            assert_eq!(args.scheduler, "fifo");
            assert_eq!(args.scheduler_config, "sync");
            assert_eq!(args.tbt_deadline_ms, 100);
        }
    }
}

#[test]
fn test_cli_parse_model_from_path() {
    let cli = Cli::try_parse_from([
        "met",
        "serve",
        "--model-from-path",
        "/tmp/model",
        "--port",
        "8888",
    ]);
    assert!(cli.is_ok());
    match cli.unwrap().command {
        Command::Benchmark(_)
        | Command::DumpServeOptions
        | Command::SyncRecipes
        | Command::Doctor => {
            unreachable!("this test parses a serve command")
        }
        Command::Serve(args) => {
            assert!(args.model.is_none());
            assert_eq!(
                args.model_from_path,
                Some(std::path::PathBuf::from("/tmp/model")),
            );
        }
    }
}

#[test]
fn test_cli_parse_slai_policy() {
    let cli = Cli::try_parse_from([
        "met",
        "serve",
        "nvidia/model",
        "--scheduler",
        "slai",
        "--tbt-deadline-ms",
        "50",
    ]);
    assert!(cli.is_ok());
    match cli.unwrap().command {
        Command::Benchmark(_)
        | Command::DumpServeOptions
        | Command::SyncRecipes
        | Command::Doctor => {
            unreachable!("this test parses a serve command")
        }
        Command::Serve(args) => {
            assert_eq!(args.scheduler, "slai");
            assert_eq!(args.tbt_deadline_ms, 50);
        }
    }
}

#[test]
fn test_build_layer_kv_dtypes_disabled() {
    let dtypes = build_layer_kv_dtypes(
        metrale_cache::kv_cache::KvCacheDtype::Nvfp4,
        12,
        0,
        metrale_cache::kv_cache::KvCacheDtype::Bf16,
    );
    assert!(dtypes.is_empty());
}

#[test]
fn test_build_layer_kv_dtypes_bf16_noop() {
    // 2026-09-26: A base dtype equal to the boundary dtype needs no overlay.
    let dtypes = build_layer_kv_dtypes(
        metrale_cache::kv_cache::KvCacheDtype::Bf16,
        12,
        2,
        metrale_cache::kv_cache::KvCacheDtype::Bf16,
    );
    assert!(dtypes.is_empty());
}

#[test]
fn test_build_layer_kv_dtypes_basic() {
    use metrale_cache::kv_cache::KvCacheDtype;
    let dtypes = build_layer_kv_dtypes(
        KvCacheDtype::Nvfp4,
        12,
        2,
        metrale_cache::kv_cache::KvCacheDtype::Bf16,
    );
    assert_eq!(dtypes.len(), 12);
    assert_eq!(dtypes[0], KvCacheDtype::Bf16);
    assert_eq!(dtypes[1], KvCacheDtype::Bf16);
    for i in 2..10 {
        assert_eq!(dtypes[i], KvCacheDtype::Nvfp4, "layer {i}");
    }
    assert_eq!(dtypes[10], KvCacheDtype::Bf16);
    assert_eq!(dtypes[11], KvCacheDtype::Bf16);
}

#[test]
fn test_build_layer_kv_dtypes_overlap() {
    use metrale_cache::kv_cache::KvCacheDtype;
    // 2026-09-26: With 4 layers, the first 3 and last 3 cover them all.
    let dtypes = build_layer_kv_dtypes(
        KvCacheDtype::Fp8,
        4,
        3,
        metrale_cache::kv_cache::KvCacheDtype::Bf16,
    );
    assert_eq!(dtypes.len(), 4);
    for d in &dtypes {
        assert_eq!(*d, KvCacheDtype::Bf16);
    }
}

#[test]
fn test_build_layer_kv_dtypes_single_layer() {
    use metrale_cache::kv_cache::KvCacheDtype;
    let dtypes = build_layer_kv_dtypes(
        KvCacheDtype::Nvfp4,
        1,
        1,
        metrale_cache::kv_cache::KvCacheDtype::Bf16,
    );
    assert_eq!(dtypes.len(), 1);
    assert_eq!(dtypes[0], KvCacheDtype::Bf16);
}

#[test]
fn test_auto_high_precision_layers_non_turbo_none() {
    use metrale_cache::kv_cache::KvCacheDtype;
    for d in [KvCacheDtype::Bf16, KvCacheDtype::Fp8, KvCacheDtype::Nvfp4] {
        assert_eq!(crate::main_modules::auto_high_precision_layers(d, 10), None);
    }
}

#[test]
fn test_auto_high_precision_layers_baseline_formula() {
    use metrale_cache::kv_cache::KvCacheDtype;
    // 2026-09-26: `ceil(n / 3)`, at least 2.
    assert_eq!(
        crate::main_modules::auto_high_precision_layers(KvCacheDtype::Turbo8, 10),
        Some(4),
    );
    assert_eq!(
        crate::main_modules::auto_high_precision_layers(KvCacheDtype::Turbo4KTurbo8V, 3),
        Some(2),
    );
}

#[test]
fn test_auto_high_precision_layers_weak_dtypes_stronger_default() {
    use metrale_cache::kv_cache::KvCacheDtype;
    // 2026-09-26: `Turbo2` and `Bf16KTurbo3V` take `ceil(4n / 5)`, at least 4.
    for d in [KvCacheDtype::Turbo2, KvCacheDtype::Bf16KTurbo3V] {
        assert_eq!(
            crate::main_modules::auto_high_precision_layers(d, 10),
            Some(8)
        );
        assert_eq!(
            crate::main_modules::auto_high_precision_layers(d, 2),
            Some(4)
        );
    }
}

#[test]
fn test_auto_high_precision_layers_every_turbo_dtype_covered() {
    use metrale_cache::kv_cache::KvCacheDtype;
    // 2026-09-26: Each turbo dtype listed gets at least 2 boundary layers;
    // only `Bf16`, `Fp8` and `Nvfp4` return `None`.
    for d in [
        KvCacheDtype::Turbo2,
        KvCacheDtype::Turbo3,
        KvCacheDtype::Turbo4,
        KvCacheDtype::Turbo8,
        KvCacheDtype::Turbo4KTurbo3V,
        KvCacheDtype::Turbo4KTurbo8V,
        KvCacheDtype::Turbo3KTurbo8V,
        KvCacheDtype::Bf16KTurbo4V,
        KvCacheDtype::Bf16KTurbo3V,
        KvCacheDtype::Fp8KTurbo4V,
        KvCacheDtype::Fp8KTurbo3V,
        KvCacheDtype::Bf16KTurbo2V,
        KvCacheDtype::Fp8KTurbo2V,
    ] {
        assert!(
            crate::main_modules::auto_high_precision_layers(d, 10).unwrap_or(0) >= 2,
            "{d:?} must auto-enable high-precision boundary layers",
        );
    }
}

#[test]
fn test_cli_parse_kv_high_precision_layers() {
    let cli = Cli::try_parse_from([
        "met",
        "serve",
        "nvidia/model",
        "--kv-high-precision-layers",
        "3",
    ]);
    assert!(cli.is_ok());
    match cli.unwrap().command {
        Command::Benchmark(_)
        | Command::DumpServeOptions
        | Command::SyncRecipes
        | Command::Doctor => {
            unreachable!("this test parses a serve command")
        }
        Command::Serve(args) => {
            assert_eq!(args.kv_high_precision_layers, "3");
        }
    }
}

#[test]
fn test_cli_default_kv_high_precision_layers() {
    let cli = Cli::try_parse_from(["met", "serve", "nvidia/model"]);
    assert!(cli.is_ok());
    match cli.unwrap().command {
        Command::Benchmark(_)
        | Command::DumpServeOptions
        | Command::SyncRecipes
        | Command::Doctor => {
            unreachable!("this test parses a serve command")
        }
        Command::Serve(args) => {
            assert_eq!(args.kv_high_precision_layers, "0");
        }
    }
}
