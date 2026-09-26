# metrale-config

**Path:** `crates/config/`

The typed model config tree: Hugging Face `config.json` parsing, GGUF metadata and hardware capabilities. `ModelConfig` (`src/lib.rs`) is the single source of truth for model shape, and the per-family parsers live under `src/parsers/`.
