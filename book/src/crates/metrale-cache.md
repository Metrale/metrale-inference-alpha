# metrale-cache

**Path:** `crates/cache/`

KV cache, KV dequant, KV spill and the radix-tree prefix cache. `PagedKvCache` (`src/kv_cache.rs`) owns the paged block pool and `RadixTree` (`src/radix_tree.rs`) indexes cached prefixes by token sequence; see [metrale-gpu-runtime](./metrale-gpu-runtime.md) for how they are used.
