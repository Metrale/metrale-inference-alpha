# API Reference

<script>window.location.replace("https://docs.dev.metrale.ai/");</script>
<noscript>
<meta http-equiv="refresh" content="0; url=https://docs.dev.metrale.ai/">
</noscript>

If you are not redirected automatically, follow this link to the [Rust API reference](https://docs.dev.metrale.ai/).

The API reference is generated from the crate source with `cargo doc --workspace --no-deps` on every merge to `main`. Top-level crates:

- [`metrale_core`](https://docs.dev.metrale.ai/metrale_core/) — target abstractions, tensor, dtype, kernel registry, host-side FP8/BF16 numerics
- [`metrale_kernels`](https://docs.dev.metrale.ai/metrale_kernels/) — embedded PTX registry
- [`spark_runtime`](https://docs.dev.metrale.ai/spark_runtime/) — GPU backend, KV cache, sampler
- [`spark_comm`](https://docs.dev.metrale.ai/spark_comm/) — collective-op trait + NCCL impl
- [`spark_model`](https://docs.dev.metrale.ai/spark_model/) — layer assembly, weight loaders, engine
- [`spark_server`](https://docs.dev.metrale.ai/spark_server/) — HTTP server, tool parsing
- [`metrale_spark_bench`](https://docs.dev.metrale.ai/metrale_spark_bench/) — benchmark client
