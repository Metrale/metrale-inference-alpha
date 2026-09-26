// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: [`DynBenchmark`], the object-safe form of [`Benchmark`].
//!
//! [`Benchmark`] and [`Plugin`] return `impl Future` / `impl Stream`, which
//! makes them not dyn-compatible, so descriptors construct
//! `Box<dyn DynBenchmark>` instead. The blanket impl below gives every
//! `Benchmark + Send` this trait with boxed futures. `Benchmark::run` has no
//! counterpart here: callers drive a `DynBenchmark` with `executor::drive`.
//!
//! Owner: bench.
//! Invariants: none beyond the types.

use anyhow::Result;
use futures::future::BoxFuture;

use crate::benchmark::{Benchmark, BenchmarkDescriptor};
use crate::metadata::PluginMetadata;
use crate::params::{ParamSpec, ParamValues};
use crate::plugin::{Plugin, PluginHandle};
use crate::result::BenchmarkResult;

/// 2026-09-26: Object-safe form of [`Benchmark`] + [`Plugin`]; each method
/// forwards to the trait method of the same name.
pub trait DynBenchmark: Send {
    fn descriptor(&self) -> &'static BenchmarkDescriptor;
    fn metadata(&self) -> &'static PluginMetadata;
    fn parameters(&self) -> Vec<ParamSpec>;
    fn configure(&mut self, values: &ParamValues) -> Result<()>;
    fn load<'a>(&'a mut self, handle: PluginHandle) -> BoxFuture<'a, Result<()>>;
    fn next<'a>(&'a mut self) -> BoxFuture<'a, Result<BenchmarkResult>>;
    fn cleanup<'a>(&'a mut self) -> BoxFuture<'a, Result<()>>;
}

impl<T> DynBenchmark for T
where
    T: Benchmark + Send,
{
    fn descriptor(&self) -> &'static BenchmarkDescriptor {
        Benchmark::descriptor(self)
    }
    fn metadata(&self) -> &'static PluginMetadata {
        Plugin::metadata(self)
    }
    fn parameters(&self) -> Vec<ParamSpec> {
        Benchmark::parameters(self)
    }
    fn configure(&mut self, values: &ParamValues) -> Result<()> {
        Benchmark::configure(self, values)
    }
    fn load<'a>(&'a mut self, handle: PluginHandle) -> BoxFuture<'a, Result<()>> {
        Box::pin(Plugin::load(self, handle))
    }
    fn next<'a>(&'a mut self) -> BoxFuture<'a, Result<BenchmarkResult>> {
        Box::pin(Benchmark::next(self))
    }
    fn cleanup<'a>(&'a mut self) -> BoxFuture<'a, Result<()>> {
        Box::pin(Benchmark::cleanup(self))
    }
}
