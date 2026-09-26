// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Model teardown: ordered, fallible release of state that owns device memory.
//!
//! Owner: metrale-core.
//! Invariants:
//! - `Teardown::release_all` calls `release` on every registered resource,
//!   newest first, even after one fails, and leaves the set empty.
//!
//! [`ModelResource`] and [`Teardown`] give an ordered, fallible release path,
//! which `Drop` cannot: it is neither ordered across independent values nor
//! able to report a failure.

/// 2026-09-25: State that owns device memory and must be released in a defined order.
///
/// `Cx` is whatever releasing needs; for GPU state that is the backend
/// (`dyn GpuBackend`). Making it a type parameter keeps `metrale-core` free of
/// a dependency on the backend crate while still handing each resource the
/// thing that owns its memory.
pub trait ModelResource<Cx: ?Sized>: Send + Sync {
    /// 2026-09-25: Human name, used to attribute a failure in `Teardown::release_all`'s error.
    fn label(&self) -> &'static str;

    /// 2026-09-25: Release everything this owns. Must be idempotent.
    fn release(&mut self, cx: &Cx) -> anyhow::Result<()>;
}

/// 2026-09-25: Releases a set of resources in reverse registration order, the
/// inverse of how they were built, which is the only safe order when later
/// resources borrow earlier ones.
///
/// One failure does not abandon the rest: every resource's release is
/// attempted, and one error naming every failure is returned afterwards.
pub struct Teardown<Cx: ?Sized> {
    resources: Vec<Box<dyn ModelResource<Cx>>>,
}

impl<Cx: ?Sized> Default for Teardown<Cx> {
    fn default() -> Self {
        Self {
            resources: Vec::new(),
        }
    }
}

impl<Cx: ?Sized> Teardown<Cx> {
    pub fn new() -> Self {
        Self::default()
    }

    /// 2026-09-25: Register a resource. Callers register in construction order.
    pub fn push(&mut self, resource: Box<dyn ModelResource<Cx>>) {
        self.resources.push(resource);
    }

    pub fn len(&self) -> usize {
        self.resources.len()
    }

    pub fn is_empty(&self) -> bool {
        self.resources.is_empty()
    }

    /// 2026-09-25: Release everything, newest first. After attempting them all,
    /// returns one error that lists every failure with its resource's label.
    pub fn release_all(&mut self, cx: &Cx) -> anyhow::Result<()> {
        let mut failures = Vec::new();
        while let Some(mut resource) = self.resources.pop() {
            if let Err(e) = resource.release(cx) {
                // 2026-09-25: Every failure is reported, not just the first: after a
                // partial teardown the operator needs the whole picture to
                // decide whether the GPU is still usable.
                failures.push(format!("{}: {e:#}", resource.label()));
            }
        }
        if failures.is_empty() {
            return Ok(());
        }
        Err(anyhow::anyhow!(
            "{} resource(s) failed to release: {}",
            failures.len(),
            failures.join("; ")
        ))
    }
}

#[cfg(test)]
#[path = "scope_tests.rs"]
mod tests;
