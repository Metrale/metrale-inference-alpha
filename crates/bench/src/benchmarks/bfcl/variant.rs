// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The inherent impl of [`Variant`]: its descriptor and plugin
//! metadata, its default draw percentages and floor, its [`DrawSpec`], and the
//! sample count its draw is pinned to. The enum is declared in `mod.rs`.
//!
//! Owner: bench, BFCL benchmark.
//! Invariants: none beyond the types.

use super::*;

impl Variant {
    pub(super) fn descriptor(self) -> &'static BenchmarkDescriptor {
        match self {
            Variant::Subset => &SUBSET_DESCRIPTOR,
            Variant::SubsetEcholp => &SUBSET_ECHOLP_DESCRIPTOR,
            Variant::Full => &FULL_DESCRIPTOR,
        }
    }
    pub(super) fn metadata(self) -> &'static PluginMetadata {
        match self {
            Variant::Subset => &SUBSET_METADATA,
            Variant::SubsetEcholp => &ECHOLP_METADATA,
            Variant::Full => &FULL_METADATA,
        }
    }
    pub(super) fn default_pct(self, category: &str) -> f64 {
        match (self, category) {
            (Variant::Full, _) => 100.0,
            (Variant::Subset, "non_live") => 62.0,
            (Variant::Subset, _) => 10.0,
            (Variant::SubsetEcholp, "non_live") => 46.0,
            (Variant::SubsetEcholp, "live") => 23.0,
            (Variant::SubsetEcholp, _) => 12.0,
        }
    }
    /// 2026-09-26: The subset floor, read from the variant's own [`DrawSpec`]
    /// rather than written out again: `configure` rebuilds the spec from the
    /// parameter defaults, so a second copy here would decide the draw.
    pub(super) fn default_floor(self) -> usize {
        self.spec().subset_floor.unwrap_or(0)
    }

    /// 2026-09-26: The draw this variant is defined by. The constructor uses it
    /// directly and the `subset_floor` parameter default reads it through
    /// `default_floor`.
    pub(super) fn spec(self) -> DrawSpec {
        match self {
            Variant::Subset => DrawSpec::golden(),
            Variant::SubsetEcholp => DrawSpec::echolp(),
            Variant::Full => DrawSpec::full(),
        }
    }

    /// 2026-09-26: The whole-draw sample count this variant is pinned to, or
    /// `None` when unpinned. A run that produces a different count logs a
    /// warning that it is not comparable to the draw's baseline.
    pub(super) fn expected_samples(self) -> Option<usize> {
        match self {
            Variant::Subset => Some(995),
            Variant::SubsetEcholp => Some(1004),
            Variant::Full => None,
        }
    }
}
