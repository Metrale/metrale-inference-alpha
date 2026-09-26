// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: The [`KvCacheDtype`] catalogue: every variant and its canonical
//! `--kv-cache-dtype` spelling.
//!
//! The flag is an `Option<String>` in clap, so the list of valid values comes
//! from here: the server's `options_for_flag` builds it from
//! [`KvCacheDtype::ALL`] for the CLI manifest and the TUI option picker.
//!
//! Owner: cache.
//! Invariants:
//! - The `name` match has no wildcard arm, so a new variant does not compile
//!   until it has a spelling.
//! - Every `ALL` entry's `name` parses back to that entry (tested below).

use super::KvCacheDtype;

impl KvCacheDtype {
    /// 2026-09-25: Every variant, in declaration order. Nothing checks that a
    /// new variant is added here; add it together with its [`KvCacheDtype::name`]
    /// arm, which the compiler does require.
    pub const ALL: [KvCacheDtype; 16] = [
        KvCacheDtype::Bf16,
        KvCacheDtype::Fp8,
        KvCacheDtype::Nvfp4,
        KvCacheDtype::Turbo4,
        KvCacheDtype::Turbo3,
        KvCacheDtype::Turbo2,
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
    ];

    /// 2026-09-25: The canonical `--kv-cache-dtype` spelling. `Display`
    /// delegates here, so a listed value is the string `FromStr` accepts; the
    /// tests below check that round trip.
    pub const fn name(self) -> &'static str {
        match self {
            KvCacheDtype::Bf16 => "bf16",
            KvCacheDtype::Fp8 => "fp8",
            KvCacheDtype::Nvfp4 => "nvfp4",
            KvCacheDtype::Turbo4 => "turbo4",
            KvCacheDtype::Turbo3 => "turbo3",
            KvCacheDtype::Turbo2 => "turbo2",
            KvCacheDtype::Turbo8 => "turbo8",
            KvCacheDtype::Turbo4KTurbo3V => "turbo4k_turbo3v",
            KvCacheDtype::Turbo4KTurbo8V => "turbo4k_turbo8v",
            KvCacheDtype::Turbo3KTurbo8V => "turbo3k_turbo8v",
            KvCacheDtype::Bf16KTurbo4V => "bf16k_turbo4v",
            KvCacheDtype::Bf16KTurbo3V => "bf16k_turbo3v",
            KvCacheDtype::Fp8KTurbo4V => "fp8k_turbo4v",
            KvCacheDtype::Fp8KTurbo3V => "fp8k_turbo3v",
            KvCacheDtype::Bf16KTurbo2V => "bf16k_turbo2v",
            KvCacheDtype::Fp8KTurbo2V => "fp8k_turbo2v",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::KvCacheDtype;

    #[test]
    fn every_listed_name_parses_back_to_the_variant_that_produced_it() {
        // 2026-09-25: Each listed name must parse, to the variant that
        // produced it, and `Display` must print that same name.
        for dtype in KvCacheDtype::ALL {
            let parsed: KvCacheDtype = dtype
                .name()
                .parse()
                .unwrap_or_else(|e| panic!("{} does not parse: {e:#}", dtype.name()));
            assert_eq!(
                parsed,
                dtype,
                "{} parses to a different dtype",
                dtype.name()
            );
            assert_eq!(
                dtype.to_string(),
                dtype.name(),
                "Display and the catalogue disagree"
            );
        }
    }

    #[test]
    fn the_catalogue_has_no_duplicates() {
        // 2026-09-25: A duplicate would show one choice twice in a picker.
        for (i, a) in KvCacheDtype::ALL.iter().enumerate() {
            for b in &KvCacheDtype::ALL[i + 1..] {
                assert_ne!(a, b, "{} is listed twice", a.name());
            }
        }
    }
}
