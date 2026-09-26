// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: GEMM-path selection: [`GemmDispatch`], resolved from `METRALE_*`
//! variables and carried to the dispatch sites on
//! [`crate::layer::ForwardContext`].
//!
//! Owner: model-layers ops.
//! Invariants:
//! - A switch variable counts as on only when its value is exactly `1`;
//!   `METRALE_CUBLAS_GEMM` has its own grammar ([`CublasScope`]) and
//!   `METRALE_W4A16_VARIANT` accepts `v1`, `v2` and `v3`.
//! - Unknown `METRALE_CUBLAS_GEMM` tokens never add a family.

/// 2026-09-25: Which projection families take a cuBLASLt GEMM arm, so one
/// family can be armed without the others.
///
/// Grammar: `METRALE_CUBLAS_GEMM=<token>[,<token>]*`, ASCII-case-insensitive,
/// whitespace around a token ignored:
///
/// | token | meaning |
/// |---|---|
/// | `ffn` | dense-FFN and MoE shared-expert projections |
/// | `attn` | attention Q/K/V, O and the output gate |
/// | `ssm` | GDN `in_proj_qkvz` and `out_proj` |
/// | `head` | LM / MTP head (see [`CublasScope::head`]) |
/// | `all`, `1`, `true` | every family |
/// | `off`, `0`, `false`, empty | the empty set |
///
/// The result is the union of the tokens, so `off` clears nothing (`ffn,off`
/// is `ffn`). Unknown tokens are dropped and reported, and never widen the
/// set.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CublasScope {
    /// 2026-09-25: Dense-FFN gate/up/down and the MoE shared expert.
    pub ffn: bool,
    /// 2026-09-25: Attention Q/K/V, O projection and the output gate.
    pub attn: bool,
    /// 2026-09-25: GDN `in_proj_qkvz` and `out_proj`, in prefill and decode.
    pub ssm: bool,
    /// 2026-09-25: LM / MTP head. Parsed and set by `all`, but no dispatch site
    /// reads it, so setting it changes nothing.
    pub head: bool,
}

impl CublasScope {
    /// 2026-09-25: No family armed: what an absent, empty or `off`
    /// `METRALE_CUBLAS_GEMM` resolves to, and the value in
    /// [`GemmDispatch::defaults`].
    pub const OFF: Self = Self {
        ffn: false,
        attn: false,
        ssm: false,
        head: false,
    };

    /// 2026-09-25: Every family: what `all`, `1` and `true` resolve to.
    pub const ALL: Self = Self {
        ffn: true,
        attn: true,
        ssm: true,
        head: true,
    };

    /// 2026-09-25: Whether any family is armed.
    pub fn any(&self) -> bool {
        self.ffn || self.attn || self.ssm || self.head
    }
}

/// 2026-09-25: Parse the [`CublasScope`] grammar. Returns the resolved set and
/// the lowercased tokens that matched nothing. Pure: the environment read and
/// the logging are in [`GemmDispatch::from_env`].
pub fn parse_cublas_scope(raw: Option<&str>) -> (CublasScope, Vec<String>) {
    let mut scope = CublasScope::OFF;
    let mut unknown = Vec::new();
    let Some(raw) = raw else {
        return (scope, unknown);
    };
    for token in raw.split(',') {
        match token.trim().to_ascii_lowercase().as_str() {
            "" | "0" | "false" | "off" => {}
            "1" | "true" | "all" => scope = CublasScope::ALL,
            "ffn" => scope.ffn = true,
            "attn" => scope.attn = true,
            "ssm" => scope.ssm = true,
            "head" => scope.head = true,
            other => unknown.push(other.to_owned()),
        }
    }
    (scope, unknown)
}

/// 2026-09-25: Which GEMM implementation each projection takes, resolved by
/// [`GemmDispatch::from_env`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemmDispatch {
    /// 2026-09-25: Block-scaled FP8 prefill (per-128-block weight scales and
    /// per-token activation scales). On unless `METRALE_FP8_SINGLE_SCALE=1`.
    pub fp8_blockscaled_prefill: bool,
    /// 2026-09-25: Which projection families take a cuBLASLt GEMM arm
    /// (`METRALE_CUBLAS_GEMM`).
    pub cublas: CublasScope,
    /// 2026-09-25: Row-wise FP8 cuBLASLt GEMM for the GDN `in_proj_qkvz`
    /// prefill (`METRALE_CUBLAS_FP8=1`).
    pub cublas_fp8: bool,
    /// 2026-09-25: CUTLASS BF16 GEMM for the GDN `in_proj_qkvz` prefill, on a
    /// memoized FP8→BF16 dequant of the weight (`METRALE_CUTLASS_GEMM=1`).
    pub cutlass_gemm: bool,
    /// 2026-09-25: CUTLASS NVFP4 GEMM (`METRALE_CUTLASS_NVFP4_GEMM=1`). Implies
    /// the per-projection NVFP4 flags below except `cutlass_nvfp4_ssm_out`.
    pub cutlass_nvfp4_gemm: bool,
    pub cutlass_nvfp4_qkvz: bool,
    pub cutlass_nvfp4_attn_q: bool,
    pub cutlass_nvfp4_attn_kv: bool,
    pub cutlass_nvfp4_attn_o: bool,
    pub cutlass_nvfp4_ssm_out: bool,
    /// 2026-09-25: `METRALE_W4A16_VARIANT`: `v1`, `v2` and `v3` give 1, 2 and 3;
    /// any other value, or none, gives 0, which the attention prefill treats
    /// like 2.
    pub w4a16_variant: u8,
}

fn from_values(mut value: impl FnMut(&str) -> Option<String>) -> GemmDispatch {
    fn on(value: &mut impl FnMut(&str) -> Option<String>, var: &str) -> bool {
        value(var).as_deref() == Some("1")
    }

    let all_nvfp4 = on(&mut value, "METRALE_CUTLASS_NVFP4_GEMM");
    GemmDispatch {
        w4a16_variant: match value("METRALE_W4A16_VARIANT").as_deref() {
            Some("v1") => 1,
            Some("v2") => 2,
            Some("v3") => 3,
            _ => 0,
        },
        fp8_blockscaled_prefill: !on(&mut value, "METRALE_FP8_SINGLE_SCALE"),
        cublas: parse_cublas_scope(value("METRALE_CUBLAS_GEMM").as_deref()).0,
        cublas_fp8: on(&mut value, "METRALE_CUBLAS_FP8"),
        cutlass_gemm: on(&mut value, "METRALE_CUTLASS_GEMM"),
        cutlass_nvfp4_gemm: all_nvfp4,
        cutlass_nvfp4_qkvz: all_nvfp4 || on(&mut value, "METRALE_CUTLASS_NVFP4_QKVZ"),
        cutlass_nvfp4_attn_q: all_nvfp4 || on(&mut value, "METRALE_CUTLASS_NVFP4_ATTN_Q"),
        cutlass_nvfp4_attn_kv: all_nvfp4 || on(&mut value, "METRALE_CUTLASS_NVFP4_ATTN_KV"),
        cutlass_nvfp4_attn_o: all_nvfp4 || on(&mut value, "METRALE_CUTLASS_NVFP4_ATTN_O"),
        // 2026-09-25: Not implied by `METRALE_CUTLASS_NVFP4_GEMM`.
        cutlass_nvfp4_ssm_out: on(&mut value, "METRALE_CUTLASS_NVFP4_SSM_OUT"),
    }
}

impl GemmDispatch {
    /// 2026-09-25: Resolve from the environment, and log the resolved
    /// `METRALE_CUBLAS_GEMM` scope when that variable is set.
    pub fn from_env() -> Self {
        let raw = std::env::var("METRALE_CUBLAS_GEMM").ok();
        let resolved = from_values(metrale_config::levers::var);
        log_cublas_scope(raw.as_deref(), resolved.cublas);
        resolved
    }

    /// 2026-09-25: Everything off except block-scaled FP8 prefill: what an
    /// environment with none of these variables resolves to.
    pub fn defaults() -> Self {
        Self {
            w4a16_variant: 0,
            fp8_blockscaled_prefill: true,
            cublas: CublasScope::OFF,
            cublas_fp8: false,
            cutlass_gemm: false,
            cutlass_nvfp4_gemm: false,
            cutlass_nvfp4_qkvz: false,
            cutlass_nvfp4_attn_q: false,
            cutlass_nvfp4_attn_kv: false,
            cutlass_nvfp4_attn_o: false,
            cutlass_nvfp4_ssm_out: false,
        }
    }

    /// 2026-09-25: Whether CUTLASS NVFP4 is on for an attention projection:
    /// `q_proj` reads the Q flag, `k_proj` and `v_proj` the KV flag, and any
    /// other label the umbrella flag.
    pub fn cutlass_nvfp4_attn_qkv(&self, label: &str) -> bool {
        match label {
            "q_proj" => self.cutlass_nvfp4_attn_q,
            "k_proj" | "v_proj" => self.cutlass_nvfp4_attn_kv,
            _ => self.cutlass_nvfp4_gemm,
        }
    }
}

/// 2026-09-25: Log the cuBLASLt families a set `METRALE_CUBLAS_GEMM` resolved
/// to, and warn about its unknown tokens. Silent when the variable is unset.
fn log_cublas_scope(raw: Option<&str>, scope: CublasScope) {
    let Some(raw) = raw else {
        return;
    };
    let (_, unknown) = parse_cublas_scope(Some(raw));
    if !unknown.is_empty() {
        tracing::warn!(
            "METRALE_CUBLAS_GEMM={raw:?}: ignoring unknown families [{}]. The grammar is a \
             comma-separated subset of all|ffn|attn|ssm|head|off (1/true = all).",
            unknown.join(", ")
        );
    }
    tracing::info!(
        "[metrale] METRALE_CUBLAS_GEMM={raw:?} -> cuBLASLt arms ffn={} attn={} ssm={} head={} \
         (head has no consumer yet){}",
        scope.ffn,
        scope.attn,
        scope.ssm,
        scope.head,
        if scope.any() {
            ""
        } else {
            " — no arm enabled"
        }
    );
}

impl Default for GemmDispatch {
    fn default() -> Self {
        Self::defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn resolve(values: &[(&str, &str)]) -> GemmDispatch {
        let values: HashMap<_, _> = values.iter().copied().collect();
        from_values(|name| values.get(name).map(|value| (*value).to_owned()))
    }

    #[test]
    fn defaults_have_only_blockscaled_prefill_on() {
        let d = GemmDispatch::defaults();
        assert_eq!(
            resolve(&[]),
            d,
            "absent environment uses the public default"
        );
        assert_eq!(
            d,
            GemmDispatch {
                fp8_blockscaled_prefill: true,
                cublas: CublasScope::OFF,
                cublas_fp8: false,
                cutlass_gemm: false,
                cutlass_nvfp4_gemm: false,
                cutlass_nvfp4_qkvz: false,
                cutlass_nvfp4_attn_q: false,
                cutlass_nvfp4_attn_kv: false,
                cutlass_nvfp4_attn_o: false,
                cutlass_nvfp4_ssm_out: false,
                w4a16_variant: 0,
            }
        );
    }

    #[test]
    fn the_umbrella_flag_implies_the_per_projection_ones() {
        let d = resolve(&[("METRALE_CUTLASS_NVFP4_GEMM", "1")]);
        assert!(d.cutlass_nvfp4_gemm);
        assert!(d.cutlass_nvfp4_qkvz);
        assert!(d.cutlass_nvfp4_attn_qkv("q_proj"));
        assert!(d.cutlass_nvfp4_attn_qkv("k_proj"));
        assert!(d.cutlass_nvfp4_attn_qkv("v_proj"));
        assert!(d.cutlass_nvfp4_attn_o);
        // 2026-09-25: SSM-out is not implied by the umbrella flag.
        assert!(!d.cutlass_nvfp4_ssm_out);
    }

    #[test]
    fn per_projection_flags_are_independent() {
        let cases = [
            (
                "METRALE_CUTLASS_NVFP4_QKVZ",
                [true, false, false, false, false],
            ),
            (
                "METRALE_CUTLASS_NVFP4_ATTN_Q",
                [false, true, false, false, false],
            ),
            (
                "METRALE_CUTLASS_NVFP4_ATTN_KV",
                [false, false, true, false, false],
            ),
            (
                "METRALE_CUTLASS_NVFP4_ATTN_O",
                [false, false, false, true, false],
            ),
            (
                "METRALE_CUTLASS_NVFP4_SSM_OUT",
                [false, false, false, false, true],
            ),
        ];
        for (name, expected) in cases {
            let d = resolve(&[(name, "1")]);
            assert_eq!(
                [
                    d.cutlass_nvfp4_qkvz,
                    d.cutlass_nvfp4_attn_q,
                    d.cutlass_nvfp4_attn_kv,
                    d.cutlass_nvfp4_attn_o,
                    d.cutlass_nvfp4_ssm_out,
                ],
                expected,
                "{name} must not enable a neighboring projection"
            );
        }
    }

    #[test]
    fn non_nvfp4_flags_map_independently_and_single_scale_is_inverted() {
        let cases = [
            ("METRALE_CUBLAS_GEMM", [true, false, false]),
            ("METRALE_CUBLAS_FP8", [false, true, false]),
            ("METRALE_CUTLASS_GEMM", [false, false, true]),
        ];
        for (name, expected) in cases {
            let d = resolve(&[(name, "1")]);
            assert_eq!(
                [d.cublas.any(), d.cublas_fp8, d.cutlass_gemm],
                expected,
                "{name} must not enable a neighboring GEMM path"
            );
            assert!(d.fp8_blockscaled_prefill);
        }
        assert!(!resolve(&[("METRALE_FP8_SINGLE_SCALE", "1")]).fp8_blockscaled_prefill);
    }

    fn scope(raw: &str) -> CublasScope {
        resolve(&[("METRALE_CUBLAS_GEMM", raw)]).cublas
    }

    /// 2026-09-25: Each spelling in the [`CublasScope`] table maps to its
    /// family set.
    #[test]
    fn the_scope_grammar_maps_each_spelling_to_its_family_set() {
        let f = |ffn, attn, ssm, head| CublasScope {
            ffn,
            attn,
            ssm,
            head,
        };
        let cases: [(&str, CublasScope); 13] = [
            ("all", CublasScope::ALL),
            ("1", CublasScope::ALL),
            ("true", CublasScope::ALL),
            ("ALL", CublasScope::ALL),
            ("off", CublasScope::OFF),
            ("0", CublasScope::OFF),
            ("false", CublasScope::OFF),
            ("", CublasScope::OFF),
            ("ffn", f(true, false, false, false)),
            ("attn", f(false, true, false, false)),
            ("ssm", f(false, false, true, false)),
            ("head", f(false, false, false, true)),
            ("ffn,attn", f(true, true, false, false)),
        ];
        for (raw, expected) in cases {
            assert_eq!(scope(raw), expected, "METRALE_CUBLAS_GEMM={raw:?}");
        }
        assert_eq!(scope(" ffn , ssm "), f(true, false, true, false));
        // 2026-09-25: A union, so `off` clears nothing; a clearing token would
        // make the result depend on token order.
        assert_eq!(scope("ffn,off"), f(true, false, false, false));
    }

    /// 2026-09-25: A typo must not widen the set; it is dropped and reported
    /// as typed, lowercased.
    #[test]
    fn unknown_families_are_dropped_and_reported_never_widening_the_set() {
        assert_eq!(scope("junk"), CublasScope::OFF);
        assert_eq!(
            scope("ffn,junk"),
            CublasScope {
                ffn: true,
                ..CublasScope::OFF
            }
        );
        let (resolved, unknown) = parse_cublas_scope(Some("ffn, FNN ,bogus"));
        assert_eq!(
            resolved,
            CublasScope {
                ffn: true,
                ..CublasScope::OFF
            }
        );
        assert_eq!(
            unknown,
            vec!["fnn".to_owned(), "bogus".to_owned()],
            "the warning must name what the operator typed, lowercased"
        );
    }

    /// 2026-09-25: An absent variable and `off` both resolve to the empty set
    /// with no unknown token to warn about.
    #[test]
    fn an_absent_variable_resolves_to_the_empty_set_silently() {
        assert_eq!(parse_cublas_scope(None), (CublasScope::OFF, Vec::new()));
        assert_eq!(
            parse_cublas_scope(Some("off")),
            (CublasScope::OFF, Vec::new())
        );
        assert!(!CublasScope::OFF.any());
        assert!(CublasScope::ALL.any());
    }

    #[test]
    fn w4a16_variants_accept_only_documented_spellings() {
        for (value, expected) in [
            ("v1", 1),
            ("v2", 2),
            ("v3", 3),
            ("1", 0),
            ("V1", 0),
            ("unknown", 0),
        ] {
            assert_eq!(
                resolve(&[("METRALE_W4A16_VARIANT", value)]).w4a16_variant,
                expected,
                "value {value}"
            );
        }
    }

    #[test]
    fn an_unknown_projection_label_falls_back_to_the_umbrella_flag() {
        assert!(!GemmDispatch::defaults().cutlass_nvfp4_attn_qkv("mystery"));
        let d = GemmDispatch {
            cutlass_nvfp4_gemm: true,
            ..GemmDispatch::defaults()
        };
        assert!(d.cutlass_nvfp4_attn_qkv("mystery"));
    }
}
