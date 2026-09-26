// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Pure resolution for a gate run: which (model variant, recipe)
//! to serve, which recipe keys the operator overrode, and which run parameters
//! the selected variant's baseline defines.
//!
//! Owner: server CLI (`met benchmark`).
//! Invariants:
//! - No I/O and no server: every function here is a function of its arguments.
//! - An explicit `--param` is never replaced by a baseline value.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail, ensure};
use metrale_bench::gate;

/// 2026-09-26: Why a `--hardware` value has no baseline entry. The two cases
/// need different fixes:
///
/// * [`Self::Unknown`]: the id is not in
///   `metrale_bench::hardware::ids::KNOWN_HARDWARE_IDS`. The spelling is
///   wrong, or the class must be registered there first.
/// * [`Self::NoRecordYet`]: the id is registered and nothing has been
///   measured on it. The fix is to run the gate on that box and commit the
///   thresholds.
#[derive(Debug, thiserror::Error)]
pub(super) enum HardwareRefusal {
    #[error(
        "{hardware:?} is not a box class Metrale Engine knows, so nothing can be scored against it. \
         Registered classes are [{registered}]; {benchmark_id} has baselines for [{measured}]."
    )]
    Unknown {
        benchmark_id: String,
        hardware: String,
        registered: String,
        measured: String,
    },
    #[error(
        "{benchmark_id} has no record yet for {hardware:?}. It is a registered box class with \
         nothing measured on it — run the gate on that box and commit one \
         (`met benchmark run {benchmark_id} --hardware {hardware} --pull-request-gate`), \
         which needs a `[[benchmark]]` entry in kernels/{hardware}/<model>/BENCH.toml. \
         Today it has baselines for [{measured}]."
    )]
    NoRecordYet {
        benchmark_id: String,
        hardware: String,
        measured: String,
    },
}

impl HardwareRefusal {
    /// 2026-09-26: Classify a `hardware` key the baseline does not carry,
    /// against the box-class registry. The caller has checked the absence.
    fn of(benchmark_id: &str, hardware: &str, baseline: &gate::GateBaseline) -> Self {
        let measured = baseline
            .hardware
            .keys()
            .cloned()
            .collect::<Vec<_>>()
            .join(", ");
        if metrale_bench::hardware::ids::is_known_hardware_id(hardware) {
            Self::NoRecordYet {
                benchmark_id: benchmark_id.to_string(),
                hardware: hardware.to_string(),
                measured,
            }
        } else {
            Self::Unknown {
                benchmark_id: benchmark_id.to_string(),
                hardware: hardware.to_string(),
                registered: metrale_bench::hardware::ids::KNOWN_HARDWARE_IDS.join(", "),
                measured,
            }
        }
    }
}

#[derive(Debug)]
pub(super) struct Resolved {
    pub model: String,
    pub recipe_id: String,
    /// 2026-09-26: The resolved variant's baseline entry, cloned.
    pub entry: gate::ModelBaseline,
    /// 2026-09-26: The box class the entry is for, whose `HARDWARE.toml`
    /// limits the serve plan reads.
    pub hardware: String,
}

/// 2026-09-26: Pick the (model, recipe) a gate run should serve.
///
/// `hardware: None` takes the baseline's only box class and refuses when
/// there are several or none. A class the baseline does not carry is refused
/// as a [`HardwareRefusal`].
///
/// `checkpoint` selects the model variant. `None` takes the one marked
/// `default = true`; baseline assembly (`gate::bench::baseline_for`) refuses
/// zero or two defaults per box class. A checkpoint the baseline does not carry
/// is refused naming the ones it does, and so is a variant with no `recipe`.
pub(super) fn resolve(
    baseline: &gate::GateBaseline,
    benchmark_id: &str,
    hardware: Option<&str>,
    checkpoint: Option<&str>,
) -> Result<Resolved> {
    let hw_key = match hardware {
        Some(h) => h.to_string(),
        None => {
            let mut keys = baseline.hardware.keys();
            match (keys.next(), keys.next()) {
                (Some(only), None) => only.clone(),
                // 2026-09-26: Several box classes and no --hardware: refuse,
                // since each class has its own thresholds.
                (Some(_), Some(_)) => bail!(
                    "{benchmark_id} has baselines for several box classes ([{}]); pass \
                     --hardware to say which one this run is for rather than guessing",
                    baseline
                        .hardware
                        .keys()
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                (None, _) => bail!("{benchmark_id} has no hardware entries in its baseline"),
            }
        }
    };

    // 2026-09-26: Classified before the baseline lookup, whose own error
    // cannot tell an unknown class from an unmeasured one.
    if !baseline.hardware.contains_key(&hw_key) {
        return Err(HardwareRefusal::of(benchmark_id, &hw_key, baseline).into());
    }
    let (model, entry) = baseline.resolve(&hw_key, checkpoint)?;
    let recipe_id = entry.recipe.clone().ok_or_else(|| {
        anyhow::anyhow!(
            "no recipe is bound to {model:?} on {hw_key:?} for {benchmark_id}. Self-start needs \
             one; either add `recipe` to the baseline entry or drive an existing server with \
             --url/--model and no --pull-request-gate."
        )
    })?;
    Ok(Resolved {
        model,
        recipe_id,
        entry: entry.clone(),
        hardware: hw_key,
    })
}

/// 2026-09-26: Set the run parameters that `BenchmarkDescriptor::threshold_params`
/// pairs with a metric, from the selected variant's bound on that metric.
///
/// Precedence:
/// 1. an operator's `--param KEY=…` is left as given;
/// 2. otherwise a `max` bound gives `max + noise` and a `min` bound gives
///    `min - noise`; a metric with both bounds is an error;
/// 3. a variant with no bound on the metric leaves the schema default.
///
/// Returns what was applied, for the caller to print.
///
/// `bench_variants::BenchState::choose_variant` (TUI) takes the same bound
/// without the noise band, and skips a metric with both bounds where this
/// refuses it.
pub(super) fn apply_threshold_params(
    descriptor: &metrale_bench::BenchmarkDescriptor,
    specs: &[metrale_bench::ParamSpec],
    values: &mut metrale_bench::ParamValues,
    entry: &gate::ModelBaseline,
    explicit: &[(String, String)],
) -> Result<Vec<(String, f64)>> {
    let mut applied = Vec::new();
    for (param, metric) in descriptor.threshold_params {
        if explicit.iter().any(|(k, _)| k == param) {
            continue;
        }
        let Some(bound) = entry.metrics.get(*metric) else {
            continue;
        };
        // 2026-09-26: The driver compares raw values against the bar it is
        // handed, while `gate::scoring::compare` passes value + noise >= min and
        // value - noise <= max. The driver gets the noise-adjusted bar so that a
        // run the gate passes does not fail its own verdict.
        let noise = bound.noise.unwrap_or(0.0);
        let derived = match (bound.min, bound.max) {
            (Some(min), Some(max)) => bail!(
                "{} couples param {param:?} to metric {metric:?}, whose baseline bound \
                 declares BOTH min ({min}) and max ({max}) — ambiguous: the driver cannot \
                 tell which one to self-verdict against. Split the metric or drop a bound.",
                descriptor.id
            ),
            (None, Some(max)) => max + noise,
            (Some(min), None) => min - noise,
            (None, None) => continue,
        };
        let spec = specs.iter().find(|s| s.key == *param).ok_or_else(|| {
            anyhow::anyhow!(
                "{} declares threshold param {param:?} but its schema has no such parameter — \
                 the declaration and the schema have drifted",
                descriptor.id
            )
        })?;
        // 2026-09-26: Through the spec's own parser, so the kind's bounds apply
        // as they do to a typed --param.
        let value = spec.kind.parse(&format!("{derived}")).with_context(|| {
            format!("deriving --param {param} from the baseline's {metric} bound {derived}")
        })?;
        values.set(param.to_string(), value);
        applied.push((param.to_string(), derived));
    }
    Ok(applied)
}

/// 2026-09-26: Apply the selected variant's `[benchmarks.param_overrides]`
/// pins: run parameters its thresholds were measured with. An operator's
/// `--param KEY=…` is left as given; otherwise the pin replaces the schema
/// default, parsed by the spec's own parser.
///
/// Refused:
/// * a pin on a `threshold_params`-coupled parameter, whose value comes from
///   the paired metric's bound;
/// * a pin naming no schema parameter.
///
/// Returns what was applied, for the caller to print. The gate check requires
/// each pinned param on the record (`gate::scoring`).
pub(super) fn apply_param_overrides(
    descriptor: &metrale_bench::BenchmarkDescriptor,
    specs: &[metrale_bench::ParamSpec],
    values: &mut metrale_bench::ParamValues,
    entry: &gate::ModelBaseline,
    explicit: &[(String, String)],
) -> Result<Vec<(String, String)>> {
    let mut applied = Vec::new();
    for (key, raw) in &entry.param_overrides {
        if descriptor.threshold_params.iter().any(|(p, _)| p == key) {
            bail!(
                "{}: baseline param override {key:?} names a threshold-coupled parameter — \
                 its value is derived from the paired metric's bound, and a second source \
                 here would fight it. Move the number into the metric's bound instead.",
                descriptor.id
            );
        }
        if explicit.iter().any(|(k, _)| k == key) {
            continue;
        }
        let spec = specs
            .iter()
            .find(|s| s.key == key.as_str())
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "{}: baseline param override {key:?} names no parameter in the schema — \
                 the BENCH.toml pin and the driver have drifted, and running without it \
                 would measure a different instrument than the thresholds describe",
                    descriptor.id
                )
            })?;
        let value = spec
            .kind
            .parse(raw)
            .with_context(|| format!("applying the baseline's param override {key}={raw}"))?;
        values.set(key.clone(), value);
        applied.push((key.clone(), raw.clone()));
    }
    Ok(applied)
}

/// 2026-09-26: Parse `--serve-override KEY=VALUE` pairs into recipe overrides.
///
/// Checks only the pair's shape. Whether the key is a serve flag is decided
/// when the recipe renders: `Recipe::argv` refuses a key that renders to no
/// flag, and clap in `serve_args` refuses an unknown flag.
///
/// `port` is refused: the gate binds a free port itself.
pub(super) fn parse_serve_overrides(pairs: &[String]) -> Result<BTreeMap<String, String>> {
    let mut out = BTreeMap::new();
    for pair in pairs {
        let (key, value) = pair.split_once('=').with_context(|| {
            format!("--serve-override {pair:?} is not KEY=VALUE (e.g. kv_cache_dtype=fp8)")
        })?;
        let key = key.trim();
        ensure!(
            !key.is_empty(),
            "--serve-override {pair:?} has an empty key"
        );
        ensure!(
            key != "port",
            "--serve-override cannot set `port`: the gate binds a free port itself and serves \
             on it, so an override here would name a port nothing is listening on."
        );
        // 2026-09-26: A repeated key: the last value wins.
        out.insert(key.to_string(), value.to_string());
    }
    Ok(out)
}

#[cfg(test)]
#[path = "bench_resolve_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "bench_resolve_hardware_tests.rs"]
mod hardware_tests;

#[cfg(test)]
#[path = "bench_resolve_params_tests.rs"]
mod params_tests;
