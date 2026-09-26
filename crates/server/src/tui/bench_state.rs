// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Benchmarks-section state: the selection, the parameter form, and what the running benchmark has reported.
//!
//! The Suite flow is List → Variants → Params → Run, the variant step only
//! for a benchmark with model variants (see [`super::bench_variants`]); the
//! History subsection reads the recorded runs. Nothing here awaits: the
//! executor owns the async side and `pump` drains its channels.
//!
//! Owner: server tui.
//! Invariants:
//! - `select` builds `edit` as one buffer per parameter, then the endpoint URL
//!   and the model; `row_count` counts the same rows.

use std::collections::{BTreeMap, VecDeque};
use std::time::Instant;

use metrale_bench::{
    BenchmarkDescriptor, BenchmarkResult, ExecutorMessage, LogLine, ParamSpec, ParamValues,
    PluginEvent, RunHandle, RunStatus, TargetEndpoint, registry,
};

/// 2026-09-26: How many log lines the run pane keeps.
const LOG_CAPACITY: usize = 500;

/// 2026-09-26: Which step of the flow the Suite subsection is showing.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub enum View {
    #[default]
    List,
    /// 2026-09-26: Which model variant of the selected benchmark; shown when it has any.
    Variants,
    Params,
    Run,
}

#[derive(Default)]
pub struct BenchState {
    /// 2026-09-26: Index into [`registry::all`].
    pub selected: usize,
    pub view: View,
    /// 2026-09-26: Provenance of the selected benchmark, cached by `select` so the renderer does
    /// not build the benchmark on every frame.
    meta: Option<&'static metrale_bench::PluginMetadata>,
    /// 2026-09-26: Schema of the selected benchmark, and the values being edited.
    pub specs: Vec<ParamSpec>,
    pub values: ParamValues,
    /// 2026-09-26: One edit buffer per row. Rows past `specs.len()` are the target fields,
    /// so the endpoint is edited with the same keys as everything else.
    pub edit: Vec<String>,
    pub row: usize,
    pub editing: bool,
    /// 2026-09-26: Per-field validation messages, keyed by parameter key, `__url` or `__model`.
    pub errors: BTreeMap<String, String>,
    pub target: TargetEndpoint,
    /// 2026-09-26: True once the target model is pinned, by typing it or by choosing a variant.
    ///
    /// Until then `follow_live_model` keeps the target on the model being served.
    pub target_model_pinned: bool,
    /// 2026-09-26: True when the current pin came from `choose_variant`, not from the keyboard.
    /// `select` releases a variant pin; a typed pin survives benchmark switches.
    pub variant_pinned: bool,
    /// 2026-09-26: The start-confirmation prompt is open (descriptor `needs_confirmation`).
    pub confirm_open: bool,
    /// 2026-09-26: The selected benchmark's model variants, from `variants_for`.
    /// Empty when it has none or no repository checkout is found.
    pub variants: Vec<super::bench_variants::VariantRow>,
    /// 2026-09-26: Cursor into `variants`.
    pub variant_row: usize,

    executor: Option<metrale_bench::BenchmarkExecutor>,
    run: Option<RunHandle>,
    /// 2026-09-26: The benchmark the in-flight (or last) run belongs to.
    pub running_id: Option<&'static str>,
    /// 2026-09-26: The descriptor of the run in flight; `persist` needs it for the record.
    running_descriptor: Option<&'static metrale_bench::BenchmarkDescriptor>,
    /// 2026-09-26: Whether to probe the endpoint before measuring. Defaults to `Probe`; `p` in the
    /// form toggles.
    pub coherence: metrale_bench::CoherencePolicy,
    pub frame: Option<BenchmarkResult>,
    pub log: VecDeque<LogLine>,
    pub status: String,
    pub progress: Option<(u64, u64)>,
    pub glow: bool,
    pub started: Option<Instant>,
    pub table_scroll: usize,
    /// 2026-09-26: Results-table scroll ceilings, set by the Run and History renderers each frame.
    /// Held here rather than on `App` because `bench_keys` sees only `BenchState`.
    pub table_scroll_max: std::cell::Cell<usize>,
    pub history_table_scroll_max: std::cell::Cell<usize>,
    /// 2026-09-26: Entries one Suite-list page holds, set by `draw_list` each frame; PgUp/PgDn page by it.
    pub suite_page: std::cell::Cell<usize>,

    pub history: Vec<metrale_bench::RunRecord>,
    pub history_row: usize,
    /// 2026-09-26: Viewport offset into the selected past run's results table (PgUp/PgDn).
    pub history_table_scroll: usize,
    history_loaded: bool,
    /// 2026-09-26: Set by `persist` and cleared by `start`, so a run's terminal frame is recorded at most once.
    persisted: bool,
    /// 2026-09-26: The endpoint check between pressing start and the run beginning.
    pub preflight: Option<crate::tui::bench_preflight::Preflight>,
}

impl BenchState {
    /// 2026-09-26: Point the target at the model that is being served; a no-op once the target is pinned.
    pub fn follow_live_model(&mut self, live: &str) {
        if self.target_model_pinned || live.is_empty() || self.target.model == live {
            return;
        }
        self.target = TargetEndpoint::new(self.target.base_url.clone(), live);
        // 2026-09-26: The form's model row shows its edit buffer, so it follows too unless being edited.
        if !self.editing
            && let Some(model_row) = self.edit.get_mut(self.specs.len() + 1)
        {
            *model_row = self.target.model.clone();
        }
    }

    pub fn attach(&mut self, executor: metrale_bench::BenchmarkExecutor, target: TargetEndpoint) {
        self.executor = Some(executor);
        self.target = target;
        self.select(0);
    }

    pub fn descriptor(&self) -> Option<&'static BenchmarkDescriptor> {
        registry::all().get(self.selected).copied()
    }

    /// 2026-09-26: True when a text buffer owns the keyboard, so `App` does not treat digits as section jumps.
    pub fn is_editing(&self) -> bool {
        self.editing
    }

    pub fn is_running(&self) -> bool {
        self.run.as_ref().is_some_and(|r| !r.is_finished())
    }

    /// 2026-09-26: Load a benchmark's schema into the form, with every value at its spec default.
    pub fn select(&mut self, index: usize) {
        let all = registry::all();
        if all.is_empty() {
            return;
        }
        self.selected = index.min(all.len() - 1);
        let Some(descriptor) = self.descriptor() else {
            return;
        };
        let bench = descriptor.build();
        self.meta = Some(bench.metadata());
        self.specs = bench.parameters();
        self.values = ParamValues::defaults(&self.specs);
        self.edit = self
            .specs
            .iter()
            .map(|s| s.default.to_edit_string())
            .chain([self.target.base_url.clone(), self.target.model.clone()])
            .collect();
        self.errors.clear();
        self.row = 0;
        self.editing = false;
        // 2026-09-26: Variants and a variant pin belong to the previous benchmark, so both are released;
        // an operator-typed pin (`variant_pinned == false`) survives.
        self.variants.clear();
        self.variant_row = 0;
        if self.variant_pinned {
            self.variant_pinned = false;
            self.target_model_pinned = false;
        }
    }

    /// 2026-09-26: Provenance of the selected benchmark.
    pub fn plugin_metadata(&self) -> &'static metrale_bench::PluginMetadata {
        // 2026-09-26: `meta` is `None` until `select` succeeds; the fallback spares the renderer an Option.
        self.meta.unwrap_or(&FALLBACK_METADATA)
    }

    /// 2026-09-26: Total form rows: one per parameter, then the two target fields.
    pub fn row_count(&self) -> usize {
        self.specs.len() + 2
    }

    /// 2026-09-26: Label, help and domain hint for a row, whether it is a parameter or a target field.
    pub fn row_meta(&self, row: usize) -> (&str, &str, String) {
        match self.specs.get(row) {
            Some(spec) => (spec.label, spec.help, spec.kind.domain_hint()),
            None if row == self.specs.len() => (
                "Endpoint URL",
                "Which server to benchmark. Defaults to this one.",
                "http://host:port".to_string(),
            ),
            _ => (
                "Model",
                "The `model` field sent in each request.",
                "model id".to_string(),
            ),
        }
    }

    /// 2026-09-26: Parse and store the row's edit buffer. A parse error is kept in `errors` under the
    /// field's key, and `start` refuses while any is present.
    pub fn commit_row(&mut self, row: usize) {
        let raw = self.edit.get(row).cloned().unwrap_or_default();
        match self.specs.get(row) {
            Some(spec) => {
                let key = spec.key.to_string();
                match spec.kind.parse(&raw) {
                    Ok(value) => {
                        self.values.set(key.clone(), value);
                        self.errors.remove(&key);
                    }
                    Err(e) => {
                        self.errors.insert(key, e.to_string());
                    }
                }
            }
            None if row == self.specs.len() => {
                let trimmed = raw.trim();
                if trimmed.is_empty() {
                    self.errors
                        .insert("__url".into(), "must not be empty".into());
                } else {
                    self.target = TargetEndpoint::new(trimmed, self.target.model.clone());
                    self.errors.remove("__url");
                    // 2026-09-26: Show the URL `TargetEndpoint::new` normalised.
                    self.edit[row] = self.target.base_url.clone();
                }
            }
            _ => {
                let trimmed = raw.trim();
                if trimmed.is_empty() {
                    self.errors
                        .insert("__model".into(), "must not be empty".into());
                } else {
                    self.target = TargetEndpoint::new(self.target.base_url.clone(), trimmed);
                    self.target_model_pinned = true;
                    // 2026-09-26: Typed by hand, so it survives benchmark switches.
                    self.variant_pinned = false;
                    self.errors.remove("__model");
                }
            }
        }
    }

    /// 2026-09-26: Validation message for a row, if it has one.
    pub fn row_error(&self, row: usize) -> Option<&str> {
        let key = match self.specs.get(row) {
            Some(spec) => spec.key,
            None if row == self.specs.len() => "__url",
            _ => "__model",
        };
        self.errors.get(key).map(String::as_str)
    }

    /// 2026-09-26: Check the endpoint, then start. Refuses a running or invalid form before the check,
    /// as `start` does. With the probe set to `Skip` this is `start`.
    pub fn begin_start(&mut self) -> Result<(), String> {
        if self.is_running() {
            return Err("a benchmark is already running".into());
        }
        if !self.errors.is_empty() {
            return Err(format!("{} field(s) need fixing", self.errors.len()));
        }
        if self.coherence == metrale_bench::CoherencePolicy::Skip {
            return self.start();
        }
        let executor = self
            .executor
            .as_ref()
            .ok_or("the benchmark executor is unavailable")?;
        let expectation = self.descriptor().and_then(|d| d.intended_for);
        self.preflight = Some(crate::tui::bench_preflight::Preflight::begin(
            executor.runtime(),
            self.target.clone(),
            expectation,
            std::time::Duration::from_secs(30),
        ));
        Ok(())
    }

    /// 2026-09-26: Drain the pre-flight. Called on every tick; starts the run when the check is clean.
    pub fn poll_preflight(&mut self) {
        let Some(pre) = self.preflight.as_mut() else {
            return;
        };
        // 2026-09-26: Some(true) starts now; a concern keeps the modal up, and None means still checking.
        if pre.poll(&self.target) == Some(true) {
            self.preflight = None;
            let _ = self.start();
        }
    }

    /// 2026-09-26: Proceed past a reported concern.
    pub fn accept_preflight(&mut self) -> Result<(), String> {
        self.preflight = None;
        self.start()
    }

    /// 2026-09-26: Abandon the run and go back to the form.
    pub fn cancel_preflight(&mut self) {
        self.preflight = None;
    }

    /// 2026-09-26: Start the selected benchmark. Refuses while a run is in flight, while any field is
    /// invalid, and without a selected benchmark or an executor.
    pub fn start(&mut self) -> Result<(), String> {
        if self.is_running() {
            return Err("a benchmark is already running".into());
        }
        if !self.errors.is_empty() {
            return Err(format!("{} field(s) need fixing", self.errors.len()));
        }
        let descriptor = self.descriptor().ok_or("no benchmark selected")?;
        let executor = self
            .executor
            .as_ref()
            .ok_or("the benchmark executor is unavailable")?;
        self.log.clear();
        self.frame = None;
        self.status = "starting".into();
        self.progress = None;
        self.table_scroll = 0;
        self.persisted = false;
        self.started = Some(Instant::now());
        self.running_id = Some(descriptor.id);
        self.running_descriptor = Some(descriptor);
        self.run = Some(executor.start(
            descriptor,
            self.values.clone(),
            self.target.clone(),
            self.coherence,
            // 2026-09-26: No temperature ceilings: the dashboard reads no `HARDWARE.toml`.
            None,
        ));
        self.view = View::Run;
        Ok(())
    }

    pub fn cancel(&mut self) {
        if let Some(run) = &self.run {
            run.cancel();
            self.status = "cancelling — the server keeps serving".into();
        }
    }

    /// 2026-09-26: Drain the executor's channels. Called on every pass of the event loop.
    pub fn pump(&mut self) {
        // 2026-09-26: Drain first, then release the borrow: the handlers below mutate `self`.
        let Some((messages, finished)) = self
            .run
            .as_ref()
            .map(|run| (run.drain(), run.is_finished()))
        else {
            return;
        };
        for message in messages {
            match message {
                ExecutorMessage::Event(PluginEvent::Log(line)) => self.push_log(line),
                ExecutorMessage::Event(PluginEvent::Status(text)) => self.status = text,
                ExecutorMessage::Event(PluginEvent::Progress { done, total }) => {
                    self.progress = Some((done, total));
                }
                ExecutorMessage::Event(PluginEvent::Glow(on)) => self.glow = on,
                ExecutorMessage::Frame(frame) => {
                    for line in &frame.log {
                        self.push_log(line.clone());
                    }
                    if let Some(p) = frame.progress {
                        self.progress = Some(p);
                    }
                    if frame.status.is_terminal() {
                        self.status = match frame.status {
                            RunStatus::Completed => "completed".into(),
                            _ => "failed".into(),
                        };
                        self.persist(&frame);
                    } else {
                        self.status = frame.phase.clone();
                    }
                    self.frame = Some(*frame);
                }
            }
        }
        // 2026-09-26: The glow follows the executor's signal, and a finished run always clears it.
        if finished {
            self.glow = false;
        }
    }

    fn push_log(&mut self, line: LogLine) {
        if self.log.len() >= LOG_CAPACITY {
            self.log.pop_front();
        }
        self.log.push_back(line);
    }

    /// 2026-09-26: Record the terminal frame, once, through `metrale_bench::history::save`, the writer
    /// the headless runner also uses.
    fn persist(&mut self, frame: &BenchmarkResult) {
        if self.persisted {
            return;
        }
        self.persisted = true;
        let (Some(executor), Some(descriptor)) = (&self.executor, self.running_descriptor) else {
            return;
        };
        let mut record = metrale_bench::RunRecord::new(
            descriptor,
            &self.values,
            &self.target,
            // 2026-09-26: Empty serve overrides: the TUI did not configure the endpoint it benchmarks, so it
            // records none rather than claiming a configuration it cannot see.
            Default::default(),
            metrale_bench::RunSource::Tui,
            crate::cli::METRALE_VERSION,
            frame.clone(),
        );
        if let Err(e) = metrale_bench::history::save(executor.artifacts(), &mut record) {
            tracing::warn!("could not record this run: {e:#}");
        }
        // 2026-09-26: The next visit to History re-reads the run directory.
        self.history_loaded = false;
    }
}

/// 2026-09-26: Shown only before `select` has loaded a benchmark. A `const` so it has a `'static`
/// address to borrow from.
const FALLBACK_METADATA: metrale_bench::PluginMetadata =
    metrale_bench::PluginMetadata::metrale("no benchmark selected");

#[path = "bench_state_history.rs"]
mod history;

#[cfg(test)]
#[path = "bench_state_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "bench_state_more_tests.rs"]
mod more_tests;
