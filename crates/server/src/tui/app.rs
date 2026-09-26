// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Dashboard state and the top-level key reducer; rendering is in `render/`.
//!
//! Owner: server tui.
//! Invariants:
//! - `on_key` clears any mouse selection before it handles a key.
//! - Ctrl+C requests shutdown before any prompt, overlay or input field sees it.

use std::time::Instant;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

pub use super::app_types::{OpsState, Toast};
pub use super::section::Section;

use super::bench_state::BenchState;
use super::chat::ChatState;
use super::data::kernels::KernelTableModel;
use super::data::library::LibraryEntry;
use super::data::metrics_poll::StatsModel;
use super::progress::ProgressModel;

#[derive(Clone, Copy, PartialEq)]
pub enum MainSub {
    Overview,
    Kernels,
}

#[derive(Clone, Copy, PartialEq)]
pub enum BenchSub {
    Suite,
    History,
}

#[derive(Clone, Copy, PartialEq)]
pub enum TermSub {
    Ops,
    Chat,
}

#[derive(Clone, Copy, PartialEq)]
pub enum Focus {
    Sidebar,
    Content,
    Input,
}

pub struct App {
    pub args: crate::cli::ServeArgs,
    /// 2026-09-26: The model cell, so the Library can start or replace a model.
    /// `App::new` leaves it `None`; the dashboard attaches it right after (`tui::mod`),
    /// and tests leave it `None`.
    pub host: Option<std::sync::Arc<crate::main_modules::model_host::ModelHost>>,
    /// 2026-09-26: Ask the event loop for a full repaint on the next frame.
    ///
    /// ratatui diffs its new buffer against the last buffer it drew, not against
    /// the terminal, so cells that diverged (a foreign write, a reflowed resize)
    /// stay stale until a clear. The clear is requested when the layout changes
    /// wholesale, not every frame.
    pub repaint: bool,
    /// 2026-09-26: No model loaded, so the startup checklist is not tracking anything.
    /// Cleared when a Library launch starts; set again if that launch fails with no
    /// live model.
    pub awaiting_model: bool,
    pub section: Section,
    pub main_sub: MainSub,
    pub bench_sub: BenchSub,
    pub term_sub: TermSub,
    pub focus: Focus,
    pub progress: ProgressModel,
    pub stats: StatsModel,
    /// 2026-09-26: Background thermal/throttle sampler. It spawns `nvidia-smi` on its
    /// own thread, and the UI reads only its latest snapshot (`data::thermal`).
    pub thermal: super::data::thermal::ThermalProbe,
    pub started: Instant,
    /// 2026-09-26: None = follow newest; Some(n) = scrolled up by n lines.
    pub log_scroll: Option<usize>,
    pub log_filter: String,
    pub log_filter_editing: bool,
    pub kernels: Option<KernelTableModel>,
    /// 2026-09-26: Which model the kernel table describes. `on_tick` rebuilds the table
    /// from the load's audit when the live model differs, including after a swap.
    pub kernels_for: Option<String>,
    pub kernel_scroll: usize,
    pub kernel_filter: String,
    pub library: Vec<LibraryEntry>,
    /// 2026-09-26: The local cache needs (re)scanning. Set at boot, when a download
    /// settles, and when the Library reducer asks; cleared when a scan starts.
    pub library_dirty: bool,
    /// 2026-09-26: The Library section: joined recipes + local weights, and the edit form.
    pub lib: crate::tui::lib_state::LibState,
    /// 2026-09-26: Model downloads and update checks.
    pub download: crate::tui::download_state::DownloadState,
    /// 2026-09-26: Where the Library's search field was drawn, published by the
    /// renderer for the mouse handler, so hit-testing uses the drawn layout rather
    /// than a copy of the layout math. `None` whenever the last frame did not draw
    /// the field.
    pub lib_search_click: std::cell::Cell<Option<ratatui::layout::Rect>>,
    /// 2026-09-26: Scroll ceilings, published by the renderer. See `app_scroll`.
    pub log_scroll_max: std::cell::Cell<usize>,
    pub kernel_scroll_max: std::cell::Cell<usize>,
    pub chat_scroll_max: std::cell::Cell<usize>,
    /// 2026-09-26: An in-progress or just-finished mouse selection, in cell
    /// coordinates. It lives on `App` because the renderer draws the highlight from
    /// it, and it must survive the ticks between button-down and button-up.
    pub selection: Option<crate::tui::selection::Selection>,
    pub network_selected: usize,
    pub ops: OpsState,
    pub chat: ChatState,
    pub bench: BenchState,
    /// 2026-09-26: The Help section: the guide pane and the issue-report pipeline.
    pub help: crate::tui::help_state::HelpState,
    /// 2026-09-26: The running scheduler's handles, taken from the newest published
    /// run each loop (`events.rs`). `None` until a run is published.
    pub run: Option<crate::tui::RunHandles>,
    pub toasts: Vec<Toast>,
    pub help_open: bool,
    /// 2026-09-26: Help-modal scroll offset, for a key table taller than the modal.
    pub help_scroll: usize,
    pub help_scroll_max: std::cell::Cell<usize>,
    /// 2026-09-26: A `q` that would have destroyed work in flight, waiting to be
    /// answered. Set only when [`App::work_in_flight`] named something; an idle
    /// dashboard quits on the first press.
    pub confirm_quit: bool,
    /// 2026-09-26: A `Ctrl+N` waiting to be answered before it discards the chat
    /// transcript. Set only when the transcript is non-empty or a reply is
    /// streaming.
    pub confirm_chat_clear: bool,
    /// 2026-09-26: An open "one download at a time" question: which job is running,
    /// and which one the user just asked for.
    pub download_switch: Option<(String, String)>,
    /// 2026-09-26: A download to start once the running one releases the slot; only
    /// one download runs at a time.
    pub pending_start: Option<String>,
    pub tick: u64,
    pub should_quit: bool,
    pub detach: bool,
}

impl App {
    pub fn new(args: crate::cli::ServeArgs) -> Self {
        // 2026-09-26: With no model, boot lands on the Library, the screen that can
        // start one.
        let awaiting_model = args.model.is_none() && args.model_from_path.is_none();
        Self {
            awaiting_model,
            repaint: false,
            args,
            host: None,
            section: if awaiting_model {
                Section::Library
            } else {
                Section::Main
            },
            main_sub: MainSub::Overview,
            bench_sub: BenchSub::Suite,
            run: None,
            term_sub: TermSub::Ops,
            focus: Focus::Content,
            progress: ProgressModel::default(),
            stats: StatsModel::default(),
            thermal: super::data::thermal::ThermalProbe::spawn(),
            started: Instant::now(),
            log_scroll: None,
            log_filter: String::new(),
            log_filter_editing: false,
            kernels: None,
            kernels_for: None,
            kernel_scroll: 0,
            kernel_filter: String::new(),
            library: Vec::new(),
            library_dirty: true,
            download: Default::default(),
            selection: None,
            lib_search_click: std::cell::Cell::new(None),
            log_scroll_max: std::cell::Cell::new(0),
            kernel_scroll_max: std::cell::Cell::new(0),
            chat_scroll_max: std::cell::Cell::new(0),
            lib: Default::default(),
            network_selected: 0,
            ops: OpsState::default(),
            chat: ChatState::default(),
            bench: BenchState::default(),
            help: Default::default(),
            toasts: Vec::new(),
            help_open: false,
            help_scroll: 0,
            help_scroll_max: std::cell::Cell::new(0),
            confirm_quit: false,
            confirm_chat_clear: false,
            download_switch: None,
            pending_start: None,
            tick: 0,
            should_quit: false,
            detach: false,
        }
    }

    pub fn toast(&mut self, text: impl Into<String>, error: bool) {
        let text = text.into();
        // 2026-09-26: An error toast is also logged at WARN, so it outlives the toast
        // in the log ring and the tee file.
        if error {
            tracing::warn!("{text}");
        }
        self.toasts.push(Toast {
            text,
            error,
            at: Instant::now(),
        });
        if self.toasts.len() > 3 {
            self.toasts.remove(0);
        }
    }

    pub fn on_tick(&mut self) {
        self.tick += 1;
        self.progress.ease_tick();
        // 2026-09-26: Info toasts dismiss at 5 s and errors at 12 s; with at most 3
        // toasts kept (`toast`), a permanent one would crowd out the rest.
        self.toasts
            .retain(|t| t.at.elapsed().as_secs() < if t.error { 12 } else { 5 });
        // 2026-09-26: A launch that failed after its thread started: reset the
        // checklist and repaint, so the dashboard does not keep showing a load.
        if let Some(err) = self.lib.poll_launch() {
            let live = self.host.as_ref().and_then(|h| h.live_model());
            self.awaiting_model = live.is_none();
            self.progress.reset();
            self.repaint = true;
            self.toast(super::lib_state::problem_line(&err), true);
        }

        // 2026-09-26: Keep the benchmark target on the model that is actually serving.
        if let Some(name) = self
            .host
            .as_ref()
            .and_then(|h| h.live_model())
            .or_else(|| self.args.model_name.clone())
            .or_else(|| self.args.model.clone())
        {
            self.bench.follow_live_model(&name);
        }
        // 2026-09-26: Rebuild the kernel table when a different model finishes
        // loading, including after a swap.
        let live = self
            .host
            .as_ref()
            .and_then(|h| h.live_model())
            .or_else(|| self.args.model_name.clone())
            .or_else(|| self.args.model.clone());
        if self.progress.ready && !self.awaiting_model && live.is_some() && self.kernels_for != live
        {
            let model = super::data::kernels::build();
            // 2026-09-26: Only missing required kernels raise a toast; the ones
            // MODEL.toml declares absent (`missing_expected`) do not.
            let n = model.missing_required.len();
            if n > 0 {
                let msg = format!("{n} kernel lookup(s) unresolved — Main ▸ Kernels");
                self.toast(msg, false);
            }
            self.kernels = Some(model);
            self.kernels_for = live;
        }
    }

    /// 2026-09-26: True when a text input owns the keyboard.
    fn in_input(&self) -> bool {
        self.log_filter_editing
            || (self.section == Section::Library && self.lib.is_editing())
            || (self.section == Section::Benchmarks && self.bench.is_editing())
            || (self.section == Section::Help && self.help.is_editing())
            || (self.section == Section::Terminal && self.focus == Focus::Input)
    }

    pub fn on_key(&mut self, key: KeyEvent) {
        // 2026-09-26: Any keystroke ends a selection: its coordinates are screen
        // cells, so once the screen changes it would highlight different text.
        self.selection = None;
        // 2026-09-26: Ctrl+C always requests clean shutdown (raw mode swallows SIGINT).
        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            super::shutdown::request("Ctrl+C");
            self.should_quit = true;
            return;
        }
        // 2026-09-26: A pending confirmation owns the keyboard until it is answered.
        if self.confirm_quit && self.answer_quit_prompt(key) {
            return;
        }
        // 2026-09-26: An open download question also owns the keyboard, after the quit
        // prompt, which outranks it.
        if self.download_switch.is_some() && self.answer_download_switch(key) {
            return;
        }
        if self.confirm_chat_clear && self.answer_chat_clear(key) {
            return;
        }
        if self.help_open {
            self.on_help_overlay_key(key);
            return;
        }
        if self.in_input() {
            self.on_input_key(key);
            return;
        }
        match key.code {
            KeyCode::Char('q') => self.on_quit_key(),
            KeyCode::Char('?') => self.help_open = true,
            KeyCode::Char('1') => self.jump(Section::Main),
            KeyCode::Char('2') => self.jump(Section::Stats),
            KeyCode::Char('3') => self.jump(Section::Network),
            KeyCode::Char('4') => self.jump(Section::Library),
            KeyCode::Char('5') => self.jump(Section::Benchmarks),
            KeyCode::Char('6') => self.jump(Section::Terminal),
            KeyCode::Char('7') => self.jump(Section::Help),
            KeyCode::Tab => self.cycle_section(1),
            KeyCode::BackTab => self.cycle_section(-1),
            KeyCode::Char('f') if self.section == Section::Main => {
                self.log_filter_editing = true;
            }
            KeyCode::Char('i') | KeyCode::Enter
                if self.section == Section::Terminal && self.focus != Focus::Input =>
            {
                self.focus = Focus::Input;
            }
            // 2026-09-26: Benchmarks and Library own every key the global bindings
            // above did not claim, Esc included.
            _ if self.section == Section::Benchmarks => self.on_section_key(key),
            _ if self.section == Section::Library => self.on_library_key(key),
            // 2026-09-26: Help owns its keys too, Esc included.
            _ if self.section == Section::Help => self.on_help_key(key),
            KeyCode::Esc => {
                self.focus = Focus::Content;
                self.log_scroll = None;
            }
            _ => self.on_section_key(key),
        }
    }

    fn on_section_key(&mut self, key: KeyEvent) {
        let down = matches!(key.code, KeyCode::Down | KeyCode::Char('j'));
        let up = matches!(key.code, KeyCode::Up | KeyCode::Char('k'));
        match self.section {
            // 2026-09-26: Both panes move through `scroll`, the same entry point as the
            // mouse wheel, so keys and wheel share one ceiling.
            Section::Main => match self.main_sub {
                // 2026-09-26: `g`/Home and `G`/End jump to the top and bottom, as the
                // help overlay advertises ("g / G").
                MainSub::Overview => {
                    if up {
                        self.scroll(-1);
                    } else if down {
                        self.scroll(1);
                    } else if matches!(key.code, KeyCode::Char('G') | KeyCode::End) {
                        self.log_scroll = None;
                    } else if matches!(key.code, KeyCode::Char('g') | KeyCode::Home) {
                        // 2026-09-26: Oldest line the pane can show; ceiling published
                        // by the renderer.
                        let max = self.log_scroll_max.get();
                        self.log_scroll = (max > 0).then_some(max);
                    }
                }
                MainSub::Kernels => {
                    if down {
                        self.scroll(1);
                    } else if up {
                        self.scroll(-1);
                    } else if matches!(key.code, KeyCode::Char('g') | KeyCode::Home) {
                        self.kernel_scroll = 0;
                    } else if matches!(key.code, KeyCode::Char('G') | KeyCode::End) {
                        self.kernel_scroll = self.kernel_scroll_max.get();
                    }
                }
            },
            // 2026-09-26: The Library's navigation is in `lib_keys` only, so one reducer
            // owns its selection.
            Section::Library => {}
            Section::Network => {
                let n = self.args.world_size.max(1);
                if matches!(key.code, KeyCode::Right | KeyCode::Char('l')) && n > 1 {
                    self.network_selected = (self.network_selected + 1).min(n - 1);
                } else if matches!(key.code, KeyCode::Left | KeyCode::Char('h')) {
                    self.network_selected = self.network_selected.saturating_sub(1);
                }
            }
            // 2026-09-26: Chat's content keys are in `app_input`, beside its
            // input-focused keys.
            Section::Terminal if self.term_sub == TermSub::Chat => self.on_chat_content_key(key),
            // 2026-09-26: Ops scrolls its own output (`app_scroll`).
            Section::Terminal => self.on_ops_content_key(key),
            Section::Benchmarks => self.on_bench_key(key),
            Section::Stats | Section::Help => {}
        }
    }

    /// 2026-09-26: Route into the Library reducer and surface whatever it wants said.
    pub(super) fn on_library_key(&mut self, key: KeyEvent) {
        match self.lib.on_key(key) {
            super::lib_keys::Outcome::Toast { text, error } => self.toast(text, error),
            super::lib_keys::Outcome::Launch => self.launch_selected_recipe(),
            super::lib_keys::Outcome::Download => self.download_selected_model(),
            super::lib_keys::Outcome::CancelDownload => match self.download.cancel() {
                Some((text, error)) => self.toast(text, error),
                None => self.toast("nothing is downloading".to_string(), false),
            },
            super::lib_keys::Outcome::CheckFresh => self.check_selected_model(),
            super::lib_keys::Outcome::None => {}
        }
    }

    /// 2026-09-26: Route into the Benchmarks reducer and surface whatever it wants said.
    pub(super) fn on_bench_key(&mut self, key: KeyEvent) {
        if let super::bench_keys::Outcome::Toast { text, error } =
            self.bench.on_key(key, self.bench_sub)
        {
            self.toast(text, error);
        }
    }
}

// 2026-09-26: Navigation-order tests.
#[cfg(test)]
#[path = "app_tests.rs"]
mod tests;

// 2026-09-26: Every global binding, driven key by key through `on_key`.
#[cfg(test)]
#[path = "app_keys_tests.rs"]
mod key_tests;
