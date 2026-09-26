// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Library section state: the joined catalogue, the recipe index refresh, and the config form.
//!
//! Owner: server tui.
//! Invariants:
//! - No method here waits on the network or on a model load: the index refresh
//!   and the swap run on their own threads and are read with `try_recv`.

use std::collections::BTreeMap;
use std::sync::mpsc::Receiver;

use crate::recipe::fetch::{self, Index};
use crate::tui::data::catalogue::{self, Entry};
use crate::tui::data::library::LibraryEntry;

/// 2026-09-26: Which pane of the Library is showing: `List` (one row per
/// model), then `Cards` (the selected model's recipes), then `Config` (one
/// recipe's settings). `open_cards` goes to `Cards` even for a model with one
/// recipe.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum View {
    #[default]
    List,
    Cards,
    Config,
}

#[derive(Default)]
pub struct LibState {
    pub view: View,
    pub index: Index,
    pub rows: Vec<Entry>,
    pub selected: usize,
    pub filter: String,
    pub filter_editing: bool,
    /// 2026-09-26: Index into `cards()` for the current model.
    pub card: usize,
    /// 2026-09-26: True while an index refresh is in flight; the list title
    /// shows a spinner for it.
    pub fetching: bool,
    pending: Option<Receiver<Index>>,
    /// 2026-09-26: Cancels the in-flight index refresh. Set and cleared
    /// together with `pending`.
    fetch_cancel: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// 2026-09-26: An in-flight scan of the local HF cache.
    pub(super) pending_scan: Option<Receiver<Vec<LibraryEntry>>>,
    /// 2026-09-26: The local cache needs re-scanning. Set by the reducer, which
    /// has no access to `App`; the event loop drains it into
    /// `App::library_dirty`.
    pub mark_dirty: bool,
    /// 2026-09-26: Starting points synthesized for a no-recipe model (see
    /// `lib_start`): the model they were built for, and the cards. `cards()`
    /// returns them only while that model is the current row.
    pub(super) starting: Option<(String, Vec<crate::recipe::Recipe>)>,
    /// 2026-09-26: The recipe id whose date is being looked up. While it is
    /// `Some`, `date_text` shows a placeholder for that recipe. One lookup at a
    /// time, for the recipe on screen (`want_date_for`).
    pub dating: Option<String>,
    pub(super) pending_date: Option<Receiver<(String, Option<String>)>>,
    /// 2026-09-26: Recipe ids already looked up, including failed lookups, so a
    /// later tick does not ask again.
    pub(super) dated: std::collections::HashSet<String>,
    /// 2026-09-26: Dates fetched from GitHub this session, by recipe id. Kept
    /// out of `index.recipes` because `poll` replaces `index` wholesale.
    pub(super) fetched_dates: std::collections::HashMap<String, String>,
    /// 2026-09-26: The loader thread's failure message, if a launch is in
    /// flight. `launch` returns once the thread is spawned; a swap that fails
    /// later reports here.
    launch_result: Option<Receiver<String>>,
    /// 2026-09-26: The store root, so a refresh knows where the cache lives.
    pub(super) root: Option<std::path::PathBuf>,
    /// 2026-09-26: There is no recipe store to attach, and there will not be
    /// one. `ArtifactStore::discover()` fails only when `METRALE_HOME` is set
    /// but empty, or when it is unset and `HOME` is unset or empty, so a retry
    /// cannot succeed; `events_rules::tick_work` stops attaching once this is set.
    pub(super) recipes_unavailable: bool,

    /// 2026-09-26: Config-form edits, keyed as the recipe keys them. Only
    /// edited keys appear; a key the recipe does not list is an added setting
    /// (`config_rows`).
    pub overrides: BTreeMap<String, String>,
    /// 2026-09-26: Recipe keys the user removed, so the flag is not passed.
    /// Removing a key also drops its override (`toggle_removed`).
    pub removed: std::collections::BTreeSet<String>,
    /// 2026-09-26: The open picker, if any. While one is open it receives
    /// every key (`lib_keys`).
    pub modal: Option<crate::tui::lib_modal::ConfigModal>,
    /// 2026-09-26: The key of a free-text setting being added that clap
    /// declares no default for, while its value is typed. No override exists
    /// until the value is committed.
    pub pending_add: Option<String>,
    /// 2026-09-26: The donor of borrowed parameters, as "`<recipe id>`
    /// (measured on `<model>`)", set when a donor's parameters are applied over
    /// this form. Independent of `Recipe::starting_point`: a synthesized card
    /// can also carry borrowed values.
    pub borrowed: Option<String>,
    pub row: usize,
    pub editing: bool,
    pub edit_buffer: String,
    /// 2026-09-26: Why the current form will not launch, if it will not.
    pub error: Option<String>,
}

impl LibState {
    /// 2026-09-26: Point the Library at a store and show the index already
    /// cached on disk (a synchronous local read).
    pub fn attach(&mut self, root: std::path::PathBuf, local: &[LibraryEntry]) {
        self.index = fetch::cached(&root);
        self.root = Some(root);
        self.rebuild(local);
    }

    /// 2026-09-26: Start a background refresh. A call while one is in flight
    /// is ignored.
    pub fn refresh(&mut self) {
        if self.fetching {
            return;
        }
        let Some(root) = &self.root else {
            return;
        };
        let (rx, cancel) = fetch::refresh_in_background(root);
        self.pending = Some(rx);
        self.fetch_cancel = Some(cancel);
        self.fetching = true;
    }

    /// 2026-09-26: Stop an in-flight refresh, if there is one. Called when the
    /// dashboard exits; the fetch workers check the flag between files, so
    /// each finishes at most the request it is in.
    pub fn cancel_refresh(&self) {
        if let Some(c) = &self.fetch_cancel {
            c.store(true, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// 2026-09-26: Poll the refresh; returns true when a new index arrived.
    pub fn poll(&mut self, local: &[LibraryEntry]) -> bool {
        let Some(rx) = &self.pending else {
            return false;
        };
        match rx.try_recv() {
            Ok(index) => {
                self.index = index;
                self.pending = None;
                self.fetch_cancel = None;
                self.fetching = false;
                self.rebuild(local);
                true
            }
            Err(std::sync::mpsc::TryRecvError::Empty) => false,
            // 2026-09-26: The fetcher thread ended without sending. Stop the
            // spinner; the cached list stays on screen.
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.pending = None;
                self.fetch_cancel = None;
                self.fetching = false;
                false
            }
        }
    }

    /// 2026-09-26: Re-join, keeping the selection on the model the user had,
    /// not on its row index: `catalogue::join` re-sorts (runnable now, then
    /// with a recipe, then the rest; by model id within each), so the same
    /// index can name another model after a refresh.
    pub fn rebuild(&mut self, local: &[LibraryEntry]) {
        let anchor = self.current().map(|e| e.model.clone());
        let card_anchor = self.selected_card().map(|r| r.id.clone());
        self.rows = catalogue::join(&self.index.recipes, local);

        if let Some(model) = anchor {
            match self.visible().iter().position(|e| e.model == model) {
                Some(i) => self.selected = i,
                // 2026-09-26: The model is no longer listed; step back to the
                // list rather than show another model's cards.
                None => self.view = View::List,
            }
        }
        // 2026-09-26: Same for the selected recipe: if it is gone, drop its
        // form edits instead of carrying them onto the recipe that inherits
        // its index.
        if let Some(id) = card_anchor {
            match self.cards().iter().position(|r| r.id == id) {
                Some(i) => self.card = i,
                None => {
                    self.overrides.clear();
                    self.removed.clear();
                    self.modal = None;
                    self.pending_add = None;
                    self.borrowed = None;
                    self.editing = false;
                    if self.view == View::Config {
                        self.view = View::Cards;
                    }
                }
            }
        }
        self.clamp();
    }

    fn clamp(&mut self) {
        let n = self.visible().len();
        self.selected = self.selected.min(n.saturating_sub(1));
        // 2026-09-26: `card` and `row` index into lists that a refresh can
        // shorten.
        let cards = self.cards().len();
        self.card = self.card.min(cards.saturating_sub(1));
        let rows = self.config_rows().len();
        self.row = self.row.min(rows.saturating_sub(1));
    }

    /// 2026-09-26: Take the loader's failure, if it reported one.
    pub fn poll_launch(&mut self) -> Option<String> {
        let rx = self.launch_result.as_ref()?;
        match rx.try_recv() {
            Ok(msg) => {
                self.launch_result = None;
                Some(msg)
            }
            // 2026-09-26: Empty: still loading. Disconnected: the loader
            // thread ended without sending a failure.
            Err(std::sync::mpsc::TryRecvError::Empty) => None,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                self.launch_result = None;
                None
            }
        }
    }

    /// 2026-09-26: Rows passing the filter.
    pub fn visible(&self) -> Vec<&Entry> {
        self.rows
            .iter()
            .filter(|r| r.matches(&self.filter))
            .collect()
    }

    pub fn current(&self) -> Option<&Entry> {
        self.visible().get(self.selected).copied()
    }

    pub fn move_selection(&mut self, delta: isize) {
        let n = self.visible().len();
        if n == 0 {
            return;
        }
        let next = self.selected as isize + delta;
        self.selected = next.clamp(0, n as isize - 1) as usize;
    }

    /// 2026-09-26: Open the card view for the selected model. A model with no
    /// recipe opens on synthesized starting points (`lib_start`).
    pub fn open_cards(&mut self) -> Result<(), String> {
        let Some(entry) = self.current() else {
            return Err("nothing selected".into());
        };
        if !entry.has_recipe() {
            self.open_starting_points();
        }
        self.card = 0;
        self.view = View::Cards;
        Ok(())
    }

    /// 2026-09-26: The cards for the selected model: its recipes, or, when it
    /// has none, the starting points built for it. Once a refresh brings a
    /// recipe for the model, its recipes replace the starting points, and
    /// `rebuild` drops an open form whose card is gone.
    pub fn cards(&self) -> &[crate::recipe::Recipe] {
        match self.current() {
            Some(entry) if entry.has_recipe() => &entry.recipes,
            Some(entry) => match &self.starting {
                Some((for_model, cards)) if *for_model == entry.model => cards,
                _ => &[],
            },
            None => &[],
        }
    }

    /// 2026-09-26: The card the cursor is on.
    pub fn selected_card(&self) -> Option<&crate::recipe::Recipe> {
        self.cards().get(self.card)
    }

    /// 2026-09-26: Open the config form for the selected card, clearing any
    /// previous edits. Refused for a recipe of another runtime.
    pub fn open_config(&mut self) -> Result<(), String> {
        let Some(recipe) = self.selected_card() else {
            return Err("nothing selected".into());
        };
        if !recipe.is_metrale() {
            return Err(format!(
                "{} is a {} recipe and cannot be configured here",
                recipe.id,
                recipe.runtime.as_deref().unwrap_or("non-metrale")
            ));
        }
        self.overrides.clear();
        self.removed.clear();
        self.modal = None;
        self.pending_add = None;
        self.borrowed = None;
        self.error = None;
        self.row = 0;
        self.editing = false;
        self.view = View::Config;
        Ok(())
    }

    /// 2026-09-26: The recipe backing the config form: the selected card.
    pub fn config_recipe(&self) -> Option<&crate::recipe::Recipe> {
        self.selected_card()
    }

    /// 2026-09-26: Start the selected recipe, replacing whatever is running.
    ///
    /// The swap runs on the `metrale-swap` thread so the render loop keeps
    /// drawing; `model_swap::swap` goes through the same `load_model` as boot.
    /// Returns once the thread is spawned; a later failure arrives through
    /// `poll_launch`.
    pub fn launch(
        &mut self,
        host: std::sync::Arc<crate::main_modules::model_host::ModelHost>,
    ) -> Result<(), String> {
        let recipe = self.config_recipe().ok_or("no recipe selected")?;
        // 2026-09-26: Refuse before anything is torn down when the weights are
        // not on disk.
        if !self.selected_has_weights() {
            return Err(format!(
                "{} is not downloaded — press Esc, then d to download it",
                recipe.model
            ));
        }
        // 2026-09-26: Validate on this thread so a bad recipe is reported
        // before a swap starts; `swap` validates again.
        let args = recipe
            .serve_args_edited(&self.overrides, &self.removed)
            .map_err(|e| problem_line(&format!("{e:#}")))?;
        let previous = host.current().map(|s| s.model_name.clone());
        // 2026-09-26: The loader sends only a failure message; `poll_launch`
        // reads it.
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        self.launch_result = Some(rx);
        std::thread::Builder::new()
            .name("metrale-swap".into())
            .spawn(move || {
                if let Err(e) = crate::main_modules::model_swap::swap(&host, args) {
                    tracing::error!(
                        "swap failed{}: {e:#}",
                        previous
                            .as_deref()
                            .map(|p| format!(" (was serving {p})"))
                            .unwrap_or_default()
                    );
                    // 2026-09-26: A disconnected receiver means the dashboard
                    // moved on.
                    let _ = tx.send(format!("{e:#}"));
                }
            })
            .map_err(|e| format!("could not start the loader thread: {e}"))?;
        Ok(())
    }

    /// 2026-09-26: A launch's result channel has not settled yet. The quit
    /// guard (`app_quit`) asks this.
    pub fn launch_in_flight(&self) -> bool {
        self.launch_result.is_some()
    }

    /// 2026-09-26: The selected model's weights are on disk: a `.safetensors`
    /// shard and a `refs/main` (`data::library`), so an unfinished download
    /// reads as false.
    pub fn selected_has_weights(&self) -> bool {
        self.current().is_some_and(|e| e.has_weights())
    }
}

/// 2026-09-26: Reduce a validation report to the one line a form field can show.
///
/// `format_violations` writes a "Metrale Engine CLI: …" header, then per
/// violation `[n] <what>`, `why:` and `fix:` lines. This returns the first
/// `what`, joined with its `fix` when there is one. Text without a `[n]` line
/// (clap's own errors) yields its first non-empty line that is not that header.
pub(crate) fn problem_line(s: &str) -> String {
    let lines: Vec<&str> = s.lines().map(str::trim).collect();
    let what = lines
        .iter()
        .find_map(|l| l.strip_prefix("[").and_then(|r| r.split_once("] ")))
        .map(|(_, what)| what);
    let fix = lines
        .iter()
        .find_map(|l| l.strip_prefix("fix: "))
        .filter(|f| !f.is_empty());
    match (what, fix) {
        (Some(what), Some(fix)) => format!("{what} — {fix}"),
        (Some(what), None) => what.to_string(),
        (None, _) => lines
            .iter()
            .find(|l| !l.is_empty() && !l.starts_with("Metrale Engine CLI:"))
            .unwrap_or(&"invalid")
            .to_string(),
    }
}

#[cfg(test)]
#[path = "lib_state_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "lib_state_more_tests.rs"]
mod more_tests;

#[cfg(test)]
#[path = "lib_state_recipe_tests.rs"]
mod recipe_tests;

#[cfg(test)]
#[path = "lib_state_list_tests.rs"]
mod list_tests;
