// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Library actions that need more than the Library's own state.
//!
//! The Library reducer (`lib_keys`) returns an `Outcome`; `App`, which holds
//! the cache root, the server handle and the download state, performs it:
//! download, the one-download-at-a-time question, the freshness check, the
//! recipe-store attach and the recipe launch.
//!
//! Owner: server tui.
//! Invariants:
//! - At most one download job exists: a second model is queued in
//!   `pending_start` and started only once `download.job` is empty.

use crossterm::event::{KeyCode, KeyEvent};

use super::app::{App, MainSub};
use super::section::Section;

impl App {
    /// 2026-09-26: The HF cache root these actions operate on.
    pub(super) fn cache_root(&self) -> Option<std::path::PathBuf> {
        crate::model_resolver::resolve_cache_root(self.args.cache_dir.as_deref()).ok()
    }

    /// 2026-09-26: Which model the Library is pointing at, for a download or a check.
    pub(super) fn selected_model(&self) -> Option<String> {
        self.lib.current().map(|e| e.model.clone())
    }

    /// 2026-09-26: Download, resume, or update the selected model.
    pub(super) fn download_selected_model(&mut self) {
        let Some(model) = self.selected_model() else {
            self.toast("no model selected".to_string(), true);
            return;
        };
        let Some(root) = self.cache_root() else {
            self.toast(
                "no HuggingFace cache directory to download into".to_string(),
                true,
            );
            return;
        };
        // 2026-09-26: `DownloadState::start` refuses a second download; ask whether to switch instead.
        if let Some(running) = self.download.job.as_ref().map(|j| j.repo.clone())
            && running != model
        {
            self.download_switch = Some((running, model));
            return;
        }
        let (text, error) = self.download.start(&model, root);
        self.toast(text, error);
    }

    /// 2026-09-26: Answer the "one download at a time" question; `false` when none is open.
    ///
    /// Only `x`, `y` or `Y` switches; every other key keeps the running
    /// download, the same rule as `answer_quit_prompt`. `x` because the help
    /// overlay lists it as "stop the running download"; `y` because the quit
    /// prompt uses it.
    pub(super) fn answer_download_switch(&mut self, key: KeyEvent) -> bool {
        let Some((running, wanted)) = self.download_switch.take() else {
            return false;
        };
        if matches!(
            key.code,
            KeyCode::Char('x') | KeyCode::Char('y') | KeyCode::Char('Y')
        ) {
            // 2026-09-26: `cancel` returns None when the job settled between the question and the answer.
            if let Some((text, error)) = self.download.cancel() {
                self.toast(text, error);
            }
            // 2026-09-26: Queue rather than start: a first cancel leaves `download.job` set until `pump`
            // sees the worker settle, and `start` refuses while it is set.
            self.pending_start = Some(wanted.clone());
            self.toast(format!("{wanted} starts when {running} settles"), false);
        }
        true
    }

    /// 2026-09-26: Start a queued download once `download.job` is empty, however the previous one ended.
    ///
    /// Called on every pass of the event loop, after `download.pump`.
    pub(super) fn start_pending_download(&mut self) {
        if self.download.job.is_some() {
            return;
        }
        let Some(model) = self.pending_start.take() else {
            return;
        };
        let Some(root) = self.cache_root() else {
            self.toast(
                "no HuggingFace cache directory to download into".to_string(),
                true,
            );
            return;
        };
        let (text, error) = self.download.start(&model, root);
        self.toast(text, error);
    }

    /// 2026-09-26: Ask the Hub whether the selected model is behind.
    pub(super) fn check_selected_model(&mut self) {
        let Some(model) = self.selected_model() else {
            self.toast("no model selected".to_string(), true);
            return;
        };
        let Some(root) = self.cache_root() else {
            self.toast("no HuggingFace cache directory".to_string(), true);
            return;
        };
        let (text, error) = self.download.check(&model, root);
        self.toast(text, error);
    }

    /// 2026-09-26: First entry into the Library: point it at the recipe store and start the recipe fetch.
    ///
    /// On failure it sets `recipes_unavailable`, which stops `tick_work` from
    /// calling it again: `ArtifactStore::discover` reads only `METRALE_HOME`
    /// and `HOME`, so a retry in the same process fails the same way. The
    /// `rebuild` still lists the locally scanned models.
    pub(super) fn attach_recipes(&mut self) {
        match metrale_bench::ArtifactStore::discover() {
            Ok(store) => {
                self.lib.attach(store.root().to_path_buf(), &self.library);
                self.lib.refresh();
            }
            Err(e) => {
                tracing::warn!("recipes unavailable: {e:#}");
                self.lib.recipes_unavailable = true;
                self.lib.rebuild(&self.library);
            }
        }
    }

    /// 2026-09-26: Start the configured recipe and switch to Main ▸ Overview, where the load checklist is drawn.
    ///
    /// `progress` is reset first, or a second load would render with every
    /// phase still `Done` from the first.
    pub(super) fn launch_selected_recipe(&mut self) {
        let Some(host) = self.host.clone() else {
            self.toast("no server attached to this dashboard".to_string(), true);
            return;
        };
        match self.lib.launch(host) {
            Ok(()) => {
                self.progress.reset();
                // 2026-09-26: A load is in flight, so the checklist tracks it again.
                self.awaiting_model = false;
                self.repaint = true;
                self.section = Section::Main;
                self.main_sub = MainSub::Overview;
            }
            Err(e) => self.toast(e, true),
        }
    }
}

#[cfg(test)]
#[path = "app_library_tests.rs"]
mod tests;
