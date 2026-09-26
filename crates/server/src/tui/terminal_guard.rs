// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Taking the terminal for the TUI (raw mode, alternate screen,
//! mouse capture, bracketed paste) and giving it back.
//!
//! - [`TerminalGuard`] takes the terminal in `enter` and restores it on drop.
//! - [`restore`] is idempotent (the `taken` flag), so the guard, the panic
//!   hook and `main`'s exit paths can all call it.
//! - [`install_panic_hook`] chains a process-wide hook that restores the
//!   terminal first, then prints the newest log-ring lines and the tee file's
//!   path to stderr, then runs the previous hook.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};

use crossterm::event::{
    DisableBracketedPaste, DisableMouseCapture, EnableBracketedPaste, EnableMouseCapture,
};
use crossterm::terminal::{
    EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode,
};

/// 2026-09-26: What this process has done to the terminal. Plain atomics, so
/// the panic hook reads them without blocking.
struct Terminal {
    /// 2026-09-26: Set by a successful `enter`; `restore()` clears it.
    taken: AtomicBool,
    /// 2026-09-26: Saved dup of the original stderr fd while it is
    /// redirected (-1 = not).
    orig_stderr: std::sync::atomic::AtomicI32,
}

// 2026-09-26: Static because the terminal belongs to the process, and
// `restore()` is called from the panic hook, the guard's drop and `main`'s
// exit paths, which share no object it could be carried on.
static TERM: Terminal = Terminal {
    taken: AtomicBool::new(false),
    orig_stderr: std::sync::atomic::AtomicI32::new(-1),
};

/// 2026-09-26: While the TUI owns the screen, fd 2 is pointed at the tee file
/// (when one is open), so an `eprintln!` from another crate or a C library
/// does not draw over the raw-mode frame. `restore()` puts the real stderr
/// back before the panic hook prints. Off Unix this is a no-op and such
/// output is not captured.
#[cfg(not(unix))]
fn redirect_stderr_to_tee() {}

#[cfg(not(unix))]
fn unredirect_stderr() {}

#[cfg(unix)]
fn redirect_stderr_to_tee() {
    if let Some(tee_fd) = super::init::tee_raw_fd() {
        // 2026-09-26: SAFETY: dup/dup2/close on fd 2, the tee fd and the dup
        // made here; the saved fd is closed in `unredirect_stderr`.
        unsafe {
            let orig = libc::dup(2);
            if orig >= 0 && libc::dup2(tee_fd, 2) >= 0 {
                TERM.orig_stderr.store(orig, Ordering::SeqCst);
            } else if orig >= 0 {
                libc::close(orig);
            }
        }
    }
}

#[cfg(unix)]
fn unredirect_stderr() {
    let orig = TERM.orig_stderr.swap(-1, Ordering::SeqCst);
    if orig >= 0 {
        // 2026-09-26: SAFETY: restores and closes the fd saved by
        // `redirect_stderr_to_tee`.
        unsafe {
            libc::dup2(orig, 2);
            libc::close(orig);
        }
    }
}

/// 2026-09-26: Put the terminal back: stderr, raw mode, bracketed paste,
/// mouse capture, the alternate screen and the cursor. Only the first call
/// after an `enter` does anything; callable from any thread and from the
/// panic hook. Errors are ignored so every step is tried.
pub fn restore() {
    if !TERM.taken.swap(false, Ordering::SeqCst) {
        return;
    }
    unredirect_stderr();
    let _ = disable_raw_mode();
    let mut out = std::io::stdout();
    let _ = crossterm::execute!(
        out,
        DisableBracketedPaste,
        DisableMouseCapture,
        LeaveAlternateScreen
    );
    let _ = crossterm::execute!(out, crossterm::cursor::Show);
    let _ = out.flush();
}

/// 2026-09-26: Terminal ownership for the TUI thread; dropping it calls
/// [`restore`].
pub struct TerminalGuard;

impl TerminalGuard {
    /// 2026-09-26: Enter raw mode, the alternate screen, mouse capture and
    /// bracketed paste, then redirect stderr.
    pub fn enter() -> std::io::Result<Self> {
        enable_raw_mode()?;
        // 2026-09-26: With bracketed paste a paste arrives whole as one
        // `Event::Paste`; without it each pasted newline would be an Enter key.
        crossterm::execute!(
            std::io::stdout(),
            EnterAlternateScreen,
            EnableMouseCapture,
            EnableBracketedPaste
        )?;
        TERM.taken.store(true, Ordering::SeqCst);
        redirect_stderr_to_tee();
        Ok(Self)
    }
}

impl Drop for TerminalGuard {
    fn drop(&mut self) {
        restore();
    }
}

/// 2026-09-26: Install the chained panic hook: it restores the terminal,
/// prints the last 50 captured log lines and the tee file's path, then runs
/// the previous hook for the message and backtrace.
///
/// `events::run` calls it before `TerminalGuard::enter`. Each call chains one
/// more hook.
pub fn install_panic_hook() {
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        // 2026-09-26: Restore first, so what follows prints on a normal
        // screen.
        restore();
        let mut err = std::io::stderr();
        let _ = writeln!(err, "\n── metrale-tui: panic — last log lines ──");
        super::log_ring::dump_to(&mut err, 50);
        if let Some(p) = super::init::tee_file_path() {
            let _ = writeln!(err, "── full log: {p} ──");
        }
        previous(info);
    }));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn restore_is_idempotent_when_never_taken() {
        // 2026-09-26: Never entered: `restore` returns before touching the
        // terminal.
        assert!(!TERM.taken.load(Ordering::SeqCst));
        restore();
        restore();
        assert!(!TERM.taken.load(Ordering::SeqCst));
    }
}
