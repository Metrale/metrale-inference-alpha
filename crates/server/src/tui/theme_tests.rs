// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests of colour-depth resolution and of the styles that must
//! keep a signal without colour. `depth_of` takes the two variables as values
//! so these need no `set_var`: the environment is process-global, and
//! `logo_tests.rs` sets `COLORTERM` for the whole test binary.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use super::*;

#[test]
fn a_preference_outranks_a_capability() {
    // 2026-09-26: The terminal advertises 24-bit and `NO_COLOR` is set.
    assert_eq!(
        depth_of(Some("1"), Some("truecolor")),
        Depth::None,
        "COLORTERM describes the terminal, NO_COLOR describes the user"
    );
}

/// 2026-09-26: `NO_COLOR` is not a boolean: `NO_COLOR=0` means no colour,
/// and only an empty value is ignored.
#[test]
fn no_color_is_presence_not_truth() {
    for set_to in ["1", "0", "false", "no", "yes", "anything at all"] {
        assert_eq!(
            depth_of(Some(set_to), None),
            Depth::None,
            "NO_COLOR={set_to}"
        );
    }
    assert_eq!(
        depth_of(Some(""), Some("truecolor")),
        Depth::True,
        "an empty value is explicitly not set"
    );
    assert_eq!(depth_of(None, Some("truecolor")), Depth::True);
}

#[test]
fn colorterm_still_picks_the_fallback_when_colour_is_allowed() {
    assert_eq!(depth_of(None, Some("24bit")), Depth::True);
    assert_eq!(depth_of(None, Some("truecolor")), Depth::True);
    assert_eq!(depth_of(None, None), Depth::Ansi256);
    assert_eq!(depth_of(None, Some("8bit")), Depth::Ansi256);
}

/// 2026-09-26: Under `NO_COLOR`, `selected()`, `border(true)` and `warn()`
/// carry a modifier instead of a hue. The live branch asserted is whichever
/// the test process's environment selects.
#[test]
fn the_signals_that_are_only_colour_get_a_modifier_instead() {
    let colourless = |s: Style| s.fg.is_none() && s.bg.is_none();

    let sel = Style::default().add_modifier(Modifier::REVERSED);
    assert!(colourless(sel) && sel.add_modifier.contains(Modifier::REVERSED));

    assert_eq!(
        depth_of(None, Some("truecolor")),
        Depth::True,
        "the coloured branch is the one the palette constants describe"
    );

    // 2026-09-26: The live functions agree with whichever branch this
    // process is in.
    match depth() {
        Depth::None => {
            assert!(colourless(selected()), "{:?}", selected());
            assert!(selected().add_modifier.contains(Modifier::REVERSED));
            assert!(border(true).add_modifier.contains(Modifier::BOLD));
            assert!(warn().add_modifier.contains(Modifier::BOLD));
            assert_eq!(gradient_at(0.5), Color::Reset);
            assert_eq!(glow(3), Color::Reset);
            assert_eq!(TEXT.color(), Color::Reset);
        }
        _ => {
            assert_eq!(selected().bg, Some(BG_SELECTION.color()));
            assert_eq!(border(true).fg, Some(CYAN.color()));
            assert_ne!(TEXT.color(), Color::Reset);
        }
    }
}
