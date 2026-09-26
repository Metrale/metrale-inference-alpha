// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Header logo (three chevrons in purple, cyan and green) and the Main tab's flag chips.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.
//!
//! `render::header` draws [`three_line`] on a tall header and [`one_line`]
//! otherwise, passing a wave step until the listener is ready and a model is
//! loaded, and `None` after.

use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};

use super::theme;

/// 2026-09-26: Rows of one half-block chevron cell; [`three_line`] draws three
/// of them side by side.
pub const CHEVRON_ROWS: [&str; 3] = ["▀█▄ ", "  ██", "▄█▀ "];

/// 2026-09-26: A brand color with each RGB channel scaled to 45%, for the
/// chevrons the wave is not on.
fn dimmed(c: theme::C) -> Color {
    let s = |v: u8| ((v as f64) * 0.45) as u8;
    match c.color() {
        Color::Rgb(r, g, b) => Color::Rgb(s(r), s(g), s(b)),
        other => other, // 2026-09-26: indexed or reset colors are not dimmed.
    }
}

/// 2026-09-26: Per-chevron colors for wave step `wave`: one chevron at full
/// brightness, the others dimmed; `None` gives all three at full brightness.
fn chevron_colors(wave: Option<usize>) -> [Color; 3] {
    let brand = [theme::PURPLE, theme::CYAN, theme::GREEN];
    match wave {
        None => brand.map(|c| c.color()),
        Some(step) => {
            let bright = step % 3;
            let mut out = [Color::Reset; 3];
            for (i, c) in brand.iter().enumerate() {
                out[i] = if i == bright { c.color() } else { dimmed(*c) };
            }
            out
        }
    }
}

/// 2026-09-26: The 1-line logo: `❯❯❯ Metrale Engine`.
pub fn one_line(wave: Option<usize>) -> Line<'static> {
    let colors = chevron_colors(wave);
    let mut spans: Vec<Span> = colors
        .iter()
        .map(|c| Span::styled("❯", Style::default().fg(*c).add_modifier(Modifier::BOLD)))
        .collect();
    spans.push(Span::styled(
        " Metrale Engine",
        theme::text().add_modifier(Modifier::BOLD),
    ));
    Line::from(spans)
}

/// 2026-09-26: The 3-row logo, with the wordmark on rows 1 and 2.
pub fn three_line(wave: Option<usize>) -> [Line<'static>; 3] {
    let colors = chevron_colors(wave);
    let row = |r: usize, with_wordmark: Option<(&'static str, Style)>| -> Line<'static> {
        let mut spans: Vec<Span> = Vec::with_capacity(7);
        spans.push(Span::raw(" "));
        for (i, color) in colors.iter().enumerate() {
            spans.push(Span::styled(CHEVRON_ROWS[r], Style::default().fg(*color)));
            if i < 2 {
                spans.push(Span::raw("  "));
            }
        }
        if let Some((text, style)) = with_wordmark {
            spans.push(Span::raw("   "));
            spans.push(Span::styled(text, style));
        }
        Line::from(spans)
    };
    [
        row(0, None),
        row(
            1,
            Some(("M E T R A L E", theme::text().add_modifier(Modifier::BOLD))),
        ),
        row(2, Some(("E N G I N E", theme::dim()))),
    ]
}

/// 2026-09-26: One header chip: its text, and the tint the Main tab colors it
/// with.
#[derive(Clone, Debug)]
pub struct Badge {
    pub text: String,
    pub tint: BadgeTint,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BadgeTint {
    Model,
    Quant,
    Role,
    Neutral,
}

/// 2026-09-26: The Main tab's chips, derived from `ServeArgs`, in display order.
///
/// With `awaiting_model` only the no-model chip and the port are returned:
/// every other chip describes a loaded model, and `ServeArgs` holds clap
/// defaults whether or not one is serving.
pub fn badges(a: &crate::cli::ServeArgs, awaiting_model: bool) -> Vec<Badge> {
    let mut out = Vec::new();
    if awaiting_model {
        out.push(Badge {
            text: "no model · press 4 for Library".into(),
            tint: BadgeTint::Neutral,
        });
        out.push(Badge {
            text: format!(":{}", a.port),
            tint: BadgeTint::Neutral,
        });
        return out;
    }
    let model = a
        .model_name
        .clone()
        .or_else(|| a.model.clone())
        .unwrap_or_else(|| "<model>".into());
    out.push(Badge {
        text: model,
        tint: BadgeTint::Model,
    });
    out.push(Badge {
        text: format!(
            "kv {} · lm {} · mtp {}",
            // 2026-09-26: These are unresolved args; an omitted
            // `--kv-cache-dtype` is resolved at load against the model's
            // behaviour default (`serve_phases::kv_cache`).
            a.kv_cache_dtype.as_deref().unwrap_or("auto"),
            a.lm_head_dtype,
            a.mtp_quantization
        ),
        tint: BadgeTint::Quant,
    });
    if a.dflash {
        out.push(Badge {
            text: match a.dflash_gamma {
                Some(g) => format!("DFlash γ={g}"),
                None => "DFlash γ=auto".to_string(),
            },
            tint: BadgeTint::Quant,
        });
    } else if a.speculative || a.self_speculative || a.ngram_speculative {
        out.push(Badge {
            // 2026-09-26: An omitted `--num-drafts` is resolved at load
            // against the model's `default_num_drafts`
            // (`apply_model_default_num_drafts`).
            text: match a.num_drafts {
                Some(n) => format!("MTP k={}", n + 1),
                None => "MTP k=auto".to_string(),
            },
            tint: BadgeTint::Quant,
        });
    } else {
        out.push(Badge {
            text: "spec off".into(),
            tint: BadgeTint::Neutral,
        });
    }
    out.push(Badge {
        text: format!("batch {}", a.max_batch_size),
        tint: BadgeTint::Neutral,
    });
    out.push(Badge {
        text: format!("ctx {}", human_tokens(a.max_seq_len)),
        tint: BadgeTint::Neutral,
    });
    let role = if a.rank == 0 { "head" } else { "worker" };
    out.push(Badge {
        text: format!(
            "{role} {}/{} · tp{} · ep{}",
            a.rank, a.world_size, a.tp_size, a.ep_size
        ),
        tint: BadgeTint::Role,
    });
    out.push(Badge {
        text: format!("sched {}", a.scheduler),
        tint: BadgeTint::Neutral,
    });
    // 2026-09-26: `--hermetic` also removes the prefix-cache chip below
    // (`prefix_caching_enabled`), so it gets a chip of its own to say why.
    if a.hermetic {
        out.push(Badge {
            text: "HERMETIC · KAT".to_string(),
            tint: BadgeTint::Quant,
        });
    }
    if a.prefix_caching_enabled() {
        out.push(Badge {
            text: format!(
                "prefix-cache · ssm {}@{}",
                a.ssm_cache_slots, a.ssm_checkpoint_interval
            ),
            tint: BadgeTint::Neutral,
        });
    }
    out.push(Badge {
        text: format!(":{}", a.port),
        tint: BadgeTint::Neutral,
    });
    out
}

fn human_tokens(n: usize) -> String {
    if n >= 1024 && n.is_multiple_of(1024) {
        format!("{}k", n / 1024)
    } else {
        n.to_string()
    }
}

#[cfg(test)]
#[path = "logo_tests.rs"]
mod more_tests;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chevron_rows_are_uniform_width() {
        // 2026-09-26: Every row must have the same width, or the chevrons
        // shear.
        let w = CHEVRON_ROWS[0].chars().count();
        assert!(CHEVRON_ROWS.iter().all(|r| r.chars().count() == w));
    }

    #[test]
    fn wave_brightens_one_chevron_at_a_time() {
        // 2026-09-26: In truecolor mode each wave step has exactly one
        // full-brightness chevron; indexed colors are not dimmed.
        unsafe { std::env::set_var("COLORTERM", "truecolor") };
        for step in 0..3 {
            let colors = chevron_colors(Some(step));
            let brand: Vec<Color> = [theme::PURPLE, theme::CYAN, theme::GREEN]
                .iter()
                .map(|c| c.color())
                .collect();
            let bright = colors.iter().zip(&brand).filter(|(a, b)| a == b).count();
            assert_eq!(bright, 1, "step {step}");
        }
    }
}
