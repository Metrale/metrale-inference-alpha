// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The recipe cards for one model (left) and the selected recipe's
//! detail (right), led by its description in full.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use ratatui::Frame;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::Paragraph;

use super::super::{panel, wrap};
use crate::recipe::Recipe;
use crate::tui::app::App;
use crate::tui::theme;

pub fn draw(f: &mut Frame, app: &App, area: Rect) {
    let cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(46), Constraint::Percentage(54)])
        .split(area);
    draw_cards(f, app, cols[0]);
    draw_detail(f, app, cols[1]);
}

/// 2026-09-26: The chips that tell sibling recipes apart: starting point,
/// quantization, topology (`EP=n` or `1 node`) and model size.
fn chips(recipe: &Recipe) -> Vec<Span<'static>> {
    let mut out = Vec::new();
    let mut chip = |text: String, style: Style| {
        out.push(Span::styled(
            format!(" {text} "),
            style.add_modifier(Modifier::REVERSED | Modifier::BOLD),
        ));
        out.push(Span::raw(" "));
    };
    // 2026-09-26: First, so a clipped line still shows it. Under `NO_COLOR`
    // the reversed block and the words remain.
    if recipe.starting_point.is_some() {
        chip("starting point".into(), theme::warn());
    }
    if !recipe.quantization.is_empty() {
        chip(recipe.quantization.clone(), theme::brand_cyan());
    }
    // 2026-09-26: Topology is shown for single-node recipes too.
    chip(
        if recipe.min_nodes > 1 {
            format!("EP={}", recipe.min_nodes)
        } else {
            "1 node".to_string()
        },
        if recipe.min_nodes > 1 {
            theme::warn()
        } else {
            theme::dim()
        },
    );
    if !recipe.model_params.is_empty() {
        chip(recipe.model_params.clone(), theme::brand_purple());
    }
    out
}

/// 2026-09-26: The recipe id after its last `/`.
fn stem(recipe: &Recipe) -> &str {
    recipe.id.rsplit('/').next().unwrap_or(&recipe.id)
}

fn draw_cards(f: &mut Frame, app: &App, area: Rect) {
    let recipes = app.lib.cards();
    let model = app
        .lib
        .current()
        .map(|e| e.model.clone())
        .unwrap_or_default();
    // 2026-09-26: `LibState::cards` returns either the model's recipes or its
    // starting points, so the first card decides the noun in the title.
    let synthesized = recipes.first().is_some_and(|r| r.starting_point.is_some());
    let noun = if synthesized {
        "starting point"
    } else {
        "recipe"
    };
    let block = panel(
        format!(
            "{} ─ {} {noun}{} ─",
            model.to_uppercase(),
            recipes.len(),
            if recipes.len() == 1 { "" } else { "s" }
        ),
        true,
    );
    let inner = block.inner(area);
    f.render_widget(block, area);
    let width = inner.width.saturating_sub(4) as usize;

    // 2026-09-26: Four lines per card: title, chips, container, spacer.
    let per_card = 4usize;
    let visible = (inner.height as usize / per_card).max(1);
    let first = app.lib.card.saturating_sub(visible.saturating_sub(1));

    let mut lines: Vec<Line> = Vec::new();
    for (i, recipe) in recipes.iter().enumerate().skip(first).take(visible) {
        let selected = i == app.lib.card;
        let bar = if selected {
            Span::styled("▌", theme::brand_purple())
        } else {
            Span::raw(" ")
        };
        let title_style = if selected {
            theme::text().add_modifier(Modifier::BOLD)
        } else {
            theme::text()
        };
        // 2026-09-26: A non-metrale recipe shows `⊘ <runtime>` on its title
        // line.
        let mut head = Line::from(vec![
            bar,
            Span::styled(format!(" {}", stem(recipe)), title_style),
            Span::styled(
                if recipe.is_metrale() {
                    String::new()
                } else {
                    format!("  ⊘ {}", recipe.runtime.as_deref().unwrap_or("non-metrale"))
                },
                theme::dim(),
            ),
        ]);
        if selected {
            head = head.style(theme::selected());
        }
        lines.push(head);

        let mut chip_line = vec![Span::raw("   ")];
        chip_line.extend(chips(recipe));
        lines.push(Line::from(chip_line));

        lines.push(Line::from(vec![
            Span::raw("   "),
            Span::styled(
                truncate(&recipe.container, width.saturating_sub(3)),
                theme::dim(),
            ),
        ]));
        lines.push(Line::from(""));
    }
    f.render_widget(Paragraph::new(lines), inner);
}

fn draw_detail(f: &mut Frame, app: &App, area: Rect) {
    let Some(recipe) = app.lib.selected_card() else {
        f.render_widget(panel("RECIPE ─".into(), false), area);
        return;
    };
    let block = panel(format!("{} ─", stem(recipe).to_uppercase()), false);
    let inner = block.inner(area);
    f.render_widget(block, area);
    let width = inner.width.saturating_sub(2) as usize;

    let mut lines: Vec<Line> = wrap(&recipe.description, width, theme::text2());
    lines.push(Line::from(""));
    for (label, value) in [
        ("model", recipe.model.clone()),
        ("maintainer", recipe.maintainer.clone()),
        // 2026-09-26: `date_text` is empty for a starting point or an undated
        // recipe, and the loop below skips empty values.
        ("updated", app.lib.date_text(recipe)),
        ("kv cache", recipe.kv_dtype.clone()),
        ("container", recipe.container.clone()),
    ] {
        if value.is_empty() {
            continue;
        }
        lines.push(Line::from(vec![
            Span::styled(format!("  {label:<12}"), theme::dim()),
            Span::styled(value, theme::text()),
        ]));
    }
    lines.push(Line::from(""));
    lines.push(Line::from(Span::styled(
        format!(" {} SETTINGS", recipe.defaults.len()),
        theme::dim(),
    )));
    for (key, value) in recipe.defaults.iter().take(8) {
        lines.push(Line::from(vec![
            Span::styled(format!("  {key:<24}"), theme::dim()),
            Span::styled(value.clone(), theme::text()),
        ]));
    }
    if recipe.defaults.len() > 8 {
        lines.push(Line::from(Span::styled(
            format!("   … {} more", recipe.defaults.len() - 8),
            theme::dim(),
        )));
    }
    lines.push(Line::from(""));
    // 2026-09-26: The last line says whether the dashboard can start this
    // recipe: only a metrale recipe with `min_nodes <= 1`.
    let launchable = recipe.is_metrale() && recipe.min_nodes <= 1;
    lines.push(Line::from(Span::styled(
        if launchable {
            " ⏎ configure and start".to_string()
        } else if !recipe.is_metrale() {
            " this runtime cannot be launched from here".to_string()
        } else {
            format!(
                " needs {} nodes — the dashboard starts single-node runs only",
                recipe.min_nodes
            )
        },
        if launchable {
            theme::brand_cyan()
        } else {
            theme::warn()
        },
    )));
    f.render_widget(Paragraph::new(lines), inner);
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max || max == 0 {
        return s.to_string();
    }
    s.chars().take(max.saturating_sub(1)).collect::<String>() + "…"
}
