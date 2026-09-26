// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The `met serve` flags the Config form can add or pick, read from clap's `ServeArgs` at runtime.
//!
//! Enumerated value sets come from `cli::flag_values::options_for_flag`, the
//! module whose lists `validate_serve_args` also checks against.
//!
//! Owner: server tui.
//! Invariants:
//! - [`serve_fields`] holds one entry per long flag of `ServeArgs`, minus
//!   `Append`/`Count` flags and help/version actions.

use std::sync::OnceLock;

use clap::{ArgAction, CommandFactory as _};

/// 2026-09-26: One addable `met serve` flag, shaped for the form.
#[derive(Clone, Debug)]
pub struct FieldSpec {
    /// 2026-09-26: The `defaults:` key: the long name with `-` replaced by `_`.
    pub key: String,
    /// 2026-09-26: clap's long name.
    pub flag: String,
    /// 2026-09-26: First line of the clap help, for the add-list row.
    pub help: String,
    /// 2026-09-26: The long clap help (else the short one), for the picker's side panel.
    pub help_full: String,
    /// 2026-09-26: clap's first default value; `None` makes the add flow ask for a value.
    pub default: Option<String>,
    /// 2026-09-26: The closed value set, empty for free-form flags.
    ///
    /// `true`/`false` for a flag with clap's bool value parser, else
    /// `cli::flag_values::options_for_flag`.
    pub options: Vec<String>,
}

/// 2026-09-26: Every `met serve` flag the form can carry, built once per process.
///
/// Excluded: arguments without a long name (the positional model), and
/// `Append`/`Count` flags, since a `defaults:` map holds one value per key.
pub fn serve_fields() -> &'static [FieldSpec] {
    static FIELDS: OnceLock<Vec<FieldSpec>> = OnceLock::new();
    FIELDS.get_or_init(build)
}

/// 2026-09-26: The spec for a form row's key, matched through `schema::flag_for`.
///
/// So `max_model_len` and `host` find the `--max-seq-len` and `--bind` specs.
pub fn spec_for_key(key: &str) -> Option<&'static FieldSpec> {
    let flag = crate::recipe::schema::flag_for(key)?;
    serve_fields().iter().find(|s| s.flag == flag)
}

fn build() -> Vec<FieldSpec> {
    let bool_parser_id = clap::builder::ValueParser::bool().type_id();
    let command = crate::cli::ServeArgs::command();
    command
        .get_arguments()
        .filter_map(|arg| {
            let long = arg.get_long()?;
            match arg.get_action() {
                ArgAction::Append | ArgAction::Count => return None,
                ArgAction::Help
                | ArgAction::HelpShort
                | ArgAction::HelpLong
                | ArgAction::Version => return None,
                _ => {}
            }
            let is_bool = arg.get_value_parser().type_id() == bool_parser_id;
            let options = if is_bool {
                // 2026-09-26: `recipe::schema::argv_for` renders `true` on a presence-only flag as the bare
                // flag and drops `false`; other flags keep the value.
                vec!["true".to_string(), "false".to_string()]
            } else {
                crate::cli::flag_values::options_for_flag(long).unwrap_or_default()
            };
            let help_full = arg
                .get_long_help()
                .or_else(|| arg.get_help())
                .map(|h| h.to_string())
                .unwrap_or_default();
            Some(FieldSpec {
                key: long.replace('-', "_"),
                flag: long.to_string(),
                help: arg
                    .get_help()
                    .map(|h| h.to_string().lines().next().unwrap_or("").to_string())
                    .unwrap_or_default(),
                help_full,
                default: arg
                    .get_default_values()
                    .first()
                    .map(|v| v.to_string_lossy().into_owned()),
                options,
            })
        })
        .collect()
}

#[cfg(test)]
#[path = "lib_fields_tests.rs"]
mod tests;
