// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The serve flag surface and the lever table as one JSON document, printed by `met dump-serve-options`.
//!
//! Owner: server CLI.
//! Invariants:
//! - Every flag entry is read from `ServeArgs::command()`, and every lever entry from
//!   `metrale_config::levers::all()`; nothing here lists a flag or lever by hand.
//! - A flag's `options` come from `cli::flag_values::options_for_flag`, the value sets
//!   `validate_serve_args` checks against.
//!
//! `ServeArgs` has no `Serialize` derive, so the document promises no stable flag names.
//! `crate::tui::lib_fields` reads the same clap metadata for the dashboard; this adds
//! whether each flag is presence-only.

use clap::{ArgAction, CommandFactory};
use serde::Serialize;

/// 2026-09-26: Version of the document's shape: the fields of `Manifest`, `Flag` and
/// `Lever`. It does not change when a flag or lever is added or removed.
pub const SCHEMA_VERSION: u32 = 2;

/// 2026-09-26: The whole document.
#[derive(Debug, Serialize)]
pub struct Manifest {
    /// 2026-09-26: `SCHEMA_VERSION`.
    pub schema_version: u32,
    /// 2026-09-26: `cli::METRALE_VERSION` of the binary that printed it.
    pub engine_version: String,
    /// 2026-09-26: Every serve flag with a long name, except clap's own help and version,
    /// in `ServeArgs` declaration order.
    pub flags: Vec<Flag>,
    /// 2026-09-26: Every `METRALE_*` lever, from `metrale_config::levers::all()`, in name
    /// order. The binary refuses to start while an undeclared `METRALE_*` variable is set
    /// (`levers::check` in `main.rs`).
    pub levers: Vec<Lever>,
}

/// 2026-09-26: One `METRALE_*` lever, copied from its `LeverSpec`.
#[derive(Debug, Serialize)]
pub struct Lever {
    pub env: &'static str,
    /// 2026-09-26: Where the setting sits in the engine's configuration: `group.name`.
    pub path: &'static str,
    /// 2026-09-26: `switch`, `int`, `float`, `text` or `path` (`levers::Ty::as_str`).
    #[serde(rename = "type")]
    pub ty: &'static str,
    /// 2026-09-26: The value in force when the variable is unset, as the table states it.
    pub default: &'static str,
    /// 2026-09-26: The serve flag that sets the same thing and wins when given.
    pub flag: Option<&'static str>,
    /// 2026-09-26: `runtime`, `harness`, `build`, `dev` or `tool` (`levers::Class::as_str`).
    pub class: &'static str,
    /// 2026-09-26: The repository file that reads it.
    pub reader: &'static str,
    pub doc: &'static str,
}

/// 2026-09-26: One flag, as clap describes it.
#[derive(Debug, Serialize)]
pub struct Flag {
    /// 2026-09-26: The long name with `-` replaced by `_`, the recipe `defaults:` spelling.
    pub key: String,
    /// 2026-09-26: The long name, without the leading dashes.
    pub flag: String,
    /// 2026-09-26: Whether the flag takes no value (clap `ArgAction::SetTrue`).
    /// `--gdn-fused-norm` is presence-only; `--ssm-h-dtype` takes a value.
    pub presence_only: bool,
    /// 2026-09-26: Whether clap's value parser for the flag is the bool parser.
    pub is_bool: bool,
    /// 2026-09-26: The accepted values, from `cli::flag_values::options_for_flag`. Empty
    /// for a free-form flag and for every bool flag.
    pub options: Vec<String>,
    /// 2026-09-26: clap's first default value, when it declares one.
    pub default: Option<String>,
    /// 2026-09-26: The first line of the flag's help.
    pub help: Option<String>,
    /// 2026-09-26: The flag's other long names (`get_all_aliases`), which `get_long()`
    /// does not return. `--bind` has the alias `host`.
    pub cli_aliases: Vec<String>,
    /// 2026-09-26: Recipe keys that map to this flag under another name
    /// (`recipe::schema::RENAMES`), such as `max_model_len` for `--max-seq-len`.
    pub recipe_aliases: Vec<String>,
}

/// 2026-09-26: Build the document from `ServeArgs::command()` and the lever table.
#[must_use]
pub fn build() -> Manifest {
    let bool_parser = clap::builder::ValueParser::bool().type_id();
    let command = super::ServeArgs::command();

    let flags = command
        .get_arguments()
        .filter_map(|arg| {
            // 2026-09-26: Skip arguments without a long name (the positional MODEL) and
            // clap's own help and version arguments.
            let long = arg.get_long()?;
            match arg.get_action() {
                ArgAction::Help
                | ArgAction::HelpShort
                | ArgAction::HelpLong
                | ArgAction::Version => return None,
                _ => {}
            }
            let presence_only = matches!(arg.get_action(), ArgAction::SetTrue);
            let is_bool = arg.get_value_parser().type_id() == bool_parser;
            // 2026-09-26: A bool flag lists no values; `bool_surface_tests.rs`
            // (`every_bool_flag_is_presence_only`) checks that every bool is presence-only.
            let options = if is_bool {
                Vec::new()
            } else {
                super::flag_values::options_for_flag(long).unwrap_or_default()
            };
            Some(Flag {
                key: long.replace('-', "_"),
                flag: long.to_owned(),
                presence_only,
                is_bool,
                options,
                default: arg
                    .get_default_values()
                    .first()
                    .map(|v| v.to_string_lossy().into_owned()),
                help: arg
                    .get_help()
                    .map(|h| h.to_string().lines().next().unwrap_or_default().to_owned()),
                cli_aliases: arg
                    .get_all_aliases()
                    .map(|a| a.iter().map(|s| (*s).to_owned()).collect())
                    .unwrap_or_default(),
                recipe_aliases: crate::recipe::schema::RENAMES
                    .iter()
                    .filter(|(_, flag)| *flag == long)
                    .map(|(key, _)| (*key).to_owned())
                    .collect(),
            })
        })
        .collect();

    let levers = metrale_config::levers::all()
        .map(|l| Lever {
            env: l.env,
            path: l.path,
            ty: l.ty.as_str(),
            default: l.default,
            flag: l.flag,
            class: l.class.as_str(),
            reader: l.reader,
            doc: l.doc,
        })
        .collect();

    Manifest {
        schema_version: SCHEMA_VERSION,
        engine_version: super::METRALE_VERSION.to_owned(),
        flags,
        levers,
    }
}

#[cfg(test)]
#[path = "manifest_tests.rs"]
mod tests;
