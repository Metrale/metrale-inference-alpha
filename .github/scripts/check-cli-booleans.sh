#!/usr/bin/env bash
# SPDX-License-Identifier: MIT OR Apache-2.0
#
# Refuse value-taking booleans in the files that declare the `met` command line.
#
# A command-line boolean is a presence flag: `--x` when the feature defaults
# off, `--no-x` when it defaults on, and an `auto|on|off` enum when its default
# lives somewhere else. `Option<bool>` and `default_missing_value` are the two
# spellings that bring back `--flag true|false` (and the absent-vs-false split
# that made documented levers silent no-ops), so neither may appear in a file
# that declares clap arguments. The clap-introspection test
# (`cli/bool_surface_tests.rs`) checks the parsed surface; this checks the
# source, so the form cannot be reintroduced even behind a custom parser.
#
# Scope: `crates/server/src/cli.rs` and every `.rs` under `crates/server/src/cli/`
# that declares clap arguments (`#[arg(`, `#[command(` or a clap derive). Other
# files there (JSON wire types, resolvers) may carry an `Option<bool>` that is
# not a command-line flag. Comment lines do not count.
set -euo pipefail
: "${CLI_ROOT:=crates/server/src}"

[ -f "$CLI_ROOT/cli.rs" ] && [ -d "$CLI_ROOT/cli" ] || {
  echo "::error::$CLI_ROOT/cli.rs or $CLI_ROOT/cli/ does not exist, so this check scanned nothing."
  echo "If the command line moved, update CLI_ROOT in the same commit."
  exit 1
}

mapfile -t files < <(
  { echo "$CLI_ROOT/cli.rs"; find "$CLI_ROOT/cli" -name '*.rs' -type f; } | sort |
    while read -r f; do
      if grep -qE '#\[(arg|command|clap)\(|derive\([^)]*\b(Parser|Args|Subcommand)\b' "$f"; then
        echo "$f"
      fi
    done
)
# The four files that hold today's arguments (cli.rs, serve_args.rs,
# bench_args.rs, bench_certify/args.rs) must all be found; fewer means the
# discovery broke, not that the arguments went away.
if [ "${#files[@]}" -lt 4 ]; then
  echo "::error::found only ${#files[@]} file(s) declaring clap arguments under $CLI_ROOT; this check cannot vouch for a surface it did not find."
  exit 1
fi

hits=$(for f in "${files[@]}"; do
  awk -v f="$f" '
    /^[[:space:]]*\/\// { next }
    /Option<bool>|default_missing_value/ { printf "%s:%d: %s\n", f, NR, $0 }
  ' "$f"
done)
if [ -n "$hits" ]; then
  echo "::error::A command-line boolean takes no value."
  echo "$hits"
  echo
  echo "Make it a presence flag (ArgAction::SetTrue) named for its non-default"
  echo "state (--x / --no-x), or an auto|on|off enum when its default is decided"
  echo "elsewhere (see cli::flag_values::Tristate)."
  exit 1
fi
echo "OK: no Option<bool> or default_missing_value in ${#files[@]} CLI source files"
