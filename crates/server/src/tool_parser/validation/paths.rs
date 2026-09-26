// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: `normalize_paths`, the clean-up of path arguments against the
//! client's working directory, and its punctuation-drift helpers.
//!
//! Owner: server (tool parser).
//! Invariants: none beyond the types.

use super::*;

/// 2026-09-26: Clean up path arguments against the client's working
/// directory `cwd`.
///
/// - For each `PATH_KEYS` value: strip a leading `=`, trailing commas, spaces
///   and tabs, and one pair of surrounding double quotes.
/// - A `workdir` whose last component equals `cwd`'s up to punctuation and
///   ASCII case, under the same parent, becomes `cwd`. In a `bash` call the
///   same spelling inside `command` is repaired too.
/// - An absolute path under `cwd` is made relative to it. Other absolute
///   paths and relative paths are left as they are.
pub fn normalize_paths(calls: &mut [ToolCall], cwd: &str) {
    const PATH_KEYS: &[&str] = &["file_path", "filePath", "path", "file", "workdir"];
    let cwd_slash = if cwd.ends_with('/') {
        cwd.to_string()
    } else {
        format!("{cwd}/")
    };

    for call in calls.iter_mut() {
        let Ok(mut args) = serde_json::from_str::<serde_json::Map<String, serde_json::Value>>(
            &call.function.arguments,
        ) else {
            continue;
        };
        let mut changed = false;
        if call.function.name.eq_ignore_ascii_case("bash")
            && let Some(serde_json::Value::String(command)) = args.get("command")
            && let Some(repaired) = repair_punctuation_drifted_cwd_in_command(command, cwd)
        {
            args.insert("command".to_string(), serde_json::Value::String(repaired));
            changed = true;
        }
        for key in PATH_KEYS {
            if let Some(serde_json::Value::String(path)) = args.get(*key) {
                // 2026-09-26: The comma strip comes before the quote strip, so
                // `"/tmp/x/Cargo.toml",` loses both.
                let trimmed = path.trim();
                let mut sanitized: &str = trimmed;
                if let Some(rest) = sanitized.strip_prefix('=') {
                    sanitized = rest.trim_start();
                }
                sanitized = sanitized.trim_end_matches([',', ' ', '\t']);
                if sanitized.len() >= 2 && sanitized.starts_with('"') && sanitized.ends_with('"') {
                    sanitized = &sanitized[1..sanitized.len() - 1];
                }
                if sanitized != path.as_str() {
                    args.insert(
                        key.to_string(),
                        serde_json::Value::String(sanitized.to_string()),
                    );
                    changed = true;
                }
                let Some(serde_json::Value::String(path)) = args.get(*key) else {
                    continue;
                };
                if *key == "workdir" && punctuation_equivalent_basename(path, cwd) {
                    if path != cwd {
                        args.insert(key.to_string(), serde_json::Value::String(cwd.to_string()));
                        changed = true;
                    }
                    continue;
                }
                if !path.starts_with('/') {
                    continue;
                }
                if !path.starts_with(&cwd_slash) {
                    continue;
                }
                let new_path = path[cwd_slash.len()..].to_string();
                if new_path != *path && !new_path.is_empty() {
                    args.insert(key.to_string(), serde_json::Value::String(new_path));
                    changed = true;
                }
            }
        }
        if changed && let Ok(new_args) = serde_json::to_string(&serde_json::Value::Object(args)) {
            call.function.arguments = new_args;
        }
    }
}

fn fold_path_component(name: &str) -> String {
    name.chars()
        .filter(|c| c.is_ascii_alphanumeric())
        .map(|c| c.to_ascii_lowercase())
        .collect()
}

fn punctuation_equivalent_basename(candidate: &str, cwd: &str) -> bool {
    let candidate = std::path::Path::new(candidate);
    let cwd = std::path::Path::new(cwd);
    if candidate.parent() != cwd.parent() {
        return false;
    }
    match (candidate.file_name(), cwd.file_name()) {
        (Some(candidate), Some(cwd)) => {
            fold_path_component(&candidate.to_string_lossy())
                == fold_path_component(&cwd.to_string_lossy())
        }
        _ => false,
    }
}

fn repair_punctuation_drifted_cwd_in_command(command: &str, cwd: &str) -> Option<String> {
    let cwd = std::path::Path::new(cwd);
    let parent = cwd.parent()?.to_str()?;
    let basename = cwd.file_name()?.to_str()?;
    let parent_prefix = if parent == "/" {
        "/".to_string()
    } else {
        format!("{parent}/")
    };
    let canonical_fold = fold_path_component(basename);
    let mut repaired = String::with_capacity(command.len());
    let mut cursor = 0;
    let mut changed = false;

    while let Some(offset) = command[cursor..].find(&parent_prefix) {
        let component_start = cursor + offset + parent_prefix.len();
        let component_end = command[component_start..]
            .find(|c: char| !(c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.')))
            .map_or(command.len(), |end| component_start + end);
        let candidate = &command[component_start..component_end];

        repaired.push_str(&command[cursor..component_start]);
        if candidate != basename
            && !candidate.is_empty()
            && fold_path_component(candidate) == canonical_fold
        {
            repaired.push_str(basename);
            changed = true;
        } else {
            repaired.push_str(candidate);
        }
        cursor = component_end;
    }

    if !changed {
        return None;
    }
    repaired.push_str(&command[cursor..]);
    Some(repaired)
}
