// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tool-schema rewrites applied before a schema is compiled
//! into a grammar.
//!
//! Owner: server (grammar).
//! Invariants: none beyond the types.

/// 2026-09-26: A copy of `schema` with an optional string property
/// `_think` inserted first in `properties`, a scratchpad the model can fill
/// before the real arguments. `_think` is not added to `required`.
///
/// Returns the copy unchanged when `schema` is not a JSON object, has no
/// `properties` object, or already has a `_think` property.
pub fn augment_schema_with_tafc_think(schema: &serde_json::Value) -> serde_json::Value {
    let mut schema = schema.clone();
    let Some(obj) = schema.as_object_mut() else {
        return schema;
    };
    let Some(props) = obj.get_mut("properties").and_then(|p| p.as_object_mut()) else {
        return schema;
    };
    if props.contains_key("_think") {
        return schema;
    }
    // 2026-09-26: Rebuilt so `_think` is the first property.
    let mut new_props = serde_json::Map::with_capacity(props.len() + 1);
    new_props.insert(
        "_think".to_string(),
        serde_json::json!({
            "type": "string",
            "description": "Optional scratchpad: brief rationale for selecting this tool and these arguments. Server-side only — not forwarded to the tool implementation.",
        }),
    );
    for (k, v) in props.iter() {
        new_props.insert(k.clone(), v.clone());
    }
    obj.insert(
        "properties".to_string(),
        serde_json::Value::Object(new_props),
    );
    schema
}

/// 2026-09-26: A copy of `schema` in which every top-level required
/// property of type `"string"` without a `minLength` gets `"minLength": 1`.
/// The grammar then generates a `{1,}` repetition for the value, so the
/// model cannot emit `""` for a required string.
pub(super) fn enforce_min_length_on_required_strings(
    schema: &serde_json::Value,
) -> serde_json::Value {
    let mut schema = schema.clone();
    let obj = match schema.as_object_mut() {
        Some(o) => o,
        None => return schema,
    };

    let required: Vec<String> = obj
        .get("required")
        .and_then(|r| r.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_str().map(String::from))
                .collect()
        })
        .unwrap_or_default();

    if required.is_empty() {
        return schema;
    }

    if let Some(props) = obj.get_mut("properties").and_then(|p| p.as_object_mut()) {
        for key in &required {
            if let Some(prop) = props.get_mut(key).and_then(|p| p.as_object_mut()) {
                let is_string = prop.get("type").and_then(|t| t.as_str()) == Some("string");
                if is_string && !prop.contains_key("minLength") {
                    prop.insert("minLength".to_string(), serde_json::Value::Number(1.into()));
                }
            }
        }
    }

    schema
}

/// 2026-09-26: Rewrite schema patterns that the grammar compiler rejects
/// or turns into an empty rule body into equivalents it compiles, walking
/// nested schemas.
///
/// Returns `None` when `schema` is `false`, or is neither a boolean nor an
/// object. A nested property whose schema returns `None` is dropped, along
/// with its `required` entry. Nesting deeper than 32 levels becomes `{}`
/// (any value).
pub(super) fn sanitize_schema_for_grammar(schema: &serde_json::Value) -> Option<serde_json::Value> {
    sanitize_recursive(schema, schema, 0)
}

fn sanitize_recursive(
    schema: &serde_json::Value,
    root: &serde_json::Value,
    depth: usize,
) -> Option<serde_json::Value> {
    if depth > 32 {
        return Some(serde_json::json!({}));
    }

    if let Some(b) = schema.as_bool() {
        return if b { Some(serde_json::json!({})) } else { None };
    }

    let obj = schema.as_object()?;
    let mut result = obj.clone();

    // 2026-09-26: A resolvable `#/` `$ref` is replaced by its target's keys;
    // keys already on the schema win.
    if let Some(ref_str) = result
        .get("$ref")
        .and_then(|v| v.as_str())
        .map(String::from)
        && let Some(resolved) = resolve_local_ref(&ref_str, root)
    {
        result.remove("$ref");
        if let Some(resolved_obj) = resolved.as_object() {
            for (k, v) in resolved_obj {
                if !result.contains_key(k) {
                    result.insert(k.clone(), v.clone());
                }
            }
        }
    }

    // 2026-09-26: Python type names ("dict", "float", ...) are mapped to
    // JSON Schema types; the compiler rejects any other type name as
    // unsupported.
    if let Some(t) = result.get("type").and_then(|t| t.as_str()) {
        let mapped = match t {
            "dict" => Some("object"),
            "tuple" | "list" => Some("array"),
            "float" | "double" => Some("number"),
            "int" | "long" => Some("integer"),
            "bool" => Some("boolean"),
            "str" => Some("string"),
            _ => None,
        };
        if let Some(m) = mapped {
            result.insert("type".to_string(), serde_json::Value::String(m.into()));
        }
    }

    if let Some(arr) = result.get("enum").and_then(|v| v.as_array())
        && arr.is_empty()
    {
        result.remove("enum");
    }

    // 2026-09-26: An empty `anyOf`/`oneOf` is removed. A one-element one is
    // replaced by its sanitized element's keys (keys already present win).
    // A longer one keeps the elements that sanitize, and is removed if none
    // does.
    for key in ["anyOf", "oneOf"] {
        if let Some(arr) = result.get(key).and_then(|v| v.as_array()).cloned() {
            if arr.is_empty() {
                result.remove(key);
            } else if arr.len() == 1 {
                if let Some(inner) = sanitize_recursive(&arr[0], root, depth + 1) {
                    result.remove(key);
                    if let Some(inner_obj) = inner.as_object() {
                        for (k, v) in inner_obj {
                            if !result.contains_key(k) {
                                result.insert(k.clone(), v.clone());
                            }
                        }
                    }
                }
            } else {
                let sanitized: Vec<serde_json::Value> = arr
                    .iter()
                    .filter_map(|el| sanitize_recursive(el, root, depth + 1))
                    .collect();
                if sanitized.is_empty() {
                    result.remove(key);
                } else {
                    result.insert(key.to_string(), serde_json::Value::Array(sanitized));
                }
            }
        }
    }

    if let Some(arr) = result.get("allOf").and_then(|v| v.as_array()).cloned() {
        if arr.is_empty() {
            result.remove("allOf");
        } else if arr.len() == 1 {
            if let Some(inner) = sanitize_recursive(&arr[0], root, depth + 1) {
                result.remove("allOf");
                if let Some(inner_obj) = inner.as_object() {
                    for (k, v) in inner_obj {
                        if !result.contains_key(k) {
                            result.insert(k.clone(), v.clone());
                        }
                    }
                }
            }
        } else {
            // 2026-09-26: Takes the first `type`, the union of `properties`
            // (a later sub-schema's property replaces an earlier one's) and the
            // union of `required`; keys already on the schema win.
            let mut merged_props = serde_json::Map::new();
            let mut merged_required: Vec<serde_json::Value> = Vec::new();
            let mut merged_type: Option<serde_json::Value> = None;
            for sub in &arr {
                if let Some(s) = sanitize_recursive(sub, root, depth + 1)
                    && let Some(o) = s.as_object()
                {
                    if let Some(t) = o.get("type") {
                        merged_type.get_or_insert_with(|| t.clone());
                    }
                    if let Some(p) = o.get("properties").and_then(|p| p.as_object()) {
                        for (k, v) in p {
                            merged_props.insert(k.clone(), v.clone());
                        }
                    }
                    if let Some(r) = o.get("required").and_then(|r| r.as_array()) {
                        for item in r {
                            if !merged_required.contains(item) {
                                merged_required.push(item.clone());
                            }
                        }
                    }
                }
            }
            result.remove("allOf");
            if let Some(t) = merged_type {
                result.entry("type").or_insert(t);
            }
            if !merged_props.is_empty() {
                result
                    .entry("properties")
                    .or_insert(serde_json::Value::Object(merged_props));
            }
            if !merged_required.is_empty() {
                result
                    .entry("required")
                    .or_insert(serde_json::Value::Array(merged_required));
            }
        }
    }

    // 2026-09-26: An object with no properties and none of
    // `patternProperties`, `additionalProperties`, `unevaluatedProperties`
    // or `propertyNames` gets `additionalProperties: true`. Without it, a
    // strict-mode object at the XML top level (no braces) generates an
    // empty rule body.
    let is_object = result.get("type").and_then(|t| t.as_str()) == Some("object");
    let has_props = result
        .get("properties")
        .and_then(|p| p.as_object())
        .is_some_and(|p| !p.is_empty());
    let has_structural_keys = result.contains_key("patternProperties")
        || result.contains_key("additionalProperties")
        || result.contains_key("unevaluatedProperties")
        || result.contains_key("propertyNames");

    if is_object && !has_props && !has_structural_keys {
        result.insert(
            "additionalProperties".to_string(),
            serde_json::Value::Bool(true),
        );
    }

    if let Some(props) = result.get("properties").cloned()
        && let Some(props_obj) = props.as_object()
    {
        let mut new_props = serde_json::Map::new();
        for (k, v) in props_obj {
            if let Some(sanitized) = sanitize_recursive(v, root, depth + 1) {
                new_props.insert(k.clone(), sanitized);
            } else {
                // 2026-09-26: The property's schema is `false` or not an
                // object: drop it from `required` as well.
                if let Some(req) = result.get_mut("required").and_then(|r| r.as_array_mut()) {
                    req.retain(|r| r.as_str() != Some(k.as_str()));
                }
            }
        }
        result.insert(
            "properties".to_string(),
            serde_json::Value::Object(new_props),
        );
    }

    if let Some(items) = result.get("items").cloned()
        && items.is_object()
        && let Some(sanitized) = sanitize_recursive(&items, root, depth + 1)
    {
        result.insert("items".to_string(), sanitized);
    }

    if let Some(addl) = result.get("additionalProperties").cloned()
        && addl.is_object()
        && let Some(sanitized) = sanitize_recursive(&addl, root, depth + 1)
    {
        result.insert("additionalProperties".to_string(), sanitized);
    }

    Some(serde_json::Value::Object(result))
}

/// 2026-09-26: Resolve a `#/`-prefixed JSON Pointer (e.g. `#/$defs/Foo`)
/// against `root`, decoding `~1` and `~0`. `None` for any other form or a
/// missing target.
fn resolve_local_ref(ref_str: &str, root: &serde_json::Value) -> Option<serde_json::Value> {
    let path = ref_str.strip_prefix("#/")?;
    let mut current = root;
    for segment in path.split('/') {
        let decoded = segment.replace("~1", "/").replace("~0", "~");
        current = current.get(&decoded)?;
    }
    Some(current.clone())
}
