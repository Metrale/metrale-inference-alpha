// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Tests for request shaping and the vision-token subtraction.
//!
//! Owner: bench, vision.
//! Invariants: none beyond the types.

use super::*;

#[test]
fn a_data_uri_is_what_the_api_accepts() {
    assert_eq!(
        data_uri(&[0x89, b'P', b'N', b'G']),
        "data:image/png;base64,iVBORw=="
    );
}

#[test]
fn images_precede_the_prompt_and_order_is_preserved() {
    // 2026-09-26: The multi-image probe asks which image came first, so the
    // builder must keep the order it was given.
    let a = b"\x89PNG-A".as_slice();
    let b = b"\x89PNG-B".as_slice();
    let v = body("m", &[a, b], "which is first?", 32);
    assert_eq!(
        v,
        serde_json::json!({
            "model": "m",
            "stream": true,
            "temperature": 0.0,
            "max_tokens": 32,
            "chat_template_kwargs": {"enable_thinking": false},
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64,iVBORy1B"
                }},
                {"type": "image_url", "image_url": {
                    "url": "data:image/png;base64,iVBORy1C"
                }},
                {"type": "text", "text": "which is first?"}
            ]}]
        })
    );
}

#[test]
fn a_probe_with_no_images_sends_only_text() {
    // 2026-09-26: The control's request: no image part at all.
    let v = body("m", &[], "no image here", 16);
    assert_eq!(
        v,
        serde_json::json!({
            "model": "m",
            "stream": true,
            "temperature": 0.0,
            "max_tokens": 16,
            "chat_template_kwargs": {"enable_thinking": false},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "no image here"}
            ]}]
        })
    );
}

#[test]
fn vision_tokens_subtracts_the_measured_overhead() {
    // 2026-09-26: 215 - 19 = 196, the count measured on 2026-08-14 for a
    // 448x448 image.
    assert_eq!(vision_tokens(215, 19).unwrap(), 196);
}

#[test]
fn an_impossible_subtraction_is_an_error_not_a_wrap() {
    // 2026-09-26: An overhead above `prompt_tokens` is an error, not a wrapped
    // `usize`.
    let e = vision_tokens(5, 19).unwrap_err();
    assert_eq!(
        e.to_string(),
        "prompt_tokens 5 is below the measured template overhead 19 — the calibration request \
         and this one did not render the same template, so the subtraction is meaningless"
    );
}
