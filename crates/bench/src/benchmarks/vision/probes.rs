// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Capability probes: did the model see the picture? Each probe
//! attaches fixtures, asks a question, and is scored by
//! `score::reply_matches` against terms the reply must and must not contain.
//! The fixtures they use come from `scripts/gen_test_images.py`, which draws a
//! size label on each.
//!
//! This leg shows that vision ran; `geometry.rs` checks the token count.
//! Geometry alone would pass an encoder that produced the right count from
//! wrong embeddings.
//!
//! Owner: bench, vision.
//! Invariants: none beyond the types.

/// 2026-09-26: A probe: which fixtures to attach, what to ask, what a correct
/// answer must and must not contain.
pub struct Probe {
    pub id: &'static str,
    /// 2026-09-26: Fixture names, looked up in `provision::FIXTURES`. Empty
    /// sends no image; see [`CONTROL`].
    pub images: &'static [&'static str],
    pub prompt: &'static str,
    /// 2026-09-26: Lowercase terms that must all appear.
    pub want_all: &'static [&'static str],
    /// 2026-09-26: Lowercase terms none of which may appear, so a reply that
    /// names every option does not pass.
    pub want_none: &'static [&'static str],
}

pub const PROBES: &[Probe] = &[
    Probe {
        id: "sees-an-image",
        images: &["01_square_224.png"],
        prompt: "Describe what you see in this image in one short sentence.",
        want_all: &[],
        // 2026-09-26: Nothing positive is asserted; the reply must not say it
        // sees no image.
        want_none: &["cannot see", "no image", "unable to see", "don't see"],
    },
    Probe {
        id: "reads-the-size-label",
        images: &["07_hd_1280x720.png"],
        prompt: "This image has a size label drawn on it. Read the label exactly.",
        want_all: &["1280"],
        want_none: &["cannot see", "no image"],
    },
    Probe {
        id: "multi-image-order",
        images: &["01_square_224.png", "08_portrait_480x854.png"],
        // 2026-09-26: The question is about the first image, so the answer
        // depends on the images arriving in order.
        prompt: "You are shown two images. Is the FIRST one square or portrait? \
                 Answer with one word.",
        want_all: &["square"],
        want_none: &["portrait"],
    },
];

/// 2026-09-26: The no-image control: the size-label question with nothing
/// attached. It passes when the reply does not contain `1280`, the term
/// `reads-the-size-label` requires; otherwise the capability leg is not
/// evidence and the run is VACUOUS.
pub const CONTROL: Probe = Probe {
    id: "control-no-image",
    images: &[],
    prompt: "This image has a size label drawn on it. Read the label exactly.",
    want_all: &[],
    want_none: &["1280"],
};

/// 2026-09-26: The probe the concurrency leg sends, `reads-the-size-label`, so
/// a reply passes only by reading the label from the image.
pub fn concurrency_probe() -> &'static Probe {
    PROBES
        .iter()
        .find(|probe| probe.id == "reads-the-size-label")
        .expect("reads-the-size-label probe is part of the fixed probe set")
}

#[cfg(test)]
#[path = "probes_tests.rs"]
mod probes_tests;
