// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The `video-fidelity` benchmark: did a clip's frames reach the
//! model, in order?
//!
//! The order legs send one color sequence forwards (`01_colors_fwd.mp4`) and
//! reversed (`02_colors_rev.mp4`), same container, duration and prompt, and
//! require the reply to reverse with the clip. A no-video control runs last;
//! if it names the whole sequence the run is VACUOUS (`score::verdict`).
//!
//! The other legs, all in `driver.rs`:
//! - Geometry: the 1 s, 2 s and 4 s clips must satisfy
//!   `t4 - t2 == 2 * (t2 - t1)` in prompt tokens
//!   (`geometry::check_proportional`). The absolute count depends on the
//!   server's `--video-fps`, which the benchmark neither sends nor reads.
//! - Backend parity: the MP4 and the GIF of the same colors must cost the same
//!   prompt tokens. The server decodes GIF in-process and every other
//!   container with ffmpeg.
//! - Mixed media: an image and a video in one request.
//! - Integrity: two videos in one request, video before image, two opposite
//!   clips in flight, and a video in an earlier turn.
//! - Concurrency: C = 1, 2, 4 copies of one request, every reply correct and
//!   every level at the C=1 prompt-token count.
//!
//! A leg whose request fails because the server cannot decode the container
//! (`request::is_decoder_unavailable`) is Skipped, not failed; ffmpeg decoding
//! is off unless the serve passes `--video-allow-ffmpeg`. A run where every leg
//! was skipped is INCONCLUSIVE, not PASS.
//!
//! Registered in `registry.rs`; model targets gate on it through
//! `gate = "video-fidelity"` entries in their BENCH.toml.
//!
//! Owner: bench, video.
//! Invariants: none beyond the types.

pub mod concurrency;
pub mod driver;
pub mod geometry;
pub mod provision;
pub mod request;
pub mod score;

pub use driver::{DESCRIPTOR, METADATA};
