// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The committed video clips, and writing them to disk.
//!
//! The clips are `include_bytes!`d into the binary and written to
//! `~/.metrale/artifacts/video` behind a content-derived stamp, so a changed
//! clip re-provisions without a version bump. The vision fixtures use the
//! same scheme.
//!
//! Owner: bench, video.
//! Invariants:
//! - `provision` commits the stamp only after every clip write succeeded.

use anyhow::Result;

use crate::artifacts::{ArtifactStore, Stamp, write_asset_bytes};

/// 2026-09-26: A clip and what it shows.
pub struct Clip {
    pub name: &'static str,
    pub bytes: &'static [u8],
    /// 2026-09-26: The colors it shows, in order, one per second.
    pub colors: &'static [&'static str],
    /// 2026-09-26: Seconds of footage. Only `provision_tests.rs` reads it, to
    /// pin the 1:2:4 durations the geometry leg relies on.
    pub seconds: u32,
    /// 2026-09-26: True for a container the server decodes with ffmpeg (every
    /// one but GIF). Only `provision_tests.rs` reads it; the driver decides a
    /// skip from the server's error (`request::is_decoder_unavailable`).
    pub needs_ffmpeg: bool,
    pub mime: &'static str,
}

/// 2026-09-26: Five clips, used in pairs and one triple:
///
/// * `01` vs `02`: temporal order. The same four colors reversed, same
///   container and duration.
/// * `01` vs `03`: backend parity. The same colors as an MP4 (ffmpeg) and a
///   GIF (decoded in-process); both must cost the same prompt tokens.
/// * `05`, `04`, `01`: geometry at 1 s, 2 s and 4 s
///   (`geometry::check_proportional`).
pub const CLIPS: &[Clip] = &[
    Clip {
        name: "01_colors_fwd.mp4",
        bytes: include_bytes!("../../../assets/video/01_colors_fwd.mp4"),
        colors: &["red", "green", "blue", "yellow"],
        seconds: 4,
        needs_ffmpeg: true,
        mime: "video/mp4",
    },
    Clip {
        name: "02_colors_rev.mp4",
        bytes: include_bytes!("../../../assets/video/02_colors_rev.mp4"),
        colors: &["yellow", "blue", "green", "red"],
        seconds: 4,
        needs_ffmpeg: true,
        mime: "video/mp4",
    },
    Clip {
        name: "03_colors_fwd.gif",
        bytes: include_bytes!("../../../assets/video/03_colors_fwd.gif"),
        colors: &["red", "green", "blue", "yellow"],
        seconds: 4,
        needs_ffmpeg: false,
        mime: "image/gif",
    },
    Clip {
        name: "05_colors_unit.mp4",
        bytes: include_bytes!("../../../assets/video/05_colors_unit.mp4"),
        colors: &["red"],
        seconds: 1,
        needs_ffmpeg: true,
        mime: "video/mp4",
    },
    Clip {
        name: "04_colors_half.mp4",
        bytes: include_bytes!("../../../assets/video/04_colors_half.mp4"),
        colors: &["red", "green"],
        seconds: 2,
        needs_ffmpeg: true,
        mime: "video/mp4",
    },
];

pub fn clip(name: &str) -> Option<&'static Clip> {
    CLIPS.iter().find(|c| c.name == name)
}

/// 2026-09-26: Content-derived over every clip's name and bytes, so a changed
/// clip re-provisions without a version bump.
fn stamp_value() -> String {
    crate::benchmarks::content_stamp("video-fixtures-v1", CLIPS.iter().map(|c| (c.name, c.bytes)))
}

pub const PLUGIN_ID: &str = "video";

pub fn provision(store: &ArtifactStore) -> Result<std::path::PathBuf> {
    let dir = store.plugin_dir(PLUGIN_ID)?;
    let stamp = Stamp::new(&dir, ".provisioned", stamp_value());
    if stamp.is_current() {
        return Ok(dir);
    }
    for c in CLIPS {
        write_asset_bytes(&dir, c.name, c.bytes)?;
    }
    // 2026-09-26: Last: a stamp written before the writes complete would mark
    // a partial directory as current.
    stamp.commit()?;
    Ok(dir)
}

#[cfg(test)]
#[path = "provision_tests.rs"]
mod provision_tests;
