// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Decodes video containers to RGB frames by running ffmpeg as a
//! subprocess, for every container that is not a GIF.
//!
//! Owner: model-layers (vision input).
//! Invariants:
//! - `decode_frames` refuses before building a process unless
//!   `FfmpegPolicy::enabled`.
//! - The binary gets its arguments as argv, never through a shell, and the
//!   container over stdin (`pipe:0`); no temp file is written.
//!
//! The payload is an untrusted request body, so a decode is bounded: the frame
//! count by `-frames:v`, the output by a capped read that kills the child once
//! it passes `max_output_bytes`, and the wall clock by a watchdog that kills
//! the child after `timeout_secs`.

use anyhow::{Context, Result, bail, ensure};
use image::RgbImage;
use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};

/// 2026-09-25: Operator policy for subprocess decoding. The serve builds it from
/// the `--video-*` flags.
#[derive(Debug, Clone)]
pub struct FfmpegPolicy {
    pub enabled: bool,
    /// 2026-09-25: Binary to run. A bare name is resolved on PATH; an absolute path
    /// is used as given.
    pub binary: String,
    pub max_frames: usize,
    pub max_output_bytes: usize,
    pub timeout_secs: u64,
}

impl Default for FfmpegPolicy {
    fn default() -> Self {
        Self {
            enabled: false,
            binary: "ffmpeg".to_string(),
            max_frames: 768,
            // 2026-09-25: No serve flag sets this cap; serve_load.rs takes it from
            // this `Default`.
            max_output_bytes: 512 * 1024 * 1024,
            timeout_secs: 120,
        }
    }
}

/// 2026-09-25: The 8-byte PNG signature. Frames arrive concatenated on stdout and
/// `split_png_stream` separates them at each occurrence.
const PNG_MAGIC: [u8; 8] = [0x89, b'P', b'N', b'G', 0x0d, 0x0a, 0x1a, 0x0a];

/// 2026-09-25: What `probe` found.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Availability {
    /// 2026-09-25: Usable, with the first line of its `-version` output.
    Ready(String),
    /// 2026-09-25: Enabled but not runnable, with the reason.
    Missing(String),
    /// 2026-09-25: `FfmpegPolicy::enabled` is false.
    Disabled,
}

/// 2026-09-25: Check whether the configured decoder runs, with one `-version`
/// invocation. The serve calls it at boot so a missing binary shows in the
/// startup log rather than on the first video request.
pub fn probe(policy: &FfmpegPolicy) -> Availability {
    if !policy.enabled {
        return Availability::Disabled;
    }
    match Command::new(&policy.binary)
        .arg("-version")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .output()
    {
        Ok(out) if out.status.success() => {
            let first = String::from_utf8_lossy(&out.stdout)
                .lines()
                .next()
                .unwrap_or("unknown version")
                .trim()
                .to_string();
            Availability::Ready(first)
        }
        Ok(out) => {
            Availability::Missing(format!("{:?} ran but exited {}", policy.binary, out.status))
        }
        Err(e) => Availability::Missing(format!("{:?} could not be run: {e}", policy.binary)),
    }
}

/// 2026-09-25: The error for a decoder that could not be started. It always
/// carries the OS error. Only `NotFound` gets the install hint;
/// `PermissionDenied` and `ExecutableFileBusy` (ETXTBSY) get their own hint,
/// and other kinds get none.
fn spawn_failure(binary: &str, err: std::io::Error) -> anyhow::Error {
    let hint = match err.kind() {
        std::io::ErrorKind::NotFound => {
            " — is ffmpeg installed and on PATH? (set --video-ffmpeg-path to point at it)"
        }
        std::io::ErrorKind::PermissionDenied => {
            " — not executable by this user, or its filesystem is mounted noexec"
        }
        std::io::ErrorKind::ExecutableFileBusy => {
            " — another process still holds it open for writing (ETXTBSY)"
        }
        _ => "",
    };
    anyhow::anyhow!("could not run {binary:?}: {err}{hint}")
}

/// 2026-09-25: Decode `bytes` to RGB frames sampled at `target_fps`.
///
/// ffmpeg samples with `-vf fps=`, so the returned frames are already at
/// `target_fps`, and the caller does not need the source frame rate. A
/// non-finite or non-positive `target_fps` samples at 2.0.
pub fn decode_frames(
    bytes: &[u8],
    target_fps: f32,
    policy: &FfmpegPolicy,
) -> Result<Vec<RgbImage>> {
    ensure!(
        policy.enabled,
        "this container needs ffmpeg to decode and subprocess decoding is disabled; \
         pass --video-allow-ffmpeg to enable it, or send an animated GIF"
    );
    let fps = if target_fps.is_finite() && target_fps > 0.0 {
        target_fps
    } else {
        2.0
    };

    let mut child = Command::new(&policy.binary)
        .args([
            "-v",
            "error",
            // 2026-09-25: Stdin carries the container, so ffmpeg must not read
            // it as keyboard input.
            "-nostdin",
            "-i",
            "pipe:0",
            "-vf",
            &format!("fps={fps}"),
            "-frames:v",
            &policy.max_frames.to_string(),
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "pipe:1",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|err| spawn_failure(&policy.binary, err))?;

    let mut stdin = child.stdin.take().context("no stdin pipe")?;
    let mut stdout = child.stdout.take().context("no stdout pipe")?;
    let mut stderr = child.stderr.take().context("no stderr pipe")?;

    // 2026-09-25: Feed the container on a thread, concurrently with the stdout
    // read below: ffmpeg writes output while it still consumes input, so
    // writing everything first can deadlock once the output pipe fills.
    let input = bytes.to_vec();
    let writer = std::thread::spawn(move || {
        // 2026-09-25: ffmpeg may stop reading once it has `-frames:v` frames, so
        // a write error is expected and ignored.
        let _ = stdin.write_all(&input);
        drop(stdin);
    });

    let child = Arc::new(Mutex::new(child));
    let watchdog = {
        let child = Arc::clone(&child);
        let secs = policy.timeout_secs.max(1);
        std::thread::spawn(move || {
            let deadline = std::time::Instant::now() + std::time::Duration::from_secs(secs);
            loop {
                std::thread::sleep(std::time::Duration::from_millis(100));
                let mut guard = match child.lock() {
                    Ok(g) => g,
                    Err(_) => return false,
                };
                match guard.try_wait() {
                    Ok(Some(_)) => return false,
                    Ok(None) => {}
                    Err(_) => return false,
                }
                if std::time::Instant::now() >= deadline {
                    let _ = guard.kill();
                    return true;
                }
            }
        })
    };

    // 2026-09-25: Read at most one byte past the cap; `over_cap` below detects
    // that byte.
    let mut out = Vec::new();
    let read_res = (&mut stdout)
        .take(policy.max_output_bytes as u64 + 1)
        .read_to_end(&mut out);

    // 2026-09-25: Kill before draining stderr. Past the cap nothing reads
    // stdout, so the child blocks on a full pipe and never closes stderr, and
    // the stderr read would wait until the watchdog fires.
    let over_cap = out.len() > policy.max_output_bytes;
    if over_cap && let Ok(mut g) = child.lock() {
        let _ = g.kill();
    }

    let mut err_text = String::new();
    let _ = stderr.read_to_string(&mut err_text);
    let _ = writer.join();

    let status = {
        let mut g = child
            .lock()
            .map_err(|_| anyhow::anyhow!("decoder lock poisoned"))?;
        g.wait().context("waiting for the decoder")?
    };
    let timed_out = watchdog.join().unwrap_or(false);

    read_res.context("reading decoded frames")?;
    ensure!(
        !timed_out,
        "decoding exceeded {}s and was stopped",
        policy.timeout_secs
    );
    ensure!(
        !over_cap,
        "decoded output exceeded the {}-byte cap",
        policy.max_output_bytes
    );
    if !status.success() {
        let why = err_text.lines().next_back().unwrap_or("no detail").trim();
        bail!("decoder failed: {why}");
    }

    let frames = split_png_stream(&out)?;
    ensure!(
        !frames.is_empty(),
        "the container decoded to zero frames (is there a video stream?)"
    );
    Ok(frames)
}

/// 2026-09-25: Split a concatenated PNG stream into images, starting a new image
/// at each `PNG_MAGIC`. An empty stream gives no images; a piece that is not a
/// valid PNG is an error naming its index.
fn split_png_stream(buf: &[u8]) -> Result<Vec<RgbImage>> {
    let mut starts = Vec::new();
    let mut i = 0usize;
    while i + PNG_MAGIC.len() <= buf.len() {
        if buf[i..i + PNG_MAGIC.len()] == PNG_MAGIC {
            starts.push(i);
            i += PNG_MAGIC.len();
        } else {
            i += 1;
        }
    }
    let mut frames = Vec::with_capacity(starts.len());
    for (n, &s) in starts.iter().enumerate() {
        let e = starts.get(n + 1).copied().unwrap_or(buf.len());
        let img = image::load_from_memory_with_format(&buf[s..e], image::ImageFormat::Png)
            .with_context(|| format!("frame {n} did not decode as PNG"))?;
        frames.push(img.to_rgb8());
    }
    Ok(frames)
}

#[cfg(test)]
#[path = "video_decode_ffmpeg_tests.rs"]
mod tests;
