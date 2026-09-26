// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-25: Tests for the ffmpeg subprocess decode backend. Tests that need a
//! real decoder encode their own clip with ffmpeg and return early when it is
//! absent.
//!
//! Owner: model-layers (vision input).
//! Invariants: none beyond the types.

use super::*;

fn have_ffmpeg() -> bool {
    matches!(probe(&enabled()), Availability::Ready(_))
}

fn enabled() -> FfmpegPolicy {
    FfmpegPolicy {
        enabled: true,
        ..Default::default()
    }
}

/// 2026-09-25: Encode a `testsrc` clip to MP4 and return its bytes, or `None` on
/// any failure.
fn make_mp4(seconds: u32, fps: u32, size: &str, codec: &str) -> Option<Vec<u8>> {
    // 2026-09-25: The directory is unique per call (`SEQ`): each call removes its
    // directory at the end, so a shared one could lose another test's clip.
    static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "metrale-vid-{}-{n}-{}-{}-{size}-{codec}",
        std::process::id(),
        seconds,
        fps
    ));
    std::fs::create_dir_all(&dir).ok()?;
    let out = dir.join("clip.mp4");
    let status = Command::new("ffmpeg")
        .args([
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            &format!("testsrc=size={size}:rate={fps}:duration={seconds}"),
            "-c:v",
            codec,
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(&out)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .ok()?;
    if !status.success() {
        return None;
    }
    let bytes = std::fs::read(&out).ok();
    let _ = std::fs::remove_dir_all(&dir);
    bytes
}

#[test]
fn the_default_policy_runs_nothing() {
    let p = FfmpegPolicy::default();
    assert!(!p.enabled, "subprocess decoding must be opt-in");
    let err = decode_frames(b"whatever", 2.0, &p).unwrap_err().to_string();
    assert!(err.contains("--video-allow-ffmpeg"), "{err}");
}

/// 2026-09-25: Disabled refuses before any process is built, so a nonexistent
/// binary is never reached.
#[test]
fn disabled_refuses_before_touching_the_binary() {
    let p = FfmpegPolicy {
        enabled: false,
        binary: "/nonexistent/definitely-not-a-decoder".to_string(),
        ..Default::default()
    };
    let err = decode_frames(b"x", 2.0, &p).unwrap_err().to_string();
    assert!(err.contains("disabled"), "{err}");
}

#[test]
fn probing_a_missing_binary_reports_it_by_name() {
    let p = FfmpegPolicy {
        enabled: true,
        binary: "/nonexistent/definitely-not-a-decoder".to_string(),
        ..Default::default()
    };
    match probe(&p) {
        Availability::Missing(why) => {
            assert!(why.contains("definitely-not-a-decoder"), "{why}");
        }
        other => panic!("expected Missing, got {other:?}"),
    }
}

#[test]
fn probing_while_disabled_says_disabled_rather_than_missing() {
    assert_eq!(probe(&FfmpegPolicy::default()), Availability::Disabled);
}

/// 2026-09-25: A missing binary fails the decode with the binary named and the
/// install hint, not as a decode error.
#[test]
fn a_missing_binary_fails_the_decode_by_name() {
    let p = FfmpegPolicy {
        enabled: true,
        binary: "/nonexistent/definitely-not-a-decoder".to_string(),
        ..Default::default()
    };
    let err = format!("{:#}", decode_frames(b"x", 2.0, &p).unwrap_err());
    assert!(err.contains("definitely-not-a-decoder"), "{err}");
    assert!(err.contains("ffmpeg installed"), "no remedy offered: {err}");
}

#[test]
fn an_h264_mp4_decodes_to_frames_at_the_requested_rate() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let mp4 = make_mp4(3, 10, "320x240", "libx264").expect("encode");
    let frames = decode_frames(&mp4, 2.0, &enabled()).expect("decode");
    // 2026-09-25: 3 s at 2 fps is 6 frames; the band allows one either side.
    assert!(
        (5..=7).contains(&frames.len()),
        "expected ~6 frames, got {}",
        frames.len()
    );
    assert_eq!(frames[0].dimensions(), (320, 240));
}

#[test]
fn the_requested_rate_actually_changes_the_frame_count() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let mp4 = make_mp4(4, 10, "160x120", "libx264").expect("encode");
    let slow = decode_frames(&mp4, 1.0, &enabled()).expect("1 fps");
    let fast = decode_frames(&mp4, 4.0, &enabled()).expect("4 fps");
    assert!(
        fast.len() > slow.len(),
        "4 fps gave {} frames, 1 fps gave {}",
        fast.len(),
        slow.len()
    );
}

/// 2026-09-25: Frames must differ: a backend that returned one frame N times
/// would pass the count and dimension checks.
#[test]
fn decoded_frames_are_not_all_identical() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let mp4 = make_mp4(3, 10, "160x120", "libx264").expect("encode");
    let frames = decode_frames(&mp4, 2.0, &enabled()).expect("decode");
    assert!(frames.len() >= 2);
    assert_ne!(
        frames[0].as_raw(),
        frames[frames.len() - 1].as_raw(),
        "first and last frame are byte-identical — the clip did not advance"
    );
}

#[test]
fn the_frame_cap_is_honoured() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let mp4 = make_mp4(5, 10, "160x120", "libx264").expect("encode");
    let p = FfmpegPolicy {
        max_frames: 3,
        ..enabled()
    };
    let frames = decode_frames(&mp4, 10.0, &p).expect("decode");
    assert_eq!(frames.len(), 3);
}

#[test]
fn an_output_cap_smaller_than_the_clip_is_refused() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let mp4 = make_mp4(3, 10, "320x240", "libx264").expect("encode");
    let p = FfmpegPolicy {
        max_output_bytes: 1024,
        ..enabled()
    };
    let err = format!("{:#}", decode_frames(&mp4, 2.0, &p).unwrap_err());
    assert!(err.contains("cap"), "{err}");
}

/// 2026-09-25: A non-video payload returns the decoder's error, not a panic or a
/// hang.
#[test]
fn a_non_video_payload_fails_cleanly() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let err = format!(
        "{:#}",
        decode_frames(b"this is not a video at all", 2.0, &enabled()).unwrap_err()
    );
    assert!(err.contains("decoder failed"), "{err}");
}

#[cfg(unix)]
#[test]
fn a_hanging_decoder_is_killed_at_timeout() {
    use std::os::unix::fs::PermissionsExt;

    static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "metrale-hanging-ffmpeg-{}-{seq}",
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let binary = dir.join("hanging-ffmpeg");
    std::fs::write(&binary, "#!/bin/sh\nexec sleep 10\n").unwrap();
    let mut permissions = std::fs::metadata(&binary).unwrap().permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&binary, permissions).unwrap();

    let policy = FfmpegPolicy {
        enabled: true,
        binary: binary.display().to_string(),
        timeout_secs: 1,
        ..Default::default()
    };
    // 2026-09-25: Retry while the spawn itself fails. This test writes a script
    // and execs it at once; a sibling test's fork can briefly hold a write
    // descriptor on it, and exec then fails with ETXTBSY. The retry keys on
    // the "could not run" prefix, which only `spawn_failure` produces, so it
    // fires whatever the errno. Each attempt is printed, and five failed
    // attempts fail the test. The elapsed-time check restarts per attempt
    // because it measures the kill.
    let mut attempts = 0;
    let err = loop {
        attempts += 1;
        let started = std::time::Instant::now();
        let err = decode_frames(b"input", 2.0, &policy)
            .unwrap_err()
            .to_string();
        if err.starts_with("could not run") && attempts < 5 {
            eprintln!("attempt {attempts}: the fake decoder would not spawn: {err}");
            std::thread::sleep(std::time::Duration::from_millis(50));
            continue;
        }
        assert!(
            started.elapsed() < std::time::Duration::from_secs(3),
            "the timeout kill took {:?}",
            started.elapsed()
        );
        break err;
    };
    let _ = std::fs::remove_dir_all(&dir);
    assert!(
        err.contains("decoding exceeded 1s"),
        "after {attempts} attempt(s): {err}"
    );
}

/// 2026-09-25: A write handle held open on the script makes exec fail with
/// ETXTBSY. Pins that `spawn_failure` names ETXTBSY and does not give the
/// install hint.
#[cfg(target_os = "linux")]
#[test]
fn a_binary_still_open_for_writing_is_reported_as_etxtbsy() {
    use std::io::Write;
    use std::os::unix::fs::PermissionsExt;

    static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let seq = SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir =
        std::env::temp_dir().join(format!("metrale-busy-ffmpeg-{}-{seq}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let binary = dir.join("busy-ffmpeg");

    // 2026-09-25: The handle stays open across the spawn attempt below; that is
    // the condition under test.
    let mut held = std::fs::File::create(&binary).unwrap();
    held.write_all(b"#!/bin/sh\nexit 0\n").unwrap();
    held.flush().unwrap();
    let mut permissions = std::fs::metadata(&binary).unwrap().permissions();
    permissions.set_mode(0o755);
    std::fs::set_permissions(&binary, permissions).unwrap();

    let policy = FfmpegPolicy {
        enabled: true,
        binary: binary.display().to_string(),
        timeout_secs: 1,
        ..Default::default()
    };
    let err = decode_frames(b"input", 2.0, &policy)
        .unwrap_err()
        .to_string();
    drop(held);
    let _ = std::fs::remove_dir_all(&dir);

    assert!(err.contains("ETXTBSY"), "{err}");
    assert!(
        !err.contains("is ffmpeg installed"),
        "a busy binary must not be reported as a missing one: {err}"
    );
}

/// 2026-09-25: H.265 in an MP4. Returns early when this ffmpeg build has no
/// libx265.
#[test]
fn an_hevc_clip_decodes_too_when_the_encoder_is_available() {
    if !have_ffmpeg() {
        eprintln!("skipping: no ffmpeg");
        return;
    }
    let Some(mp4) = make_mp4(2, 10, "160x120", "libx265") else {
        eprintln!("skipping: no libx265 in this ffmpeg build");
        return;
    };
    let frames = decode_frames(&mp4, 2.0, &enabled()).expect("decode hevc");
    assert!(!frames.is_empty());
    assert_eq!(frames[0].dimensions(), (160, 120));
}

#[test]
fn an_empty_stream_splits_to_nothing() {
    assert!(
        split_png_stream(b"")
            .expect("empty is not an error")
            .is_empty()
    );
}

fn spawn_message(kind: std::io::ErrorKind) -> String {
    spawn_failure("/opt/ff", std::io::Error::new(kind, "boom")).to_string()
}

#[test]
fn a_missing_binary_still_says_install_ffmpeg() {
    let m = spawn_message(std::io::ErrorKind::NotFound);
    assert!(m.contains("is ffmpeg installed"), "{m}");
    assert!(m.contains("/opt/ff"), "the binary must be named: {m}");
}

#[test]
fn a_present_but_unrunnable_binary_is_not_reported_as_missing() {
    for (kind, expected) in [
        (std::io::ErrorKind::PermissionDenied, "noexec"),
        (std::io::ErrorKind::ExecutableFileBusy, "ETXTBSY"),
    ] {
        let m = spawn_message(kind);
        assert!(
            !m.contains("is ffmpeg installed"),
            "{kind:?} must not be blamed on a missing install: {m}"
        );
        assert!(m.contains(expected), "{kind:?} must name its cause: {m}");
    }
}

#[test]
fn an_unclassified_spawn_error_still_carries_the_os_message() {
    // 2026-09-25: Other kinds get no hint, but the OS error text is kept.
    let m = spawn_message(std::io::ErrorKind::Other);
    assert!(
        m.contains("boom"),
        "the OS error must reach the operator: {m}"
    );
    assert!(!m.contains("is ffmpeg installed"), "{m}");
}

#[test]
fn a_stream_of_two_pngs_splits_into_two_frames() {
    let mut buf = Vec::new();
    for pixel in [[1, 2, 3], [4, 5, 6]] {
        let img = image::RgbImage::from_pixel(4, 3, image::Rgb(pixel));
        let mut one = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut std::io::Cursor::new(&mut one), image::ImageFormat::Png)
            .expect("encode");
        buf.extend_from_slice(&one);
    }
    let frames = split_png_stream(&buf).expect("split");
    assert_eq!(frames.len(), 2);
    assert_eq!(frames[0].dimensions(), (4, 3));
    assert_eq!(frames[0].get_pixel(0, 0).0, [1, 2, 3]);
    assert_eq!(frames[1].get_pixel(0, 0).0, [4, 5, 6]);
}
