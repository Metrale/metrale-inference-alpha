// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Normalising `bash` tool output so that identical work reads
//! identically: progress lines are dropped, and timestamps, dates, durations,
//! pids and `ps` bookkeeping columns become placeholders. Every tool output is
//! fed back to the model, so a build that reports `in 0.00s` on one run and
//! `in 1.23s` on the next gives two greedy runs different contexts.
//!
//! Each rule is anchored on a shape only tool bookkeeping has: a cargo status
//! word at the start of a line, a `time(1)` row, curl's progress meter, a
//! timestamp. Compiler diagnostics pass through. Outside a `fuser` line, the
//! pid rules rewrite only runs of four or more digits, so `(exit status: 101)`
//! and `2 passed` survive. A port survives when it is glued to `/tcp`, `:` or
//! `.`, or follows the word `port`.
//!
//! Only `bash` results are normalised (`agent_shell.rs` `run_shell`), before
//! truncation. `read`, `grep` and `glob` results are not, so an `edit` built
//! from a `read` matches the file. No rule can fix a `ps` COMMAND column that
//! shows another process on the box, or timing that changes what a command
//! prints.
//!
//! Owner: bench, agentic.
//! Invariants:
//! - `normalize` is idempotent: every placeholder it writes is digit-free.
//! - It drops only progress lines, and reorders only runs of libtest result
//!   lines; every other line keeps its place.

/// 2026-09-26: Prefixes (after leading whitespace) of cargo progress lines,
/// which differ between a cold build and a warm one. `Finished`, which says
/// whether the build worked, is not among them.
const PROGRESS: [&str; 7] = [
    "Compiling ",
    "Fresh ",
    "Downloading ",
    "Downloaded ",
    "Updating ",
    "Locking ",
    "Blocking waiting",
];

const MONTHS: [&str; 12] = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
];

/// 2026-09-26: Two-letter units first, so `ms` is tried before `m`.
const UNITS: [&str; 6] = ["ns", "µs", "us", "ms", "s", "m"];

/// 2026-09-26: Normalise one shell result. Idempotent: every placeholder it
/// writes is digit-free, so no rule matches its own output.
pub fn normalize(text: &str) -> String {
    let mut lines: Vec<String> = text
        .split('\n')
        .filter(|line| !is_noise(line))
        .map(scrub)
        .collect();
    sort_test_results(&mut lines);
    lines.join("\n")
}

/// 2026-09-26: Sort each run of consecutive `test <name> ... <result>` lines,
/// which libtest prints in the order tests finish. The `test result:` summary
/// and a failing test's `---- <name> stdout ----` block do not have that shape
/// and keep their place.
fn sort_test_results(lines: &mut [String]) {
    let mut i = 0;
    while i < lines.len() {
        if !is_test_result(&lines[i]) {
            i += 1;
            continue;
        }
        let mut end = i;
        while end < lines.len() && is_test_result(&lines[end]) {
            end += 1;
        }
        lines[i..end].sort();
        i = end;
    }
}

fn is_test_result(line: &str) -> bool {
    line.starts_with("test ") && line.contains(" ... ")
}

/// 2026-09-26: A progress line: a [`PROGRESS`] prefix or a curl meter row.
fn is_noise(line: &str) -> bool {
    let head = line.trim_start();
    PROGRESS.iter().any(|p| head.starts_with(p))
        // 2026-09-26: curl's progress meter: its two header rows, and every row
        // with a `--:--:--` clock.
        || line.contains("--:--:--")
        || head.starts_with("% Total")
        || head.starts_with("Dload")
}

fn scrub(line: &str) -> String {
    let line = replace_spans(line);
    let line = trailing_duration(&line);
    let line = time_builtin(&line);
    pids(&line)
}

/// 2026-09-26: Rewrite every timestamp- or date-shaped span, at word
/// boundaries only, so digits inside an identifier (a cargo metadata hash, a
/// version) are never taken for one.
fn replace_spans(line: &str) -> String {
    let mut out = String::with_capacity(line.len());
    let mut i = 0;
    while i < line.len() {
        let rest = &line[i..];
        let boundary = !line[..i].chars().next_back().is_some_and(identifier_char);
        if boundary && let Some((len, replacement)) = span_at(rest) {
            let ends_at_boundary = !rest[len..].chars().next().is_some_and(identifier_char);
            if ends_at_boundary {
                out.push_str(replacement);
                i += len;
                continue;
            }
        }
        let c = rest.chars().next().unwrap_or('\0');
        out.push(c);
        i += c.len_utf8();
    }
    out
}

fn identifier_char(c: char) -> bool {
    c.is_ascii_alphanumeric() || c == '_'
}

fn span_at(s: &str) -> Option<(usize, &'static str)> {
    if let Some(n) = timestamp(s) {
        return Some((n, "<timestamp>"));
    }
    if let Some(n) = listing_date(s) {
        return Some((n, "<date>"));
    }
    // 2026-09-26: `curl: (7) Failed to connect to localhost port 3001 after 0
    // ms: …`: the port stays, the duration after `after` does not.
    if let Some(rest) = s.strip_prefix("after")
        && let Some(n) = duration(rest.as_bytes(), 0)
    {
        return Some((5 + n, "after <elapsed>"));
    }
    None
}

/// 2026-09-26: `2026-08-06T21:09:12.345Z`, and the same with a space for the
/// `T`. A bare date with no clock is left alone: it could be data.
fn timestamp(s: &str) -> Option<usize> {
    let b = s.as_bytes();
    let mut i = digits(b, 0, 4, 4)?;
    i = byte(b, i, b'-')?;
    i = digits(b, i, 2, 2)?;
    i = byte(b, i, b'-')?;
    i = digits(b, i, 2, 2)?;
    if !matches!(b.get(i), Some(b'T') | Some(b' ')) {
        return None;
    }
    let mut j = clock(b, i + 1)?;
    if b.get(j) == Some(&b'.') {
        j = digits(b, j + 1, 1, 9)?;
    }
    match b.get(j) {
        Some(b'Z') => j += 1,
        Some(b'+') | Some(b'-') => {
            if let Some(k) = digits(b, j + 1, 2, 2)
                .and_then(|k| byte(b, k, b':'))
                .and_then(|k| digits(b, k, 2, 2))
            {
                j = k;
            }
        }
        _ => {}
    }
    Some(j)
}

/// 2026-09-26: A month name, day, and clock or year: `Aug  6 21:09`,
/// `Aug  6 21:09:12`, `Aug  6  2025` (`ls -l`, `date`).
fn listing_date(s: &str) -> Option<usize> {
    let month = MONTHS.iter().find(|m| s.starts_with(**m))?;
    let b = s.as_bytes();
    let mut i = spaces(b, month.len())?;
    i = digits(b, i, 1, 2)?;
    let j = spaces(b, i)?;
    clock(b, j).or_else(|| digits(b, j, 4, 4))
}

fn clock(b: &[u8], i: usize) -> Option<usize> {
    let mut j = digits(b, i, 2, 2)?;
    j = byte(b, j, b':')?;
    j = digits(b, j, 2, 2)?;
    if b.get(j) == Some(&b':') {
        j = digits(b, j + 1, 2, 2)?;
    }
    Some(j)
}

/// 2026-09-26: A line that ends in `… in <duration>`: cargo's `Finished` line,
/// libtest's `test result: … finished in 0.00s`. Only a whole duration at the
/// end of the line matches, which keeps prose containing " in " intact.
fn trailing_duration(line: &str) -> String {
    let trimmed = line.trim_end();
    let Some(cut) = trimmed.rfind(" in ") else {
        return line.to_string();
    };
    let tail = &trimmed[cut + 4..];
    match whole_duration(tail) {
        true => format!("{} in <elapsed>", &trimmed[..cut]),
        false => line.to_string(),
    }
}

/// 2026-09-26: `time(1)`'s three rows: `real\t0m1.234s`, `user …`, `sys …`.
fn time_builtin(line: &str) -> String {
    let head = line.trim_start();
    let Some(word) = ["real", "user", "sys"]
        .iter()
        .find(|w| head.starts_with(**w))
    else {
        return line.to_string();
    };
    let rest = &head[word.len()..];
    let value = rest.trim_start();
    if value.len() == rest.len() || !whole_duration(value.trim_end()) {
        return line.to_string();
    }
    format!("{}{word}\t<elapsed>", &line[..line.len() - head.len()])
}

fn whole_duration(s: &str) -> bool {
    duration(s.as_bytes(), 0) == Some(s.len()) && !s.is_empty()
}

/// 2026-09-26: One or more `<number><unit>` groups, each optionally preceded
/// by a space: `0.42s`, `1m 03s`, `0m1.234s`, `0 ms`.
fn duration(b: &[u8], start: usize) -> Option<usize> {
    let (mut i, mut any) = (start, false);
    loop {
        let mut j = i;
        if b.get(j) == Some(&b' ') {
            j += 1;
        }
        let Some(n) = number(b, j) else { break };
        let mut k = n;
        if b.get(k) == Some(&b' ') {
            k += 1;
        }
        let Some(end) = unit(b, k) else { break };
        i = end;
        any = true;
    }
    any.then_some(i)
}

/// 2026-09-26: A unit counts only when no letter follows it, so `3 more` is not
/// 3 minutes and `2 seconds` is not `2 s` + `econds`.
fn unit(b: &[u8], i: usize) -> Option<usize> {
    let rest = std::str::from_utf8(b.get(i..)?).ok()?;
    let u = UNITS.iter().find(|u| rest.starts_with(**u))?;
    let end = i + u.len();
    match b.get(end) {
        Some(c) if c.is_ascii_alphabetic() => None,
        _ => Some(end),
    }
}

fn number(b: &[u8], i: usize) -> Option<usize> {
    let j = digits(b, i, 1, 12)?;
    match b.get(j) {
        Some(b'.') => digits(b, j + 1, 1, 12),
        _ => Some(j),
    }
}

/// 2026-09-26: Rewrite process ids.
///
/// [`ps_line`] and [`fuser_line`] are tried first. `fuser`'s `3001/tcp:  12345`
/// keeps the port and rewrites the free-standing numbers after the colon.
/// Otherwise, runs of four to seven digits are rewritten on a line that
/// consists only of such numbers (`echo $!`, `pgrep`, `lsof -t`) or that names
/// a pid with a [`NAMES_A_PID`] word (`kill: (12345) - No such process`).
///
/// A four-digit number alone on a line (`… | wc -c`, or a port) is rewritten
/// too. `ps` output is covered only by [`ps_line`].
fn pids(line: &str) -> String {
    if let Some(rewritten) = ps_line(line).or_else(|| fuser_line(line)) {
        return rewritten;
    }
    // 2026-09-26: A line that already carries `<pid>` has been through here;
    // matching the word inside it would let a second pass rewrite digits the
    // first pass kept.
    let named = !line.contains("<pid>")
        && line
            .split(|c: char| !c.is_ascii_alphabetic())
            .any(|w| NAMES_A_PID.iter().any(|k| w.eq_ignore_ascii_case(k)));
    match named || bare_numbers(line) {
        true => digit_runs(line, 4, 7),
        false => line.to_string(),
    }
}

/// 2026-09-26: Words that mark a line as naming a pid. Matched as whole words,
/// so `Processing` and `skill` do not match.
const NAMES_A_PID: [&str; 7] = [
    "kill",
    "killed",
    "killing",
    "pid",
    "pids",
    "process",
    "processes",
];

/// 2026-09-26: `ps aux` output with the pid, %CPU, %MEM, VSZ, RSS, START and
/// TIME columns replaced; USER, TTY, STAT and COMMAND are kept.
///
/// Recognised by shape, because `| grep` strips the header: eleven or more
/// fields, an all-digit pid, two `N.N` percentages, two integer sizes, and a
/// colon-separated TIME in the tenth field.
fn ps_line(line: &str) -> Option<String> {
    let f: Vec<&str> = line.split_whitespace().collect();
    if f.len() < 11 {
        return None;
    }
    let int = |s: &str| !s.is_empty() && s.bytes().all(|c| c.is_ascii_digit());
    let decimal = |s: &str| s.split_once('.').is_some_and(|(a, b)| int(a) && int(b));
    let elapsed = |s: &str| s.contains(':') && s.split(':').all(int);
    let shaped =
        int(f[1]) && decimal(f[2]) && decimal(f[3]) && int(f[4]) && int(f[5]) && elapsed(f[9]);
    shaped.then(|| {
        format!(
            "{} <pid> <cpu> <mem> <vsz> <rss> {} {} <start> <time> {}",
            f[0],
            f[6],
            f[7],
            f[10..].join(" ")
        )
    })
}

fn fuser_line(line: &str) -> Option<String> {
    let head = line.trim_start();
    let n = digits(head.as_bytes(), 0, 1, 5)?;
    let rest = ["/tcp:", "/udp:"]
        .iter()
        .find_map(|p| head[n..].strip_prefix(*p))?;
    let keep = &line[..line.len() - rest.len()];
    Some(format!("{keep}{}", digit_runs(rest, 1, 9)))
}

fn bare_numbers(line: &str) -> bool {
    let mut fields = line.split_whitespace().peekable();
    fields.peek().is_some()
        && fields.all(|f| (4..=7).contains(&f.len()) && f.bytes().all(|c| c.is_ascii_digit()))
}

/// 2026-09-26: Replace every free-standing run of `min..=max` digits.
///
/// A run that touches a letter, a digit or a [`GLUED`] byte on either side is
/// part of something else and is kept: `pingpong-9a8b7c6d` (a cargo metadata
/// hash), `0.0.0.0:3001` and `3001/tcp` (an address and a port). So is a run
/// right after the word `port` ([`after_port_word`]).
fn digit_runs(s: &str, min: usize, max: usize) -> String {
    let (b, mut out, mut i) = (s.as_bytes(), String::with_capacity(s.len()), 0);
    while i < s.len() {
        if b[i].is_ascii_digit() {
            let end = digits(b, i, 1, usize::MAX).unwrap_or(i);
            let glued_before = i > 0 && glued(b[i - 1]);
            let glued_after = b.get(end).copied().is_some_and(glued);
            let free = !(glued_before || glued_after || after_port_word(s, i));
            match free && (min..=max).contains(&(end - i)) {
                true => out.push_str("<pid>"),
                false => out.push_str(&s[i..end]),
            }
            i = end;
            continue;
        }
        let c = s[i..].chars().next().unwrap_or('\0');
        out.push(c);
        i += c.len_utf8();
    }
    out
}

/// 2026-09-26: Is the number at `i` right after the word `port` (ignoring
/// spaces, tabs, `=` and `:`)? Then it is a port: `# kill whatever is on port
/// 3001`.
fn after_port_word(s: &str, i: usize) -> bool {
    let head = s[..i].trim_end_matches([' ', '\t', '=', ':']);
    head.rsplit(|c: char| !c.is_ascii_alphabetic())
        .next()
        .is_some_and(|w| w.eq_ignore_ascii_case("port"))
}

/// 2026-09-26: Non-alphanumeric neighbours that make a number part of a larger
/// token.
const GLUED: [u8; 4] = [b'.', b':', b'/', b'-'];

fn glued(c: u8) -> bool {
    c.is_ascii_alphanumeric() || GLUED.contains(&c)
}

fn digits(b: &[u8], i: usize, min: usize, max: usize) -> Option<usize> {
    let mut end = i;
    while end < b.len() && b[end].is_ascii_digit() && end - i < max {
        end += 1;
    }
    (end - i >= min).then_some(end)
}

fn byte(b: &[u8], i: usize, want: u8) -> Option<usize> {
    (b.get(i) == Some(&want)).then_some(i + 1)
}

fn spaces(b: &[u8], i: usize) -> Option<usize> {
    let mut end = i;
    while b.get(end) == Some(&b' ') {
        end += 1;
    }
    (end > i).then_some(end)
}

#[cfg(test)]
#[path = "norm_tests.rs"]
mod tests;
