// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The Stats sampler. Each `sample` reads the prometheus counters
//! and diffs them against the previous sample into rates, reads the TTFT
//! histogram, prefix-cache and sampler figures, GPU and host memory and the
//! scheduler snapshot, and pushes onto bounded histories.
//!
//! Owner: server tui.
//! Invariants: none beyond the types.

use std::collections::VecDeque;
use std::time::Instant;

use crate::metrics;
use metrale_speculative::snapshot::SchedulerSnapshot;

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// 2026-09-26: Numeric history for a sparkline or chart. `new(cap)` with
/// `cap >= 1` keeps at most `cap` points: a push at capacity drops the oldest.
#[derive(Default)]
pub struct History {
    pub points: VecDeque<f64>,
    cap: usize,
}

impl History {
    fn new(cap: usize) -> Self {
        Self {
            points: VecDeque::with_capacity(cap),
            cap,
        }
    }
    fn push(&mut self, v: f64) {
        if self.points.len() == self.cap {
            self.points.pop_front();
        }
        self.points.push_back(v);
    }
    pub fn as_u64(&self) -> Vec<u64> {
        self.points
            .iter()
            .map(|v| (v.max(0.0) * 10.0) as u64)
            .collect()
    }
}

/// 2026-09-26: One TTFT histogram snapshot: (upper_bound_secs, cumulative_count).
pub type TtftBuckets = Vec<(f64, u64)>;

pub struct StatsModel {
    pub requests_total: u64,
    pub requests_active: i64,
    pub gen_tokens_total: u64,
    pub prompt_tokens_total: u64,
    pub tool_calls_total: u64,
    pub bytes_in_total: u64,
    pub bytes_out_total: u64,
    pub gen_tps: f64,
    pub prompt_tps: f64,
    pub req_rate: f64,
    pub bytes_in_rate: f64,
    pub bytes_out_rate: f64,
    pub gen_tps_history: History,
    pub req_history: History,
    pub queue_history: History,
    pub entropy_history: History,
    pub ttft_buckets: TtftBuckets,
    pub ttft_p50_ms: Option<f64>,
    pub ttft_p90_ms: Option<f64>,
    pub prefix_hit_rate: Option<f64>,
    pub prefix_hit_tokens: u64,
    pub entropy: f64,
    // 2026-09-26: Per `k` label of `metrale_spec_decode_verify_total`:
    // (k, count of outcomes containing "accept", count of all outcomes).
    pub spec_accept: Vec<(String, u64, u64)>,
    /// 2026-09-26: True when the last sample read both free GPU memory and the
    /// baseline. The three GPU fields below are 0.0 until the first such
    /// sample and keep their previous values after a failed one.
    pub gpu_known: bool,
    pub gpu_free_gb: f64,
    pub gpu_total_gb: f64,
    pub metrale_used_gb: f64,
    pub host_avail_gb: f64,
    pub host_total_gb: f64,
    pub sched: Option<SchedulerSnapshot>,
    last_sample: Option<(Instant, Prev)>,
}

#[derive(Clone, Copy)]
struct Prev {
    requests: u64,
    gen_tok: u64,
    decoded: u64,
    prompt: u64,
    bytes_in: u64,
    bytes_out: u64,
}

impl Default for StatsModel {
    fn default() -> Self {
        Self {
            requests_total: 0,
            requests_active: 0,
            gen_tokens_total: 0,
            prompt_tokens_total: 0,
            tool_calls_total: 0,
            bytes_in_total: 0,
            bytes_out_total: 0,
            gen_tps: 0.0,
            prompt_tps: 0.0,
            req_rate: 0.0,
            bytes_in_rate: 0.0,
            bytes_out_rate: 0.0,
            gen_tps_history: History::new(120),
            req_history: History::new(16),
            queue_history: History::new(24),
            entropy_history: History::new(24),
            ttft_buckets: Vec::new(),
            ttft_p50_ms: None,
            ttft_p90_ms: None,
            prefix_hit_rate: None,
            prefix_hit_tokens: 0,
            entropy: 0.0,
            spec_accept: Vec::new(),
            gpu_known: false,
            gpu_free_gb: 0.0,
            gpu_total_gb: 0.0,
            metrale_used_gb: 0.0,
            host_avail_gb: 0.0,
            host_total_gb: 0.0,
            sched: None,
            last_sample: None,
        }
    }
}

/// 2026-09-26: Percentile from cumulative histogram buckets: the upper bound
/// (in ms) of the first bucket whose count reaches the rank. The total is the
/// last bucket's count; `None` when it is 0.
fn hist_percentile(buckets: &TtftBuckets, p: f64) -> Option<f64> {
    let total = buckets.last().map(|(_, c)| *c)?;
    if total == 0 {
        return None;
    }
    let rank = (total as f64 * p).ceil() as u64;
    for (ub, cum) in buckets {
        if *cum >= rank {
            return Some(ub * 1000.0);
        }
    }
    None
}

impl StatsModel {
    /// 2026-09-26: Take one sample. The event loop calls it every
    /// `events_rules::SAMPLE_EVERY` ticks.
    pub fn sample(&mut self, run: Option<&crate::tui::RunHandles>) {
        let now = Instant::now();
        self.requests_total = metrics::REQUESTS_TOTAL.get();
        self.requests_active = metrics::REQUESTS_ACTIVE.get();
        self.gen_tokens_total = metrics::GENERATION_TOKENS_TOTAL.get();
        let decoded_total = metrics::DECODED_TOKENS_TOTAL.get();
        self.prompt_tokens_total = metrics::PROMPT_TOKENS_TOTAL.get();
        self.tool_calls_total = metrics::TOOL_CALLS_TOTAL.get();
        self.bytes_in_total = metrics::HTTP_BYTES_IN.get();
        self.bytes_out_total = metrics::HTTP_BYTES_OUT.get();

        if let Some((t0, prev)) = self.last_sample {
            let dt = now.duration_since(t0).as_secs_f64().max(0.05);
            // 2026-09-26: The larger of two deltas: `DECODED_TOKENS_TOTAL`
            // counts each token the streaming chat handler processes, and
            // `GENERATION_TOKENS_TOTAL` adds a request's tokens when it ends
            // (streaming or blocking).
            let decoded_delta = decoded_total.saturating_sub(prev.decoded);
            let gen_delta = self.gen_tokens_total.saturating_sub(prev.gen_tok);
            self.gen_tps = decoded_delta.max(gen_delta) as f64 / dt;
            self.prompt_tps = (self.prompt_tokens_total.saturating_sub(prev.prompt)) as f64 / dt;
            self.req_rate = (self.requests_total.saturating_sub(prev.requests)) as f64 / dt;
            self.bytes_in_rate = (self.bytes_in_total.saturating_sub(prev.bytes_in)) as f64 / dt;
            self.bytes_out_rate = (self.bytes_out_total.saturating_sub(prev.bytes_out)) as f64 / dt;
            self.gen_tps_history.push(self.gen_tps);
            self.req_history.push(self.req_rate);
        }
        self.last_sample = Some((
            now,
            Prev {
                requests: self.requests_total,
                gen_tok: self.gen_tokens_total,
                decoded: decoded_total,
                prompt: self.prompt_tokens_total,
                bytes_in: self.bytes_in_total,
                bytes_out: self.bytes_out_total,
            },
        ));

        self.ttft_buckets.clear();
        for mf in prometheus::gather() {
            if mf.name() != "metrale_time_to_first_token_seconds" {
                continue;
            }
            if let Some(m) = mf.get_metric().first() {
                for b in m.get_histogram().get_bucket() {
                    self.ttft_buckets
                        .push((b.upper_bound(), b.cumulative_count()));
                }
            }
        }
        self.ttft_p50_ms = hist_percentile(&self.ttft_buckets, 0.50);
        self.ttft_p90_ms = hist_percentile(&self.ttft_buckets, 0.90);

        self.spec_accept = spec_accept_from_gather();

        // 2026-09-26: Prefix-cache counts since the current model loaded
        // (`cache_counts_this_run`), not the process-lifetime counters.
        let (hits, misses, hit_tokens) = metrale_telemetry::run_metrics::cache_counts_this_run();
        self.prefix_hit_tokens = hit_tokens;
        self.prefix_hit_rate = (hits + misses > 0).then(|| hits as f64 / (hits + misses) as f64);
        self.entropy = metrale_sampling::last_entropy() as f64;
        self.entropy_history.push(self.entropy);

        // 2026-09-26: Both reads must succeed: `metrale_used_gb` is their
        // difference.
        match (
            super::gpu_free_bytes(),
            metrale_gpu_runtime::gpu::baseline_free_bytes(),
        ) {
            (Some(free), Some(baseline)) => {
                self.gpu_free_gb = free as f64 / GIB;
                self.gpu_total_gb = baseline as f64 / GIB;
                self.metrale_used_gb = (self.gpu_total_gb - self.gpu_free_gb).max(0.0);
                self.gpu_known = true;
            }
            _ => self.gpu_known = false,
        }
        if let Some((avail, total)) = host_mem_gb() {
            self.host_avail_gb = avail;
            self.host_total_gb = total;
        }

        self.sched = run.and_then(|r| r.snapshot.read());
        if let Some(s) = self.sched {
            self.queue_history.push(s.pending_len as f64);
        }
    }
}

fn spec_accept_from_gather() -> Vec<(String, u64, u64)> {
    use std::collections::BTreeMap;
    let mut per_k: BTreeMap<String, (u64, u64)> = BTreeMap::new();
    for mf in prometheus::gather() {
        if mf.name() != "metrale_spec_decode_verify_total" {
            continue;
        }
        for m in mf.get_metric() {
            let mut k = String::new();
            let mut outcome = String::new();
            for l in m.get_label() {
                match l.name() {
                    "k" => k = l.value().to_string(),
                    "outcome" => outcome = l.value().to_string(),
                    _ => {}
                }
            }
            let e = per_k.entry(k).or_default();
            let v = m.get_counter().get_value() as u64;
            e.1 += v;
            if outcome.contains("accept") {
                e.0 += v;
            }
        }
    }
    per_k.into_iter().map(|(k, (a, t))| (k, a, t)).collect()
}

/// 2026-09-26: (available, total) host RAM in GiB, from /proc/meminfo.
fn host_mem_gb() -> Option<(f64, f64)> {
    let text = std::fs::read_to_string("/proc/meminfo").ok()?;
    let grab = |key: &str| -> Option<f64> {
        text.lines()
            .find(|l| l.starts_with(key))?
            .split_whitespace()
            .nth(1)?
            .parse::<f64>()
            .ok()
            .map(|kb| kb / (1024.0 * 1024.0))
    };
    Some((grab("MemAvailable:")?, grab("MemTotal:")?))
}
