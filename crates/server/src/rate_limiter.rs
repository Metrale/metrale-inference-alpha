// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: Per-identity token-bucket rate limiter for `/v1/*` requests: one bucket metered in requests and one in tokens, each refilling linearly with elapsed time up to its burst.
//!
//! The identity (`extract_identity`) is, first match wins: a hash of the
//! `Authorization: Bearer` token, the first `X-Forwarded-For` entry, then the
//! peer IP. `X-Forwarded-For` is trusted as sent.
//!
//! Configuration, read once at startup (`RateLimitConfig::from_env`): unset
//! `METRALE_RATE_LIMIT_RPM` and `METRALE_RATE_LIMIT_TPM` mean 0, which
//! disables that bucket; `METRALE_RATE_LIMIT_BURST_RPM` and
//! `METRALE_RATE_LIMIT_BURST_TPM` default to the matching rate. With both
//! rates 0 every request is admitted and the headers carry fixed "unlimited"
//! values.
//!
//! Owner: server rate limiter.
//! Invariants:
//! - A bucket never holds more than its burst: refill and refund both cap at it.
//! - A denied request consumes nothing: a request debited before a token
//!   denial is refunded.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::Mutex;

/// 2026-09-26: Limiter configuration. A rate of 0 disables that bucket; bursts
/// are at least 1.
#[derive(Clone, Copy, Debug)]
pub struct RateLimitConfig {
    pub rpm: u64,
    pub tpm: u64,
    pub burst_rpm: u64,
    pub burst_tpm: u64,
}

impl RateLimitConfig {
    /// 2026-09-26: Read the four `METRALE_RATE_LIMIT_*` variables.
    ///
    /// # Errors
    /// When a variable is set to something that is not a whole number
    /// (`env_config::parse_min`); unset or blank means the default.
    pub fn from_env() -> Result<Self, String> {
        Self::from_raw(
            std::env::var("METRALE_RATE_LIMIT_RPM").ok().as_deref(),
            std::env::var("METRALE_RATE_LIMIT_TPM").ok().as_deref(),
            std::env::var("METRALE_RATE_LIMIT_BURST_RPM")
                .ok()
                .as_deref(),
            std::env::var("METRALE_RATE_LIMIT_BURST_TPM")
                .ok()
                .as_deref(),
        )
    }

    /// 2026-09-26: [`Self::from_env`] without the environment, so tests need not
    /// set process variables.
    ///
    /// # Errors
    /// As [`Self::from_env`].
    pub fn from_raw(
        rpm: Option<&str>,
        tpm: Option<&str>,
        burst_rpm: Option<&str>,
        burst_tpm: Option<&str>,
    ) -> Result<Self, String> {
        use crate::env_config::parse_min;
        let rpm = parse_min(
            "METRALE_RATE_LIMIT_RPM",
            rpm,
            0,
            "requests per minute per client; 0 disables the request-rate limit",
        )?
        .unwrap_or(0);
        let tpm = parse_min(
            "METRALE_RATE_LIMIT_TPM",
            tpm,
            0,
            "tokens per minute per client; 0 disables the token-rate limit",
        )?
        .unwrap_or(0);
        // 2026-09-26: An unset burst takes the rate; a set `0` is floored to 1
        // below.
        let burst_rpm = parse_min(
            "METRALE_RATE_LIMIT_BURST_RPM",
            burst_rpm,
            0,
            "request-bucket depth; defaults to METRALE_RATE_LIMIT_RPM",
        )?
        .unwrap_or(rpm);
        let burst_tpm = parse_min(
            "METRALE_RATE_LIMIT_BURST_TPM",
            burst_tpm,
            0,
            "token-bucket depth; defaults to METRALE_RATE_LIMIT_TPM",
        )?
        .unwrap_or(tpm);
        Ok(Self {
            rpm,
            tpm,
            burst_rpm: burst_rpm.max(1),
            burst_tpm: burst_tpm.max(1),
        })
    }

    pub fn is_enabled(&self) -> bool {
        self.rpm > 0 || self.tpm > 0
    }

    /// 2026-09-26: The token limit to advertise: the burst when the token
    /// bucket is enforced, else the same "unlimited" value as the disabled path.
    /// A floored burst of 1 on an unenforced axis would read as one token left.
    fn advertised_tpm(&self) -> u64 {
        if self.tpm > 0 {
            self.burst_tpm
        } else {
            1_000_000_000
        }
    }

    /// 2026-09-26: As [`Self::advertised_tpm`], for the request bucket.
    fn advertised_rpm(&self) -> u64 {
        if self.rpm > 0 {
            self.burst_rpm
        } else {
            1_000_000
        }
    }

    /// 2026-09-26: Remaining budget to advertise: the real figure when the axis
    /// is enforced, else a value consistent with the "unlimited" limit.
    fn advertised_remaining_tpm(&self, avail: u64) -> u64 {
        if self.tpm > 0 { avail } else { 999_999_999 }
    }

    fn advertised_remaining_rpm(&self, avail: u64) -> u64 {
        if self.rpm > 0 { avail } else { 999_999 }
    }
}

/// 2026-09-26: One bucket's limit, remaining budget and reset time, for the
/// `x-ratelimit-*` response headers.
#[derive(Clone, Copy, Debug)]
pub struct BucketSnapshot {
    pub limit: u64,
    pub remaining: u64,
    /// 2026-09-26: Seconds until this bucket is full again (0 when its rate
    /// is 0).
    pub reset_secs: u64,
}

#[derive(Clone, Copy, Debug)]
pub struct RateDecision {
    pub allowed: bool,
    pub requests: BucketSnapshot,
    pub tokens: BucketSnapshot,
    /// 2026-09-26: Seconds for the `Retry-After` header: at least 1 when
    /// denied, 0 when allowed.
    pub retry_after_secs: u64,
    /// 2026-09-26: The bucket that denied the request; `None` when allowed.
    pub denied_by: Option<DenialReason>,
}

#[derive(Clone, Copy, Debug)]
pub enum DenialReason {
    Requests,
    Tokens,
}

/// 2026-09-26: What the rate-limit middleware stores in the request's
/// extensions when the limiter is enabled and admitted the request: the
/// identity and the tokens reserved (`max_seq_len`). The chat handlers use it
/// to refund the unused part (`api/chat_blocking.rs`,
/// `api/chat_stream/handle_done.rs`, `handle_error.rs`).
#[derive(Clone, Debug)]
pub struct RequestContext {
    pub identity: String,
    pub reserved_tokens: u64,
}

struct Bucket {
    /// 2026-09-26: Budget left; fractional because refill is continuous.
    available: f64,
    /// 2026-09-26: When `available` was last refilled.
    last_refill: Instant,
}

impl Bucket {
    fn new(burst: u64) -> Self {
        Self {
            available: burst as f64,
            last_refill: Instant::now(),
        }
    }

    /// 2026-09-26: Refill at `rate_per_sec` up to `burst`, then debit `cost` if
    /// it fits. Returns whether it was debited.
    fn try_consume(&mut self, cost: f64, rate_per_sec: f64, burst: f64, now: Instant) -> bool {
        let dt = now
            .saturating_duration_since(self.last_refill)
            .as_secs_f64();
        if dt > 0.0 {
            self.available = (self.available + dt * rate_per_sec).min(burst);
            self.last_refill = now;
        }
        if self.available >= cost {
            self.available -= cost;
            true
        } else {
            false
        }
    }

    fn snapshot(&self, rate_per_sec: f64, burst: f64, now: Instant) -> (f64, u64) {
        let dt = now
            .saturating_duration_since(self.last_refill)
            .as_secs_f64();
        let available = (self.available + dt * rate_per_sec).min(burst);
        let deficit = (burst - available).max(0.0);
        let reset = if rate_per_sec > 0.0 {
            (deficit / rate_per_sec).ceil() as u64
        } else {
            0
        };
        (available, reset)
    }

    /// 2026-09-26: Add `amount` back, up to `burst`.
    fn refund(&mut self, amount: f64, burst: f64) {
        self.available = (self.available + amount).min(burst);
    }
}

struct KeyState {
    requests: Bucket,
    tokens: Bucket,
}

/// 2026-09-26: The limiter, shared across requests.
pub struct RateLimiter {
    cfg: RateLimitConfig,
    inner: Mutex<HashMap<String, KeyState>>,
    /// 2026-09-26: When idle keys were last evicted; `admit` evicts again once
    /// `SCRUB_INTERVAL` has passed.
    last_scrub: Mutex<Instant>,
}

const SCRUB_INTERVAL: Duration = Duration::from_secs(120);
/// 2026-09-26: A key whose buckets were both last refilled this long ago is
/// evicted.
const IDLE_EVICT: Duration = Duration::from_secs(600);
/// 2026-09-26: Map size at which a new key forces an idle-key eviction
/// before `SCRUB_INTERVAL` is due. It is not a hard cap: the new key is
/// inserted even when nothing was evicted.
const MAX_KEYS: usize = 100_000;

impl RateLimiter {
    /// 2026-09-26: A limiter configured from the environment.
    ///
    /// # Errors
    /// As [`RateLimitConfig::from_env`].
    pub fn from_env() -> Result<Arc<Self>, String> {
        Ok(Arc::new(Self {
            cfg: RateLimitConfig::from_env()?,
            inner: Mutex::new(HashMap::new()),
            last_scrub: Mutex::new(Instant::now()),
        }))
    }

    #[cfg(test)]
    pub fn with_config(cfg: RateLimitConfig) -> Arc<Self> {
        Arc::new(Self {
            cfg,
            inner: Mutex::new(HashMap::new()),
            last_scrub: Mutex::new(Instant::now()),
        })
    }

    pub fn config(&self) -> RateLimitConfig {
        self.cfg
    }

    /// 2026-09-26: Admit or deny one request from `key`, debiting one request
    /// and `estimated_tokens` tokens from the enforced buckets. The middleware
    /// passes `max_seq_len`; handlers refund the unused part with
    /// [`Self::refund_tokens`].
    pub fn admit(&self, key: &str, estimated_tokens: u64) -> RateDecision {
        let now = Instant::now();
        self.scrub_if_due(now);

        let rpm = self.cfg.rpm;
        let tpm = self.cfg.tpm;

        if !self.cfg.is_enabled() {
            return RateDecision {
                allowed: true,
                requests: BucketSnapshot {
                    limit: 1_000_000,
                    remaining: 999_999,
                    reset_secs: 0,
                },
                tokens: BucketSnapshot {
                    limit: 1_000_000_000,
                    remaining: 999_999_999,
                    reset_secs: 0,
                },
                retry_after_secs: 0,
                denied_by: None,
            };
        }

        let req_rate = rpm as f64 / 60.0;
        let tok_rate = tpm as f64 / 60.0;
        let req_burst = self.cfg.burst_rpm as f64;
        let tok_burst = self.cfg.burst_tpm as f64;

        let mut map = self.inner.lock();
        // 2026-09-26: At `MAX_KEYS`, a new key first evicts every idle key.
        if map.len() >= MAX_KEYS && !map.contains_key(key) {
            map.retain(|_, state| {
                state.requests.last_refill.elapsed() < IDLE_EVICT
                    || state.tokens.last_refill.elapsed() < IDLE_EVICT
            });
        }
        let state = map.entry(key.to_string()).or_insert_with(|| KeyState {
            requests: Bucket::new(self.cfg.burst_rpm),
            tokens: Bucket::new(self.cfg.burst_tpm),
        });

        let req_allowed = if rpm > 0 {
            state.requests.try_consume(1.0, req_rate, req_burst, now)
        } else {
            true
        };
        if !req_allowed {
            let (req_avail, req_reset) = state.requests.snapshot(req_rate, req_burst, now);
            let (tok_avail, tok_reset) = state.tokens.snapshot(tok_rate, tok_burst, now);
            return RateDecision {
                allowed: false,
                requests: BucketSnapshot {
                    limit: self.cfg.advertised_rpm(),
                    remaining: self.cfg.advertised_remaining_rpm(req_avail.max(0.0) as u64),
                    reset_secs: req_reset,
                },
                tokens: BucketSnapshot {
                    limit: self.cfg.advertised_tpm(),
                    remaining: self.cfg.advertised_remaining_tpm(tok_avail.max(0.0) as u64),
                    reset_secs: tok_reset,
                },
                retry_after_secs: req_reset.max(1),
                denied_by: Some(DenialReason::Requests),
            };
        }

        let tok_allowed = if tpm > 0 {
            state
                .tokens
                .try_consume(estimated_tokens as f64, tok_rate, tok_burst, now)
        } else {
            true
        };
        if !tok_allowed {
            // 2026-09-26: Return the request just debited: a denial costs
            // nothing.
            if rpm > 0 {
                state.requests.refund(1.0, req_burst);
            }
            let (req_avail, req_reset) = state.requests.snapshot(req_rate, req_burst, now);
            let (tok_avail, tok_reset) = state.tokens.snapshot(tok_rate, tok_burst, now);
            return RateDecision {
                allowed: false,
                requests: BucketSnapshot {
                    limit: self.cfg.advertised_rpm(),
                    remaining: self.cfg.advertised_remaining_rpm(req_avail.max(0.0) as u64),
                    reset_secs: req_reset,
                },
                tokens: BucketSnapshot {
                    limit: self.cfg.advertised_tpm(),
                    remaining: self.cfg.advertised_remaining_tpm(tok_avail.max(0.0) as u64),
                    reset_secs: tok_reset,
                },
                retry_after_secs: tok_reset.max(1),
                denied_by: Some(DenialReason::Tokens),
            };
        }

        let (req_avail, req_reset) = state.requests.snapshot(req_rate, req_burst, now);
        let (tok_avail, tok_reset) = state.tokens.snapshot(tok_rate, tok_burst, now);
        RateDecision {
            allowed: true,
            requests: BucketSnapshot {
                limit: self.cfg.advertised_rpm(),
                remaining: self.cfg.advertised_remaining_rpm(req_avail.max(0.0) as u64),
                reset_secs: req_reset,
            },
            tokens: BucketSnapshot {
                limit: self.cfg.advertised_tpm(),
                remaining: self.cfg.advertised_remaining_tpm(tok_avail.max(0.0) as u64),
                reset_secs: tok_reset,
            },
            retry_after_secs: 0,
            denied_by: None,
        }
    }

    /// 2026-09-26: Return `amount` tokens to `key`'s token bucket, up to its
    /// burst. No-op when the token bucket is disabled or the key was evicted.
    pub fn refund_tokens(&self, key: &str, amount: u64) {
        if amount == 0 || !self.cfg.is_enabled() || self.cfg.tpm == 0 {
            return;
        }
        let mut map = self.inner.lock();
        if let Some(state) = map.get_mut(key) {
            state
                .tokens
                .refund(amount as f64, self.cfg.burst_tpm as f64);
        }
    }

    fn scrub_if_due(&self, now: Instant) {
        let mut last = self.last_scrub.lock();
        if now.saturating_duration_since(*last) < SCRUB_INTERVAL {
            return;
        }
        *last = now;
        drop(last);
        let mut map = self.inner.lock();
        map.retain(|_, state| {
            state.requests.last_refill.elapsed() < IDLE_EVICT
                || state.tokens.last_refill.elapsed() < IDLE_EVICT
        });
    }
}

#[path = "rate_limiter/identity.rs"]
mod identity;
pub use identity::extract_identity;

#[cfg(test)]
#[path = "rate_limiter/tests.rs"]
mod tests;

#[cfg(test)]
#[path = "rate_limiter/advertised_tests.rs"]
mod advertised_limit_tests;
