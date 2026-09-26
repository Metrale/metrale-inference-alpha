# ADR-0017: License — MIT OR Apache-2.0

**Status:** Accepted
**Date:** 2026-09-24
**Supersedes:** [ADR-0001](0001-license-agpl-with-cla.md)

## Context

[ADR-0001](0001-license-agpl-with-cla.md) chose AGPL-3.0-only with a CLA, to
close the SaaS loophole while keeping a path to commercial terms. Its own
"Worse" section named the cost: AGPL-3.0 cannot be embedded in a permissively
licensed library, and enforcing the network-use clause depends on noticing
violations.

Metrale Engine is built on a Rust ecosystem that is overwhelmingly
MIT OR Apache-2.0 (`cudarc`, and most entries in `deny.toml`'s allow-list). The
owner decided the engine should carry the same licence.

## Decision

- Metrale Engine is licensed **MIT OR Apache-2.0**, at the recipient's option:
  `LICENSE-MIT` and `LICENSE-APACHE` in the repository root, copyright holder
  Metrale Corp.
- Every project-owned source file carries
  `SPDX-License-Identifier: MIT OR Apache-2.0`. `.licenserc.yaml` states the
  rule and `scripts/check_spdx.py` enforces it in the `license-headers` job.
- Third-party code keeps its own licence and header, and is listed in
  `THIRD_PARTY_NOTICES.md` with its full text under `LICENSES/`.
- The workspace crates declare `license = "MIT OR Apache-2.0"`, and
  `deny.toml` no longer needs an entry for the project's own licence.
- The CLA stays as it is.

## Consequences

**Better:**
- Metrale Engine can be embedded in permissively licensed and proprietary
  projects without a copyleft analysis.
- The licence matches the Rust convention, which most legal reviews already
  clear.

**Worse:**
- The SaaS-loophole protection ADR-0001 was written for is gone: a hosted
  fork owes nothing back. This is the trade-off accepted here.
- A separately licensed commercial edition no longer has a copyleft boundary
  to sell across.

**New problems we created:**
- Nothing in CI fails when a stray "AGPL" string is added to prose. The SPDX
  check covers headers only.
