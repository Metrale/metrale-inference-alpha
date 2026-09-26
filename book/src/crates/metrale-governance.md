# metrale-governance

**Path:** `crates/governance/`

The PR journey ledger: an append-only record of how a change reached `main`. `.benchmarks/` answers whether a commit passed; the ledger answers how a pull request got there — which gates were re-opened and by what, which runs superseded which. The canonical form is `governance/pr-<n>.jsonl`, a grow-only set keyed `(head_sha, run_id, attempt)`; a graph is materialised from it on demand and never stored. It is advisory: nothing in the pull-request gate check reads it.
