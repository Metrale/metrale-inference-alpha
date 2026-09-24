# How a Change Lands

Every performance claim this project makes is a measurement, and measurements are only worth the
process that produced them. A change reaches `main` through three stages, and
nothing about that is manual goodwill — it is enforced.

<p align="center">
  <img src="https://raw.githubusercontent.com/Metrale/metrale-inference-alpha/main/docs/diagrams/pr-certification.png" alt="PR certification flow: Stage 1 Verification, Stage 2 Certification requiring an engineer's seal and benchmark records, Stage 3 Ready to merge, then the merge queue. A table shows what survives a new commit, main advancing, and conflicts." width="960">
</p>

**Stage 1 — Verification** runs on every push: formatting, clippy, typos, licence
headers, kernel structure, tests, merge ancestry. Certification and the nine
release-matrix build legs are held back, so an early draft does not burn an hour
of runners.

**Stage 2 — Certification** opens on `/stamp` from anyone with write access **or
the PR's own author** — nobody is better placed to say their own branch has
stopped churning, and the author is precisely who the hold was protecting from
burnt runners. It needs two things. An **engineer's seal** — a codeowner comments `/seal`, and the
sealers' owned paths must cover the whole diff. And **benchmark records** — the
campaign runs on a real GPU box and the records are committed; CI only checks
that they cover what changed. Records are Ed25519-signed over their own bytes and
the commit they name, and CI requires every record a PR adds to share one commit
and one signer — so an altered record, one re-pointed at a different commit, or
results spliced from two campaigns are all detectable, and every record is
attributable to a key in `.github/record-signers/`.

That is an integrity check, not proof the numbers were measured. The key is
generated on the operator's own box and signing happens there, after the record
is written, so a determined insider can sign whatever they like; CI has no GPU
and verifies signatures and internal consistency rather than witnessing
execution. What this makes nearly impossible is the *accident* — shipping the
wrong records — and what it makes attributable is everything else.

**Stage 3 — Ready to merge**, then the queue, which re-runs the whole pipeline
against its own merge commit.

The bot takes five comment commands, and they are the whole interface:

| command | who may use it | what it does |
|---|---|---|
| `/help` | anyone | prints this table on the PR, with the current state |
| `/stamp` | write access, or the PR author | releases certification and the nine release-matrix legs. **Survives new commits** |
| `/seal` | a codeowner with write access whose owned paths cover the whole diff | records the engineer's seal. **Voided by the next commit** |
| `/review` | anyone | an advisory LLM read of the diff. Never gates anything |
| `/expedite` | admin only, and it requires a stated reason | skips certification and lets the PR merge once the pipeline's own checks pass. Purely administrative, and it announces itself on the PR |

`/expedite` exists because a release should not be hostage to a GPU box being
busy. It is deliberately loud: it mints its own check run, posts a comment naming
who used it and why, and the reason is required rather than optional.

The asymmetry in the table is the part worth reading twice. **A seal survives
`main` moving** — the sealer vouched for this diff, and the queue re-runs
everything against the composed tree. **Records never do.** Two campaigns
measured apart do not compose: the interaction between them was never measured,
and calling it measured is the exact class of silent regression the gate exists
to prevent. Freeze the branch, run the campaign, queue alone.
