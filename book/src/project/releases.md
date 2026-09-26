# Release Notes

Metrale Engine records what it ships in three places. This chapter points to them; for the latest release, check the repository — this page is a stable pointer, not a ticker.

## Where to read them

| Source | What it holds |
|---|---|
| [`CHANGELOG.md`](https://github.com/Metrale/metrale-inference-alpha/blob/main/CHANGELOG.md) | Notable changes, in [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) form |
| [`docs/releases/`](https://github.com/Metrale/metrale-inference-alpha/tree/main/docs/releases) | One record per shipped image, `docs/releases/<git-sha>.md`: the SHA it was built from, the tags it received, the serve-matrix verdict and the notable engine changes since the previous shipped SHA |
| GitHub Releases | `https://github.com/Metrale/metrale-inference-alpha/releases` — tagged `vX.Y.Z` by the `release.yml` workflow |
| Docker Hub | `https://hub.docker.com/r/metrale/metrale-inference-gb10/tags` |

## Versions and tags

The workspace version is in the root `Cargo.toml` (`met --version` prints it). A release is cut by the `release.yml` workflow, which takes a bare semver and tags `vX.Y.Z`.

The GB10 image moves three tags — `:latest`, `:dev` and `:nightly` — and can carry a `:<semver>` tag. Every image built from `docker/gb10/Dockerfile` carries its source commit as the `org.opencontainers.image.revision` label:

```bash
docker inspect --format '{{ index .Config.Labels "org.opencontainers.image.revision" }}' metrale/metrale-inference-gb10:latest
```

That label, cross-checked against the newest file in `docs/releases/`, answers "is `:latest` the merged code?".

## Background

The architecture decision records under [`docs/adr/`](https://github.com/Metrale/metrale-inference-alpha/tree/main/docs/adr) explain why the subsystems look the way they do, and [`docs/METRALE_JOURNEY.md`](https://github.com/Metrale/metrale-inference-alpha/blob/main/docs/METRALE_JOURNEY.md) tells the benchmark story on GB10.
