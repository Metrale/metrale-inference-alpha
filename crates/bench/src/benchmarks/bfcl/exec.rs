// SPDX-License-Identifier: MIT OR Apache-2.0

//! 2026-09-26: The two phases of a BFCL run that talk to the outside world:
//! issuing one sample's request, and handing the collected responses to
//! `score.py`.
//!
//! Owner: bench, BFCL benchmark.
//! Invariants: none beyond the types.

use super::*;

impl Bfcl {
    pub(super) async fn generate_one(&mut self) -> Result<()> {
        let handle = self.handle()?.clone();
        let sample = self.samples[self.cursor].clone();
        let target = handle.target();
        let body = json!({
            "model": target.model,
            "stream": true,
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
            "messages": sample.messages,
            "tools": sample.tools,
            "tool_choice": sample.tool_choice,
        });
        let outcome = http::chat_stream(target, &body, self.request_timeout).await;
        let (tool_calls, has_tool_calls) = match &outcome {
            Ok(o) => (
                o.tool_calls
                    .iter()
                    .map(|c| json!({"name": c.name, "arguments": c.arguments}))
                    .collect::<Vec<_>>(),
                !o.tool_calls.is_empty(),
            ),
            Err(e) => {
                // 2026-09-26: A transport failure is scored as "no call",
                // logged, and counted in `transport_errors`, which `metrics()`
                // publishes. "No call" is the correct answer on the irrelevance
                // subsets, so a shard with failed requests could score higher;
                // `gate::check_group` refuses a shard that has any.
                self.transport_errors += 1;
                handle.warn(one_line(format!("sample {}: {e:#}", sample.sample_id)));
                (Vec::new(), false)
            }
        };
        if has_tool_calls {
            self.tool_call_samples += 1;
        }
        // 2026-09-26: A sample in `sensitive::KNOWN_PARTITION_SENSITIVE` is
        // counted and warned about where it runs.
        if super::sensitive::is_known(&sample.sample_id) {
            self.known_sensitive_seen += 1;
            handle.warn(one_line(format!(
                "sample {} is KNOWN partition-sensitive (#936): its answer can differ \
                 between the whole draw and a shard because of cross-request SSM \
                 snapshot reuse",
                sample.sample_id
            )));
        }
        self.responses.push(json!({
            "sample_id": sample.sample_id,
            "subset": sample.subset,
            "has_tool_calls": has_tool_calls,
            "tool_calls": tool_calls,
        }));
        self.cursor += 1;
        Ok(())
    }

    pub(super) async fn score(&mut self) -> Result<Scores> {
        let artifacts = self
            .artifacts
            .clone()
            .context("artifacts were not provisioned")?;
        let path = artifacts
            .dir
            .join(responses_file(self.descriptor().id, self.shard));
        let mut text = String::new();
        for r in &self.responses {
            text.push_str(&serde_json::to_string(r)?);
            text.push('\n');
        }
        std::fs::write(&path, text).with_context(|| format!("writing {}", path.display()))?;
        self.responses_path = Some(path.clone());

        let out = crate::python::run(
            &artifacts.python,
            &[
                artifacts.scorer.to_str().context("scorer path")?,
                "--dataset",
                artifacts.dataset.to_str().context("dataset path")?,
                "--responses",
                path.to_str().context("responses path")?,
            ],
            Some(&artifacts.dir),
        )
        .await
        .with_context(|| {
            format!(
                "scoring failed — {} is kept, so this can be rescored",
                path.display()
            )
        })?;
        serde_json::from_str(out.stdout.trim())
            .with_context(|| format!("scorer printed unexpected output: {}", out.stdout))
    }
}

/// 2026-09-26: Where one run's per-sample output is written in the shared
/// artifact directory, keyed by benchmark id and shard. A shard reports its
/// group's id (`Bfcl::descriptor` returns the variant's descriptor), so without
/// the shard in the name the whole draw and its shards would overwrite each
/// other's output, which a rescore or a per-sample diff between them needs.
pub(super) fn responses_file(benchmark_id: &str, shard: Option<super::dataset::Shard>) -> String {
    match shard {
        None => format!("responses-{benchmark_id}.jsonl"),
        Some(s) => format!("responses-{benchmark_id}-{}of{}.jsonl", s.index, s.count),
    }
}

#[cfg(test)]
#[path = "exec_tests.rs"]
mod exec_tests;
