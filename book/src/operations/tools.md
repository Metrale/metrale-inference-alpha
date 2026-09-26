# Tool Calling & Streaming

Metrale Engine supports OpenAI-compatible **function calling** across several wire formats and full **SSE streaming** (OpenAI + Anthropic conventions). This chapter is the operator reference for running agents against Metrale Engine — how to enable tools, stream responses, handle multi-turn tool results, and recognise the failure modes that used to bite real agents.

## Enable tools

Include `tools: [...]` in your request. The wire format comes from `--tool-call-parser <FORMAT>`, else the model's `MODEL.toml` `[behavior].tool_call_parser`, else the model type's entry in `crates/server/tool_defaults.toml`; a model with none of the three has tool calling off.

| Parser | Wire format |
|---|---|
| `hermes` | `<tool_call>{...}</tool_call>` JSON |
| `qwen3_coder` | XML-in-tool-call with `<function=...><parameter=...>` |
| `qwen3_xml` | the Qwen3 XML variant |
| `gemma4` | Gemma-4's tool-call syntax |
| `mistral` | JSON after the `[TOOL_CALLS]` prefix |
| `minimax_xml` | `<minimax:tool_call>` XML |
| `bare_json` | an unwrapped JSON object |
| `poolside_v1` | Poolside's format |

Every format is parsed on the server and emitted to the client as standard OpenAI `tool_calls` blocks — you do not need to handle the wire format in your client.

## Minimal tool-call request

```bash
curl -s http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "metrale",
    "messages": [{"role": "user", "content": "What is the weather in Paris?"}],
    "tools": [{
      "type": "function",
      "function": {
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
          "type": "object",
          "properties": {
            "location": {"type": "string", "description": "City name"}
          },
          "required": ["location"]
        }
      }
    }],
    "max_tokens": 512
  }'
```

Response (abridged):

```json
{
  "choices": [{
    "message": {
      "role": "assistant",
      "content": null,
      "tool_calls": [{
        "id": "call_00000000",
        "type": "function",
        "function": {
          "name": "get_weather",
          "arguments": "{\"location\":\"Paris\"}"
        }
      }]
    },
    "finish_reason": "tool_calls"
  }]
}
```

## Multi-turn: sending tool results back

Standard OpenAI pattern — append the assistant's `tool_calls` message and then a `role: "tool"` message with the tool's output:

```json
{
  "messages": [
    {"role": "user", "content": "What is the weather in Paris?"},
    {"role": "assistant", "content": null, "tool_calls": [{
      "id": "call_00000000", "type": "function",
      "function": {"name": "get_weather", "arguments": "{\"location\":\"Paris\"}"}
    }]},
    {"role": "tool", "tool_call_id": "call_00000000", "name": "get_weather",
     "content": "{\"temperature\": 15, \"condition\": \"cloudy\"}"}
  ],
  "tools": [...]
}
```

Metrale Engine expands the multi-turn conversation through the chat template and runs a fresh forward.

## `tool_choice`

| Value | Meaning |
|---|---|
| `"auto"` (default) | Model decides |
| `"none"` | Disable tool calling for this request |
| `"required"` | Force the model to call any tool |
| `{"type": "function", "function": {"name": "X"}}` | Force a specific tool |

`"required"` is implemented via the XGrammar grammar (see [XGrammar](../deep-dives/xgrammar.md)) — the grammar masks the "no-tool-call" path, so the sampler can only produce a valid tool-call opening.

## Streaming

Streaming is enabled with `"stream": true`:

```bash
curl -sN http://localhost:8888/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"metrale","messages":[...],"stream":true}'
```

Output is standard OpenAI SSE:

```
data: {"choices":[{"delta":{"role":"assistant","content":"Once "}}]}

data: {"choices":[{"delta":{"content":"upon "}}]}

...

data: {"choices":[{"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

Tool calls stream as `delta.tool_calls` chunks:

```
data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_00000000","type":"function","function":{"name":"get_weather","arguments":""}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\""}}]}}]}

data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"location"}}]}}]}

...

data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}

data: [DONE]
```

## Anthropic Messages

For clients built against Anthropic's API:

```bash
curl -s http://localhost:8888/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "metrale",
    "max_tokens": 500,
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

Streaming uses Anthropic's event conventions — `message_start`, `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`. Metrale Engine populates `stop_sequence` on `message_delta` when a stop token was hit.

Tool use on `/v1/messages` uses Anthropic's nested content-block format:

```json
{
  "content": [
    {"type": "text", "text": "Let me check that."},
    {"type": "tool_use", "id": "toolu_...", "name": "get_weather",
     "input": {"location": "Paris"}}
  ],
  "stop_reason": "tool_use"
}
```

## Reasoning / `<think>` blocks

Models that emit `<think>` (Qwen3.5, Nemotron-H, MiniMax) stream reasoning content as a separate channel:

```
data: {"choices":[{"delta":{"reasoning_content":"Let me think step by step. First, ..."}}]}

data: {"choices":[{"delta":{"reasoning_content":" the user is asking about..."}}]}

data: {"choices":[{"delta":{"content":"The answer is 42."}}]}

data: {"choices":[{"delta":{},"finish_reason":"stop"}]}
```

Clients that don't parse `reasoning_content` chunks will ignore them cleanly.

`--max-thinking-budget` caps the total reasoning tokens; `--disable-thinking` strips them entirely. For agent workloads that want reasoning but don't want unbounded think time, a budget of 2048–4096 is typical.

## Vision requests

Qwen3-VL and Qwen3.6 accept images in OpenAI content-parts format:

```json
{
  "model": "metrale",
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "Describe this image."},
      {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,<DATA>"}}
    ]
  }],
  "max_tokens": 512
}
```

`image_url` accepts `data:` URLs (base64-encoded). `http(s):` URLs are refused with a 400 unless the server runs with `--vision-allow-remote-images`, because fetching them lets any client make the server issue outbound requests. Multiple images per message are supported.

## Known pitfalls and how Metrale Engine addresses them

- **Tool calls inside markdown fences.** The parser is markdown-fence aware, and XGrammar enforcement keeps the call well-formed.
- **Broken tool-call XML (Qwen3-coder format).** The parser tolerates literal `</tool_call>` inside JSON string values, missing `</parameter>` tags, and empty `{}` tool-call bodies.
- **Streaming Responses store tool_calls.** The server persists tool calls to the Responses-API session store mid-stream, so multi-turn conversations across the Responses API see them on the next turn.
- **Balanced markdown URL parens.** The citation extractor uses a balanced parser, so URLs containing parentheses survive.
- **Template-forced thinking.** A `<think>\n\n</think>\n\n` template prologue does not trigger the reasoning parser; the opening `<think>` must be unclosed to count.
- **Spontaneous `<think>` outside the template position.** A `<think>` the model emits mid-stream is detected on every affected code path.

All of these have regression tests under `crates/server/src/tool_parser/` and `crates/server/src/reasoning_parser/`.

## Running against real agents

Minimum Metrale Engine config for running Claude Code, OpenCode, Cline, or nanobot:

- `--max-seq-len 16384` or higher (agents regularly exceed 4k).
- `--enable-prefix-caching` (massive TTFT win on tool schemas).
- `--scheduler slai` (keeps streaming smooth).
- `--speculative --mtp-quantization nvfp4` if the model supports it (agents are 50%+ tool calls; MTP + constrained decoding = +37% throughput).
- `--auto-compact 0.85` so long agent sessions don't crash into the seq-len wall.

## Files to read

- `crates/server/src/tool_parser.rs` and `tool_parser/` — the parser impls.
- `crates/server/src/reasoning_parser/` — `<think>` detection + extraction.
- `crates/server/src/openai/`, `anthropic/` — request/response structs.
- `crates/server/src/api/` — the HTTP handlers.
- `docs/ARCHITECTURE.md` — system overview covering the tool-call path.
- [XGrammar deep dive](../deep-dives/xgrammar.md) for the constrained-decoding side.
