# inkling-s: an OpenAI-compatible server for Inkling 2 Mini on Strata's Metal engine

`strata-inkling --serve` reads token ids on stdin and writes token ids on stdout; it has no HTTP API. Strata's own
API layer (Strata Control) installs with an auto-updater; this shim is a small alternative for a single local user,
listening on `127.0.0.1:8097`. It renders the chat template, tokenizes, runs the engine, parses the reply into text,
reasoning and tool calls, and keeps the engine's KV cache useful across turns (see "Prefix reuse" below, the part
worth reading if you write a shim for another engine).

The model loads on the first request and unloads after 5 minutes idle. Requests are served one at a time, in
arrival order. It makes no outbound network connections.

Status: the CPU test suite (a fake engine that speaks the engine's protocol) passes. A live test on an M5 Pro with
64 GB, with the edited pack on the stock engine at 64K context, measured through the shim:
- engine allocation (TotalAlloc) 17.46 GiB;
- a 28K-token prompt read at about 1,000 tok/s (28 s); decode at 39 tok/s;
- a Cline-shaped second turn in 3.3 s with prefix reuse, against 34 s without;
- cancel latency 21 ms mid-reply and 227 ms mid-prompt.

`tests/live_check.py` runs those live checks against a running shim.

## Requirements

- macOS on Apple silicon, a `strata-inkling` binary (Strata 0.1.65, or a build from `engine/`), and a `.strata` pack.
- The model folder from the pack's Hugging Face repository: `tokenizer.json` and `chat_template.jinja` (and, for the
  template tests and live checks, `tools/prompts/`). Point `INKLING_S_MODEL_DIR` at it. These files are not in this
  repository.
- [uv](https://docs.astral.sh/uv/). Python 3.12 through `uv run --offline --no-project --with jinja2==3.1.6
  --with tokenizers==0.23.3`; the tests add `transformers==5.18.0` and `pytest==9.2.0`. Everything else is the
  standard library.

## Quick start

```sh
export INKLING_S_MODEL_DIR=/path/to/Inkling-2-Mini-strata      # tokenizer.json, chat_template.jinja, the pack
export INKLING_S_ENGINE=/path/to/strata-inkling                 # default: strata-inkling on PATH
shim/inkling-s setup      # once: fetch the pinned Python packages into uv's cache (the only step that uses the network)
shim/inkling-s start      # HTTP up at once; the model loads on the first request
shim/inkling-s ask "Write a haiku about Metal"
shim/inkling-s status     # engine state, binary, pack, args, memory guard, counters
shim/inkling-s stop       # stops the shim and the engine; no process is left behind
```

Other commands: `unload` (drop the model now, keep the shim running), `logs` (follow `run/shim.log`; the engine's
own stderr is in `run/engine.log`), `test` (the CPU test suite).

To serve the edited model: either point `INKLING_S_PACK` at the patched pack, or keep the original pack and run the
patched engine with `INKLING_S_ENGINE=/path/to/patched/strata-inkling INKLING_S_ENGINE_ARGS="--ablate /path/DIRS.bin"`.

## Settings (environment variables)

| Variable | Default | Meaning |
|---|---|---|
| `INKLING_S_MODEL_DIR` | (required) | folder with `tokenizer.json` and `chat_template.jinja` |
| `INKLING_S_PACK` | `$INKLING_S_MODEL_DIR/Inkling-2-Mini-SQ-15.strata` | the `.strata` pack |
| `INKLING_S_ENGINE` | `strata-inkling` (on PATH) | engine binary |
| `INKLING_S_ENGINE_ARGS` | (none) | extra engine arguments, shell-style, appended last, e.g. `--ablate /path/DIRS.bin` |
| `INKLING_S_PORT` | 8097 | port (the host is always 127.0.0.1) |
| `INKLING_S_CTX` | 65536 | context length (`--max-context`); see memory below |
| `INKLING_S_IDLE` | 300 | seconds idle before the engine is unloaded; `<= 0` never |
| `INKLING_S_DRAFT` | off | speculative decoding (DSpark); `auto` turns it on. It costs memory and is not bit-exact (docs/upstream-bugs.md) |
| `INKLING_S_EFFORT` | (template default 0.9) | default `reasoning_effort`: `none`, `minimal`, `low`, `medium`, `high`, `xhigh`, `max` or 0–0.99 |
| `INKLING_S_REUSE` | splice | `off` disables reasoning substitution (see prefix reuse) |
| `INKLING_S_LITERALS` | text | `special`: special-token text inside messages becomes control tokens (transformers / llama.cpp behaviour) |
| `INKLING_S_ANTI_LOOP` | 1 | `0` passes `--no-anti-loop` to the engine |
| `INKLING_S_UNDECLARED_CALLS` | xml | a native tool call in reply to a request without `tools`: `xml` = Cline's XML tool text, `json` = the raw block |
| `INKLING_S_DEBUG` | 0 | `1` logs every raw reply, special tokens included |
| `INKLING_S_ALIAS` | inkling-2-mini-strata | model id reported by `/v1/models` |
| `INKLING_S_NEED_GB` | from ctx | memory the engine needs, for the guard (64K: 18.7 GB) |
| `INKLING_S_RESERVE_GB` | 8 | free memory required on top of that before loading |
| `INKLING_S_FLOOR_PCT` | 12 | the watchdog unloads the engine when free or reclaimable memory drops under this |
| `INKLING_S_WARN_GRACE` | 0 | seconds a memory-pressure warning may last before the watchdog unloads (0 = the next poll) |
| `INKLING_S_GUARD_LOCK` | (off) | optional: an external lock directory; see memory below |
| `INKLING_S_HEALTH_URL` | (off) | optional: a service that must stay healthy; see memory below |
| `INKLING_S_PAUSE_FILE` | (off) | optional: marks that service as paused |
| `INKLING_S_WATCH_FILE` | (off) | optional: a log to watch for an out-of-memory marker |
| `INKLING_S_WATCH_PATTERN` | `insufficient memory` | the marker (Metal's out-of-memory message) |
| `INKLING_S_NO_GUARD` | 0 | `1` disables the guard and the watchdog (tests only) |

`/v1/models` (under `meta`) and `/health` report the binary, pack, extra args and guard settings in use, and
`inkling-s status` prints them. The engine always runs with one slot (`--parallel 1`); extra args that set
`--parallel`, `--slots` or `--batch` are refused, because the stock engine returns a wrong first token for a request
admitted next to another one on a multi-slot engine (docs/upstream-bugs.md), and reuse depends on slot 0.

## Endpoints

- `POST /v1/chat/completions`, streaming (SSE) or not. It honours `messages`, `tools`, `tool_choice: "none"`,
  `max_tokens` / `max_completion_tokens`, `temperature`, `top_p`, `top_k`, `min_p`, `presence_penalty`,
  `frequency_penalty`, `repeat_penalty`, `seed`, `stop`, `stream_options.include_usage`, `reasoning_effort`,
  `reasoning.effort`, and `chat_template_kwargs` (`reasoning_effort`, `enable_thinking: false`).
- `GET /v1/models`, `GET /health`, and `POST /unload`.
- Replies carry `usage`, including `prompt_tokens_details.cached_tokens` (the reused prefix), and llama.cpp-style
  `timings`, including `cache_n` and `spliced_turns`.

## How it behaves

- **Prompt.** `chat_template.jinja` is rendered exactly as transformers' `apply_chat_template` renders it: the same
  Jinja environment, `tojson` and `{% generation %}` handling. The tests check this character for character and
  token for token over 12 conversations (in both literal modes) and 11 reasoning levels. The prompt ends after the last message, and the
  model writes `<|message_model|>` itself.
- **Reasoning** is on by default (template default 0.9, as with llama.cpp). It is returned as
  `reasoning_content`, the answer as `content`, the same split llama.cpp makes.
- **Native tool calls**
  (`<|message_model|>NAME<|content_invoke_tool_json|>{"name":..,"args":{..}}<|end_message|>`) become OpenAI
  `tool_calls`, with the arguments streamed in pieces and `finish_reason: tool_calls`. Cline's own XML tool format
  is plain text and passes through untouched.
- **A native call without declared tools.** Measured live: with a Cline-style system prompt and no `tools` in the
  request, this model often answers with its native call
  (`<|message_model|>search_files<|content_invoke_tool_json|>{...}`) instead of Cline's XML. The shim then
  writes the call as Cline's XML (`<search_files>\n<path>...</path>\n<regex>...</regex>\n</search_files>`), so
  Cline can execute it; prefix reuse still works on the next turn. `INKLING_S_UNDECLARED_CALLS=json` passes
  the raw block instead. A damaged call head, seen once live as `{"name"""search_files","args":{...}}`, is still
  parsed.
- **Parsing works on token ids,** so a marker can never arrive half-written. Text is decoded incrementally from
  bytes, so a character split across tokens is held back until it is whole. `stop` strings hold back any tail that
  could start one, and cancel the engine when one matches.
- **Sampling** parameters a request leaves out get llama.cpp server's defaults (temperature 0.8, top_k 40, top_p
  0.95, min_p 0.05), so a client sees the same sampling as on llama-server.
- **Special-token text inside a message,** for example a file that contains `<|end_message|>`, is encoded as plain
  text. This matches Strata Control; `INKLING_S_LITERALS=special` switches it off.
- **Context overflow** gets a 400 with `code: context_length_exceeded` (OpenAI wording), and nothing is ever
  truncated. Without `max_tokens`, a reply may use all the context that is left.
- **Not supported:**
  - images: replaced by a note (the shim does not load Strata's separate vision pack);
  - `n > 1`: answered with 400;
  - `logprobs` and `response_format` grammars: ignored;
  - `tool_choice: "required"` or a named function: not enforced.
- **Anti-loop.** The engine's anti-loop check stays on. It ends a reply when the last tokens repeat a short
  pattern (period 1–4 × 16, or 5–32 × 6, after 64 tokens).
  - Symptom: output cut off in the middle of very repetitive code (long tables, repeated literals, padding).
  - The shim logs `note: the engine's anti-loop check ended this reply`, and `finish_reason` is `stop`.
  - Use `INKLING_S_ANTI_LOOP=0` if that bites.

## Prefix reuse: why "splice"

The engine keeps the KV cache of its last request, all of it or none of it. If the new prompt starts with every
token it has already read (the previous prompt and all generated tokens but the last), it reads only the rest.
Otherwise it starts over, which at 28K tokens took about 35 s in our measurements (784–817 tok/s).

Cline resends the whole conversation each turn and drops the reasoning from earlier answers. A plain re-rendering
therefore never matches what the engine read, and every turn would cost the full 35 s.

So the shim remembers each reply it generated, keyed by the exact prompt it answered. When a client sends that
reply back, it substitutes the exact tokens the model produced, reasoning included. "Back" means unchanged, with
trimmed whitespace, with reformatted tool-call JSON, with text appended (Cline's "[Response interrupted...]"), or
cut short (a cancelled reply). Turn N+1 then reads only its new tokens.

The cost of splicing:
- The model sees its earlier reasoning in the history, which llama.cpp does not show it.
- The context fills somewhat faster. If a spliced prompt would not fit, the shim falls back to the plain rendering
  (one full read).
- `INKLING_S_REUSE=off` disables it.

What splicing cannot fix: a new Cline task, any edit to earlier history (Cline's context condensing or file-read
deduplication), or the first request after an idle unload. Each of these reads the whole prompt again, because the
engine cannot branch from a shared system prompt.

Every request logs `prompt N tok: reused R, re-read N-R in Xs`. A miss is logged as `REUSE MISSED` or
`cache miss: ... at token i: cached '...' vs new '...'`, so it is visible.

## Memory: the guard and the watchdog

The engine allocates its whole KV cache at load: fp16 only (the Metal build ignores `--kv`), about 1 GiB per 32K
of context. DSpark drafting adds 0.375 GiB per 32K.

| Context | Engine TotalAlloc (drafting off) | With drafting on |
|---|---|---|
| 32K | 16.46 GiB (measured) | 16.90 GiB (measured) |
| 64K (default) | ≈17.5 GiB (17.46 GiB measured in the live test) | ≈18.3 GiB |
| 128K | ≈19.5 GiB | ≈21.0 GiB |

The 32K figures and the 64K live figure are measured; the rest are computed from the source.

**Why the guard exists.** On a unified-memory Mac a GPU allocation that does not fit is not refused: the system
compresses or swaps other processes' memory instead. On a 64 GB machine, the 17.5 GiB engine at 64K context could
not share memory with another GPU workload of about 27 GB: the other workload's memory went to the compressor while
the engine stayed up. So the shim checks before loading, and a watchdog unloads the engine when memory runs short.

Before loading, the shim answers HTTP 503 with the reason if one of these fails:
- memory pressure is normal (`kern.memorystatus_vm_pressure_level` = 1);
- free or reclaimable memory (free + speculative + purgeable + file-backed pages from `vm_stat`) is at least the
  engine's need (64K: 18.7 GB) plus `INKLING_S_RESERVE_GB`.

While the engine is loading or loaded, a watchdog polls every 2 s and kills the engine if:
- memory pressure becomes critical (at once), or stays at warning for `INKLING_S_WARN_GRACE` seconds (default 0:
  the next poll; a 20 s grace was tried and the warning did not clear while the other workload was compressed);
- free or reclaimable memory drops under `INKLING_S_FLOOR_PCT`.

Optional checks, all off by default, for a machine that runs something else that matters:
- `INKLING_S_GUARD_LOCK=DIR`: another tool's lock directory (with `owner.json` holding `{"pid": N, "name": ...}`).
  The shim does not load while a live process holds it. A lock that appears while the engine is loaded does not
  unload it. (The shim never takes the lock itself, because it serves a long-lived engine.)
- `INKLING_S_HEALTH_URL=URL`: a service sharing the machine that must stay healthy (HTTP 200). No load while it is
  not; and while it is up, no load while another large model is in memory (llama-server, an Ollama model, another
  strata-inkling).
- `INKLING_S_PAUSE_FILE=PATH`: while this file exists, that service counts as paused: its health is not required,
  and other models may share the machine when the memory check passes.
- `INKLING_S_WATCH_FILE=PATH` (with `INKLING_S_WATCH_PATTERN`): a log to watch, for example the other service's.
  A new occurrence of the pattern (default `insufficient memory`, Metal's out-of-memory message) unloads the
  engine at once.

After a watchdog unload, a request in flight fails with 503 `memory_watchdog`; the shim refuses to load again for
120 s, and after that only when the checks pass. `inkling-s status` shows the last watchdog unload.

## Using it from Cline (VS Code)

The shim does not change Cline's settings; set them by hand.

1. Start the shim: `shim/inkling-s start`.
2. In Cline's settings (gear icon in the Cline panel), set **API Provider** to **OpenAI Compatible**.
3. Set **Base URL** to `http://127.0.0.1:8097/v1`.
4. Set **API Key** to any non-empty text, for example `local`. The shim does not check it; it only listens on
   127.0.0.1.
5. Set **Model ID** to `inkling-2-mini-strata`. Any value works, but this one matches `/v1/models`.
6. Under the model configuration:
   - **Context Window Size**: 65536, the same as `INKLING_S_CTX`. Cline uses it to decide when to condense the
     conversation; a larger value than the shim's makes requests fail with `context_length_exceeded`.
   - **Max Output Tokens**: 8192 (or Cline's default).
   - **Supports Images**: off.
   - Temperature: leave Cline's value. The shim honours it, and if Cline sends none, llama.cpp's defaults apply.
7. If Cline offers native tool calling for this provider, both modes work. Its default XML tools arrive as plain
   text.
8. Expect a slow first request in each new task: engine load about 5 s, plus about 35 s to read a 28K prompt.
   Later turns of the same task should read only their new tokens (watch `inkling-s logs`).

## Files

- `inkling-s`: the control script.
- `inkling_shim/vocab.py`: special tokens and the tokenizer wrapper (byte-level decoding, text-only encoding).
- `inkling_shim/prompt.py`: request normalization, template rendering, tokenization, reuse memory and splicing.
- `inkling_shim/parse.py`: the token-level reply parser (reasoning, text, tool calls) and the `stop` matcher.
- `inkling_shim/engine.py`: the engine process, one `GENERATE` run, the FIFO queue, idle unload, the watchdog
  loop, shutdown.
- `inkling_shim/memguard.py`: the preflight checks and the watchdog checks.
- `inkling_shim/service.py`: one completion from request to events, and logging.
- `inkling_shim/server.py`: HTTP, SSE, settings.
- `tests/`, which `inkling-s test` runs (set `INKLING_S_MODEL_DIR`, or the tests that need the tokenizer and
  template are skipped):
  - template parity against transformers;
  - the parser;
  - splicing;
  - HTTP end to end with `tests/fake_engine.py`, a CPU stand-in that speaks the engine's stdin protocol including
    all-or-nothing prefix reuse;
  - the memory guard and watchdog with faked readings.
- `tests/live_check.py`: the live GPU checks against a running shim (standard library, Python 3.9+).
- The engine protocol is described in `../docs/serve-protocol.md`; the source is Strata's
  `engine/src/inkling2/ink_serve.h`.

## Troubleshooting

- **503 `model_blocked`.** The message names the check that failed: pressure, free memory, or one of the optional
  checks (lock, health, another model).
- **503 `memory_watchdog`.** The watchdog unloaded the engine. See `run/shim.log` (`MEMORY WATCHDOG`).
- **500 `engine_error`.** The engine exited or failed; see `run/engine.log`. The next request starts it again.
- **`inkling-s start` fails with an offline or download error.** Run `inkling-s setup` once.
- **Every turn is slow.** Look for `cache miss` in `run/shim.log`; the line shows where the prompt stopped
  matching.
