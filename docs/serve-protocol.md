# strata-inkling `--serve` protocol (engine 0.1.65, commit 26350e4)

The line protocol of `strata-inkling --serve`, as used by the shim in `shim/`. Everything here is **read from
source** unless marked "verified". Source: `engine/src/inkling2/ink_serve.h` (the whole protocol, 887 lines),
`ink_main.cpp` (flags), `ink_backend.h` (defaults), `metal/ink_metal_backend.mm` (Metal specifics). Line numbers
refer to `engine/src/inkling2/` in Strata at commit 26350e4.

The shipped Mac binary is the unified `strata-inkling` (ink_main.cpp + Metal plugin), not the
standalone `strata-inkling-metal` of `metal/build_mac.sh` (that one has no `--serve`).
Verified: shipped `--help` matches `ink_main.cpp:352-360` exactly.

## 1. Launch

```
strata-inkling --serve --native PACK.strata [--max-context N] [--parallel 1..64] [--chunk N]
               [--draft auto|dspark|off] [--no-anti-loop] [--vision VISION.strata]
```

| Flag | Default | Notes |
|---|---|---|
| `--native`/`--model` | required | `.strata` (Metal only reads .strata) |
| `--max-context`/`--ctx` | 32768 (`ink_backend.h:18`) | Metal allocates global-layer KV **eagerly** for the full length (`ink_metal_runner.h:399-406`): 8 global visits x N x 8 x 128 x fp16 x (K,V) = 1 GiB per 32K, plus DSpark ctx KV 3 x N x 2 KiB x 2 (0.375 GiB per 32K). `--kv` is accepted but **ignored by Metal** (`alloc_slot` ignores dtype, KV always fp16). |
| `--parallel`/`--slots`/`--batch` | 1 | slots; KV scales linearly with it. Use 1 for a single-user shim (see prefix reuse, and docs/upstream-bugs.md issue 2: with several slots, a request admitted next to another gets a wrong first token). |
| `--chunk` | 512 | prefill tokens per step; Metal batch cap = max(256, chunk, parallel). |
| `--draft` | auto | `off` disables DSpark speculative decode. auto/dspark = on if the pack has DSpark tensors (SQ-15 does). |
| `--no-anti-loop` | anti-loop ON | anti-loop ends generation (reason `stop`) when the last tokens repeat a period 1..4 x16 or 5..32 x6, after >= 64 generated tokens (`ink_serve.h:138-154`). Long code with repetitive lines could trip it; consider `--no-anti-loop` for a coding agent. |

There is **no** flag for temperature defaults, stop strings, chat template, or reasoning; all of that is the client's job.

stderr carries `[ink_metal] ...` load logs; stdout carries only protocol lines (each flushed, max 1023 chars).

## 2. Startup lines (stdout)

```
INFO engine=strata-inkling-0.1.65 arch=inkling2_loop backend=metal model=<basename> layers=24 visits=48 vocab=200058 chunk=512 batch_slots=<P> slot_read=1 kv_dtype=fp16 drafter=dspark|none images=0|1 image_pad=200054 parallel=64
READY <max_ctx> stop hold holdb
```
On failure before READY: `ERR open: ...` or `ERR init: ...` / `ERR alloc_slot ...` and exit 1.
Wait for `READY` before sending anything (stdin is read by a thread started after READY, but lines
written earlier just queue in the pipe; still, READY = model loaded).

## 3. Commands (stdin, one per line, `\r` stripped)

Token lists are comma-separated decimal ids with no spaces: `200002,200004,100`. The token list is
always the **last** whitespace-separated word; `k=v` options sit between max_new and the list.

### `GEN <max_new> [k=v ...] <ids>` (solo, always slot 0)
Output:
```
[RESUME <n_reused>]                     only if KV prefix reuse kicked in (section 5)
PP <prefill_pos> <prompt_len> <prompt_ms_cum> <ppts_this_step>   one per prefill chunk
T <id>                                  one per generated token (the first T follows the last PP)
DONE <gen> <prompt_len> <prompt_ms> <decode_ms> <reason>
```
Errors: `ERR <msg>` with **no** DONE (bad syntax, prompt too long, slot busy).
`ERR fatal GPU error: ...` means the process exits.

### `GENERATE <req_id> <max_new> [k=v ...] <ids>` (extended, picks first free slot)
```
[RESUME <n>]                             no req_id on this line
PP <prefill_pos> <prompt_len> <ms> <ppts> no req_id on this line either (ext requests are never "admitted", ink_serve.h:809)
T <req_id> <id>
DONE <req_id> <gen> <prompt_len> <prompt_ms> <decode_ms> <reason>
```
`ERR busy: all P parallel slots active` if none free. `req_id` is any token without spaces.
Differences from GEN: `hold=` is ignored; `stop` covers both stop-token and anti-loop.
With `--parallel 1`, GEN and GENERATE behave the same except for line prefixes. GENERATE is the
better choice for a shim (lets you CANCEL by id).

### `BGEN <slot> <max_new> [k=v ...] <ids>` (native Strata multi-slot protocol)
Used by Strata's Python server. First token goes out as `T`, then `DONE 1 <plen> <ms> 0.0 admit` +
`BADM <slot> 1`, then `BT <slot> <id>` lines and `BDONE <slot> <gen> <reason> <dec_ms> read=<prompt_ms>[ drafts=o:a]`.
With `slot_read=1`, admission is immediate (`DONE 0 <plen> 0.0 0.0 admit`, `BADM <slot> 1`) and all tokens come as `BT`.
Not needed for a single-user shim.

### Control
| Line | Effect |
|---|---|
| `STOP` | cancel the active `GEN` (solo) request -> `DONE ... cancel` |
| `CANCEL <req_id>` | cancel a `GENERATE` request -> `DONE <req_id> ... cancel` |
| `BSTOP <slot>` | cancel a BGEN slot -> `BDONE <slot> ... cancel` |
| `STATS` | `STATS active_slots=<n> max_slots=<P> vram_experts=3072 ram_experts=0 total_experts=3072 kv=fp16|fp8` (active_slots = allocated slots, i.e. P, not busy ones) |
| `HELLO` | `HELLO ok engine=strata-inkling-0.1.65 arch=inkling2_loop backend=metal max_ctx=<N> parallel=<P>` |
| `BCO <slot> <ids>` | resets that slot's KV, replies `BCOK <slot> <n>` (does not prefill anything) |
| `BYIELD <x>` | no-op |
| `QUIT` | stop reading; **active requests run to completion first**, then exit 0. Send CANCEL/STOP first for a fast exit. stdin EOF behaves the same. |
| anything else | `ERR unknown command` |

Commands are polled between GPU steps (one prefill chunk or one decode/speculative step), so
CANCEL/STOP latency is one step (a 512-token prefill chunk is ~0.4-0.6 s at 28K context).

## 4. `k=v` options (`ink_serve.h:219-249`)

| Key | Default | Meaning |
|---|---|---|
| `temperature` | 0 | <= 0 means greedy argmax |
| `top_k` | **64** | clamped to 1..128. Sampling never looks beyond the top 128 tokens. Send it explicitly. |
| `top_p` | 1.0 | applied after min_p, within top_k |
| `min_p` | 0 | relative to the top token's probability |
| `penalty_repeat` | 1.0 | llama.cpp-style divide/multiply on logits |
| `penalty_freq` | 0 | `logit -= count*freq + pres` |
| `penalty_present` | 0 | |
| `penalty_last_n` | 64 | window over prompt+generated history |
| `pen_from` | 0 | history index where the penalty window may start (e.g. prompt_len to ignore the prompt) |
| `seed` | 20261121 when 0/absent | per-request mt19937_64 |
| `hold=N:ID` | off | GEN/BGEN only: after N generated tokens, finish with reason `hold` unless the token just produced is ID. Strata uses it as a reasoning budget: client then appends the end-of-thinking text itself and calls GEN again (prefix reuse makes that cheap). Not checked on the DSpark greedy fast path. |
| `image=<pct-encoded path>` | | needs `--vision`; the prompt must contain `200054` placeholders |
| `prefill=…`, `slot_read=…` | | `prefill` is parsed but **unused**. For a prefill-only/warm-up call use `max_new=0` instead (prefills, then `DONE 0 <plen> <ms> 0.0 length`). |
| `pcie_frac`, `spec_min_p`, `prefill_pipe` | | accepted and ignored |
| any other key | | `ERR unsupported key <k>` |

There is **no** stop-string or extra stop-token option. The engine stops only on token
**200006** `<|content_model_end_sampling|>` (manifest eos) or **199999** `<|endoftext|>`
(`ink_serve.h:391-393`, `ink_manifest.h:115-116`), on max_new, on max_ctx, on anti-loop, or on hold.
The stop token **is emitted** as a `T` line before DONE. Anything else (e.g. stopping after a tool
call's `<|end_message|>`) must be done by the shim with CANCEL.

Speculative decode (DSpark) is used only when exactly one request is active, it is past prefill,
`temperature<=0`, and `max_new - generated >= 2`; the fused fast path additionally needs no
penalties. It is designed as verify-then-accept, which would make it equal to plain greedy decoding, but measured
greedy output with drafting on diverges from drafting off (docs/upstream-bugs.md issue 3); with temperature > 0 it
is simply not used. It may emit several `T` lines at once.

## 5. KV / prefix cache reuse across requests (`ink_serve.h:475-495`)

Per slot, the engine remembers `cached_prefix` = every token it has fed through the model in that
slot: the prompt plus all generated tokens **except the last one** (the last sampled token, usually
the stop token 200006, was never fed back). On a new request in the same slot:

- reuse = length of the common prefix, capped at `prompt_len - 1` (at least one new token is always prefilled);
- **reuse is all-or-nothing**: if the new prompt does not start with the *entire* `cached_prefix`,
  the slot is reset and the whole prompt is re-prefilled (ShortConv/SWA state cannot be rewound).
- When reuse happens, `RESUME <n>` is printed and prefill starts at token n.

Consequences for the shim:
- Use `--parallel 1` (GENERATE always takes the first free slot, so with P>1 the slot is not stable).
- For multi-turn reuse, the next prompt must reproduce the engine's token history **exactly**:
  previous prompt ids + the generated ids (minus the final sampled token) + new tokens. Re-rendering the
  chat template from parsed messages will usually NOT match (the template re-renders reasoning as
  `<|message_model|><|content_thinking|>...<|end_message|>`, tool calls as canonical sorted JSON, and
  appends `<|content_model_end_sampling|>`; clients like Cline also drop or rewrite earlier turns).
  The robust approach: keep the last ids sent + generated ids, then for a new request compute the
  token-level common prefix with what you would send; if the full cached history is a prefix, send
  it (it will hit); otherwise accept a full re-prefill (~35 s at 28K).
- A cancelled request keeps whatever it already fed (still consistent).
- After `DONE ... stop`, the stop token itself is not in the cache; if your next prompt contains it
  right after the generated text (as the template does), it is just a new token to prefill. Fine.

## 6. Prompt construction (the engine does none of it)

- Tokenizer: `tokenizer.json` from the pack's Hugging Face folder (or `strata-inkling --export-tokenizer PACK OUT_DIR`, which writes the pack's own `chat_template.jinja` + `tokenizer.hf.json`). Encode with `add_special_tokens=False` (as the pack folder's `tools/strata_real.py` does).
- Template: `chat_template.jinja`; strip `{% generation %}`/`{% endgeneration %}` (strata_real.py does this). There is no `add_generation_prompt` branch and no trailing `<|message_model|>`: the prompt ends after the last user/tool `<|end_message|>` (or the effort system line) and the model itself emits `<|message_model|>` (200001).
- Reasoning: controlled only through the template's `reasoning_effort` (default 0.9), which renders a system message `<|message_system|><|content_text|>Thinking effort level: X<|end_message|>` before the first non-system message. `"none"` -> `0`. No engine switch exists.
- Output token grammar (from the template): `<|message_model|><|content_thinking|>...<|end_message|>` (reasoning), `<|message_model|><|content_text|>...<|end_message|>` (answer), `<|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|>` (tool call), then `<|content_model_end_sampling|>` (200006, stop).
- Relevant ids: 199999 `<|endoftext|>`, 200000 user, 200001 model, 200002 system, 200003 tool, 200004 content_text, 200006 end_sampling (eos), 200008 content_thinking, 200010 end_message, 200024 content_xml, 200049 content_invoke_tool_json, 200057 content_invoke_tool_text.

## 7. Example (as used by the pack folder's `tools/strata_real.py`)

The `DONE` line is from a real run on the 6,630-token prompt `tools/prompts/p_4k.txt`; `PP` values are
placeholders that only show the field layout.

```
> GEN 512 temperature=0 200002,200004,...,200010
< PP 512 6630 <prompt_ms_cum> <tok/s this chunk>
< ...                                  (13 PP lines for 6630 tokens at chunk=512)
< T 200001
< T 200008
< ...
< DONE 512 6630 5929.3 12282.1 length
> QUIT
```

## 8. Environment variables (all in `engine/metal/src/mtl_rt.mm`)

| Variable | Default | Effect |
|---|---|---|
| `STRATA_METAL_KEEPALIVE_MS` | 250 | GPU keep-alive tick after the last command buffer (keeps clocks up between requests); `0` = off |
| `STRATA_METAL_KEEPALIVE_WINDOW_S` | 120 | how long after the last request the keep-alive runs |
| `STRATA_METAL_SPIN` | on | spin-wait up to 20 ms on command-buffer completion (~+5% decode per upstream comment); `0` = blocking wait |
| `STRATA_METAL_MATH` | safe | `fast` = fast-math shader compile (not recommended; breaks the bitwise M=1 vs batch equality the kernels rely on) |
| `STRATA_METAL_MSL_DUMP=<dir>` | off | writes every compiled MSL library to `<dir>/NNN.msl` |
| `STRATA_METAL_WRAP_SPLIT`, `STRATA_METAL_MAX_THREADS_CAP` | | test hooks, leave unset |
| `IK2_EP_SPLIT` | | CUDA dual-GPU only, ignored on Metal |
