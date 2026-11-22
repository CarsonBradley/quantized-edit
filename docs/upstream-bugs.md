# Three issues in Strata's Inkling engine (0.1.65, commit 26350e4)

Found while evaluating the edited model on an Apple M5 Pro (Metal backend). All three affect measurement more than
everyday use, but each one can make an evaluation report the wrong thing. They have not been filed upstream.

Commands below use the stock `strata-inkling` binary and the original pack `Inkling-2-Mini-SQ-15.strata`. Token ids
come from the model's tokenizer; `engine/validate/compare_ref.py` and `engine/validate/serve_ids.py` are the helpers
used to observe each issue.

## 1. `--dump-ref` and `--kld-eval` return all-zero logits for rows 64 and up (Metal)

**What happens.** When a forward pass asks for logits at more than 64 positions, the Metal runner computes logits
for at most 64 rows of each batch it is given: `n_lg = std::min(M, 64)` in
`engine/src/inkling2/metal/ink_metal_runner.h` (line 1983). The backend (`MetalInkBackend::forward_batch`,
`ink_metal_backend.mm`) slices batches only at `max_batch` (at least 256 tokens), then copies every requested row
out of a zero-initialised buffer (around lines 240–261). Rows 64 and up of each sub-batch therefore come back as
zeros, with no error.

**Effect.** `--kld-eval` compares a reference against these rows, so mean KL is understated and top-1 agreement
overstated whenever more than 64 positions are scored; a reference written by `--dump-ref` has the same zero rows.
Serving is not affected (it asks for at most a few logit rows per step).

**Reproduction.**

```sh
# 128 token ids of any text (e.g. from eval/prep_data.py's WikiText windows)
strata-inkling --dump-ref ref128.bin --native Inkling-2-Mini-SQ-15.strata --max-context 8192 --prompt-ids "$IDS128"
python engine/validate/compare_ref.py ref128.bin ref128.bin      # prints zero_rows_A_from64
```

Observed: 64 of the 64 rows 64–127 are all zero. With the fix below, none are.

**Fix.** `engine/kldeval-64row-fix.patch` (one hunk in `ink_metal_backend.mm`): when more than 64 rows want logits,
slice the batch into 64-token sub-batches. With the fix, rows 0–63 are bit-identical to the stock binary's, and the
output is identical run to run (`engine/validate/RESULTS.md`, steps 1 and 2a).

## 2. `--parallel` above 1: a request admitted next to another gets a wrong first token

**What happens.** With `--serve --parallel 4`, every `GENERATE` request that was admitted while another request was
active produced token id 0 (`!`) as its first token, instead of `<|message_model|>` (id 200001), which every
single-slot run produces first. The rest of its output also differed from the single-slot run of the same prompt.
The cause was not investigated.

**Effect.** Any multi-slot evaluation or server built on `--parallel` > 1 silently gets different, wrong outputs
for concurrent requests. Our harness and shim always use `--parallel 1`.

**How it was observed.** The refusal harness (`eval/gate1.py`) running greedy requests through `GENERATE` with
`--parallel 4`. A reduced form, not run in exactly this form:

```sh
strata-inkling --serve --native Inkling-2-Mini-SQ-15.strata --max-context 4096 --parallel 2 --draft off
# wait for READY, then send two lines back to back (IDS = any chat-rendered prompt):
GENERATE a 8 temperature=0 $IDS
GENERATE b 8 temperature=0 $IDS
# expected: both requests start with "T a 200001" / "T b 200001"
# observed with --parallel 4 in the harness: the request admitted second started with token 0
```

## 3. Speculative decoding is not bit-exact

**What happens.** With DSpark drafting on (`--draft auto`, the default for this pack), greedy generation
(temperature 0) diverges from plain greedy generation (`--draft off`). The serve loop (`ink_serve.h`, around lines
690–730) scores the drafted tokens in one multi-row forward pass and accepts each drafted token that equals the
target's choice, which reproduces plain greedy decoding exactly only if a multi-row pass gives the same logits as a
single-token decode step. It does not quite: in our cross-path measurements the 8-row and 1-row paths' logits differ
from the 64-row path's (KL 7.1e-4 and 6.7e-4 with no edit applied; `engine/validate/RESULTS.md`, step 5), so a
near-tie can resolve differently. That is a likely cause, not a confirmed one.

**Reproduction.**

```sh
python engine/validate/serve_ids.py strata-inkling Inkling-2-Mini-SQ-15.strata MODEL_DIR \
    MODEL_DIR/tools/prompts/p_short.txt 200 off.json  --draft off --max-context 8192
python engine/validate/serve_ids.py strata-inkling Inkling-2-Mini-SQ-15.strata MODEL_DIR \
    MODEL_DIR/tools/prompts/p_short.txt 200 auto.json --draft auto --max-context 8192
# compare the "ids" arrays of off.json and auto.json
```

(`MODEL_DIR` is the pack's Hugging Face folder, with `tokenizer.json`, `chat_template.jinja` and
`tools/prompts/`.) Observed with the stock binary: the outputs first differ at generated token 29 of 200.

**Effect.** Greedy results depend on whether drafting is on, so evaluations must fix `--draft`. Ours use
`--draft off` throughout; the shim defaults to it.
