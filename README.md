# Editing a model after it has been quantized: a rank-1 edit at 3 bits, without the packer

**Summary.** We applied a published rank-1 weight edit (an abliteration LoRA) to a 13 GB model pack
that was already quantized to 3–4 bits per weight, without the tool that built it (it is not in the
public Strata repository). Rounding the edited weights back to the nearest code throws the edit away:
at 3 bits, 3% of it survives. A rounding pass that is aware of the edit's direction keeps 98.8% of it
at the worst site and more than 99% at the median, changes 1–8% of the codes, and leaves every scale
untouched (the only header bytes that change are two section hashes). The patched pack runs on the stock engine
at the stock speed. We also built the same edit as a runtime projection inside the engine, and the
two implementations agree.

Everything here was done on one Mac (M5 Pro, 64 GB) with the
[Strata](https://github.com/Niko1221/Strata) Metal engine and its `Inkling-2-Mini-SQ-15` pack. The
method is not specific to either.

## The setup

- **The model.** Inkling 2 Mini is a 26B mixture-of-experts model: 24 layers, 128 experts per layer.
- **The pack.** Strata's SQ-15 build stores each expert's matrices in one of several low-bit codecs:
  an NVFP4-style 4-bit float format (SF4, 1,538 experts) or 3-, 4- or 5-bit block-scaled integers
  (S3_K 1,022 experts, S4_K 510, S5_K 2). Trunk tensors are 8-bit per-row or BF16.
- **The engine.** Strata's Metal engine reads long prompts 3–4× faster than llama.cpp on this
  machine, but it only loads its own pack format, and the tool that built this pack is not in the public
  repository.
- **The edit.** [AtomicChat's uncensored build](https://huggingface.co/AtomicChat/Inkling-2-Mini-Uncensored-GGUF)
  of the model ships its change as a rank-1 LoRA over 49 sites: each layer's attention output, each layer's 128 expert
  down-projections, and the token embedding.

So the usual route (edit the full-precision weights, then quantize) was closed. The edit had to go
into weights that were already on a coarse grid.

## What the edit is

Reading the LoRA shows it is simpler than 49 separate low-rank updates:

- Every site in a layer shares one unit vector `r`: the attention output and all 128 experts.
- The other factor is `−Wᵀr`, to within quantization noise of the weights it was computed from.

So the whole edit is `W' = (I − r rᵀ) W`: remove the component along `r` from everything each layer
writes into the residual stream. There are 25 directions in total, 24 layers plus the embedding.

Two consequences:

1. **Recompute the edit from the pack's own weights**, not from the LoRA's stored factors. The LoRA
   was computed from BF16 weights. Added as-is to the 3-bit weights, it leaves 24% of the component
   along `r` in place before any rounding happens.
2. **What matters is one number per column**: `rᵀW'` should be zero. That turns "preserve the edit"
   into a constraint the rounding step can target directly.

## Why nearest rounding fails

The edit changes each weight by about 2% of the weight norm. On a 3- or 4-bit grid that is far less
than half a step for almost every weight, and the pack's weights already sit exactly on the grid. So
nearest rounding returns almost every weight to its old code.

Measured as the fraction of the component along `r` that is left (0 is a perfect edit, 1 is no edit):

| Format | Edit in float, no rounding | Nearest rounding, scales kept | Direction-aware rounding |
|---|---|---|---|
| SF4 (4-bit float) | 0.0017 | 0.775 | 0.0009 |
| S3_K (3-bit) | 0.240 | 0.968 | 0.0056 |
| S4_K (4-bit) | 0.131 | 0.836 | 0.0031 |
| S5_K (5-bit) | 0.069 | 0.675 | 0.0014 |
| Attention output, 8-bit rows | 0.012 | 0.108 | 0.0002 |
| Embedding, 8-bit rows | 0.0091 | 0.088 | 0.0003 |

The first column uses the LoRA's own factors, which is why it is not zero. Samples: 24–25 experts per
format, all trunk tensors.

Nothing looks wrong after nearest rounding. The file loads, the model runs, and it behaves as if it
had not been edited.

`patch/demo_synthetic.py` shows the effect on a random matrix in about a second, with numpy only. On a 3-bit
block-scaled grid with a random direction whose energy is spread evenly, nearest rounding leaves 1.000 of the
component and direction-aware rounding 0.005, at 0.057 added noise; with a heavy-tailed direction, closer to the
real ones, nearest leaves 0.93 and direction-aware 0.010. Why the two cases differ, and why 1.0 is the limiting case
rather than a typical one, is in [docs/method.md](docs/method.md).

## Direction-aware rounding

For each column of `W` (one input unit), with every block scale held fixed:

1. Round the target `W'` to the nearest code, as usual.
2. Compute the leftover component along `r`: `e = rᵀ(Q − W')`, where `Q` is the rounded column.
3. Move individual weights by one grid step, cheapest first, until `e` is close to zero. A move's
   benefit is how much it reduces `|e|` (its step size times its entry of `r`); its cost is the
   squared error it adds.
4. Repeat for three passes, with a guard against overshooting.

This is a small knapsack per column: buy a required amount of correction along `r` with as little
off-axis error as possible. It needs no calibration data and no Hessian, because the constraint comes
from the edit itself.

Cost, measured on all 3,072 experts and 25 trunk tensors of the finished pack:

| Format | Sites | Residual along `r` (median / max) | Added noise relative to ‖W‖ (median / max) |
|---|---|---|---|
| SF4 | 1,538 | 0.00085 / 0.0019 | 0.031 / 0.039 |
| S3_K | 1,022 | 0.0056 / 0.0122 | 0.050 / 0.073 |
| S4_K | 510 | 0.0032 / 0.0040 | 0.035 / 0.039 |
| S5_K | 2 | 0.0014 / 0.0014 | 0.025 / 0.025 |
| Attention output, 8-bit | 20 | 0.00019 / 0.00027 | 0.0088 / 0.0096 |
| Embedding | 1 | 0.00030 | 0.0071 |

For scale: the 3-bit format's existing quantization error is about 0.25 of the weight norm, so the
added 0.05 raises it by about 2%. AtomicChat's own shipped Q4_K model, which works, has a residual of
0.05–0.10 on its expert matrices.

The build takes under 9 minutes on four CPU cores. 488 MB of the 13.4 GB file changes, all of it
inside the intended tensors plus two section hashes.

## Encoders that change only the codes

Strata publishes its decoders (C++ and Metal shaders) under MIT. Its repository also contains reference encoders
written for an earlier pack layout (`.strata` v1.0): `encode_sk` for the block-scaled integer codes
(`tools/squant/codecs.py`) and a per-row 8-bit encoder (`tools/squant/pipeline.py`). Both choose new scales from the
weights they are given. There is no encoder for the 4-bit float format (SF4). The tool that built this Inkling pack
(its bit allocation, labelled `global_hessian_knapsack_v2` in the pack's manifest, and the pack writer for this
layout) is not in the public repository.

An edit needs something narrower: encoders that keep every existing scale and header byte (and the per-expert
vectors) and choose only the per-weight codes. We wrote those in Python from the decoder source, for the formats the
edit touches: SF4, S3_K / S4_K / S5_K, the 8-bit per-row format (Q8_ROW) and BF16. The format notes are in
[codecs/FORMAT.md](codecs/FORMAT.md).

The test that makes this trustworthy: **decode → encode must reproduce the original bytes** for every
tensor we touch. It does, for all 3,072 expert down-matrices and all 25 trunk tensors. A per-element
transcription of the Metal kernels agrees with the Python decoder on 389,184 values. As a cross-check, Strata's own
reference `decode_sk` agrees bit for bit with our decoder on 26.5 million values from this pack when both compute the
block scale in the same order.

Details that would have silently broken the patch:

- The expert directory is not in expert order. Look experts up by their id field.
- The low-bit integer formats store the down-matrix at a per-unit scale, with the inverse applied to
  the activation at runtime. Edits must be applied to the matrix the engine actually multiplies by.
- SF4 has a signed zero, and the packer uses both signs.
- Section hashes are only checked on request, so a stale hash does not stop the pack loading.

## The same edit, inside the engine

Because the edit is a projection of each writer's output, it can also be applied at runtime:
`y ← y − r (rᵀy)` after the attention output projection, after the experts are combined, and on the
embedding row. That is five insertion points across the prefill and decode paths, about 370 added lines
behind an opt-in flag (`engine/ablate.patch`), with no change to the MoE kernels.

We built it for two reasons: it is exact (no rounding at all), and it is an independent check on the
patched pack.

- With the flag off, the local build is bit-identical to the shipped binary (logits and tokens).
- With the flag on at strength 0, it is bit-identical to flag off.
- A consistency test compares the edit's effect on logits across the prefill and decode paths.
  Calibration: at 90% strength the test reads 0.92; the real paths read 0.99–1.00.

## Results

Three gates, each against a reference. Refusal counts use the same 60 prompts from the test split of
`mlabonne/harmful_behaviors` for every model, at temperature 0, with the model's reasoning off and on.

| | Original pack | AtomicChat Q4_K (llama.cpp) | Patched pack, stock engine | Original pack, patched engine |
|---|---|---|---|---|
| Refusals, reasoning off | 60 / 60 | 5 / 60 | 4 / 60 | 5 / 60 |
| Refusals, reasoning on | 60 / 60 | 5 / 60 | 5 / 60 | 2 / 60 |
| Mean KL vs original pack | 0 | n/a | 0.0102 | 0.0094 |
| Top-1 agreement | 100% | n/a | 95.14% | 95.30% |
| Perplexity, WikiText-2 | 9.126 | n/a | 9.173 | 9.163 |
| Prefill, 6.6K-token prompt (tok/s) | 1,130 | 346 | 1,141 | 1,185 |
| Prefill, 28K-token prompt (tok/s) | 962 | 215 | 956 | 990 |
| Decode after a 4K prompt (tok/s) | 43.5 | n/a | 44.4 | 43.9 |

- KL is over 20,480 positions against a top-256 reference, which reads about 10% low; the exact
  figures are roughly 0.011 and 0.010. AtomicChat reports 0.008 for the edit on BF16 weights.
- None of the models refused any of 30 harmless control prompts.
- The llama.cpp speeds were measured earlier, with another GPU workload running; the Strata speeds
  in this table were measured on an otherwise idle machine, so the two columns are not matched.
  Under matched conditions (both with the other workload running) Strata read the same prompts at
  1,118 and about 800 tok/s, which is the 3–4× gap.
- The two routes differ by less than the noise on every behavioural measure. The 3–5% of added
  rounding noise in the patched pack costs about 0.001 of KL.

## Bugs found in the engine along the way

All three are in Strata 0.1.65 (commit `26350e4`), and all three affect evaluation more than use:

1. **`--kld-eval` and `--dump-ref` return zero logits for rows 64 and up** of each batch on Metal.
   KL is understated and top-1 agreement overstated whenever more than 64 positions are scored. A
   small fix (9 added lines) is in `engine/kldeval-64row-fix.patch`.
2. **`--parallel` above 1 corrupts the first token** of a request admitted alongside another.
3. **Speculative decoding is not exact**: with drafting on, greedy output diverges from plain greedy.

Reproductions and the details are in [docs/upstream-bugs.md](docs/upstream-bugs.md).

## A shim for coding agents

The engine speaks a line protocol over token ids, so we wrote a small OpenAI-compatible server in
front of it. One finding is worth passing on to anyone doing the same for another engine.

The engine keeps its KV cache between requests only if the new prompt begins with exactly the tokens
it has already seen, including the model's reasoning. Agent clients resend the conversation without
the reasoning, so the cache never matches and every turn re-reads the whole prompt. The shim
remembers each reply's exact tokens and splices them back in when the client returns that reply. On
a 29K-token agent prompt, the second turn went from 34.0 s to 3.3 s.

The shim also guards memory, for a reason. On a unified-memory Mac a GPU allocation that does not fit is not refused:
the system compresses or swaps other processes' memory instead. On a 64 GB machine, the 17.5 GiB engine at 64K
context could not share memory with another GPU workload of about 27 GB without the system compressing that
workload's memory. So the shim checks free memory and memory pressure before it loads the engine, and unloads it when
memory runs short; optional checks protect a named service (see [shim/README.md](shim/README.md)).

## What is in this repository

| Folder | What |
|---|---|
| `patch/` | The pack route: `rounding.py` (direction-aware rounding, format-agnostic), `demo_synthetic.py` and its test, `build_abl.py` (writes the edited pack), `verify_abl.py`, `survival.py` (the measurement in the tables above), `lora_map.py` |
| `codecs/` | Bit-exact Python codecs for the formats the edit touches, their tests, and `FORMAT.md` |
| `engine/` | The runtime route: patches against Strata `26350e4`, a build script, the direction-file writer, and the GPU validation (`engine/README.md`) |
| `eval/` | The three gates (refusal rate, KL / coherence, speed) and `RESULTS.md` with every number above and how it was produced |
| `shim/` | The OpenAI-compatible server in front of `strata-inkling --serve` |
| `docs/` | `method.md` (the derivation), `upstream-bugs.md` (the three engine issues), `serve-protocol.md` |

No model weights, direction files, edited packs, prompts or transcripts are included; the tools build or fetch them.

## Reproduce

You need, besides this repository: Strata's Inkling 2 Mini pack folder from Hugging Face
(`Inkling-2-Mini-SQ-15.strata`, `tokenizer.json`, `chat_template.jinja` and `tools/`), AtomicChat's LoRA in GGUF form,
a checkout of [Strata](https://github.com/Niko1221/Strata) at commit `26350e4` (for the engine route and two of the
codec tests), [uv](https://docs.astral.sh/uv/), and for the engine and the gates a Mac with Apple silicon. Steps 1–4
run on the CPU only.

```sh
MODEL_DIR=/path/to/inkling-2-mini-pack            # the Hugging Face folder
PACK=$MODEL_DIR/Inkling-2-Mini-SQ-15.strata
LORA=/path/to/Inkling-2-Mini-Uncensored-LoRA.gguf
PY="uv run --no-project --with numpy --with zstandard --with gguf python -I"

# 1. The method on a random matrix: numpy only, a few seconds.
uv run --no-project --with numpy python -I patch/demo_synthetic.py
uv run --no-project --with numpy python -I patch/test_rounding.py

# 2. Codec tests on the real pack (about 10 s and 4 GB peak for the sample; --all round-trips every expert).
$PY codecs/test_codecs.py --pack $PACK --strata-src /path/to/Strata --lora $LORA

# 3. How much of the edit survives each rounding strategy.
$PY patch/survival.py --pack $PACK --lora $LORA --out results/survival.json

# 4. Build and verify the edited pack: about 9 minutes on four cores. The output starts as a copy-on-write clone
#    where the file system supports it (3.3 GB of new blocks on APFS); the script wants 46 GiB free before cloning.
$PY patch/build_abl.py  --pack $PACK --lora $LORA --out Inkling-2-Mini-SQ-15-abl.strata --procs 4
$PY patch/verify_abl.py --orig $PACK --new Inkling-2-Mini-SQ-15-abl.strata --lora $LORA --procs 4

# 5. The runtime route: patch and build the engine (Command Line Tools only), write the direction file.
git clone https://github.com/Niko1221/Strata && git -C Strata checkout 26350e4
engine/build.sh --patch Strata out
$PY engine/make_dirs.py $LORA $PACK DIRS.bin

# 6. The gates (GPU). prep_data.py downloads the prompt sets and WikiText-2 at pinned revisions.
uv run --no-project --with pyarrow --with tokenizers --with huggingface_hub python -I eval/prep_data.py --model-dir $MODEL_DIR
G="uv run --no-project --with jinja2 --with tokenizers --with numpy python eval/run_gates.py --model-dir $MODEL_DIR \
   --strata-bin /path/to/stock/strata-inkling --kl-bin out/strata-inkling"
$G --pack $PACK --gate all                                     # the original; records the gate-2 reference
$G --pack $PWD/Inkling-2-Mini-SQ-15-abl.strata --gate all      # the patched pack on the stock engine
$G --pack $PACK --engine out/strata-inkling --engine-args "--ablate $PWD/DIRS.bin --ablate-scale 1" --gate all
#    engine/validate/gpu_validate.sh runs the engine-route checks (flag-off parity, scale 0, path consistency).

# 7. The shim: tests (CPU, a fake engine), then serve the edited pack on 127.0.0.1:8097.
INKLING_S_MODEL_DIR=$MODEL_DIR shim/inkling-s setup      # once: fetches the pinned Python packages
INKLING_S_MODEL_DIR=$MODEL_DIR shim/inkling-s test
INKLING_S_MODEL_DIR=$MODEL_DIR INKLING_S_ENGINE=/path/to/strata-inkling \
  INKLING_S_PACK=$PWD/Inkling-2-Mini-SQ-15-abl.strata shim/inkling-s start
```

`--kl-bin` points the KL gate at a build with the 64-row fix (see the engine bugs below). The patched engine with
its flag off serves: in the validation its logits were bit-identical to the stock binary's on rows 0–63, the rows the
stock binary gets right, and its greedy tokens identical. Each
script's header lists its options; the gates and the shim can be wrapped in an external memory or GPU lock
(`--guard`, `INKLING_S_GUARD_LOCK`), off by default.

## Limits

- **One model, one pack, one edit.** The rounding method is general, but it has been measured on a
  single rank-1 projection per layer. A higher-rank edit turns the scalar constraint per column into
  a small vector one; the same greedy approach should apply but is untested.
- **The refusal test is small.** 60–100 prompts, cut-off generations, and a phrase-based classifier
  checked by hand on about 120 transcripts. It shows the patched models match the reference; it does
  not rank them.
- **Quality is measured by KL and three code prompts**, not by a task benchmark.
- **Clipping was not optimised.** 1–5% of targets fall outside their block's range. The greedy pass
  reaches its residuals anyway; refitting scales might lower the noise.
- **Edit-specific leftovers.** Where the original LoRA does not fully remove the direction, neither
  do we.

## Related work

We looked for prior work before writing this up and did not find a measurement of how much of an
edit survives re-rounding, or a rounding method constrained by the edit. Closest, and worth knowing:

- **Strata's own reference encoders** (`tools/squant/` in its repository) quantize full-precision weights into
  `.strata` codes, choosing scales and rounding to nearest. They were written for an earlier pack layout and do not
  cover SF4. We did not use them; our encoders keep the existing scales instead.
- **Editing quantized GGUFs in place with scales kept.** A Hugging Face release by SC117 transplants
  abliterated tensors between GGUFs and, for 9 two-bit tensors, re-encodes "using [the format's] own
  learned block scales … only the codes". It does not describe how codes are chosen or measure what
  survives.
- **[heretic-gguf](https://github.com/MoriNoNushi/heretic-gguf)** runs directional ablation against
  quantized GGUFs as hot-swapped LoRA adapters, and recommends the adapter over its merged export
  because the merge "requantizes edited tensors".
- **Edit first, then quantize with error compensation** (GPTQ-style), or keep the edited tensors at
  higher precision. Both need the full-precision weights or the packer.
- **Merging adapters into low-bit weights** is a known problem in quantization-aware fine-tuning:
  [QA-LoRA](https://arxiv.org/abs/2309.14717) and
  [LoTA-QAF](https://arxiv.org/abs/2505.18724) constrain the adapter during training so the merge is
  exact. Our case is the reverse: the edit is given, and only the rounding can adapt.
- **Adaptive rounding** in post-training quantization
  ([YAQA](https://arxiv.org/abs/2505.22988), GPTQ, AdaRound) chooses codes to minimise output error on
  calibration data. Direction-aware rounding is the same idea with a constraint supplied by the edit
  in place of calibration.

## Credits and licences

- [Strata](https://github.com/Niko1221/Strata) by Niko1221 and contributors, MIT. The codecs here are
  derived from its published decoders; the codec tests cross-check against its reference S_K codec
  (`tools/squant/codecs.py`); the patches in `engine/` modify its engine.
- Inkling 2 Mini by Thinking Machines Lab, Apache-2.0 with a Model Acceptable Use Policy.
- The abliteration LoRA by AtomicChat, Apache-2.0.
- [llama.cpp](https://github.com/ggml-org/llama.cpp), MIT: the reference model was served with it, and its `gguf`
  Python package reads the LoRA.
- This repository ships tools and patches only, no model weights. Our code is MIT-licensed (LICENSE); third-party
  notices are in NOTICE.
