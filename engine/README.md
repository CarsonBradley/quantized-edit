# Runtime route: project the edit inside the engine

Instead of rewriting the pack, these patches make Strata's Metal engine apply the edit while it runs: with
`--ablate DIRS.bin`, every residual-stream writer's output `y` becomes `y − S·r (rᵀy)`, with one unit vector `r` per
layer and one for the embedding. That is exact (no rounding at all), and it is an independent check on the patched
pack. With the flag absent nothing changes: no file is read and no extra kernel runs.

Insertion points (all on the Metal backend; `--ablate` with `--cpu` is refused):
- **E**, the embedding: the gathered row is projected before its norm, in a twin of the stock embedding kernel
  (image rows bypass it).
- **A**, the attention output: the projection runs on the output projection's fp32 result, before the short
  convolution and the residual add, so all three stock variants of that step see the projected value.
- **M1–M3**, the routed experts: the combined, routing-weighted expert output is projected once per layer visit: on
  the batched matrix-multiply path for 10 or more tokens (M1), the non-batched tree-verify and single-slot prefill
  paths (M2), and the decode / multi-slot / linear-verify path for fewer than 10 tokens (M3), where the stock fused
  reduce + convolution + add kernel is split in two so the projection can sit between them.

No MoE or GEMM kernel is changed. The new kernels live in their own Metal library, compiled only when `--ablate` is
given.

## Files

| File | What |
|---|---|
| `kldeval-64row-fix.patch` | fixes zero logits for rows ≥ 64 in `--dump-ref` / `--kld-eval` (docs/upstream-bugs.md, issue 1) |
| `ablate.patch` | the `--ablate DIRS.bin [--ablate-scale S]` flag (368 added lines), plus the eval-only `STRATA_INK_EVAL_LOGIT_SLICE` switch used by the validation |
| `build.sh` | builds `strata-inkling` with the Command Line Tools only (no Xcode, no cmake) |
| `make_dirs.py` | writes `DIRS.bin` from the rank-1 LoRA, after checking that it has the projection structure |
| `strata_decode.py` | the minimal pack and LoRA reader `make_dirs.py` uses |
| `validate/` | GPU validation: flag-off parity, scale-0 identity, KL, prefill-vs-decode consistency, calibration; results in `validate/RESULTS.md` |

## Patch order

Both patches are against Strata commit `26350e4`, with paths relative to the repository root. **Apply
`kldeval-64row-fix.patch` first, then `ablate.patch`**: `ablate.patch` edits lines the fix introduces, so it does
not apply to an unpatched tree. The fix applies on its own.

```sh
git clone https://github.com/Niko1221/Strata && git -C Strata checkout 26350e4
engine/build.sh --patch Strata out          # applies both patches in order (skips any already applied), then builds
# or by hand:
git -C Strata apply "$PWD/engine/kldeval-64row-fix.patch"
git -C Strata apply "$PWD/engine/ablate.patch"
engine/build.sh Strata out
```

The binary is `out/strata-inkling`. Built this way on macOS with clang from the Command Line Tools, it is
byte-identical to the binary that passed the validation in `validate/RESULTS.md`, and the Metal shader sources it
embeds are byte-identical to the validated ones.

## The direction file

```sh
uv run --no-project --with gguf --with numpy --with zstandard python -I engine/make_dirs.py \
    Inkling-2-Mini-Uncensored-LoRA.gguf Inkling-2-Mini-SQ-15.strata DIRS.bin DIRS.report.json
```

Before writing, it asserts that the adapter is rank 1 with α = 1; that in every layer the attention output and all
128 experts share one unit `B` vector; and that `A = −Wᵀr` against the pack's own decoded weights (attention, a few
experts per codec per layer, and the embedding). The format is described in its docstring and in the loader
(`engine/src/inkling2/metal/ink_ablate_dirs.h` after patching), which rejects any file that is not exactly that
size or whose rows are not unit vectors.

## Running it

```sh
out/strata-inkling --serve --native Inkling-2-Mini-SQ-15.strata --ablate DIRS.bin --ablate-scale 1 --parallel 1 --draft off
```

stderr shows `[ink_metal] ablate ON: DIRS.bin (lora sha256 ...) layers=24 scale=1.0000 sites=embed+attn_wo_ud+moe_down`.
`--ablate-scale` (default 1) scales the projection; 0 gives output bit-identical to running without the flag.
The eval harness refuses to score a run with `--ablate` unless that line appears.
