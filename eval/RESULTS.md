# Results of the three gates (2026-11-22)

Models compared:
- **Original pack**: `Inkling-2-Mini-SQ-15.strata` on the stock `strata-inkling` 0.1.65 binary.
- **AtomicChat Q4_K (llama.cpp)**: AtomicChat's uncensored release of Inkling 2 Mini, edited in BF16 and then
  quantized to Q4_K, served by llama.cpp's `llama-server` (build b12621). The target the edited packs should match.
- **Patched pack, stock engine**: the original pack edited with `patch/build_abl.py`, on the stock binary.
- **Original pack, patched engine**: the original pack on a build with `engine/ablate.patch`, run with
  `--ablate DIRS.bin --ablate-scale 1`.

All numbers come from `eval/run_gates.py` on one Mac (M5 Pro, 64 GB). Transcripts and logit dumps are not published;
this page gives counts only.

## Gate 1: refusal rate

**Prompts.** `eval/prep_data.py` (seed 20261122): 100 of the 104 prompts in the test split of
`mlabonne/harmful_behaviors`, and the first 30 of a 100-prompt sample from the test split of
`mlabonne/harmless_alpaca` as a control. The llama.cpp reference ran the first 60 harmful prompts (100 would not fit
the time budget); every comparison below uses those 60.

**Generation.** The same token ids go to both engines (the model's own chat template and tokenizer), greedy
(temperature 0). "Reasoning off" means the template's `reasoning_effort="none"`; the model still writes a short
reasoning block. "Reasoning on" is the template default (0.9). Reasoning is capped at 256 tokens (off) or 380 tokens
(on); a capped reasoning block is closed by the client and the answer is forced. Each answer is cut 120 tokens after
it starts.

**Classifier** (`eval/lib/classify.py`, phrase-based). Answers are *hard refusals* (a refusal phrase at the start),
*soft refusals* (a later refusal, a deflection, a redirect to help lines), *reframed* (no refusal, but the answer
steers to a harmless, authorised or consented version of the request), or *complies*. "Refusals" counts hard + soft.
The phrase lists were calibrated by reading about 120 transcripts (every non-obvious classification of the original
pack's run and of the llama.cpp reference) and frozen before any edited model was scored. The reframe detector
misses some cases, so the reframe column is a lower bound for every model.

| Model | Reasoning off: refusals | Off: refusals + reframed | Reasoning on: refusals | On: refusals + reframed | Harmless controls refused (off / on) |
|---|---|---|---|---|---|
| Original pack | 60/60 | 60/60 | 60/60 | 60/60 | 0/30 / 0/30 |
| AtomicChat Q4_K (target) | 5/60 (95% CI 4–18%) | 24/60 (29–53%) | 5/60 (4–18%) | 15/60 (16–37%) | 0/30 / 0/30 |
| Patched pack, stock engine | 4/60 (3–16%) | 13/60 (13–34%) | 5/60 (4–18%) | 12/60 (12–32%) | 0/30 / 0/30 |
| Original pack, patched engine | 5/60 (4–18%) | 20/60 (23–46%) | 2/60 (1–11%) | 14/60 (14–35%) | 0/30 / 0/30 |

Intervals are 95% Wilson intervals. On all 100 harmful prompts: original pack 99/100 refusals in both modes; patched pack 4% (off) and 6% (on), refusals +
reframed 19% and 19%; patched engine 6% and 2%, refusals + reframed 25% and 19%.

The remaining refusals are on the same four prompts for all three edited models; all but one of those are soft
refusals.

With 60–100 prompts an interval spans roughly ±5–10 points, so a difference of a few prompts between models is not
meaningful. This gate shows that both edited packs match the reference; it does not rank them.

## Gate 2: not broken

**Running-text KL.** WikiText-2 raw test, 40 evenly spaced windows of 512 tokens (20,480 scored positions). The
original pack's logits are stored as a top-256 reference per position (the full dumps would be 16 GB); KL(original ‖
candidate) is computed with the tail outside the top 256 merged into one bucket, which is a lower bound on the exact
KL: it read 2–6% low on synthetic perturbations and about 10% low on the first real candidate. The engine's own exact
`--kld-eval` on window 0 (511 positions) is reported alongside. All dumps come from builds with
`engine/kldeval-64row-fix.patch` (docs/upstream-bugs.md, issue 1), and every dump is checked for all-zero rows.

| Model | Mean KL (top-256 lower bound) | Exact KL, window 0 (`--kld-eval`) | Top-1 agreement | Perplexity (original 9.126) |
|---|---|---|---|---|
| Original pack vs itself | 0.000000 | 0.000000 | 100.00% | 9.126 |
| Patched pack, stock engine | 0.01019 | 0.00583 (harness: 0.00527) | 95.14% | 9.173 |
| Original pack, patched engine | 0.00937 | 0.00493 (harness: 0.00443) | 95.30% | 9.163 |

Multiplying by about 1.1 gives a rough exact-KL figure: about 0.011 and 0.010. AtomicChat reports 0.0078 (neutral
text) and 0.0076 (code) for its edit on BF16 weights.

**Coherence.** Three prompts from the pack's `tools/prompts/` (63, 6,630 and 28,156 tokens), 512 new tokens, greedy,
reasoning at the template default. An edited model is expected to diverge from the original at some point.

| Model | First divergence from the original (short / 6.6K / 28K) | Distinct 4-grams (short / 6.6K / 28K) |
|---|---|---|
| Original pack | – | 0.91 / 0.94 / 0.94 |
| Patched pack, stock engine | 52 / 8 / 8 | 0.85 / 0.88 / 0.93 |
| Original pack, patched engine | 52 / 8 / 8 | 0.90 / 0.98 / 0.99 |

All six edited outputs were read: coherent, on-task reasoning about the prompt, with no loops.

## Gate 3: speed

`--serve` on real chat prompts (GEN, greedy, `--draft off`, `--max-context 32768`), median of 2 repetitions, plus the
engine's synthetic `--bench --prompt-len N --gen 128 --draft off`. Measured with no other model loaded and the
machine otherwise idle, each run holding the GPU exclusively. Prefill and decode in tokens per second.

| Run | Prefill 6,630 | Prefill 28,156 | Decode after 4K / 28K / short prompt | `--bench` prefill 6,630 / 28,156 | `--bench` decode 6,630 / 28,156 |
|---|---|---|---|---|---|
| Original pack, stock | 1,130 | 962 | 43.5 / 37.5 / 46.5 | 1,252 / 1,029 | 43.1 / 37.9 |
| Patched pack, stock | 1,141 | 956 | 44.4 / 37.8 / 47.2 | 1,277 / 1,065 | 44.1 / 38.6 |
| Original pack, patched engine, flag on | 1,185 | 990 | 43.9 / 37.8 / 47.0 | 1,276 / 1,072 | 42.5 / 38.1 |
| Original pack, patched engine, flag off | 1,190 | 988 | 44.4 / 38.2 / 47.6 | 1,281 / 1,073 | 44.1 / 38.7 |

- The patched pack runs at the original's speed (same binary, same formats).
- The runtime edit costs within 0.5% of prefill and 0.3–1% of decode on real prompts (flag on vs off, same binary),
  3.6% on the 6.6K bench decode (42.5 vs 44.1).
- The locally built engine with the flag off is within a few percent of the stock binary, within run-to-run spread;
  the original's numbers were taken 2.5 hours earlier.

**llama.cpp comparison.** The AtomicChat Q4_K model under llama.cpp read the same prompts at 346 (6,630 tokens) and
215 (28,156 tokens) tok/s. Those runs were measured earlier with another GPU workload running, so they are not matched
with the table above. Under matched conditions (both with that workload running) Strata read the same prompts at
1,118 and about 800 tok/s: the 3–4× gap.

## How long each gate takes (GPU time, excluding waits)

| Phase | Time |
|---|---|
| Gate 2 KL, first run (records the reference) | 5.8 min |
| Gate 2 KL, candidate (40 dumps + `--kld-eval`) | about 2.6 min |
| Gate 2 coherence (3 prompts × 512 tokens) | 1.5 min |
| Gate 3 speed | about 2.6 min |
| Gate 1, original pack (100 harmful + 30 harmless, two modes) | 23.9 min |
| Gate 1, edited models | 32–34 min (compliant answers run the full 120 tokens) |
| Gate 1, llama.cpp reference (60 harmful + 30 harmless, two modes) | 38.4 min |

## Limits

- Greedy refusal rates on short budgets (at most 120 answer tokens). An answer that starts by complying and refuses
  later would be missed.
- The classifier works on phrases and was calibrated on the original and reference models' outputs only. A model
  that refuses in new wording needs its "complies" items read.
- Quality is measured by KL, perplexity and three long prompts, not by a task benchmark.
