# Method: putting a rank-1 projection into weights that are already quantized

## The edit

AtomicChat's abliteration of Inkling 2 Mini is distributed as a rank-1 LoRA over 49 sites per model: in each of
the 24 layers, the attention output projection and the down projection of each of the 128 routed experts (stored as
one tensor per layer), plus the token embedding. A LoRA site adds `ΔW = B A` to a weight `W` of shape [out, in],
here with `B` a column of length `out` and `A` a row of length `in` (scale α/rank = 1).

Reading the adapter shows more structure than 49 independent updates:

- Within a layer, every site has the same unit vector `B = r_L`: the attention output and all 128 experts. Cosine
  between them 1.00000000; `|B| = 1`.
- The other factor is `A = −r_Lᵀ W`, computed from the unquantized weights. Against the pack's weights this holds to
  a relative error of 3e-7 on the attention matrices stored in BF16, about 1e-2 on those stored as 8-bit rows (which
  is their quantization noise), and 1e-3 to 2e-3 on decoded 4-bit-float expert matrices.
- The embedding site has the same form on the other side: each looked-up embedding row `e` becomes
  `e − (e·r_emb) r_emb`.

So the whole edit is

    W' = (I − r rᵀ) W

for every matrix that writes into the residual stream: it removes the component along `r` from each writer's
output. There are 25 directions (24 layers + the embedding). The projection is linear and commutes with the
router's weighted sum of expert outputs, so it can equally be applied once to the combined MoE output; that is what
the runtime route in `engine/` does.

Two consequences shape everything that follows.

1. **Recompute the edit on the weights you have.** The LoRA was computed from BF16 weights. Added as-is to the 3-bit
   weights of this pack, `W_q + B A` leaves 24% of the component along `r` in place before any rounding happens
   (`|rᵀ(W_q + BA)| / |rᵀW_q|` = 0.24 on S3_K experts, 0.13 on S4_K, 0.0017 on SF4), because `A` was computed from
   different weights. Computing `W' = W_q − r (rᵀW_q)` on the decoded pack weights instead removes it exactly, in float.
2. **The edit is one number per column.** What must hold after rounding is `rᵀ Q[:, j] = 0` for every input column
   `j` of the rounded matrix `Q`. That is a constraint a rounding step can target directly.

## Why nearest rounding loses the edit

The pack's weights already sit exactly on their quantization grid, with fixed per-block scales. After the edit, a
weight `W_ij` moves by `−r_i (rᵀW[:, j])`. Nearest rounding with the scales kept returns the weight to its old code
unless that move exceeds half a grid step.

How many weights cross depends on how the direction's energy is spread:

- **Spread evenly** (`|r_i| ≈ 1/√n` for all n = 2,304 rows): the move is about 2% of a typical weight, while half a
  step of a 3- or 4-bit grid is tens of percent of one. Essentially no weight crosses, and nearest rounding returns
  the original matrix: the fraction of the component left is 1.00. This is the first case in
  `patch/demo_synthetic.py` (1.0000 at 3 bits, 0.9775 at 4 bits).
- **Concentrated** (a few rows of `r` carry much of its norm): the weights in those rows move by more, some of them
  across half a step, so part of the edit survives. Real edit directions are like this. Measured on the pack, the
  fraction left after nearest rounding is 0.968 (S3_K), 0.836 (S4_K), 0.775 (SF4), 0.675 (S5_K), and about 0.1 on
  the 8-bit attention and embedding matrices, whose steps are much finer. The demo's second case, a
  heavy-tailed random `r` (Student-t, 2 degrees of freedom), gives 0.93 and 0.72 at 3 and 4 bits with the default
  seed; this number varies with how concentrated the draw is (0.51 to 0.93 at 3 bits over seeds 0–2).

So the 1.0000 of the uniform synthetic case is the limiting case, not a general figure: the loss depends on the
grid and on the direction. In every case measured, nearest rounding kept less than a third of the edit on the 3- to
5-bit formats. Nothing visibly breaks: the pack loads, the model runs, and it behaves as if it had not been edited.

## Direction-aware rounding

Work column by column (one input unit at a time), with every block scale held fixed. Let `t` be the edited target
column, `q` the rounded column, and `e = rᵀ(q − t)` the leftover component along `r`. Choose codes to

    minimise  Σ_i (q_i − t_i)²     subject to  |rᵀ(q − t)| ≈ 0,

with each weight at its nearest grid value or one step away. Starting from nearest rounding, moving weight `i` one
step `δ_i` (toward the side that reduces `|e|`) changes `e` by `b_i = r_i δ_i` (its *benefit*) and adds
`c_i = (q_i + δ_i − t_i)² − (q_i − t_i)²` to the squared error (its *cost*). Buying enough benefit to cancel `e` at
the least cost is a small knapsack problem; the greedy solution sorts the candidate moves by `c_i / |b_i|` and takes
them in order until `|e|` is smallest. Moves whose benefit alone would overshoot (`|b_i| ≥ 2|e|`) are excluded. Three
passes clean up what the first leaves (a pass changes the candidate set). The implementation is
`patch/rounding.greedy_round`; all columns are processed at once with array operations.

It needs no calibration data and no Hessian, because the constraint comes from the edit itself. It is the same
idea as adaptive-rounding methods in post-training quantization (choose codes, not nearest values, to minimise
an error), with a constraint supplied by the edit in place of calibration statistics.

Format details enter only through the grid: the list of representable values around each weight. For block-scaled
integers that is `(q − half) × scale`; for SF4 it is the E2M1 table times the block scale (including its signed
zero); for 8-bit rows it is `q × row_scale`. The embedding is constrained per row on the hidden side, so its
transpose is passed in.

### Measured cost

On all 3,072 experts and 25 trunk tensors of the finished pack, as fraction of the component along `r` left
(residual) and rounding noise added relative to the weight norm, `|Q − W'| / |W|`:

| Format | Sites | Residual (median / max) | Added noise (median / max) | Codes changed (sampled sites, mean) |
|---|---|---|---|---|
| SF4 | 1,538 | 0.00085 / 0.0019 | 0.031 / 0.039 | 7.7% |
| S3_K | 1,022 | 0.0056 / 0.0122 | 0.050 / 0.073 | 1.1% |
| S4_K | 510 | 0.0032 / 0.0040 | 0.035 / 0.039 | 1.7% |
| S5_K | 2 | 0.0014 / 0.0014 | 0.025 / 0.025 | 3.8% |
| Attention output, 8-bit rows | 20 | 0.00019 / 0.00027 | 0.0088 / 0.0096 | |
| Embedding, 8-bit rows | 1 | 0.00030 | 0.0071 | |

The four attention matrices stored in BF16 are rounded to nearest (a float format needs no special rounding):
residual 0.0058, noise 0.0016. For scale, the 3-bit format's own quantization error is about 0.25 of the weight
norm, so an added 0.05 raises it by about 2% in quadrature. AtomicChat's own Q4_K release, which was edited in BF16
and then quantized, has a residual of 0.05–0.10 on its expert matrices.

### Alternatives that were measured

On the same sampled sites (24–25 experts per format), against the same target:

| Format | Greedy (used) residual / noise | Sequential error feedback residual / noise |
|---|---|---|
| SF4 | 0.0009 / 0.031 | 0.0000 / 0.076 |
| S3_K | 0.0056 / 0.051 | 0.0000 / 0.186 |
| S4_K | 0.0031 / 0.035 | 0.0000 / 0.110 |
| S5_K | 0.0014 / 0.025 | 0.0000 / 0.063 |
| Attention, 8-bit rows | 0.0002 / 0.009 | 0.0000 / 0.014 |

Error feedback (`patch/rounding.ef_round`: process rows in order of decreasing `|r_i|`, pre-compensating each row
for the residual so far, as GPTQ does with the inverse Hessian `I − r rᵀ` restricted to the remaining rows) removes
the component exactly but adds two to four times the noise. A single-pass variant (round to nearest, then flip the
cheapest weights to their other neighbour once; `patch/rounding.single_pass_round`) was compared on 3 sites per
format: residual 2–10× higher than greedy at similar noise (SF4 0.0113 vs 0.0011; S3_K 0.0189 vs 0.0078; S4_K
0.0119 vs 0.0030; S5_K 0.0060 vs 0.0014), about three times faster. Greedy was chosen; speed was not the
constraint (the whole pack builds in under 9 minutes on four CPU cores).

### Clipping

1–5% of the edited targets fall outside the representable range of their block (SF4 5.5%, S3_K 4.5%, S4_K 1.8%,
S5_K 1.1% on the sample). They are clipped to the outermost level; the greedy pass reaches its residuals anyway.
For the 8-bit rows, clipping (scale bytes untouched) was compared with rescaling only the rows that overflow: noise
0.00830 vs 0.00828 on one attention matrix and 0.00722 vs 0.00725 on embedding slices, with equal or lower residual
for clipping, so the scales were left alone everywhere. Refitting the scales of 3- and 4-bit blocks might lower the
noise; that was not tried.

## What this does not cover

- One model, one pack, one edit: a rank-1 projection per layer. A rank-k edit turns the scalar constraint per column
  into a k-vector; the same greedy approach should extend (choose moves by cost per unit of reduction of `|e|`), but
  it is untested.
- The rounding sees each matrix in isolation. It does not use activation statistics, so it cannot tell which
  rounding errors matter more to the model's output; the KL measurements in `eval/RESULTS.md` are the check that the
  added noise is harmless here.
- Where the original edit does not fully remove a direction, neither does this.
