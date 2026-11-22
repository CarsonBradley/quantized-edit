"""demo_synthetic.py - the headline result on a random matrix, with numpy only. Runs in a few seconds.

    python patch/demo_synthetic.py [--rows 2304] [--cols 512] [--block 32] [--seed 0]

1. Draw a random matrix W0 and quantize it to a block-scaled integer grid (3-bit and 4-bit, one scale per block of
   `block` consecutive weights in a row, symmetric levels (q - half) * scale as in K-quant-style formats).
   Call the result W: it sits exactly on its grid, like the weights in an already-quantized model file.
2. Apply a rank-1 projection W' = (I - r r^T) W along a unit vector r, in two cases:
     uniform r       Gaussian entries: the energy of r is spread evenly over all rows
     concentrated r  Student-t entries with 2 degrees of freedom: a few rows carry much of r, as in the real
                     edit's directions (see docs/method.md for why this matters for nearest rounding)
3. Re-encode W' with every scale held fixed, two ways:
     nearest          : each weight to its nearest representable value
     direction-aware  : nearest, then patch.rounding.greedy_round (cheapest one-step moves per column until
                        r^T Q[:, j] is close to 0)
4. Print, for each:
     residual  |r^T Q| / |r^T W|   fraction of the component along r that is left (0 = edit kept, 1 = edit lost)
     noise     |Q - W'| / |W|      rounding error added, relative to the weight norm
     changed   fraction of codes that differ from W's codes

For scale, it also prints the size of the edit |W' - W| / |W| and the grid's own quantization error
|W - W0| / |W0|.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rounding import UniformGrid, added_noise, greedy_round, project, residual  # noqa: E402

FORMATS = {3: 3.5, 4: 7.5}          # bits -> half-range: levels (q - half) for q in 0 .. 2*half


def block_quantize(W0: np.ndarray, bits: int, block: int):
    """Block-scaled symmetric integer quantization along axis 1. Returns (grid, codes, values)."""
    half = FORMATS[bits]
    rows, cols = W0.shape
    assert cols % block == 0
    amax = np.abs(W0).reshape(rows, cols // block, block).max(axis=2)
    scale = np.repeat(amax / half, block, axis=1)                 # extremes land on the outermost levels
    grid = UniformGrid(scale, 0, int(2 * half), offset=half)
    q, v = grid.nearest(W0)
    return grid, q, v


def run(rows=2304, cols=512, block=32, seed=0, bits=(3, 4), case="uniform"):
    rng = np.random.default_rng(seed)
    W0 = rng.standard_normal((rows, cols))
    r = rng.standard_normal(rows)
    if case == "concentrated":
        r = rng.standard_t(2, rows)
    elif case != "uniform":
        raise ValueError(case)
    r /= np.linalg.norm(r)
    out = {}
    for b in bits:
        grid, q0, W = block_quantize(W0, b, block)
        T = project(W, r)                                          # the edit, in float
        qn, vn = grid.nearest(T)                                   # nearest, scales fixed
        t0 = time.time()
        qg, vg = greedy_round(T, r, grid, passes=3)                # direction-aware, scales fixed
        dt = time.time() - t0
        out[b] = {
            "max_r2": float(np.max(r ** 2)),
            "quant_error": added_noise(W, W0, W0),
            "edit_size": added_noise(T, W, W),
            "nearest": {"residual": residual(vn, W, r), "noise": added_noise(vn, T, W),
                        "changed": float((qn != q0).mean())},
            "direction_aware": {"residual": residual(vg, W, r), "noise": added_noise(vg, T, W),
                                "changed": float((qg != q0).mean()), "seconds": dt},
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", type=int, default=2304, help="size of the constrained (output) axis")
    ap.add_argument("--cols", type=int, default=512)
    ap.add_argument("--block", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    print(f"random {a.rows} x {a.cols} matrix, Gaussian entries, block size {a.block}, seed {a.seed}")
    for case in ("uniform", "concentrated"):
        res = run(a.rows, a.cols, a.block, a.seed, case=case)
        print(f"\n=== {case} r (largest share of |r|^2 in one row: {res[3]['max_r2']:.3f})")
        for b, d in res.items():
            print(f"{b}-bit grid: own quantization error |W - W0|/|W0| = {d['quant_error']:.3f}; "
                  f"edit size |W' - W|/|W| = {d['edit_size']:.4f}")
            print(f"  {'rounding':17s} {'residual':>9s} {'noise':>8s} {'changed':>8s}")
            for k in ("nearest", "direction_aware"):
                m = d[k]
                print(f"  {k.replace('_', '-'):17s} {m['residual']:9.4f} {m['noise']:8.4f} {m['changed']:8.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
