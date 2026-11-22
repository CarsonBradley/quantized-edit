"""test_rounding.py - checks for rounding.py and the synthetic demo. numpy only, a few seconds.

    python patch/test_rounding.py          (or: pytest patch/test_rounding.py)
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import demo_synthetic  # noqa: E402
from rounding import (CallbackGrid, TableGrid, UniformGrid, added_noise, greedy_round, project,  # noqa: E402
                      residual, single_pass_round)

E2M1 = np.array([-6, -4, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 2, 3, 4, 6], np.float64)


def test_demo_reproduces_the_effect():
    """Nearest rounding keeps most of the component along r; direction-aware rounding removes it at a few
    percent added noise. Thresholds are loose on purpose: this checks the qualitative claim, not the digits."""
    t0 = time.time()
    res = demo_synthetic.run()
    assert time.time() - t0 < 60
    for bits, d in res.items():
        assert d["nearest"]["residual"] > 0.9, (bits, d)
        assert d["direction_aware"]["residual"] < 0.02, (bits, d)
        assert d["direction_aware"]["noise"] < 0.08, (bits, d)
        assert d["direction_aware"]["changed"] < 0.1, (bits, d)


def test_demo_concentrated_direction():
    """With a heavy-tailed r a few weights move by more than half a step, so nearest rounding keeps less of the
    edit than in the uniform case, but still most of the component; direction-aware rounding removes it."""
    res = demo_synthetic.run(case="concentrated")
    uni = demo_synthetic.run(case="uniform")
    for bits, d in res.items():
        assert 0.3 < d["nearest"]["residual"] < uni[bits]["nearest"]["residual"], (bits, d)
        assert d["direction_aware"]["residual"] < 0.03, (bits, d)
        assert d["direction_aware"]["noise"] < 0.08, (bits, d)


def test_codes_stay_on_the_grid_with_scales_fixed():
    rng = np.random.default_rng(1)
    W0 = rng.standard_normal((256, 64))
    grid, q0, W = demo_synthetic.block_quantize(W0, 3, 32)
    r = rng.standard_normal(256)
    r /= np.linalg.norm(r)
    q, v = greedy_round(project(W, r), r, grid)
    assert q.min() >= 0 and q.max() <= 7
    assert np.allclose(v, (q.astype(np.float64) - 3.5) * grid.scale)
    _, v_again = grid.nearest(v)                         # every value is representable
    assert np.allclose(v_again, v)


def test_callback_grid_matches_uniform_grid():
    """The same grid described by a neighbour callback gives the same rounding."""
    rng = np.random.default_rng(2)
    W0 = rng.standard_normal((128, 48))
    ug, _, W = demo_synthetic.block_quantize(W0, 4, 16)
    scale = ug.scale

    def neighbours(x):
        # Decide in value space: floor(x / scale) alone is not exact one ulp away from a grid point, and
        # CallbackGrid.step asks for the neighbours of nextafter(v).
        x = np.asarray(x, np.float64)
        q = np.clip(np.floor(x / scale + 7.5), 0, 15)
        q = np.where((q < 15) & ((q + 1 - 7.5) * scale <= x), q + 1, q)
        q = np.where((q > 0) & ((q - 7.5) * scale > x), q - 1, q)
        lo = (q - 7.5) * scale
        hi = np.where(lo < x, np.minimum(q + 1, 15), q)
        return lo, (hi - 7.5) * scale

    cg = CallbackGrid(neighbours)
    r = rng.standard_normal(128)
    r /= np.linalg.norm(r)
    T = project(W, r)
    _, vu = greedy_round(T, r, ug)
    _, vc = greedy_round(T, r, cg)
    assert np.array_equal(np.asarray(vu, np.float32), np.asarray(vc, np.float32))


def test_table_grid_fp4_levels():
    """A non-uniform grid (FP4 E2M1 levels with one scale per 16 weights): same qualitative behaviour."""
    rng = np.random.default_rng(3)
    W0 = rng.standard_normal((1024, 128))
    amax = np.abs(W0).reshape(1024, 8, 16).max(axis=2)
    grid = TableGrid(E2M1, np.repeat(amax / 6.0, 16, axis=1))
    _, W = grid.nearest(W0)
    r = rng.standard_normal(1024)
    r /= np.linalg.norm(r)
    T = project(W, r)
    _, vn = grid.nearest(T)
    _, vg = greedy_round(T, r, grid)
    assert residual(vn, W, r) > 0.5
    assert residual(vg, W, r) < 0.05 and added_noise(vg, T, W) < 0.1


def test_single_pass_alternative_reduces_the_residual():
    rng = np.random.default_rng(4)
    W0 = rng.standard_normal((512, 64))
    grid, _, W = demo_synthetic.block_quantize(W0, 3, 32)
    r = rng.standard_normal(512)
    r /= np.linalg.norm(r)
    T = project(W, r)
    v = single_pass_round(T, W, r, grid)
    assert residual(v, W, r) < 0.1


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                fails += 1
                print(f"FAIL {name}: {e}")
    print(f"{'all passed' if not fails else f'{fails} failed'}")
    sys.exit(1 if fails else 0)
