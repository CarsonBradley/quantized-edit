"""rounding.py - choose quantization codes for an edited weight matrix so that a rank-1 projection edit survives.

The problem. A weight matrix W already sits exactly on a quantization grid (its codes and block scales are
given). The edit is a projection W' = (I - r r^T) W: remove the component along a unit vector r from every
column. The per-weight change is usually far smaller than half a grid step, so rounding W' to the nearest
representable value returns almost every weight to its old code and the edit is lost.

What the edit has to achieve is one number per column: r^T Q[:, j] = 0 for the rounded matrix Q. This module
rounds to nearest and then, per column, moves the cheapest weights one grid step until that number is close to
zero (see docs/method.md). Scales are never changed; only codes are chosen.

Everything here is format-agnostic. The format enters through a *grid* object with two methods:

    grid.nearest(T)        -> (codes, values)   nearest representable value for every target in T
    grid.step(codes, d, T) -> (codes, values)   move every weight one representable value up (d = +1) or
                                                down (d = -1); a weight that cannot move stays where it is

`codes` can be any array the grid understands (integer codes, nibbles, or the float values themselves). Three
grids are provided:

    UniformGrid(scale, qmin, qmax, offset)  value = (q - offset) * scale, integer q in [qmin, qmax]
                                            (block-scaled integer formats: Q8_0-style, K-quant-style, ...)
    TableGrid(levels, scale)                value = levels[i] * scale, a sorted table of levels
                                            (non-uniform formats such as FP4 E2M1)
    CallbackGrid(neighbours)                neighbours(x) -> (lo, hi): the largest representable value <= x and
                                            the smallest representable value >= x, element-wise. Use this when
                                            the format is easier to describe by a function than by a table.

Bit-exact grids for the Strata codecs live in codecs/strata_codecs.py (SKGrid, SF4Grid, Q8Grid); they follow
the same protocol.

Functions
    project(W, r)                       W - r (r^T W)
    greedy_round(T, r, grid, passes=3)  direction-aware rounding (the method used for the published results)
    ef_round(T, r, nearest_row, dtype)  sequential error-feedback rounding (an alternative; noisier)
    single_pass_round(T, W, r, grid)    one pass of cheapest flips to the other neighbour (an alternative)
    residual(Q, W, r)                   |r^T Q| / |r^T W|   (0 = edit fully kept, 1 = edit lost)
    added_noise(Q, T, W)                |Q - T| / |W|       (rounding error relative to the weight norm)

Axis convention: axis 0 of T is the constrained axis (the axis r lives on). For an [out, in] matrix whose
output is projected, that is the output axis, and the constraint is one number per input column. For an
embedding table [tokens, hidden] projected on the hidden side, pass the transpose.
"""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------------------------------
# Grids
# ----------------------------------------------------------------------------------------------
class UniformGrid:
    """value = (q - offset) * scale with integer q in [qmin, qmax]; scale per element (broadcastable to T),
    assumed > 0. Ties in nearest() go to the lower code."""

    def __init__(self, scale, qmin: int, qmax: int, offset: float = 0.0, dtype=np.int16):
        self.scale = np.asarray(scale, np.float64)
        self.qmin, self.qmax, self.offset, self.dtype = int(qmin), int(qmax), float(offset), dtype

    def _vals(self, q):
        return ((q.astype(np.float64) - self.offset) * self.scale)

    def nearest(self, T):
        T = np.asarray(T, np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            x = T / self.scale + self.offset
        x = np.where(np.isfinite(x), x, 0.0)
        qa = np.clip(np.floor(x), self.qmin, self.qmax)
        qb = np.clip(qa + 1, self.qmin, self.qmax)
        va, vb = self._vals(qa), self._vals(qb)
        pick_b = np.abs(vb - T) < np.abs(va - T)
        q = np.where(pick_b, qb, qa).astype(self.dtype)
        return q, np.where(pick_b, vb, va)

    def step(self, q, d, T):
        q2 = np.clip(q.astype(np.int64) + d, self.qmin, self.qmax).astype(self.dtype)
        return q2, self._vals(q2)


class TableGrid:
    """value = levels[i] * scale for a sorted 1-D table of levels; codes are indices into the table.
    scale per element (broadcastable to T), assumed > 0. Ties in nearest() go to the lower index."""

    def __init__(self, levels, scale, dtype=np.int16):
        self.levels = np.asarray(levels, np.float64)
        assert np.all(np.diff(self.levels) > 0), "levels must be strictly increasing"
        self.scale = np.asarray(scale, np.float64)
        self.dtype = dtype

    def _vals(self, i):
        return self.levels[i] * self.scale

    def nearest(self, T):
        T = np.asarray(T, np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            x = T / self.scale
        x = np.where(np.isfinite(x), x, 0.0)
        hi = np.clip(np.searchsorted(self.levels, x, side="left"), 0, len(self.levels) - 1)
        lo = np.clip(hi - 1, 0, len(self.levels) - 1)
        vlo, vhi = self._vals(lo), self._vals(hi)
        pick_hi = np.abs(vhi - T) < np.abs(vlo - T)
        i = np.where(pick_hi, hi, lo).astype(self.dtype)
        return i, np.where(pick_hi, vhi, vlo)

    def step(self, i, d, T):
        i2 = np.clip(i.astype(np.int64) + d, 0, len(self.levels) - 1).astype(self.dtype)
        return i2, self._vals(i2)


class CallbackGrid:
    """A grid described by a function. neighbours(x) -> (lo, hi), element-wise: lo = largest representable
    value <= x, hi = smallest representable value >= x (lo == hi when x is representable; at the ends of the
    range return the end value for both). Codes are the float64 values themselves.

    The callback must be exact in value space: step() asks for the neighbours of nextafter(v), one ulp from a
    grid point, and an implementation based on floor(x / scale) alone can round that back onto v and report
    that the weight cannot move. Compare candidate values against x instead (see patch/test_rounding.py)."""

    def __init__(self, neighbours):
        self.neighbours = neighbours

    def nearest(self, T):
        T = np.asarray(T, np.float64)
        lo, hi = (np.asarray(a, np.float64) for a in self.neighbours(T))
        v = np.where(np.abs(hi - T) < np.abs(lo - T), hi, lo)
        return v.copy(), v

    def step(self, v, d, T):
        v = np.asarray(v, np.float64)
        up = np.asarray(self.neighbours(np.nextafter(v, np.inf))[1], np.float64)
        dn = np.asarray(self.neighbours(np.nextafter(v, -np.inf))[0], np.float64)
        v2 = np.where(d > 0, np.maximum(up, v), np.where(d < 0, np.minimum(dn, v), v))
        return v2.copy(), v2


# ----------------------------------------------------------------------------------------------
# Edit and metrics
# ----------------------------------------------------------------------------------------------
def project(W, r):
    """W' = (I - r r^T) W for a unit vector r on axis 0."""
    W = np.asarray(W, np.float64)
    r = np.asarray(r, np.float64)
    return W - np.outer(r, r @ W)


def residual(Q, W, r):
    """Fraction of the component along r that is left: |r^T Q| / |r^T W|. 0 = perfect edit, 1 = no edit."""
    r = np.asarray(r, np.float64)
    return float(np.linalg.norm(r @ np.asarray(Q, np.float64)) / np.linalg.norm(r @ np.asarray(W, np.float64)))


def added_noise(Q, T, W):
    """Rounding error against the edited target, relative to the original weights: |Q - T| / |W|."""
    Q, T, W = (np.asarray(x, np.float64) for x in (Q, T, W))
    return float(np.linalg.norm(Q - T) / np.linalg.norm(W))


# ----------------------------------------------------------------------------------------------
# Direction-aware rounding
# ----------------------------------------------------------------------------------------------
def greedy_round(T: np.ndarray, r: np.ndarray, grid, passes: int = 3, codes=None, vals=None):
    """Ablation-aware rounding with minimum added error.

    T [n_c, m]: targets, constraint axis first; r [n_c] unit vector.  Start from nearest rounding,
    then per column pick the set of weights to move one grid step in the helpful direction that
    drives s_j = r . (v_:j - T_:j) to ~0 at the least increase of sum (v - T)^2 (greedy knapsack on
    cost / |benefit|).  Every weight stays within one grid step of its nearest value.  A few passes
    clean up what the first leaves."""
    r = np.asarray(r, np.float64)
    T64 = np.asarray(T, np.float64)
    if codes is None:
        codes, vals = grid.nearest(T)
    for _ in range(passes):
        e = vals.astype(np.float64) - T64
        s = r @ e                                                     # [m]
        d = (-np.sign(s)[None, :] * np.sign(r)[:, None]).astype(np.int16)
        c2, v2 = grid.step(codes, d, T)
        b = r[:, None] * (v2.astype(np.float64) - vals.astype(np.float64))       # change of s if flipped
        cost = (v2.astype(np.float64) - T64) ** 2 - e ** 2
        useful = (b * s[None, :] < 0) & (np.abs(b) < 2.0 * np.abs(s)[None, :])   # a single flip must not overshoot
        ratio = np.where(useful, cost / np.where(useful, np.abs(b), 1.0), np.inf)
        order = np.argsort(ratio, axis=0, kind="stable")
        bs = np.take_along_axis(np.where(useful, b, 0.0), order, axis=0)
        cum = s[None, :] + np.cumsum(bs, axis=0)                      # residual after taking first k+1
        allres = np.vstack([s[None, :], cum])
        k = np.argmin(np.abs(allres), axis=0)                         # flips to take per column (0 = none)
        if not k.any():
            break
        rank = np.empty_like(order)
        np.put_along_axis(rank, order, np.arange(T.shape[0])[:, None].repeat(T.shape[1], 1), axis=0)
        flip = rank < k[None, :]
        codes = np.where(flip, c2, codes).astype(codes.dtype)
        vals = np.where(flip, v2, vals).astype(np.float32)
    return codes, vals


# ----------------------------------------------------------------------------------------------
# Error-feedback (ablation-aware) rounding.
# target: [n_c, m] where axis 0 is the constrained axis (the r axis).  nearest(i, t_row) returns
# (codes_row, values_row) for row i.  Keeps s = r . (values - target) per column near zero by
# processing rows in order of decreasing |r_i| and pre-compensating each row with
# t_i - s * r_i / sum_{k not yet done} r_k^2  (GPTQ with H^-1 = I - r r^T / |r|^2 restricted to the tail).
# ----------------------------------------------------------------------------------------------
def ef_round(target: np.ndarray, r: np.ndarray, nearest, code_dtype, damp: float = 1.0):
    n_c, m = target.shape
    r = r.astype(np.float64)
    order = np.argsort(-np.abs(r), kind="stable")
    tail = np.cumsum((r[order] ** 2)[::-1])[::-1]          # tail[k] = sum_{j>=k} r_order[j]^2
    s = np.zeros(m, np.float64)
    codes = np.empty((n_c, m), code_dtype)
    vals = np.empty((n_c, m), np.float32)
    for k, i in enumerate(order):
        ri = r[i]
        ti = target[i].astype(np.float64)
        if tail[k] > 0 and ri != 0.0:
            ti = ti - damp * s * (ri / tail[k])
        c, v = nearest(i, ti)
        codes[i] = c
        vals[i] = v
        s += ri * (v.astype(np.float64) - target[i].astype(np.float64))
    return codes, vals


# ----------------------------------------------------------------------------------------------
# Single-pass alternative: round to nearest, then per column flip the cheapest weights to their other
# neighbouring grid value until the realised component along r matches the target. Kept for comparison;
# on the real pack it left a 2-10x larger residual than greedy_round at similar noise (docs/method.md).
# ----------------------------------------------------------------------------------------------
def single_pass_select(t, near, alt, scale, b, W, Delta, valid_alt=None):
    """Inputs in grid units: t = target / scale, near/alt = nearest and other-neighbour grid points,
    scale = per-element step multiplier, b = unit direction along axis 0, Delta = target - W.
    Returns the realised float change (W_new - W) [rows, cols]."""
    real = near * scale - W
    rho = b @ (real - Delta)                          # residual b-component per column
    gain = b[:, None] * (alt - near) * scale          # change of rho if element flipped to alt
    cost = (np.abs(alt - t) - np.abs(near - t)) * scale
    if valid_alt is not None:
        gain = np.where(valid_alt, gain, 0.0)
    for j in range(t.shape[1]):
        g = gain[:, j]
        useful = np.nonzero(np.sign(g) == -np.sign(rho[j]))[0]
        if len(useful) == 0 or rho[j] == 0:
            continue
        ratio = cost[useful, j] / np.abs(g[useful])
        order = useful[np.argsort(ratio, kind="stable")]
        cum = np.cumsum(np.abs(g[order]))
        need = abs(rho[j])
        k = int(np.searchsorted(cum, need))
        if k < len(order):
            prev = cum[k - 1] if k > 0 else 0.0
            if abs(cum[k] - need) < abs(need - prev):
                k += 1
        sel = order[:k]
        real[sel, j] = alt[sel, j] * scale[sel, j] - W[sel, j]
    return real


def single_pass_round(T, W, r, grid):
    """single_pass_select in value units, with the nearest and other-neighbour values taken from `grid`.
    Returns the rounded values (codes are not tracked: map them back with grid.nearest)."""
    T = np.asarray(T, np.float64)
    W = np.asarray(W, np.float64)
    qn, vn = grid.nearest(T)
    d = np.where(T >= vn, 1, -1).astype(np.int16)
    qa, va = grid.step(qn, d, T)
    valid = np.asarray(va, np.float64) != np.asarray(vn, np.float64)
    real = single_pass_select(T, np.asarray(vn, np.float64), np.asarray(va, np.float64), np.ones(T.shape),
                              np.asarray(r, np.float64), W, T - W, valid)
    return W + real
