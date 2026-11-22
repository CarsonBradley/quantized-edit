"""compare_ref.py - compare two --dump-ref files (same tokens) row by row.

usage: uv run --no-project --with numpy python -I compare_ref.py A.bin B.bin [--rows LO:HI]
File format (ink_main.cpp cmd_dump_ref): int32 n, int32 V, int32 tokens[n], float32 logits[n*V].
Prints one JSON object, over rows LO..HI-1 (default all):
  zero_rows_A/B   rows that are entirely 0.0 (the Metal 64-row logits bug produces these for rows >= 64)
  max_abs_diff    max |A - B|
  mean_kl/max_kl  KL(softmax A || softmax B) per row (A = reference), mean and worst
  top1_agree      fraction of rows with equal argmax
  bitwise_equal   compared rows byte-identical
  ppl_A/ppl_B     perplexity of the actual next token under A / B (rows with a next token)
Also zero_rows_A_from64 / zero_rows_B_from64 = all-zero rows among rows >= 64 of the whole file.
"""
import json
import sys

import numpy as np


def load(p):
    b = open(p, "rb").read()
    n, V = (int(x) for x in np.frombuffer(b[:8], dtype=np.int32))
    toks = np.frombuffer(b[8:8 + 4 * n], dtype=np.int32)
    lg = np.frombuffer(b[8 + 4 * n:], dtype=np.float32)
    assert lg.size == n * V, f"{p}: truncated ({lg.size} != {n}*{V})"
    return n, V, toks, lg.reshape(n, V)


def logsm(x):
    x = x.astype(np.float64)
    x = x - x.max(1, keepdims=True)
    return x - np.log(np.exp(x).sum(1, keepdims=True))


def main():
    a_path, b_path = sys.argv[1:3]
    na, Va, ta, A = load(a_path)
    nb, Vb, tb, B = load(b_path)
    assert (na, Va) == (nb, Vb) and np.array_equal(ta, tb), "different token sequences"
    lo, hi = 0, na
    if "--rows" in sys.argv:
        s = sys.argv[sys.argv.index("--rows") + 1].split(":")
        lo, hi = int(s[0] or 0), int(s[1] or na)
    zA64 = int(np.sum(np.all(A[64:] == 0, axis=1)))
    zB64 = int(np.sum(np.all(B[64:] == 0, axis=1)))
    A, B = A[lo:hi], B[lo:hi]
    la, lb = logsm(A), logsm(B)
    kl = (np.exp(la) * (la - lb)).sum(1)
    # perplexity of the actual next token (rows whose next token is inside the file)
    rows = np.arange(lo, hi)
    has_next = rows + 1 < na
    nxt = ta[rows[has_next] + 1]
    ppl_a = float(np.exp(-la[has_next, :][np.arange(int(has_next.sum())), nxt].mean())) if has_next.any() else None
    ppl_b = float(np.exp(-lb[has_next, :][np.arange(int(has_next.sum())), nxt].mean())) if has_next.any() else None
    res = {"A": a_path, "B": b_path, "rows": f"{lo}:{hi}", "n_rows": int(A.shape[0]),
           "zero_rows_A": int(np.sum(np.all(A == 0, axis=1))), "zero_rows_B": int(np.sum(np.all(B == 0, axis=1))),
           "zero_rows_A_from64": zA64, "zero_rows_B_from64": zB64,
           "max_abs_diff": float(np.max(np.abs(A.astype(np.float64) - B.astype(np.float64)))),
           "mean_kl": float(kl.mean()), "max_kl": float(kl.max()),
           "top1_agree": float(np.mean(A.argmax(1) == B.argmax(1))), "bitwise_equal": bool(np.array_equal(A, B)),
           "ppl_A": ppl_a, "ppl_B": ppl_b}
    print(json.dumps(res))


if __name__ == "__main__":
    main()
