"""edit_delta.py - is the edit applied equally on two code paths?

usage: uv run --no-project --with numpy python -I edit_delta.py OFF_A.bin ON_A.bin OFF_B.bin ON_B.bin
A and B are --dump-ref files of the SAME tokens through two code paths (e.g. slice 64 and slice 1), with the flag
off and on. The edit's effect on the logits is D_A = ON_A - OFF_A and D_B = ON_B - OFF_B (log-softmax space).
If path B applies the full edit, D_B ~= D_A: correlation ~1 and norm ratio |D_B|/|D_A| ~1. A missed insertion point on
path B shows as ratio < 1 (and lower correlation). Also reports KL(OFF||ON) per path.
"""
import json
import sys

import numpy as np


def load(p):
    b = open(p, "rb").read()
    n, V = (int(x) for x in np.frombuffer(b[:8], dtype=np.int32))
    return np.frombuffer(b[8 + 4 * n:], dtype=np.float32).reshape(n, V).astype(np.float64)


def logsm(x):
    x = x - x.max(1, keepdims=True)
    return x - np.log(np.exp(x).sum(1, keepdims=True))


oa, na, ob, nb = (logsm(load(p)) for p in sys.argv[1:5])
da, db = na - oa, nb - ob
# compare per row on the top-200 tokens of the reference (where the probability mass is)
rows = []
for t in range(da.shape[0]):
    idx = np.argsort(-oa[t])[:200]
    x, y = da[t, idx] - da[t, idx].mean(), db[t, idx] - db[t, idx].mean()
    rows.append((float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30)), float(np.linalg.norm(y) / (np.linalg.norm(x) + 1e-30))))
c = np.array(rows)
kl = lambda p, q: float((np.exp(p) * (p - q)).sum(1).mean())
print(json.dumps({"rows": len(rows), "median_corr": float(np.median(c[:, 0])), "mean_corr": float(c[:, 0].mean()),
                  "median_norm_ratio_B_over_A": float(np.median(c[:, 1])),
                  "kl_edit_pathA": kl(oa, na), "kl_edit_pathB": kl(ob, nb),
                  "kl_off_A_vs_B": kl(oa, ob), "kl_on_A_vs_B": kl(na, nb)}))
