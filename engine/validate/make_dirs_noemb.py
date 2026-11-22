"""make_dirs_noemb.py - calibration variant of DIRS.bin with the embedding site effectively disabled.

usage: uv run --no-project --with numpy --with zstandard python -I engine/validate/make_dirs_noemb.py PACK.strata DIRS.bin OUT.bin
The loader requires unit rows, so the embedding row cannot be zeroed. Instead it is replaced by the eigenvector of
E^T E with the smallest eigenvalue (the direction along which the token-embedding table has the least energy), so the
projection removes almost nothing from any embedding row. The 24 layer rows are copied unchanged. Prints how much
of the real edit's per-row removal remains: rms(E v_min) / rms(E r_emb).
"""
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from strata_decode import Pack  # noqa: E402

pack, dirs_in, out = sys.argv[1:4]
P = Pack(pack)
b = bytearray(open(dirs_in, "rb").read())
D = 2304
r_emb = np.frombuffer(bytes(b[64 + 24 * D * 4:64 + 25 * D * 4]), dtype=np.float32).astype(np.float64)
t = P.T["model.llm.embed.weight"]
off = P.secs[1][0] + t["offset"]
N = 200058                                   # unpadded vocab: the rows that are ever looked up
sc = P.mm[off:off + t["shape"][0] * 4].view(np.float32)
q = P.mm[off + t["shape"][0] * 4: off + t["shape"][0] * 4 + t["shape"][0] * D].view(np.int8).reshape(-1, D)
G = np.zeros((D, D))
er = np.zeros(N)
for i in range(0, N, 16384):
    j = min(N, i + 16384)
    E = q[i:j].astype(np.float32) * sc[i:j, None]
    G += (E.T @ E).astype(np.float64)
    er[i:j] = E.astype(np.float64) @ r_emb
w, V = np.linalg.eigh(G)
v = V[:, 0] / np.linalg.norm(V[:, 0])
ev = np.sqrt(w[0] / N)
print(f"rms(E r_emb) = {np.sqrt((er ** 2).mean()):.5f}; rms(E v_min) = {ev:.6f}; ratio = {ev / np.sqrt((er ** 2).mean()):.4f}; "
      f"cos(v_min, r_emb) = {abs(v @ r_emb):.4f}")
b[64 + 24 * D * 4:64 + 25 * D * 4] = v.astype(np.float32).tobytes()
open(out, "wb").write(bytes(b))
assert len(b) == 64 + 25 * D * 4
print(f"wrote {out}")
