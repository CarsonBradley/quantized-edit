"""lora_map.py - AtomicChat's rank-1 abliteration LoRA (GGUF adapter) -> Strata weight sites.

Everything that depends on the mapping goes through `site_delta()`, so a correction only needs to change this
file.  The mapping is checked end to end by codecs/test_codecs.py --lora (the LoRA's A vector of each sampled
site against -(r^T W) decoded from the pack, with a wrong-expert control).

  Strata expert gid = layer*128 + e   <->  blk.<layer>.ffn_down_exps  expert e
  Strata layers.<L>.attn.wo_ud        <->  blk.<L>.attn_output
  Strata embed                        <->  token_embd  (llama.cpp stores A/B flipped for token_embd)

Each site is returned as (u, v) with Delta = outer(u, v) in Strata's [rows, cols] orientation and
the LoRA scale (alpha / rank = 1.0) already applied, plus `r`: the unit 2304-vector the edit ablates.
"""
from __future__ import annotations

import numpy as np


class Lora:
    def __init__(self, path: str):
        import gguf
        rd = gguf.GGUFReader(path)
        self.t = {t.name: t for t in rd.tensors}
        alpha = float(rd.fields["adapter.lora_alpha"].contents())
        self.scale = alpha / 1.0          # rank 1

    def _arr(self, name):
        return np.asarray(self.t[name].data, dtype=np.float32)

    def site_delta(self, kind: str, layer: int | None = None, expert: int | None = None):
        """kind in {'expert', 'wo', 'embed'} -> (u, v, r).  Delta = scale * outer(u, v)."""
        if kind == "expert":
            B = self._arr(f"blk.{layer}.ffn_down_exps.weight.lora_b")[expert, :, 0]   # [2304]
            A = self._arr(f"blk.{layer}.ffn_down_exps.weight.lora_a")[expert, 0, :]   # [1152]
            u, v, r = B * self.scale, A, B
        elif kind == "wo":
            B = self._arr(f"blk.{layer}.attn_output.weight.lora_b")[:, 0]             # [2304]
            A = self._arr(f"blk.{layer}.attn_output.weight.lora_a")[0, :]             # [3072]
            u, v, r = B * self.scale, A, B
        elif kind == "embed":
            a = self._arr("token_embd.weight.lora_a")[:, 0]                           # [201024] per-token
            b = self._arr("token_embd.weight.lora_b")[:, 0]                           # [2304]
            u, v, r = a * self.scale, b, b
        else:
            raise ValueError(kind)
        r = r.astype(np.float64)
        r = r / np.linalg.norm(r)
        return u.astype(np.float64), v.astype(np.float64), r


def expert_site(gid: int):
    """Strata global expert id -> (layer, LoRA expert index)."""
    return gid // 128, gid % 128
