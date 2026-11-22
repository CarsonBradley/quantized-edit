"""make_dirs.py - build the direction file (DIRS.bin) for `strata-inkling --ablate` from AtomicChat's rank-1
abliteration LoRA.  CPU only.

usage: uv run --no-project --with gguf --with numpy --with zstandard python -I engine/make_dirs.py \
           LORA.gguf PACK.strata OUT.bin [REPORT.json]

File format (little-endian; read by engine/src/inkling2/metal/ink_ablate_dirs.h in the patched engine):
  off  0  char[8]  magic "INKABL\0\1"
  off  8  u32 x 6  version = 1, n_layers = 24, hidden = 2304, n_vec = 25, flags = 1 (last row is the embedding
                   direction), header_bytes = 64
  off 32  u8[32]   sha256 of the LoRA file (provenance only)
  off 64  f32[25][2304]  unit vectors: rows 0..23 = r_L per layer (shared by the attention output and all 128
                   expert down-projections of layer L), row 24 = r_emb (the token embedding)

Asserts, before writing anything:
  * adapter metadata: LoRA, alpha = 1, rank 1  -> llama.cpp scale = alpha / rank = 1  -> --ablate-scale 1 = the published edit
  * per layer: |lora_b(attn_output)| = 1 and all 128 lora_b(ffn_down_exps) equal it (cos > 1 - 1e-9)
  * lora_a = -W^T r on every site that can be decoded from the pack:
      - all 24 attn wo_ud (BF16 layers to 1e-5 relative, Q8_ROW layers to 3e-2 = Q8 noise),
      - expert down matrices, N_PER_CODE per down codec per layer (SF4 to 1e-2; S3/S4/S5_K up to their 4x storage
        scale and quantization noise, see below),
      - the token embedding (Q8_ROW, 3e-2 relative, cosine > 0.9999)
  * embedding site form: lora_a has one entry per vocab row (ne = [1, 201024]), lora_b is a unit 2304-vector
    (ne = [1, 2304]); llama.cpp build_inp_embd adds  scale * lora_b * get_rows(lora_a, tok)  to the gathered row,
    so with lora_a = -E r_emb that is  E' = E - (E r_emb) r_emb^T  : the same output projection, applied to the
    gathered embedding row before embed_norm.
"""
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from strata_decode import Lora, Pack  # noqa: E402

N_LAYERS, D, D_ATT, D_FF, N_EXP = 24, 2304, 3072, 1152, 128
N_PER_CODE = 4


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(a))


def cos(a, b):
    return float(a @ b / np.linalg.norm(a) / np.linalg.norm(b))


def main():
    lora_path, pack_path, out_path = sys.argv[1:4]
    report_path = sys.argv[4] if len(sys.argv) > 4 else None
    L = Lora(lora_path)
    P = Pack(pack_path)
    rep = {"lora": os.path.abspath(lora_path), "lora_sha256": L.sha256.hex(), "pack": os.path.abspath(pack_path),
           "layers": [], "embedding": {}}

    # ---- adapter metadata ----
    assert L.meta.get("general.type") == "adapter" and L.meta.get("adapter.type") == "lora", L.meta
    alpha = float(L.meta["adapter.lora_alpha"])
    assert alpha == 1.0, alpha
    assert L.ne["blk.0.attn_output.weight.lora_a"] == [D_ATT, 1], L.ne["blk.0.attn_output.weight.lora_a"]
    assert L.ne["blk.0.attn_output.weight.lora_b"] == [1, D], L.ne["blk.0.attn_output.weight.lora_b"]
    assert L.ne["blk.0.ffn_down_exps.weight.lora_a"] == [D_FF, 1, N_EXP]
    assert L.ne["blk.0.ffn_down_exps.weight.lora_b"] == [1, D, N_EXP]
    assert L.ne["token_embd.weight.lora_a"] == [1, 201024], L.ne["token_embd.weight.lora_a"]
    assert L.ne["token_embd.weight.lora_b"] == [1, D], L.ne["token_embd.weight.lora_b"]
    rank = 1
    rep["alpha"], rep["rank"], rep["llama_cpp_scale"] = alpha, rank, alpha / rank
    rep["meta"] = {k: str(v) for k, v in L.meta.items() if k.startswith(("atomic.", "adapter.", "general.name"))}

    dirs = np.zeros((N_LAYERS + 1, D), dtype=np.float64)
    for layer in range(N_LAYERS):
        a_att, b_att = L.attn(layer)
        a_exp, b_exp = L.experts(layer)
        nb = np.linalg.norm(b_att)
        assert abs(nb - 1.0) < 1e-6, (layer, nb)
        r = b_att / nb
        nbe = np.linalg.norm(b_exp, axis=1)
        assert np.all(np.abs(nbe - 1.0) < 1e-6), (layer, nbe.min(), nbe.max())
        ce = (b_exp @ r) / nbe
        assert ce.min() > 1 - 1e-9, (layer, ce.min())
        # attention: lora_a = -W^T r
        wname = f"model.llm.layers.{layer}.attn.wo_ud.weight"
        fmt = P.T[wname]["format"]
        W = P.trunk(wname)
        pred = -(W.T @ r)
        e_att = rel(a_att, pred)
        tol = 1e-5 if fmt == "BF16" else 3e-2
        assert e_att < tol and cos(a_att, pred) > 0.999, (layer, fmt, e_att)
        # experts: decode N_PER_CODE experts of every down codec present in the layer, keyed by expert_id.
        # SF4 is stored at the original scale (a = -W^T r to ~2e-3). S3/S4/S5_K down matrices are stored at 4x the
        # original scale (measured k = a.p/p.p = 0.24-0.25; the engine compensates elsewhere), so for those the check
        # is a = -k W^T r with k in [0.22, 0.27] and cosine >= 0.92 (S3, 3-bit noise) / 0.985 (S4/S5).
        # The output projection is invariant to that scale, so the direction r is what matters.
        e_exp = {}
        for code in sorted(set(P.expert_dn_code(layer, e) for e in range(N_EXP))):
            es = [e for e in range(N_EXP) if P.expert_dn_code(layer, e) == code][:N_PER_CODE]
            for e in es:
                We = P.expert_down(layer, e)
                pe = -(We.T @ r)
                k = float(a_exp[e] @ pe / (pe @ pe))
                c = cos(a_exp[e], pe)
                if code == 15:
                    er = rel(a_exp[e], pe)
                    assert er < 1e-2 and c > 0.9999, (layer, e, code, er)
                else:
                    assert 0.22 < k < 0.27 and c > (0.92 if code == 3 else 0.985), (layer, e, code, k, c)
                e_exp.setdefault(code, []).append((e, round(c, 6), round(k, 5)))
        dirs[layer] = r
        rep["layers"].append({"layer": layer, "wo_ud_format": fmt, "attn_rel_err_a_vs_-WTr": e_att,
                              "experts_b_min_cos_to_r": float(ce.min()),
                              "experts_checked_by_code (expert_id, cos, k)": {str(k_): v for k_, v in e_exp.items()}})
        print(f"L{layer:2d}: |r|=1, 128 experts share r (min cos {ce.min():.12f}); wo_ud {fmt:6s} a=-W^T r rel.err {e_att:.2e}; "
              "experts " + " ".join(f"code{c_}:n={len(v)},min_cos={min(x[1] for x in v):.4f},k={min(x[2] for x in v):.3f}..{max(x[2] for x in v):.3f}" for c_, v in e_exp.items()))

    # ---- embedding site ----
    a_emb, b_emb = L.embd()
    assert a_emb.size == 201024 and b_emb.size == D
    nb = np.linalg.norm(b_emb)
    assert abs(nb - 1.0) < 1e-6, nb
    r_emb = b_emb / nb
    E = P.trunk("model.llm.embed.weight")          # Q8_ROW [201024, 2304]
    pred = -(E @ r_emb)
    e_emb, c_emb = rel(a_emb, pred), cos(a_emb, pred)
    assert e_emb < 3e-2 and c_emb > 0.9999, (e_emb, c_emb)
    # padded vocab rows (>= 200058) are never looked up; report their a anyway
    rep["embedding"] = {"a_len": int(a_emb.size), "b_len": int(b_emb.size), "rel_err_a_vs_-E_r": e_emb, "cos": c_emb,
                        "form": "E' = E - (E r_emb) r_emb^T  (projection of each gathered row, before embed_norm)",
                        "max_abs_cos_r_emb_vs_layer_dirs": float(np.max(np.abs(dirs[:N_LAYERS] @ r_emb)))}
    dirs[N_LAYERS] = r_emb
    print(f"embed: lora_a[201024] = -E r_emb (rel.err {e_emb:.2e}, cos {c_emb:.6f}); lora_b = unit r_emb; "
          f"max |cos(r_emb, r_L)| = {rep['embedding']['max_abs_cos_r_emb_vs_layer_dirs']:.3f} (its own direction)")

    # ---- write DIRS.bin ----
    d32 = dirs.astype(np.float32)
    d32 /= np.linalg.norm(d32.astype(np.float64), axis=1, keepdims=True).astype(np.float32)
    norms = np.linalg.norm(d32.astype(np.float64), axis=1)
    assert np.all(np.abs(norms - 1) < 1e-6), norms
    hdr = b"INKABL\x00\x01" + struct.pack("<6I", 1, N_LAYERS, D, N_LAYERS + 1, 1, 64) + L.sha256
    assert len(hdr) == 64
    with open(out_path, "wb") as f:
        f.write(hdr)
        f.write(d32.tobytes(order="C"))
    size = os.path.getsize(out_path)
    assert size == 64 + (N_LAYERS + 1) * D * 4, size
    rep["out"] = os.path.abspath(out_path)
    rep["out_bytes"] = size
    print(f"wrote {out_path} ({size} bytes)")
    if report_path:
        json.dump(rep, open(report_path, "w"), indent=1)


if __name__ == "__main__":
    main()
