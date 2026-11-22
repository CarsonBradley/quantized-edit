"""survival.py - how well the rank-1 abliteration edit survives fixed-scale re-quantization.  CPU only.

  uv run --no-project --with numpy --with zstandard --with gguf python -I patch/survival.py \
      --pack Inkling-2-Mini-SQ-15.strata --lora LORA.gguf [--ac-gguf AC_Q4_K.gguf] [--per-code N] [--no-embed] \
      [--out results/survival.json]

For each sampled site (expert down matrices per codec, all 24 wo_ud, the embedding):
  W_old   decoded effective weights from the pack
  r       unit ablation direction (= normalised lora_b)
  targets
    lora  : W_old + outer(u, v)                 (the LoRA exactly as published, computed on BF16 weights)
    proj  : W_old - r (r^T W_old)               (the same projection re-derived on the pack's own weights)
  strategies
    lora_float   : lora target, no re-quantization (only the ablation number is meaningful)
    lora_nearest : lora target, fixed scales, nearest code
    proj_nearest : proj target, fixed scales, nearest code
    proj_greedy  : proj target, fixed scales, direction-aware rounding (rounding.greedy_round)
    proj_ef      : proj target, fixed scales, error-feedback rounding (rounding.ef_round)
  metrics (Delta = the strategy's intended delta, dW = W_new - W_old)
    keep      <dW, Delta> / |Delta|^2            fraction of the intended delta that survives
    off       |dW - keep*Delta| / |Delta|        energy off the intended delta, relative to |Delta|
    abl       |r^T W_new| / |r^T W_old|          (embed: |W_new r| / |W_old r|); 0 = perfect ablation
    abl16     same with the per-weight fp16 rounding the prefill kernels apply
    changed   fraction of weights whose code changed
    clip      fraction of target values outside the block's representable range
    noise     |W_new - target| / |W_old|          rounding noise added by the edit, relative to the weight norm
Calibration ("ac_q5", optional, --ac-gguf): AtomicChat's shipped Q4_K GGUF (whose ffn_down_exps are Q5_0,
attn_output Q8_0) was abliterated in BF16 and then quantized; |r^T W_ac| / |r^T W_old| on the same sites is the
residual of a quantized model whose edit is known to work.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "codecs"))
import strata_codecs as sc  # noqa: E402
from lora_map import Lora, expert_site  # noqa: E402


STRATS = (("lora_nearest", None, None), ("proj_nearest", None, None), ("proj_greedy", None, "greedy"),
          ("proj_ef", None, "ef"))
ALL = ("lora_float", "lora_nearest", "proj_nearest", "proj_greedy", "proj_ef", "ac_q5", "ac_q8")


def metrics(W_old, W_new, target, delta, r, axis, codes_old=None, codes_new=None, lo=None, hi=None,
            W_new16=None):
    dW = (W_new - W_old)
    dd = float((delta * delta).sum())
    keep = float((dW * delta).sum() / dd) if dd > 0 else float("nan")
    off = float(np.sqrt(max(((dW - keep * delta) ** 2).sum(), 0.0) / dd)) if dd > 0 else float("nan")
    if axis == 0:
        p_old, p_new = r @ W_old, r @ W_new
        p16 = r @ W_new16 if W_new16 is not None else None
    else:
        p_old, p_new = W_old @ r, W_new @ r
        p16 = W_new16 @ r if W_new16 is not None else None
    out = dict(keep=keep, off=off,
               abl=float(np.linalg.norm(p_new) / np.linalg.norm(p_old)),
               noise=float(np.linalg.norm(W_new - target) / np.linalg.norm(W_old)))
    if p16 is not None:
        out["abl16"] = float(np.linalg.norm(p16) / np.linalg.norm(p_old))
    if codes_old is not None:
        out["changed"] = float((codes_old != codes_new).mean())
    if lo is not None:
        out["clip"] = float(((target < lo) | (target > hi)).mean())
    return out


def ac_expert(ac, layer, e):
    import gguf
    t = ac[f"blk.{layer}.ffn_down_exps.weight"]
    return gguf.quants.dequantize(np.asarray(t.data[e]), t.tensor_type).astype(np.float64)   # [2304, 1152]


def ac_wo(ac, layer):
    import gguf
    t = ac[f"blk.{layer}.attn_output.weight"]
    return gguf.quants.dequantize(np.asarray(t.data), t.tensor_type).astype(np.float64)       # [2304, 3072]


def run_expert(pk, lo_, gid, ac=None):
    L, e = expert_site(gid)
    u, v, r = lo_.site_delta("expert", L, e)
    W_old = pk.decode_expert_down(gid, space="eff").astype(np.float64)
    dn = pk.codes(gid)[1]
    _, q_old = pk.decode_expert_down_codes(gid)
    lo, hi = pk.down_grid_range(gid, space="eff")
    dL = np.outer(u, v)
    dP = -np.outer(r, r @ W_old)
    res = {"gid": gid, "code": sc.CODE_NAME[dn], "layer": L}
    res["lora_float"] = metrics(W_old, W_old + dL, W_old + dL, dL, r, 0)
    for name, delta, meth in STRATS:
        delta = dL if name.startswith("lora") else dP
        T = W_old + delta
        t0 = time.time()
        b, Wn = pk.encode_expert_down(gid, T, space="eff", r=(r if meth else None), method=meth or "greedy",
                                      return_values=True)
        dt = time.time() - t0
        _, q_new = pk.decode_expert_down_codes(gid, b)
        Wn16 = pk.decode_expert_down(gid, space="eff", fp16=True, raw=b).astype(np.float64)
        # sanity: the returned values equal a fresh decode of the bytes
        assert np.array_equal(pk.decode_expert_down(gid, space="eff", raw=b), Wn)
        res[name] = metrics(W_old, Wn.astype(np.float64), T, delta, r, 0, q_old, q_new, lo, hi, Wn16)
        res[name]["sec"] = dt
    if ac is not None:
        Wac = ac_expert(ac, L, e)
        res["ac_q5"] = dict(abl=float(np.linalg.norm(r @ Wac) / np.linalg.norm(r @ W_old)),
                            rel_diff=float(np.linalg.norm(Wac - (W_old + dP)) / np.linalg.norm(W_old)))
    return res


def run_wo(pk, lo_, L, ac=None):
    name = sc.wo_name(L)
    u, v, r = lo_.site_delta("wo", L)
    W_old = pk.decode_trunk(name).astype(np.float64)
    fmt = pk.trunk_format(name)
    dL = np.outer(u, v)
    dP = -np.outer(r, r @ W_old)
    res = {"site": f"wo{L}", "code": fmt, "layer": L}
    res["lora_float"] = metrics(W_old, W_old + dL, W_old + dL, dL, r, 0)
    if fmt == "Q8_ROW":
        _, q_old = pk.trunk_q8(name)
        lo, hi = pk.trunk_grid_range(name)
        lo, hi = lo[:, None], hi[:, None]
    for nm, _, meth in STRATS:
        if fmt != "Q8_ROW" and meth:
            continue
        delta = dL if nm.startswith("lora") else dP
        T = W_old + delta
        b, Wn = pk.encode_trunk(name, T, r=(r if meth else None), method=meth or "greedy", return_values=True)
        Wn = Wn.astype(np.float64)
        Wn16 = Wn.astype(np.float16).astype(np.float64)
        if fmt == "Q8_ROW":
            N, K = W_old.shape
            q_new = np.frombuffer(b, np.int8)[4 * N:].reshape(N, K)
            res[nm] = metrics(W_old, Wn, T, delta, r, 0, np.asarray(q_old), q_new, lo, hi, Wn16)
        else:
            old_bits = pk._raw(*pk.trunk_span(name)).view("<u2")
            res[nm] = metrics(W_old, Wn, T, delta, r, 0, np.asarray(old_bits), np.frombuffer(b, "<u2"), None, None, Wn16)
    if ac is not None:
        Wac = ac_wo(ac, L)
        res["ac_q8"] = dict(abl=float(np.linalg.norm(r @ Wac) / np.linalg.norm(r @ W_old)))
    return res


def run_embed(pk, lo_, chunk=16384):
    """Embed edit is per row (E' = E - (E r) r^T); rows are independent, so work in row chunks."""
    u, v, r = lo_.site_delta("embed")
    N = 201024
    sc_all, q_all = pk.trunk_q8(sc.EMBED)
    acc = {k: dict(dd=0.0, dWd=0.0, dW2=0.0, pnew=0.0, p16=0.0, pold=0.0, noise=0.0, wold=0.0, chg=0, clip=0, n=0)
           for k in ("lora_float", "lora_nearest", "proj_nearest", "proj_greedy", "proj_ef")}
    for a in range(0, N, chunk):
        b_ = min(N, a + chunk)
        W_old = pk.decode_trunk(sc.EMBED, rows=slice(a, b_)).astype(np.float64)
        p_old = W_old @ r
        dL = np.outer(u[a:b_], v)
        dP = -np.outer(p_old, r)
        scl = np.asarray(sc_all[a:b_], np.float32)
        lo, hi = -127.0 * scl[:, None], 127.0 * scl[:, None]
        qo = np.asarray(q_all[a:b_])
        for nm, delta, mode in (("lora_float", dL, None), ("lora_nearest", dL, "n"), ("proj_nearest", dP, "n"),
                                ("proj_greedy", dP, "g"), ("proj_ef", dP, "ef")):
            T = W_old + delta
            if mode is None:
                Wn, qn = T, qo
            elif mode == "n":
                qn, Wn = sc.nearest_q8(T, np.broadcast_to(scl[:, None], T.shape))
            elif mode == "g":
                qt, vt = sc.greedy_round(np.ascontiguousarray(T.T), r, sc.Q8Grid(np.broadcast_to(scl[None, :], (T.shape[1], T.shape[0]))))
                qn, Wn = qt.T, vt.T
            else:
                qt, vt = sc.ef_round(np.ascontiguousarray(T.T), r, lambda i, tt: sc.nearest_q8(tt, scl), np.int8)
                qn, Wn = qt.T, vt.T
            Wn = np.asarray(Wn, np.float64)
            dW = Wn - W_old
            s = acc[nm]
            s["dd"] += float((delta * delta).sum())
            s["dWd"] += float((dW * delta).sum())
            s["dW2"] += float((dW * dW).sum())
            s["pnew"] += float(((Wn @ r) ** 2).sum())
            s["p16"] += float(((Wn.astype(np.float16).astype(np.float64) @ r) ** 2).sum())
            s["pold"] += float((p_old ** 2).sum())
            s["noise"] += float(((Wn - T) ** 2).sum())
            s["wold"] += float((W_old ** 2).sum())
            s["chg"] += int((qn != qo).sum())
            s["clip"] += int(((T < lo) | (T > hi)).sum())
            s["n"] += T.size
    out = {"site": "embed", "code": "Q8_ROW"}
    for nm, s in acc.items():
        keep = s["dWd"] / s["dd"]
        off2 = s["dW2"] - 2 * keep * s["dWd"] + keep * keep * s["dd"]
        out[nm] = dict(keep=keep, off=float(np.sqrt(max(off2, 0) / s["dd"])), abl=float(np.sqrt(s["pnew"] / s["pold"])),
                       abl16=float(np.sqrt(s["p16"] / s["pold"])), noise=float(np.sqrt(s["noise"] / s["wold"])),
                       changed=s["chg"] / s["n"], clip=s["clip"] / s["n"])
    return out


def summarize(rows, key):
    by = {}
    for rw in rows:
        by.setdefault(rw["code"], []).append(rw)
    out = {}
    for code, rr in by.items():
        d = {"n": len(rr)}
        for strat in ALL:
            ms = [x[strat] for x in rr if strat in x]
            if not ms:
                continue
            d[strat] = {m: dict(mean=float(np.mean([x[m] for x in ms])), max=float(np.max([x[m] for x in ms])),
                                min=float(np.min([x[m] for x in ms])))
                        for m in ms[0] if m != "sec"}
        out[code] = d
    return out


def main():
    ap = argparse.ArgumentParser(description="measure how much of the edit survives each rounding strategy")
    ap.add_argument("--pack", required=True, help="the original Inkling-2-Mini-SQ-15.strata")
    ap.add_argument("--lora", required=True, help="the rank-1 abliteration LoRA (.gguf)")
    ap.add_argument("--ac-gguf", help="optional: AtomicChat's edited Q4_K GGUF, for the calibration column")
    ap.add_argument("--per-code", type=int, default=24, help="sampled experts per down codec")
    ap.add_argument("--no-embed", action="store_true", help="skip the embedding table")
    ap.add_argument("--out", default=os.path.join("results", "survival.json"))
    a = ap.parse_args()
    per_code = a.per_code
    pk = sc.StrataPack(a.pack)
    lo_ = Lora(a.lora)
    ac = None
    if a.ac_gguf:
        import gguf
        ac = {t.name: t for t in gguf.GGUFReader(a.ac_gguf).tensors}
    rng = np.random.default_rng(0)
    gids = []
    for code in (15, 3, 4):
        g = np.array([i for i in range(3072) if pk.codes(i)[1] == code])
        gids += [int(x) for x in np.sort(rng.choice(g, min(per_code, len(g)), replace=False))]
    gids += [i for i in range(3072) if 5 in pk.codes(i) and i not in gids]
    rows = []
    t0 = time.time()
    for k, gid in enumerate(gids):
        rows.append(run_expert(pk, lo_, gid, ac))
        if k % 10 == 0:
            print(f"expert {k + 1}/{len(gids)} gid={gid} {rows[-1]['code']} "
                  f"abl lora_n={rows[-1]['lora_nearest']['abl']:.3f} proj_n={rows[-1]['proj_nearest']['abl']:.3f} "
                  f"proj_greedy={rows[-1]['proj_greedy']['abl']:.4f} noise={rows[-1]['proj_greedy']['noise']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    trunk = [run_wo(pk, lo_, L, ac) for L in range(24)]
    print("wo done", flush=True)
    emb = run_embed(pk, lo_) if not a.no_embed else None
    out = dict(experts=rows, wo=trunk, embed=emb,
               summary=dict(experts=summarize(rows, "code"), wo=summarize(trunk, "code")))
    fn = a.out
    os.makedirs(os.path.dirname(os.path.abspath(fn)), exist_ok=True)
    json.dump(out, open(fn, "w"), indent=1)
    print("wrote", fn)
    for grp, sm in out["summary"].items():
        for code, d in sm.items():
            print(f"== {grp} {code} n={d['n']}")
            for strat in ALL:
                if strat in d:
                    print(f"   {strat:13s} " + " ".join(f"{m}={v['mean']:.4g}(max {v['max']:.3g})" for m, v in d[strat].items()))
    if emb:
        print("== embed")
        for strat in ("lora_float", "lora_nearest", "proj_nearest", "proj_greedy", "proj_ef"):
            print(f"   {strat:13s} " + " ".join(f"{m}={v:.4g}" for m, v in emb[strat].items()))


if __name__ == "__main__":
    main()
