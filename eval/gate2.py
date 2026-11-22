"""Gate 2: "not broken" against the original SQ-15.

(a) Running-text KL / top-1 on WikiText-2 (40 windows x 512 tokens = 20,480 positions).
    The engine's own teacher-forced tools are used for the forward passes:
      strata-inkling --dump-ref OUT.bin --native PACK --prompt-ids a,b,c   (full fp32 logits, every position)
      strata-inkling --kld-eval REF.bin --native PACK                       (exact KL vs a full-logit ref file)
    A full-logit reference is 512 x 200,058 x 4 B = 410 MB per window (16 GB for all 40), so the stored
    reference keeps, per position, the original's top-K (K=256) token ids + log-probs, its argmax and
    its log-prob of the actual next token. A candidate is scored by dumping its logits per window
    (temporary file, deleted after reduction) and computing
      KL_lumped = sum_{v in topK} p_v (log p_v - log q_v) + p_tail log(p_tail / q_tail)
    which is the exact KL of the distributions with the non-top-K tail merged into one bucket:
    a lower bound on the true KL (data-processing inequality), tight when the tail mass is small.
    Window 0's full-logit dump is also kept as an engine-format ref so `--kld-eval` gives an exact
    engine-computed KL on 512 positions as a cross-check.
(b) Coherence: temperature-0 outputs on the model folder's tools/prompts/p_short, p_4k, p_28k (reasoning at template default),
    stored as references; a candidate is compared by first-divergence position and token agreement,
    and its text is written out for reading.
"""
import json, os, subprocess, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.common import DATA_DIR, REFS_DIR, ROOT, STRATA_BIN, Renderer, assert_guarded, log, check_ablate_marker, read_from

# The stock 0.1.65 Metal backend returns all-zero logits for rows >= 64 of any batch that asks for
# more than one logit row (runner computes min(M, 64) rows) -- confirmed on the GPU 2026-11-22
# (128-token --dump-ref: rows 64..127 all zero). A build with engine/kldeval-64row-fix.patch slices such batches
# into 64-token sub-batches; it is bitwise identical to stock on rows 0..63 and deterministic run to run.
# Reference and candidate dumps must come from binaries with the fix; every dump is checked for zero rows, so the
# default (the stock binary) fails loudly rather than silently. Set KL_BIN to the fixed build.
KL_BIN = os.environ.get("KL_BIN") or STRATA_BIN

K = 256
DUMP_CTX = 1024
REF_DIR = os.path.join(REFS_DIR, "sq15")
PROMPTS = {"p_short": "tools/prompts/p_short.txt", "p_4k": "tools/prompts/p_4k.txt", "p_28k": "tools/prompts/p_28k.txt"}
COHERENCE_MAX_NEW = 512


def windows():
    return json.load(open(os.path.join(DATA_DIR, "wikitext2_windows.json")))["windows"]


def dump_logits(pack, ids, path, log_path, binary=None, extra=()):
    assert_guarded()
    cmd = [binary or KL_BIN, "--dump-ref", path, "--native", pack, "--max-context", str(DUMP_CTX),
           "--prompt-ids", ",".join(map(str, ids)), *extra]
    with open(log_path, "a") as lf:
        off = lf.tell()
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=lf, text=True, timeout=900)
    check_ablate_marker(extra, read_from(log_path, off), "dump-ref")
    if r.returncode != 0:
        raise RuntimeError(f"dump-ref failed rc={r.returncode}: {r.stdout[-500:]}")
    with open(path, "rb") as f:
        n, V = np.frombuffer(f.read(8), dtype=np.int32)
        got_ids = np.frombuffer(f.read(4 * n), dtype=np.int32)
        lg = np.frombuffer(f.read(), dtype=np.float32).reshape(n, V)
    assert list(got_ids) == list(ids), "dump ids mismatch"
    zero = np.flatnonzero(~lg.any(axis=1))
    if zero.size:
        raise RuntimeError(f"{zero.size} all-zero logit rows (first {zero[0]}): this binary has the 64-row "
                           f"--dump-ref bug; set KL_BIN to a build with engine/kldeval-64row-fix.patch")
    return lg


def log_softmax_rows(lg, rows=64):
    out = np.empty(lg.shape, dtype=np.float32)
    for i in range(0, lg.shape[0], rows):
        x = lg[i:i + rows].astype(np.float64)
        m = x.max(1, keepdims=True)
        out[i:i + rows] = (x - m - np.log(np.exp(x - m).sum(1, keepdims=True))).astype(np.float32)
    if not np.isfinite(out).all():
        raise RuntimeError("non-finite logits")
    return out


def reduce_ref(lg, ids):
    lp = log_softmax_rows(lg)
    top = np.argpartition(-lp, K, axis=1)[:, :K]
    tlp = np.take_along_axis(lp, top, 1)
    order = np.argsort(-tlp, axis=1)
    top, tlp = np.take_along_axis(top, order, 1), np.take_along_axis(tlp, order, 1)
    nxt = np.asarray(ids[1:])
    nll = -lp[np.arange(len(ids) - 1), nxt]
    return top.astype(np.int32), tlp.astype(np.float32), top[:, 0].astype(np.int32), nll.astype(np.float32)


def kl_lumped(ref_top, ref_tlp, cand_lp):
    p = np.exp(ref_tlp.astype(np.float64))
    q_lp = np.take_along_axis(cand_lp, ref_top, 1).astype(np.float64)
    kl = (p * (ref_tlp - q_lp)).sum(1)
    pt = np.clip(1 - p.sum(1), 0, 1)
    qt = np.clip(1 - np.exp(q_lp).sum(1), 1e-30, 1)
    kl += np.where(pt > 1e-12, pt * np.log(np.maximum(pt, 1e-30) / qt), 0.0)
    return np.maximum(kl, 0.0), pt


def kl_exact(ref_lp, cand_lp, rows=64):
    out = []
    for i in range(0, ref_lp.shape[0], rows):
        a = ref_lp[i:i + rows].astype(np.float64)
        out.append((np.exp(a) * (a - cand_lp[i:i + rows].astype(np.float64))).sum(1))
    return np.concatenate(out)


def build_ref(pack, tmp_dir, binary=None):
    """Record the original pack's reference (run once, ~10 min of GPU)."""
    os.makedirs(REF_DIR, exist_ok=True)
    W = windows()
    tops, tlps, top1s, nlls = [], [], [], []
    tmp = os.path.join(tmp_dir, "dump.bin")
    t0 = time.time()
    for w, ids in enumerate(W):
        lg = dump_logits(pack, ids, tmp, os.path.join(tmp_dir, "dump.stderr"), binary)
        top, tlp, top1, nll = reduce_ref(lg, ids)
        tops.append(top); tlps.append(tlp); top1s.append(top1); nlls.append(nll)
        if w == 0:
            os.replace(tmp, os.path.join(REF_DIR, "kld_ref_w0.bin"))     # engine-format full ref (410 MB)
            validate_truncation(lg, ids)
        log(f"  ref window {w+1}/{len(W)}: tail mass mean {1-np.exp(tlp).sum(1).mean():.2e}, ppl {np.exp(nll.mean()):.2f}")
    if os.path.exists(tmp): os.remove(tmp)
    np.savez_compressed(os.path.join(REF_DIR, "kl_ref.npz"), top=np.stack(tops), tlp=np.stack(tlps),
                        top1=np.stack(top1s), nll=np.stack(nlls), K=K)
    meta = {"pack": pack, "pack_size": os.path.getsize(pack), "engine": binary or KL_BIN, "windows": len(W),
            "window_len": len(W[0]), "K": K, "dump_ctx": DUMP_CTX, "built": time.strftime("%Y-%m-%d %H:%M"),
            "seconds": round(time.time() - t0), "ref_ppl": float(np.exp(np.concatenate(nlls).mean()))}
    json.dump(meta, open(os.path.join(REF_DIR, "kl_ref_meta.json"), "w"), indent=1)
    meta["serve_crosscheck"] = serve_crosscheck(pack, W[0], np.stack(top1s)[0], tmp_dir)
    json.dump(meta, open(os.path.join(REF_DIR, "kl_ref_meta.json"), "w"), indent=1)
    log(f"gate2 reference written to {REF_DIR} in {time.time()-t0:.0f}s, ref PPL {meta['ref_ppl']:.3f}")
    return meta


def serve_crosscheck(pack, ids, top1, tmp_dir, ks=(64, 65, 100, 200, 300, 400, 512)):
    """Independent check of dump rows (incl. >= 64): the stock engine's serve path (last-row logits only)
    must pick the same greedy next token as the dump's argmax at row k-1."""
    from lib.common import StrataServer
    out = []
    with StrataServer(pack, ctx=DUMP_CTX, parallel=1, stderr_path=os.path.join(tmp_dir, "serve_check.stderr")) as S:
        for k in ks:
            toks = S.gen(ids[:k], 1)[0]
            out.append({"k": k, "serve": toks[0] if toks else None, "dump_argmax": int(top1[k - 1])})
    ok = sum(o["serve"] == o["dump_argmax"] for o in out)
    log(f"  serve cross-check: {ok}/{len(out)} greedy next tokens match dump argmax: {out}")
    return {"match": ok, "n": len(out), "detail": out}


def validate_truncation(lg, ids):
    """CPU-only check of the top-K lumped KL against exact KL, using synthetic perturbations of window 0."""
    lp = log_softmax_rows(lg)
    top, tlp, _, _ = reduce_ref(lg, ids)
    rng = np.random.default_rng(0)
    res = {}
    perts = [("temp_1.05", lambda: lg / np.float32(1.05)), ("temp_1.25", lambda: lg / np.float32(1.25)),
             ("noise_0.3", lambda: lg + rng.standard_normal(lg.shape, dtype=np.float32) * np.float32(0.3)),
             ("noise_1.0", lambda: lg + rng.standard_normal(lg.shape, dtype=np.float32))]
    for name, make in perts:
        q = log_softmax_rows(make())
        ex = kl_exact(lp, q).mean()
        lu, pt = kl_lumped(top, tlp, q)
        agree = (q.argmax(1) == lp.argmax(1)).mean()
        res[name] = {"kl_exact": float(ex), "kl_lumped": float(lu.mean()), "ratio": float(lu.mean() / max(ex, 1e-12)),
                     "top1": float(agree)}
    res["tail_mass_mean"] = float(pt.mean()); res["tail_mass_max"] = float(pt.max())
    json.dump(res, open(os.path.join(REF_DIR, "truncation_check.json"), "w"), indent=1)
    log(f"  truncation check (window 0): {json.dumps(res)}")


def score_kl(pack, out_dir, tmp_dir, max_windows=None, binary=None, extra=()):
    ref = np.load(os.path.join(REF_DIR, "kl_ref.npz"))
    W = windows()[: max_windows or None]
    tmp = os.path.join(tmp_dir, "dump.bin")
    kls, agree, nll, tails, per = [], [], [], [], []
    t0 = time.time()
    for w, ids in enumerate(W):
        lg = dump_logits(pack, ids, tmp, os.path.join(out_dir, "dump.stderr"), binary, extra)
        lp = log_softmax_rows(lg)
        kl, pt = kl_lumped(ref["top"][w], ref["tlp"][w], lp)
        ag = (lp.argmax(1) == ref["top1"][w])
        cn = -lp[np.arange(len(ids) - 1), np.asarray(ids[1:])]
        kls.append(kl); agree.append(ag); nll.append(cn); tails.append(pt)
        per.append({"window": w, "kl": float(kl.mean()), "top1": float(ag.mean()), "ppl": float(np.exp(cn.mean()))})
        os.remove(tmp)
    kl = np.concatenate(kls); ag = np.concatenate(agree); cn = np.concatenate(nll)
    rn = ref["nll"][: len(W)].reshape(-1)
    res = {"pack": pack, "engine": binary or KL_BIN, "engine_args": list(extra), "positions": int(kl.size), "windows": len(W), "kl_mean": float(kl.mean()),
           "kl_median": float(np.median(kl)), "kl_p99": float(np.percentile(kl, 99)), "kl_max": float(kl.max()),
           "top1_agreement": float(ag.mean()), "ppl_candidate": float(np.exp(cn.mean())), "ppl_reference": float(np.exp(rn.mean())),
           "ref_tail_mass_mean": float(np.concatenate(tails).mean()), "seconds": round(time.time() - t0), "per_window": per}
    # exact engine-native cross-check on window 0
    w0 = os.path.join(REF_DIR, "kld_ref_w0.bin")
    if os.path.exists(w0):
        assert_guarded()
        r = subprocess.run([binary or KL_BIN, "--kld-eval", w0, "--native", pack, "--max-context", str(DUMP_CTX), *extra],
                           capture_output=True, text=True, timeout=900)
        check_ablate_marker(extra, r.stderr, "kld-eval")
        res["engine_kld_eval_w0"] = (r.stdout.strip().splitlines() or [""])[-1]
        res["harness_w0"] = per[0]
    json.dump(res, open(os.path.join(out_dir, "kl.json"), "w"), indent=1)
    log(f"gate2 KL: mean {res['kl_mean']:.5f}, top-1 {100*res['top1_agreement']:.2f}%, PPL {res['ppl_candidate']:.3f} "
        f"(ref {res['ppl_reference']:.3f}); engine w0: {res.get('engine_kld_eval_w0')}")
    return res


# ------------------------------------------------------------------ coherence
def coherence_prompt_ids(rnd, name):
    return rnd.ids(open(os.path.join(ROOT, PROMPTS[name])).read(), reasoning=True)


def run_coherence(server, out_dir, record_ref=False):
    """server: started StrataServer with ctx >= 32768. Returns dict per prompt (also includes timings)."""
    rnd = Renderer()
    res = {}
    for name in PROMPTS:
        ids = coherence_prompt_ids(rnd, name)
        toks, gen, plen, pms, dms, why = server.gen(ids, COHERENCE_MAX_NEW)
        text = rnd.decode(toks, special=True)
        grams = [tuple(toks[i:i + 4]) for i in range(max(0, len(toks) - 3))]
        r = {"prompt": name, "prompt_tokens": plen, "tokens": toks, "text": text, "finish": why,
             "prefill_tok_s": plen / (pms / 1000) if pms else None, "decode_tok_s": gen / (dms / 1000) if dms else None,
             "distinct_4gram": len(set(grams)) / max(1, len(grams))}
        ref_p = os.path.join(REF_DIR, f"coherence_{name}.json")
        if record_ref:
            os.makedirs(REF_DIR, exist_ok=True)
            json.dump(r, open(ref_p, "w"), indent=1)
        elif os.path.exists(ref_p):
            ref = json.load(open(ref_p))["tokens"]
            n = min(len(ref), len(toks))
            div = next((i for i in range(n) if ref[i] != toks[i]), n)
            r["first_divergence"] = div
            r["ref_len"] = len(ref)
            r["identical"] = (ref == toks)
        open(os.path.join(out_dir, f"coherence_{name}.txt"), "w").write(text)
        res[name] = r
        log(f"coherence {name}: {plen} prompt tok, {gen} gen, finish={why}, distinct4={r['distinct_4gram']:.2f}"
            + (f", first divergence at {r.get('first_divergence')}" if "first_divergence" in r else ""))
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "tokens"} for k, v in res.items()},
              open(os.path.join(out_dir, "coherence.json"), "w"), indent=1)
    return res
