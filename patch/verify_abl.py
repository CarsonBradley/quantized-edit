"""verify_abl.py - verify the written edited pack against the original, from the files on disk.  CPU only.

  uv run --no-project --with numpy --with zstandard --with gguf python -I patch/verify_abl.py \
      --orig Inkling-2-Mini-SQ-15.strata --new Inkling-2-Mini-SQ-15-abl.strata --lora LORA.gguf \
      [--build-log NEW.build_log.jsonl] [--inspect PATH/strata_inspect.py] [--out verify.json] [--procs 4]

 (a) byte diff: every differing byte must be a code byte of an edited span (SF4 down nibbles, S_K down
     super-block payload bytes (offset % SB >= 6), Q8_ROW int8 codes of wo_ud / embed, BF16 wo_ud words) or one
     of the two rewritten header hashes (sec1, sec4).  Anything else is a violation.
 (b) header parses (fresh StrataPack, plus the pack distribution's tools/strata_inspect.py if --inspect is
     given), every section's header sha256 == fresh sha256.
 (c) re-decode all 49 sites (3,072 experts, 24 wo_ud, embed) from the NEW file and measure the ablation
     residual and added noise against the target computed from the ORIGINAL file.
 (d) file sizes identical.
Writes --out (default NEW + ".verify.json").  If --build-log is given (default NEW.build_log.jsonl when it
exists), the sha256 of every written expert span is also compared with the build log.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "codecs"))
import strata_codecs as sc  # noqa: E402
from lora_map import Lora, expert_site  # noqa: E402

# set in main() (and in each worker by _init)
ORIG = NEW = LORA = INSPECT = None


# ------------------------------------------------------------------ (a)
def allowed_intervals(pk):
    iv = []   # (start, end, kind, sb)
    for g in range(3072):
        off, n = pk.expert_down_span(g)
        dn = pk.codes(g)[1]
        if dn == 15:
            iv.append((off, off + sc.SF4_DN_P, "sf4_nibbles", 0))
        else:
            iv.append((off, off + n, f"{sc.CODE_NAME[dn]}_payload", sc.SK_SB_BYTES[dn]))
    for L in range(24):
        name = sc.wo_name(L)
        off, n = pk.trunk_span(name)
        N = pk.trunk[name]["shape"][0]
        if pk.trunk_format(name) == "Q8_ROW":
            iv.append((off + 4 * N, off + n, "wo_q8_codes", 0))
        else:
            iv.append((off, off + n, "wo_bf16", 0))
    off, n = pk.trunk_span(sc.EMBED)
    iv.append((off + 4 * 201024, off + n, "embed_codes", 0))
    for i in (1, 4):
        h = 0x10 + 64 * i + 16
        iv.append((h, h + 32, f"header_sha_sec{i}", 0))
    iv.sort()
    return iv


def byte_diff(pk, bs=64 << 20):
    iv = allowed_intervals(pk)
    starts = np.array([x[0] for x in iv], np.int64)
    ends = np.array([x[1] for x in iv], np.int64)
    sbs = np.array([x[3] for x in iv], np.int64)
    kinds = [x[2] for x in iv]
    per_kind = {}
    violations = 0
    first_viol = []
    total = 0
    spans_touched = np.zeros(len(iv), bool)
    size = os.path.getsize(ORIG)
    with open(ORIG, "rb") as fa, open(NEW, "rb") as fb:
        pos = 0
        while pos < size:
            a = np.frombuffer(fa.read(bs), np.uint8)
            b = np.frombuffer(fb.read(bs), np.uint8)
            d = np.nonzero(a != b)[0].astype(np.int64) + pos
            pos += len(a)
            if d.size == 0:
                continue
            total += d.size
            k = np.searchsorted(starts, d, side="right") - 1
            ok = (k >= 0) & (d < ends[np.clip(k, 0, None)])
            kk = np.clip(k, 0, None)
            sb = sbs[kk]
            ok &= (sb == 0) | (((d - starts[kk]) % np.where(sb == 0, 1, sb)) >= 6)
            violations += int((~ok).sum())
            if (~ok).any() and len(first_viol) < 20:
                first_viol += d[~ok][:20].tolist()
            spans_touched[np.unique(kk[ok])] = True
            for u, c in zip(*np.unique(kk[ok], return_counts=True)):
                per_kind[kinds[u]] = per_kind.get(kinds[u], 0) + int(c)
    n_by_kind = {}
    for i, kd in enumerate(kinds):
        n_by_kind.setdefault(kd, [0, 0])
        n_by_kind[kd][0] += 1
        n_by_kind[kd][1] += int(spans_touched[i])
    return dict(total_diff_bytes=total, violations=violations, first_violations=first_viol,
                diff_bytes_by_kind=per_kind, spans_by_kind={k: dict(spans=v[0], touched=v[1]) for k, v in n_by_kind.items()})


# ------------------------------------------------------------------ (b)
def sha_section(path, off, ln, bs=64 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(off)
        left = ln
        while left:
            buf = f.read(min(bs, left))
            h.update(buf)
            left -= len(buf)
    return h.hexdigest()


def header_check():
    import tempfile
    pn, po = sc.StrataPack(NEW), sc.StrataPack(ORIG)
    out = {"sections": []}
    for i, (s, so) in enumerate(zip(pn.secs, po.secs)):
        fresh = sha_section(NEW, s["off"], s["len"])
        out["sections"].append(dict(i=i, off=s["off"], len=s["len"], same_off_len_as_orig=(s["off"], s["len"]) == (so["off"], so["len"]),
                                    header_sha=s["sha"].hex(), fresh_sha=fresh, match=s["sha"].hex() == fresh,
                                    orig_header_sha=so["sha"].hex(), changed_vs_orig=s["sha"] != so["sha"]))
    out["manifest_equal"] = pn.manifest == po.manifest
    out["directory_equal"] = bool(np.array_equal(pn.dir, po.dir)) and bool(np.array_equal(pn.vram_order, po.vram_order))
    if INSPECT:
        with tempfile.TemporaryDirectory() as scratch:
            p = subprocess.run([sys.executable, "-I", INSPECT, NEW], cwd=scratch, capture_output=True, text=True)
        out["strata_inspect_rc"] = p.returncode
        out["strata_inspect_tail"] = p.stdout.strip().splitlines()[-3:] + p.stderr.strip().splitlines()[-3:]
    else:
        out["strata_inspect_rc"] = None
    return out


# ------------------------------------------------------------------ (c)
_po = _pn = _lo = None


def _init(orig, new, lora):
    global _po, _pn, _lo
    _po, _pn, _lo = sc.StrataPack(orig), sc.StrataPack(new), Lora(lora)


def v_expert(gid):
    L, e = expert_site(gid)
    r = _lo.site_delta("expert", L, e)[2]
    W = _po.decode_expert_down(gid, space="eff").astype(np.float64)
    T = W - np.outer(r, r @ W)
    Wn = _pn.decode_expert_down(gid, space="eff").astype(np.float64)
    W16 = _pn.decode_expert_down(gid, space="eff", fp16=True).astype(np.float64)
    p0 = np.linalg.norm(r @ W)
    off, n = _pn.expert_down_span(gid)
    return dict(site=f"expert:{gid}", code=sc.CODE_NAME[_pn.codes(gid)[1]], layer=L,
                abl=float(np.linalg.norm(r @ Wn) / p0), abl16=float(np.linalg.norm(r @ W16) / p0),
                noise=float(np.linalg.norm(Wn - T) / np.linalg.norm(W)),
                sha=hashlib.sha256(bytes(_pn.buf[off:off + n])).hexdigest())


def v_wo(L):
    name = sc.wo_name(L)
    r = _lo.site_delta("wo", L)[2]
    W = _po.decode_trunk(name).astype(np.float64)
    T = W - np.outer(r, r @ W)
    Wn = _pn.decode_trunk(name).astype(np.float64)
    W16 = _pn.decode_trunk(name, fp16=True).astype(np.float64)
    p0 = np.linalg.norm(r @ W)
    return dict(site=f"wo:{L}", code=_pn.trunk_format(name), layer=L, abl=float(np.linalg.norm(r @ Wn) / p0),
                abl16=float(np.linalg.norm(r @ W16) / p0), noise=float(np.linalg.norm(Wn - T) / np.linalg.norm(W)))


def v_embed(a, chunk=4096):
    r = _lo.site_delta("embed")[2]
    sl = slice(a, min(201024, a + chunk))
    W = _po.decode_trunk(sc.EMBED, rows=sl).astype(np.float64)
    Wn = _pn.decode_trunk(sc.EMBED, rows=sl).astype(np.float64)
    W16 = _pn.decode_trunk(sc.EMBED, rows=sl, fp16=True).astype(np.float64)
    p = W @ r
    T = W - np.outer(p, r)
    s = np.asarray(_pn.trunk_q8(sc.EMBED)[0][sl], np.float64)
    return dict(site=f"embed:{a}", pold2=float((p ** 2).sum()), pnew2=float(((Wn @ r) ** 2).sum()),
                p162=float(((W16 @ r) ** 2).sum()), err2=float(((Wn - T) ** 2).sum()), w2=float((W ** 2).sum()),
                row_res_steps=np.abs(Wn @ r / s).tolist(), row_noise=(np.linalg.norm(Wn - T, axis=1) / np.linalg.norm(W, axis=1)).tolist())


def dist(x):
    x = np.asarray(x)
    return dict(n=int(x.size), median=float(np.median(x)), p99=float(np.percentile(x, 99)), max=float(x.max()), mean=float(x.mean()))


def main():
    global ORIG, NEW, LORA, INSPECT
    ap = argparse.ArgumentParser(description="verify an edited pack against the original (CPU only)")
    ap.add_argument("--orig", required=True, help="the original pack")
    ap.add_argument("--new", required=True, help="the edited pack")
    ap.add_argument("--lora", required=True, help="the LoRA the edit was built from (.gguf)")
    ap.add_argument("--build-log", help="build_abl.py's log (default NEW.build_log.jsonl if it exists)")
    ap.add_argument("--inspect", help="optional: the pack distribution's tools/strata_inspect.py, run on NEW")
    ap.add_argument("--out", help="result JSON (default NEW.verify.json)")
    ap.add_argument("--procs", type=int, default=4)
    a = ap.parse_args()
    ORIG, NEW, LORA = os.path.abspath(a.orig), os.path.abspath(a.new), os.path.abspath(a.lora)
    INSPECT = os.path.abspath(a.inspect) if a.inspect else None
    procs = a.procs
    t0 = time.time()
    res = {}
    res["d_size"] = dict(orig=os.path.getsize(ORIG), new=os.path.getsize(NEW), equal=os.path.getsize(ORIG) == os.path.getsize(NEW))
    print("(d)", res["d_size"], flush=True)
    res["b_header"] = header_check()
    print("(b)", [(s["i"], s["match"], s["changed_vs_orig"]) for s in res["b_header"]["sections"]],
          "manifest_equal", res["b_header"]["manifest_equal"], "dir_equal", res["b_header"]["directory_equal"],
          "inspect rc", res["b_header"]["strata_inspect_rc"], flush=True)
    res["a_diff"] = byte_diff(sc.StrataPack(ORIG))
    print("(a)", {k: v for k, v in res["a_diff"].items()}, f"({time.time() - t0:.0f}s)", flush=True)
    # (c)
    ctx = mp.get_context("spawn")
    with ctx.Pool(procs, initializer=_init, initargs=(ORIG, NEW, LORA)) as pool:
        ex = list(pool.imap_unordered(v_expert, range(3072), chunksize=8))
        print(f"(c) experts decoded ({time.time() - t0:.0f}s)", flush=True)
        wo = list(pool.imap_unordered(v_wo, range(24)))
        em = list(pool.imap_unordered(v_embed, range(0, 201024, 4096)))
    # build-log consistency
    log = {}
    lp = a.build_log or NEW + ".build_log.jsonl"
    if os.path.exists(lp):
        for line in open(lp):
            rec = json.loads(line)
            log[rec["site"]] = rec
        sha_mismatch = [x["site"] for x in ex if log.get(x["site"], {}).get("sha") != x["sha"]]
    else:
        print(f"no build log at {lp}: skipping the per-span sha256 comparison", flush=True)
        sha_mismatch = None
    summ = {}
    for code in sorted({x["code"] for x in ex}):
        xs = [x for x in ex if x["code"] == code]
        summ[code] = {m: dist([x[m] for x in xs]) for m in ("abl", "abl16", "noise")}
    for code in ("Q8_ROW", "BF16"):
        xs = [x for x in wo if x["code"] == code]
        summ[f"wo_{code}"] = {m: dist([x[m] for x in xs]) for m in ("abl", "abl16", "noise")}
    agg = {k: sum(x[k] for x in em) for k in ("pold2", "pnew2", "p162", "err2", "w2")}
    rr = np.concatenate([x["row_res_steps"] for x in em])
    rn = np.concatenate([x["row_noise"] for x in em])
    summ["embed"] = dict(abl=float(np.sqrt(agg["pnew2"] / agg["pold2"])), abl16=float(np.sqrt(agg["p162"] / agg["pold2"])),
                         noise=float(np.sqrt(agg["err2"] / agg["w2"])),
                         row_residual_in_steps=dist(rr), row_noise=dist(rn))
    # outliers: per-format median-relative flags
    outl = []
    for x in ex + wo:
        s = summ[x["code"]] if x["site"].startswith("expert") else summ[f"wo_{x['code']}"]
        if x["abl"] > 0.02 or x["noise"] > 1.5 * s["noise"]["median"] or x["abl16"] > 0.02:
            outl.append({k: x[k] for k in ("site", "code", "abl", "abl16", "noise")})
    res["c_sites"] = dict(summary=summ, outliers=outl, n_experts=len(ex), n_wo=len(wo), n_embed_chunks=len(em),
                          build_log_sha_mismatch=sha_mismatch, per_expert=ex, wo=wo)
    print("(c)", json.dumps(summ, indent=1), flush=True)
    print("outliers:", outl[:30], "count", len(outl), flush=True)
    print("build-log sha mismatches:", "not checked" if sha_mismatch is None else len(sha_mismatch), flush=True)
    out = a.out or NEW + ".verify.json"
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out, flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
