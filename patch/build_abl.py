"""build_abl.py - write the edited (abliterated) SQ-15 pack.  CPU only, no model is loaded.

  uv run --no-project --with numpy --with zstandard --with gguf python -I patch/build_abl.py \
      --pack Inkling-2-Mini-SQ-15.strata --lora LORA.gguf --out Inkling-2-Mini-SQ-15-abl.strata [--procs 4]

Peak memory is a few GB per worker process (the pack is memory-mapped); about 9 minutes on four cores.

Method (see docs/method.md):
  * every edited site: target = projection of the pack's OWN decoded weights
      expert down / wo_ud [out, in]:  W' = W - r r^T W        (r = unit lora_b of that layer)
      embed [token, hidden]:          E' = E - (E r) r^T      (r = unit token_embd lora_b)
  * re-encode with every scale / header / sq byte fixed:
      SF4, S_K, Q8_ROW: strata_codecs.greedy_round (ablation-aware, 3 passes); Q8 values beyond +-127 steps are
      clipped (scale bytes untouched); BF16: round-to-nearest-even of the float64 target.
  * the new pack starts as a copy of the original (`cp -c`, a copy-on-write clone where the file system supports
    it, e.g. APFS); only code bytes are overwritten with pwrite, then the sha256 of sections 1 and 4 is
    recomputed into the header.  The original is only ever opened read-only.
Resumable: one JSON line per finished site in the build log (--log, default OUT + ".build_log.jsonl"); finished
sites are skipped.  The build is deterministic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "codecs"))
import strata_codecs as sc  # noqa: E402
from lora_map import Lora, expert_site  # noqa: E402

EMB_CHUNK = 4096

_pk = _lo = _fd = None


def _init(orig, new, lora):
    global _pk, _lo, _fd
    _pk = sc.StrataPack(orig)
    _lo = Lora(lora)
    _fd = os.open(new, os.O_WRONLY)


def _write(off, data: bytes):
    n = os.pwrite(_fd, data, off)
    assert n == len(data)
    os.fsync(_fd)


def f64_to_bf16_rne(x: np.ndarray) -> np.ndarray:
    """Correct single rounding of float64 to bf16 (avoids f64->f32->bf16 double rounding)."""
    x32 = x.astype(np.float32)
    b = sc.f32_to_bf16(x32)
    lo16 = x32.view(np.uint32) & 0xFFFF
    tie = (lo16 == 0x8000) & (x != x32.astype(np.float64))
    if tie.any():   # x32 sits on a bf16 midpoint only because of the first rounding: go toward x
        up = (np.abs(x) > np.abs(x32.astype(np.float64)))
        hi = (x32.view(np.uint32) >> 16).astype(np.uint16)
        b = np.where(tie, np.where(up, hi + 1, hi), b).astype(np.uint16)
    return b


def task_expert(gid):
    t0 = time.time()
    L, e = expert_site(gid)
    r = _lo.site_delta("expert", L, e)[2]
    W = _pk.decode_expert_down(gid, space="raw").astype(np.float64)
    T = W - np.outer(r, r @ W)
    b, vals = _pk.encode_expert_down(gid, T, space="raw", r=r, method="greedy", return_values=True)
    off, n = _pk.expert_down_span(gid)
    _write(off, b)
    v = vals.astype(np.float64)
    _, q_old = _pk.decode_expert_down_codes(gid)
    _, q_new = _pk.decode_expert_down_codes(gid, b)
    v16 = _pk.decode_expert_down(gid, space="raw", fp16=True, raw=b).astype(np.float64)
    p0 = np.linalg.norm(r @ W)
    return dict(site=f"expert:{gid}", kind="expert", gid=gid, layer=L, code=sc.CODE_NAME[_pk.codes(gid)[1]],
                off=off, n=n, sha=hashlib.sha256(b).hexdigest(),
                abl=float(np.linalg.norm(r @ v) / p0), abl16=float(np.linalg.norm(r @ v16) / p0),
                noise=float(np.linalg.norm(v - T) / np.linalg.norm(W)),
                changed=float((q_old != q_new).mean()), sec=round(time.time() - t0, 3))


def task_wo(L):
    t0 = time.time()
    name = sc.wo_name(L)
    r = _lo.site_delta("wo", L)[2]
    W = _pk.decode_trunk(name).astype(np.float64)
    T = W - np.outer(r, r @ W)
    off, n = _pk.trunk_span(name)
    fmt = _pk.trunk_format(name)
    if fmt == "BF16":
        bits = f64_to_bf16_rne(T)
        b = bits.tobytes()
        v = sc.bf16_to_f32(bits).astype(np.float64)
        changed = float((np.asarray(_pk._raw(off, n).view("<u2")).reshape(bits.shape) != bits).mean())
        clip = 0.0
    else:
        b, v = _pk.encode_trunk(name, T, r=r, method="greedy", return_values=True)
        v = v.astype(np.float64)
        s, q_old = _pk.trunk_q8(name)
        N, K = W.shape
        changed = float((np.asarray(q_old) != np.frombuffer(b, np.int8)[4 * N:].reshape(N, K)).mean())
        clip = float((np.abs(T) > 127.5 * np.asarray(s, np.float64)[:, None]).mean())
        assert b[:4 * N] == bytes(_pk._raw(off, 4 * N)), "Q8 scales must be untouched"
    _write(off, b)
    p0 = np.linalg.norm(r @ W)
    v16 = v.astype(np.float16).astype(np.float64)
    return dict(site=f"wo:{L}", kind="wo", layer=L, code=fmt, off=off, n=n, sha=hashlib.sha256(b).hexdigest(),
                abl=float(np.linalg.norm(r @ v) / p0), abl16=float(np.linalg.norm(r @ v16) / p0),
                noise=float(np.linalg.norm(v - T) / np.linalg.norm(W)), changed=changed, clip=clip,
                sec=round(time.time() - t0, 3))


def task_embed(a):
    """Rows [a, a+EMB_CHUNK) of the embedding; constraint per row on the hidden side (E r)."""
    t0 = time.time()
    N, K = _pk.trunk[sc.EMBED]["shape"]
    b_ = min(N, a + EMB_CHUNK)
    r = _lo.site_delta("embed")[2]
    s_all, q_all = _pk.trunk_q8(sc.EMBED)
    scl = np.asarray(s_all[a:b_], np.float32)
    W = _pk.decode_trunk(sc.EMBED, rows=slice(a, b_)).astype(np.float64)       # [rows, hidden]
    p_old = W @ r
    T = W - np.outer(p_old, r)
    Tt = np.ascontiguousarray(T.T)                                             # [hidden, rows]: constraint axis first
    qt, vt = sc.greedy_round(Tt, r, sc.Q8Grid(np.broadcast_to(scl[None, :], Tt.shape)))
    q = np.ascontiguousarray(qt.T)
    v = vt.T.astype(np.float64)
    off0, _ = _pk.trunk_span(sc.EMBED)
    off = off0 + 4 * N + a * K
    b = q.tobytes()
    _write(off, b)
    p_new = v @ r
    p16 = v.astype(np.float16).astype(np.float64) @ r
    q_old = np.asarray(q_all[a:b_])
    return dict(site=f"embed:{a}", kind="embed", rows=[a, b_], code="Q8_ROW", off=off, n=len(b),
                sha=hashlib.sha256(b).hexdigest(),
                pold2=float((p_old ** 2).sum()), pnew2=float((p_new ** 2).sum()), p162=float((p16 ** 2).sum()),
                err2=float(((v - T) ** 2).sum()), w2=float((W ** 2).sum()),
                row_res_steps_max=float(np.max(np.abs(p_new) / scl)),
                changed=int((q != q_old).sum()), nel=int(q.size),
                clip=int((np.abs(T) > 127.5 * scl[:, None].astype(np.float64)).sum()),
                sec=round(time.time() - t0, 3))


def run(fn, items, procs, done, logf, label, init_args):
    items = [x for x in items if x[1] not in done]
    if not items:
        print(f"{label}: nothing to do", flush=True)
        return
    t0 = time.time()
    with mp.get_context("spawn").Pool(procs, initializer=_init, initargs=init_args) as pool:
        for k, rec in enumerate(pool.imap_unordered(fn, [x[0] for x in items])):
            logf.write(json.dumps(rec) + "\n")
            logf.flush()
            if k % 100 == 0 or k == len(items) - 1:
                print(f"{label}: {k + 1}/{len(items)} {rec['site']} abl={rec.get('abl', float('nan')):.4f} "
                      f"({time.time() - t0:.0f}s)", flush=True)


def section_sha(path, off, ln, bs=64 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(off)
        left = ln
        while left:
            buf = f.read(min(bs, left))
            h.update(buf)
            left -= len(buf)
    return h.digest()


def free_gb(path):
    """Free space on the disk that holds `path` (symlinks resolved)."""
    st = shutil.disk_usage(os.path.dirname(os.path.realpath(path)))
    return st.free / 2 ** 30


def main():
    ap = argparse.ArgumentParser(description="write the edited pack (CPU only)")
    ap.add_argument("--pack", required=True, help="the original Inkling-2-Mini-SQ-15.strata (opened read-only)")
    ap.add_argument("--lora", required=True, help="the rank-1 abliteration LoRA (.gguf); only its directions are used")
    ap.add_argument("--out", required=True, help="the edited pack to write (created as a copy of --pack)")
    ap.add_argument("--log", help="build log (default OUT.build_log.jsonl); a rerun skips the sites in it")
    ap.add_argument("--procs", type=int, default=4, help="worker processes")
    ap.add_argument("--min-free-gb", type=float, default=40,
                    help="stop before cloning / before the header update when the output disk has less free space")
    a = ap.parse_args()
    ORIG, NEW = os.path.abspath(a.pack), os.path.abspath(a.out)
    LOG = a.log or NEW + ".build_log.jsonl"
    procs, MIN_FREE_GB = a.procs, a.min_free_gb
    init_args = (ORIG, NEW, os.path.abspath(a.lora))
    t_start = time.time()
    print(f"free disk before: {free_gb(NEW):.1f} GiB", flush=True)
    if not os.path.exists(NEW):
        if free_gb(NEW) < MIN_FREE_GB + 6:
            sys.exit(f"not enough free disk for the edited blocks (need > {MIN_FREE_GB + 6:.0f} GiB free)")
        if sys.platform == "darwin":
            subprocess.run(["cp", "-c", ORIG, NEW], check=True)       # APFS clone: only edited blocks use space
        else:
            subprocess.run(["cp", "--reflink=auto", ORIG, NEW], check=True)
        os.chmod(NEW, 0o644)
        print(f"cloned (cp -c) -> {NEW}; free {free_gb(NEW):.1f} GiB", flush=True)
    assert os.path.getsize(NEW) == os.path.getsize(ORIG)
    done = set()
    if os.path.exists(LOG):
        for line in open(LOG):
            try:
                done.add(json.loads(line)["site"])
            except Exception:
                pass
    print(f"{len(done)} sites already in {LOG}", flush=True)
    logf = open(LOG, "a")
    N = 201024
    run(task_wo, [(L, f"wo:{L}") for L in range(24)], procs, done, logf, "wo", init_args)
    run(task_embed, [(x, f"embed:{x}") for x in range(0, N, EMB_CHUNK)], procs, done, logf, "embed", init_args)
    run(task_expert, [(g, f"expert:{g}") for g in range(3072)], procs, done, logf, "experts", init_args)
    if free_gb(NEW) < MIN_FREE_GB:
        sys.exit(f"free disk {free_gb(NEW):.1f} GiB < {MIN_FREE_GB}: stopping before the header update")
    # header hashes for sections 1 and 4
    pk = sc.StrataPack(NEW)
    with open(NEW, "r+b") as f:
        for i in (1, 4):
            s = pk.secs[i]
            d = section_sha(NEW, s["off"], s["len"])
            f.seek(0x10 + 64 * i + 16)
            f.write(d)
            print(f"sec{i} sha256 -> {d.hex()}", flush=True)
        f.flush()
        os.fsync(f.fileno())
    print(f"free disk after: {free_gb(NEW):.1f} GiB; build time {time.time() - t_start:.0f}s", flush=True)


if __name__ == "__main__":
    main()
