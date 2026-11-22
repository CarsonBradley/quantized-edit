"""test_codecs.py - proofs for strata_codecs.py.  CPU only; the pack is memory-mapped, never read whole.

  uv run --no-project --with numpy --with zstandard python -I codecs/test_codecs.py --pack PACK.strata \
      [--strata-src STRATA_CHECKOUT] [--lora LORA.gguf] [--all] [--quick] [--per-code 24] [--out results/tests.json]

  --strata-src  a checkout of Strata at commit 26350e4; test 1 parses the shader tables from
                engine/src/inkling2/metal/ink_metal_codecs.metal (skipped without it)
  --lora        the rank-1 LoRA in GGUF form; test 6 needs it (skipped without it; needs `--with gguf`)
  --all         round-trip every expert instead of a sample
  --quick       skip the trunk round trip and the LoRA check

Tests
  1. shader_reference : a literal per-element transcription of the Metal down-projection kernels
     (sk_w2_paired2 + sk_w8 with the LUT tables parsed from ink_metal_codecs.metal, sf4_w2_pair_s,
     decode_sb_sk_16 / ink_moe_dn_mm_gen indexing) agrees bit-for-bit with the vectorised decoder.
  2. roundtrip_experts : decode -> encode == original bytes (eff and raw space), every sampled expert.
  3. roundtrip_trunk   : same for embed (Q8_ROW) and all 24 wo_ud (Q8_ROW / BF16).
  4. synthetic_sk      : pack/unpack bijection for S2_K..S8_K on random codes.
  5. cpu_vs_shader_eff : how often D*(u/255) (CPU) != D*(u*(1/255)) (shader), and whether it ever
     changes a nearest-code decision.
  6. lora_consistency  : the LoRA A vector of each sampled site vs -(r^T W) decoded from the pack
     (end-to-end check of decode semantics incl. the inv_sq column scale, and of the site mapping).
  7. reference_codec   : Strata's own public reference S_K codec (tools/squant/codecs.py, written for the
     earlier .strata v1.0 layout) on real down-matrix bytes from this pack: its decode_sk against ours, value by
     value, with the shader's and the CPU's order of the scale product; and whether its encode_sk, given our
     decoded weights, keeps the pack's super-block headers (scales) and codes.  Needs --strata-src.
Results go to --out (default results/tests.json, or results/tests_all.json with --all, under the current
directory). Exit code 1 if any check fails.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "patch"))
import strata_codecs as sc  # noqa: E402

METAL_REL = os.path.join("engine", "src", "inkling2", "metal", "ink_metal_codecs.metal")
SQUANT_REL = os.path.join("tools", "squant", "codecs.py")


def parse_lut(text, name):
    m = re.search(r"constant half(\d) " + name + r"\[(\d+)\] = \{(.*?)\};", text, re.S)
    width, n = int(m.group(1)), int(m.group(2))
    nums = [float(x) for x in re.findall(r"(-?\d+\.\d+)h", m.group(3))]
    a = np.array(nums, np.float32).reshape(n, width)
    return a


def sample_gids(pk, per_code=24, seed=0):
    rng = np.random.default_rng(seed)
    out = {}
    for code in (15, 3, 4, 5):
        g = np.array([i for i in range(3072) if pk.codes(i)[1] == code])
        if len(g) > per_code:
            g = np.sort(rng.choice(g, per_code, replace=False))
        out[code] = [int(x) for x in g]
    # every expert that has S5_K anywhere (gate/up or down)
    out["S5_any"] = [i for i in range(3072) if 5 in pk.codes(i)]
    return out


# ------------------------------------------------------------------------------------------------
def test_shader_reference(pk, gids, res, metal_path):
    text = open(metal_path).read()
    S2 = parse_lut(text, "kS2Lut")
    S3H = parse_lut(text, "kS3Hi")
    S4 = parse_lut(text, "kS4Lut")
    S5H = parse_lut(text, "kS5Hi")
    E2 = np.array([float(x.replace("f", "")) for x in
                   re.search(r"constant float kE2M1\[16\] = \{(.*?)\};", text, re.S).group(1).replace("\n", "").split(",")],
                  np.float32)

    def sk_w8(pl, j, F):   # literal transcription
        if F == 3:
            p = pl[(j >> 2):]
            h = int(pl[64 + (j >> 3)])
            return np.concatenate([S2[p[0]] + S3H[h & 15], S2[p[1]] + S3H[h >> 4]])
        if F == 4:
            p = pl[(j >> 1):]
            return np.concatenate([S4[p[0]], S4[p[1]], S4[p[2]], S4[p[3]]])
        if F == 5:
            p = pl[(j >> 1):]
            h = int(pl[128 + (j >> 3)])
            return np.concatenate([np.concatenate([S4[p[0]], S4[p[1]]]) + S5H[h & 15],
                                   np.concatenate([S4[p[2]], S4[p[3]]]) + S5H[h >> 4]])
        raise ValueError

    n_cmp = 0
    bad = 0
    rng = np.random.default_rng(1)
    for code, gl in gids.items():
        for gid in gl[:3]:
            dn = pk.codes(gid)[1]
            W = pk.decode_expert_down(gid, space="raw")
            Wh = pk.decode_expert_down(gid, space="raw", fp16=True)
            off, n = pk.expert_down_span(gid)
            body_dn = pk.buf[off:off + n]
            for rp in rng.choice(1152, 6, replace=False):
                ref = np.zeros((2, 1152), np.float32)
                if dn == 15:     # sf4_w2_pair_s: rows 2rp, 2rp+1; nibble k of the 64-bit word -> column 16b+k
                    outer = pk.sf4_dn_outer(gid)
                    for row in (2 * rp, 2 * rp + 1):
                        for b in range(72):
                            scv = sc.E4M3[body_dn[sc.SF4_DN_P + row * 72 + b]] * outer
                            word = int.from_bytes(bytes(body_dn[row * 576 + b * 8: row * 576 + b * 8 + 8]), "little")
                            for k in range(16):
                                ref[row - 2 * rp, 16 * b + k] = E2[(word >> (4 * k)) & 15] * scv
                else:            # sk_w2_paired2: unit = sbi*32 + g*8 + (u&7); col = (unit or unit-144)*8
                    SB = sc.SK_SB_BYTES[dn]
                    pr = body_dn[rp * 9 * SB:(rp + 1) * 9 * SB]
                    for unit in range(288):
                        sbi, u = unit >> 5, unit & 31
                        g, j = u >> 3, ((u >> 3) << 6) + ((u & 7) << 3)
                        blk = pr[sbi * SB:(sbi + 1) * SB]
                        D = np.frombuffer(bytes(blk[:2]), "<f2").astype(np.float32)[0]
                        eff = D * (np.float32(blk[2 + g]) * (np.float32(1.0) / np.float32(255.0)))
                        w = sk_w8(blk[6:], j, dn).astype(np.float32)
                        is_r1 = unit >= 144
                        col = ((unit - 144) if is_r1 else unit) << 3
                        ref[int(is_r1), col:col + 8] = w * eff
                got = W[2 * rp:2 * rp + 2]
                bad += int((ref.view(np.uint32) != got.view(np.uint32)).sum())
                # prefill path (decode_sb_* -> half): half(w*eff) / half(kE2M1*sc)
                bad += int((ref.astype(np.float16).astype(np.float32) != Wh[2 * rp:2 * rp + 2]).sum())
                n_cmp += 2 * ref.size
            # mm_gen indexing: row gn -> pair gn>>1, col_p = (gn&1)*1152 + col  (checked on 4 rows)
            if dn != 15:
                SB = sc.SK_SB_BYTES[dn]
                for gn in (0, 1, 1151, 2303):
                    rp = gn >> 1
                    for col in (0, 64, 1136):
                        col_p = (1152 if gn & 1 else 0) + col
                        sbi, j = col_p >> 8, col_p & 255
                        g = j >> 6
                        blk = body_dn[rp * 9 * SB + sbi * SB: rp * 9 * SB + (sbi + 1) * SB]
                        D = np.frombuffer(bytes(blk[:2]), "<f2").astype(np.float32)[0]
                        eff = D * (np.float32(blk[2 + g]) * (np.float32(1.0) / np.float32(255.0)))
                        w = np.concatenate([sk_w8(blk[6:], j, dn), sk_w8(blk[6:], j + 8, dn)]).astype(np.float32)
                        ref = (w * eff).astype(np.float16).astype(np.float32)
                        bad += int((ref != Wh[gn, col:col + 16]).sum())
                        n_cmp += 16
    res["shader_reference"] = dict(compared_values=n_cmp, mismatches=bad, passed=bad == 0)
    print(f"[shader_reference] {n_cmp} values compared, {bad} mismatches -> {'PASS' if bad == 0 else 'FAIL'}")


def test_roundtrip_experts(pk, gl_all, res):
    out = {}
    for code, gl in gl_all.items():
        ok_eff = ok_raw = 0
        fails = []
        t0 = time.time()
        for gid in gl:
            off, n = pk.expert_down_span(gid)
            orig = bytes(pk.buf[off:off + n])
            a = pk.encode_expert_down(gid, pk.decode_expert_down(gid, space="eff"), space="eff") == orig
            b = pk.encode_expert_down(gid, pk.decode_expert_down(gid, space="raw"), space="raw") == orig
            ok_eff += a
            ok_raw += b
            if not (a and b):
                fails.append(gid)
        name = sc.CODE_NAME.get(code, code) if isinstance(code, int) else code
        out[str(name)] = dict(n=len(gl), eff_identical=ok_eff, raw_identical=ok_raw, fails=fails[:20],
                              seconds=round(time.time() - t0, 1))
        print(f"[roundtrip_experts] {name}: {len(gl)} experts, eff identical {ok_eff}, raw identical {ok_raw}"
              f" -> {'PASS' if not fails else 'FAIL'} ({time.time() - t0:.1f}s)")
    res["roundtrip_experts"] = out


def test_roundtrip_trunk(pk, res):
    names = [sc.EMBED] + [sc.wo_name(L) for L in range(24)]
    out = {}
    for nm in names:
        off, n = pk.trunk_span(nm)
        N, K = pk.trunk[nm]["shape"]
        if pk.trunk_format(nm) == "Q8_ROW" and N > 65536:
            # The embedding table (201,024 rows): round-trip it in row chunks with the encoder's own per-row
            # nearest-code function, so the test stays at a few hundred MB instead of ~10 GB. encode_trunk(r=None)
            # does exactly this, chunk by chunk, then writes the unchanged scales followed by the codes.
            s_all, q_all = pk.trunk_q8(nm)
            ok = True
            for a in range(0, N, 8192):
                b = min(N, a + 8192)
                W = pk.decode_trunk(nm, rows=slice(a, b))
                q, _ = sc.nearest_q8(W, np.broadcast_to(np.asarray(s_all[a:b], np.float32)[:, None], W.shape))
                ok &= bool(np.array_equal(q, np.asarray(q_all[a:b])))
        else:
            orig = bytes(pk.buf[off:off + n])
            W = pk.decode_trunk(nm)
            ok = pk.encode_trunk(nm, W) == orig
        del W
        out[nm] = dict(format=pk.trunk_format(nm), identical=bool(ok))
    for fmt in ("Q8_ROW", "BF16"):
        sel = [v["identical"] for v in out.values() if v["format"] == fmt]
        print(f"[roundtrip_trunk] {fmt}: {sum(sel)}/{len(sel)} identical -> {'PASS' if all(sel) else 'FAIL'}")
    res["roundtrip_trunk"] = out


def test_synthetic_sk(res):
    rng = np.random.default_rng(2)
    out = {}
    for code in (2, 3, 4, 5, 6, 8):
        q = rng.integers(sc.SK_QMIN[code], sc.SK_QMAX[code] + 1, size=(512, 256))
        q = q.astype(np.int8 if code == 8 else np.uint8)
        pl = sc.sk_pack_q(q, code)
        assert pl.shape[1] == sc.SK_SB_BYTES[code] - 6
        back = sc.sk_unpack_q(pl, code)
        out[sc.CODE_NAME[code]] = bool(np.array_equal(back, q)) and bool(np.array_equal(sc.sk_pack_q(back, code), pl))
    res["synthetic_sk"] = out
    print(f"[synthetic_sk] {out}")


def test_cpu_vs_shader_eff(pk, gl_all, res):
    out = {}
    for code in (3, 4, 5):
        diff = tot = changed = 0
        for gid in gl_all[code][:8]:
            off, n = pk.expert_down_span(gid)
            raw = pk.buf[off:off + n]
            _, e_sh = sc.sk_split(raw, code)
            e_cpu = sc.sk_eff_cpu(raw, code)
            diff += int((e_sh != e_cpu).sum())
            tot += e_sh.size
            # would the CPU-valued decode re-encode to the same codes under the shader grid?
            eff_cpu_full = np.repeat(e_cpu.reshape(-1), 64).reshape(2304, 1152)
            _, q = pk.decode_expert_down_codes(gid)
            Wcpu = (q.astype(np.float32) - np.float32(sc.SK_HALF[code])) * eff_cpu_full
            q2, _ = sc.nearest_sk(Wcpu, pk._sk_eff_matrix(gid, code), code)
            changed += int((q2 != q).sum())
        out[sc.CODE_NAME[code]] = dict(subscales=tot, eff_differs=diff, frac=diff / tot, code_changes=changed)
    res["cpu_vs_shader_eff"] = out
    print(f"[cpu_vs_shader_eff] {out}")


def test_lora_consistency(pk, gl_all, res, lora_path):
    from lora_map import Lora, expert_site
    lo = Lora(lora_path)
    rows = []
    for code in (15, 3, 4, 5):
        for gid in gl_all[code][:8]:
            L, e = expert_site(gid)
            u, v, r = lo.site_delta("expert", L, e)
            W = pk.decode_expert_down(gid, space="eff").astype(np.float64)
            p = r @ W                     # [1152]
            # Delta = outer(u, v) and u = +-r; if Delta = -alpha r r^T W then v = -alpha*sign * p
            sgn = np.sign(u @ r)
            alpha = -(sgn * v) @ p / (p @ p)
            cos = -(sgn * v) @ p / np.linalg.norm(v) / np.linalg.norm(p)
            # the same with the raw (no inv_sq) matrix, to show the sq scale matters
            Wr = pk.decode_expert_down(gid, space="raw").astype(np.float64)
            alpha_raw = -(sgn * v) @ (r @ Wr) / ((r @ Wr) @ (r @ Wr))
            rows.append(dict(gid=gid, code=sc.CODE_NAME[code], alpha=float(alpha), cos=float(cos),
                             alpha_raw=float(alpha_raw)))
            # wrong-expert control: neighbour expert's A against this W
            u2, v2, _ = lo.site_delta("expert", L, (e + 1) % 128)
            rows[-1]["cos_wrong_expert"] = float(-(sgn * v2) @ p / np.linalg.norm(v2) / np.linalg.norm(p))
    for L in (0, 5, 11, 12, 23):
        u, v, r = lo.site_delta("wo", L)
        W = pk.decode_trunk(sc.wo_name(L)).astype(np.float64)
        p = r @ W
        sgn = np.sign(u @ r)
        rows.append(dict(site=f"wo{L}", fmt=pk.trunk_format(sc.wo_name(L)),
                         alpha=float(-(sgn * v) @ p / (p @ p)),
                         cos=float(-(sgn * v) @ p / np.linalg.norm(v) / np.linalg.norm(p))))
    # embed: Delta = outer(a, b) with b = r -> a should be -E r
    u, v, r = lo.site_delta("embed")
    sl = slice(0, 201024)
    Er = np.zeros(201024)
    for a in range(0, 201024, 32768):
        Er[a:a + 32768] = pk.decode_trunk(sc.EMBED, rows=slice(a, min(201024, a + 32768))).astype(np.float64) @ r
    sgn = np.sign(v @ r)
    rows.append(dict(site="embed", alpha=float(-(sgn * u[sl]) @ Er / (Er @ Er)),
                     cos=float(-(sgn * u[sl]) @ Er / np.linalg.norm(u[sl]) / np.linalg.norm(Er))))
    res["lora_consistency"] = rows
    for rw in rows:
        print("[lora_consistency]", {k: (round(x, 4) if isinstance(x, float) else x) for k, x in rw.items()})


def test_reference_codec(pk, gl_all, res, squant_path, per_code=4):
    import importlib.util
    spec = importlib.util.spec_from_file_location("strata_squant_codecs", squant_path)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    out = {}
    for code in (3, 4, 5):
        name, SB = sc.CODE_NAME[code], sc.SK_SB_BYTES[code]
        n = same_shader = same_cpu = n_sb = hdr_same = pl_same = 0
        max_rel = 0.0
        for gid in gl_all[code][:per_code]:
            off, nb = pk.expert_down_span(gid)
            raw = np.asarray(pk.buf[off:off + nb])
            theirs = np.asarray(ref.decode_sk(raw, name), np.float32).reshape(sc.HIDDEN, sc.INTER)
            ours = pk.decode_expert_down(gid, space="raw")                    # shader order: D * (u * (1/255))
            _, q = pk.decode_expert_down_codes(gid)
            eff_cpu = np.repeat(sc.sk_eff_cpu(raw, code).reshape(-1), 64).reshape(sc.HIDDEN, sc.INTER)
            ours_cpu = ((q.astype(np.float32) - np.float32(sc.SK_HALF[code])) * eff_cpu).astype(np.float32)
            n += theirs.size
            same_shader += int((theirs.view(np.uint32) == ours.view(np.uint32)).sum())
            same_cpu += int((theirs.view(np.uint32) == ours_cpu.view(np.uint32)).sum())
            nz = ours != 0
            max_rel = max(max_rel, float(np.max(np.abs(theirs[nz] - ours[nz]) / np.abs(ours[nz]))))
            enc = np.frombuffer(ref.encode_sk(ours.reshape(-1), name), np.uint8).reshape(-1, SB)
            orig = raw.reshape(-1, SB)
            n_sb += orig.shape[0]
            hdr_same += int((enc[:, :6] == orig[:, :6]).all(axis=1).sum())
            pl_same += int((enc[:, 6:] == orig[:, 6:]).all(axis=1).sum())
        out[name] = dict(experts=min(per_code, len(gl_all[code])), values=n,
                         decode_identical_to_ours_shader_order=same_shader,
                         decode_identical_to_ours_cpu_order=same_cpu, max_rel_diff_vs_shader_order=max_rel,
                         superblocks=n_sb, reencode_header_identical=hdr_same, reencode_payload_identical=pl_same)
        print(f"[reference_codec] {name}: decode_sk vs ours: {same_cpu}/{n} bit-identical with the CPU scale order, "
              f"{same_shader}/{n} with the shader order (max rel diff {max_rel:.2e}); encode_sk(our decode): "
              f"{hdr_same}/{n_sb} super-block headers and {pl_same}/{n_sb} payloads identical to the pack")
    res["reference_codec"] = out


def passed(res):
    """False if any recorded check failed (the LoRA check is a measurement, not a pass/fail test)."""
    ok = all(res.get("synthetic_sk", {}).values())
    ok &= res.get("shader_reference", {}).get("passed", True)
    for v in res.get("roundtrip_experts", {}).values():
        ok &= v["eff_identical"] == v["n"] and v["raw_identical"] == v["n"]
    ok &= all(v["identical"] for v in res.get("roundtrip_trunk", {}).values())
    ok &= all(v["code_changes"] == 0 for v in res.get("cpu_vs_shader_eff", {}).values())
    ok &= all(v["decode_identical_to_ours_cpu_order"] == v["values"] for v in res.get("reference_codec", {}).values())
    return bool(ok)


def main():
    ap = argparse.ArgumentParser(description="bit-exactness tests for strata_codecs.py")
    ap.add_argument("--pack", required=True, help="Inkling-2-Mini-SQ-15.strata")
    ap.add_argument("--strata-src", help="Strata checkout at commit 26350e4 (for the shader transcription test)")
    ap.add_argument("--lora", help="the rank-1 LoRA .gguf (for the LoRA consistency check)")
    ap.add_argument("--all", action="store_true", help="round-trip every expert")
    ap.add_argument("--quick", action="store_true", help="skip the trunk round trip and the LoRA check")
    ap.add_argument("--per-code", type=int, default=24, help="sampled experts per down codec")
    ap.add_argument("--out", help="results JSON (default results/tests.json or results/tests_all.json)")
    a = ap.parse_args()
    pk = sc.StrataPack(a.pack)
    res = {}
    gl = sample_gids(pk, per_code=a.per_code)
    if a.all:
        gl_rt = {c: [i for i in range(3072) if pk.codes(i)[1] == c] for c in (15, 3, 4, 5)}
        gl_rt["S5_any"] = gl["S5_any"]
    else:
        gl_rt = gl
    test_synthetic_sk(res)
    metal = os.path.join(a.strata_src, METAL_REL) if a.strata_src else None
    if metal and os.path.exists(metal):
        test_shader_reference(pk, gl, res, metal)
    else:
        print("[shader_reference] SKIPPED (pass --strata-src pointing at a Strata checkout of commit 26350e4)")
    test_roundtrip_experts(pk, gl_rt, res)
    if not a.quick:
        test_roundtrip_trunk(pk, res)
    test_cpu_vs_shader_eff(pk, gl, res)
    squant = os.path.join(a.strata_src, SQUANT_REL) if a.strata_src else None
    if squant and os.path.exists(squant):
        test_reference_codec(pk, gl, res, squant)
    else:
        print("[reference_codec] SKIPPED (pass --strata-src)")
    if not a.quick:
        if a.lora:
            test_lora_consistency(pk, gl, res, a.lora)
        else:
            print("[lora_consistency] SKIPPED (pass --lora)")
    fn = a.out or os.path.join("results", "tests_all.json" if a.all else "tests.json")
    os.makedirs(os.path.dirname(os.path.abspath(fn)), exist_ok=True)
    json.dump(res, open(fn, "w"), indent=1)
    ok = passed(res)
    print("wrote", fn)
    print("ALL PASSED" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
