"""strata_codecs.py - bit-exact numpy codecs for the weights a rank-1 output-projection edit touches in an
Inkling-2-Mini `.strata` v1.1 pack (SQ-15): Q8_ROW and BF16 trunk tensors, and the expert down matrices in
SF4 / S3_K / S4_K / S5_K (plus S2_K, S6_K, S8_K for completeness).

Everything works on a read-only mmap of the pack.  Nothing here writes a pack.

Value semantics follow the Metal shaders (ink_metal_codecs.metal / ink_metal_mpp.metal in Strata commit
26350e4), which is what runs on Apple GPUs; codecs/FORMAT.md describes the bit layouts and the CPU/shader
difference in the S_K scale product.

Public API
----------
    pk = StrataPack(path)
    pk.decode_expert_down(gid, space="eff", fp16=False) -> float32 [2304, 1152]
    pk.encode_expert_down(gid, W, space="eff", r=None, ...) -> bytes   (same length as the original span)
    pk.decode_trunk(name, fp16=False) -> float32 [N, K]
    pk.encode_trunk(name, W, r=None) -> bytes
    pk.expert_down_span(gid) -> (abs_file_offset, nbytes)
    pk.trunk_span(name)      -> (abs_file_offset, nbytes)

gid = layer * 128 + local_expert_id (the id the router produces).  NOTE: the Section-3 directory is
NOT sorted by expert id inside a layer; entry i belongs to layer i // 128 and carries its local id in
`expert_id`.  Always go through pk.entry(gid).

space="eff": the effective down matrix the model computes with, i.e. for S_K experts the decoded
    values times inv_sq[col] = 1/sq[col] (the per-expert fp16 vector at body[0:2304], called sq_xd_dn in the
    Strata source, that the engine multiplies into the gate/up activation before the down GEMM).
    For SF4 eff == raw.
space="raw": the decoded down codes x block scales, without the inv_sq column scale.

r (optional, unit vector of length 2304): direction-aware rounding.  Codes are chosen so that
r . (W_new - W) ~ 0 for every column (down / wo_ud) or (W_new - W) . r ~ 0 for every row (embed), instead of
plain per-weight nearest rounding: method="greedy" (default, patch/rounding.greedy_round) or "ef" (sequential
error feedback, patch/rounding.ef_round).  All side information (scales, headers, sq) is kept bit-identical in
every mode.
"""
from __future__ import annotations

import json
import mmap
import os
import struct
import sys

import numpy as np

_PATCH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "patch")
if _PATCH not in sys.path:
    sys.path.insert(0, _PATCH)
from rounding import ef_round, greedy_round  # noqa: E402,F401  (re-exported)

HIDDEN = 2304
INTER = 1152
N_EXPERTS = 128
N_LAYERS = 24

# ----------------------------------------------------------------------------------------------
# Format tables
# ----------------------------------------------------------------------------------------------
SK_SB_BYTES = {2: 70, 3: 102, 4: 134, 5: 166, 6: 198, 8: 262}
SK_HALF = {2: 1.5, 3: 3.5, 4: 7.5, 5: 15.5, 6: 31.5, 8: 0.0}
SK_QMIN = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 8: -127}
SK_QMAX = {2: 3, 3: 7, 4: 15, 5: 31, 6: 63, 8: 127}
CODE_NAME = {2: "S2_K", 3: "S3_K", 4: "S4_K", 5: "S5_K", 6: "S6_K", 8: "S8_K", 15: "SF4", 16: "MX8"}

E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                 -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)
E2M1_MAG = E2M1[:8].copy()
INV255 = np.float32(1.0) / np.float32(255.0)   # the shader's compile-time constant (1.0f / 255.0f)

# SF4 expert body: 16 B header (4 x f32: gu_amax, gu_denom, dn_amax, dn_denom)
SF4_GU_P = 2 * INTER * (HIDDEN // 2)       # 2,654,208
SF4_GU_S = 2 * INTER * (HIDDEN // 16)      # 331,776
SF4_DN_P = HIDDEN * (INTER // 2)           # 1,327,104
SF4_DN_S = HIDDEN * (INTER // 16)          # 165,888
SQ_BYTES = 2 * INTER                       # S_K experts: fp16 sq[1152] at body[0:2304]


def e4m3_table() -> np.ndarray:
    """F8_E4M3FN -> f32 exactly as e4m3_to_f32() in the shader (0x7F/0xFF NaN -> 0)."""
    out = np.zeros(256, np.float32)
    for c in range(256):
        mag = c & 0x7F
        if mag == 0x7F:
            v = 0.0
        else:
            e, m = mag >> 3, mag & 7
            if e == 0:
                v = float(np.float32(m) * np.float32(1.0 / 512.0))
            else:
                v = float(np.array([((e + 120) << 23) | (m << 20)], np.uint32).view(np.float32)[0])
        out[c] = -v if (c & 0x80) else v
    return out


E4M3 = e4m3_table()


def sk_dn_bytes(code: int) -> int:
    return INTER * 9 * SK_SB_BYTES[code]


def dn_bytes(code: int) -> int:
    if code == 15:
        return SF4_DN_P + SF4_DN_S
    if code == 16:
        return HIDDEN * INTER + HIDDEN * (INTER // 32)
    return sk_dn_bytes(code)


# ----------------------------------------------------------------------------------------------
# S_K super-blocks: [fp16 D][u8 u[4]][payload]; 256 weights = 4 sub-groups of 64;
# value = (q - half) * eff, eff = f32(D) * (f32(u[g]) * (1.0f/255.0f))   (shader formula)
# ----------------------------------------------------------------------------------------------
def sk_unpack_q(pl: np.ndarray, code: int) -> np.ndarray:
    """payload [n, SB-6] uint8 -> q [n, 256] (uint8, or int8 for S8_K)."""
    n = pl.shape[0]
    if code == 8:
        return pl[:, :256].view(np.int8).copy()
    if code in (2, 3):
        lo = pl[:, :64]
        q = np.stack([lo & 3, (lo >> 2) & 3, (lo >> 4) & 3, lo >> 6], axis=-1).reshape(n, 256)
        if code == 3:
            q = q | (np.unpackbits(pl[:, 64:96], axis=1, bitorder="little") << 2)
        return q
    lo = pl[:, :128]
    q = np.stack([lo & 15, lo >> 4], axis=-1).reshape(n, 256)
    if code == 5:
        q = q | (np.unpackbits(pl[:, 128:160], axis=1, bitorder="little") << 4)
    elif code == 6:
        h = pl[:, 128:192]
        hq = np.stack([h & 3, (h >> 2) & 3, (h >> 4) & 3, h >> 6], axis=-1).reshape(n, 256)
        q = q | (hq << 4)
    return q


def sk_pack_q(q: np.ndarray, code: int) -> np.ndarray:
    """q [n, 256] -> payload [n, SB-6] uint8 (inverse of sk_unpack_q)."""
    n = q.shape[0]
    if code == 8:
        return q.astype(np.int8).view(np.uint8).reshape(n, 256)
    q = q.astype(np.uint8)
    if code in (2, 3):
        b = (q & 3).reshape(n, 64, 4)
        lo = b[..., 0] | (b[..., 1] << 2) | (b[..., 2] << 4) | (b[..., 3] << 6)
        if code == 2:
            return lo
        hi = np.packbits((q >> 2) & 1, axis=1, bitorder="little")
        return np.concatenate([lo, hi], axis=1)
    b = (q & 15).reshape(n, 128, 2)
    lo = b[..., 0] | (b[..., 1] << 4)
    if code == 4:
        return lo
    if code == 5:
        return np.concatenate([lo, np.packbits((q >> 4) & 1, axis=1, bitorder="little")], axis=1)
    h = ((q >> 4) & 3).reshape(n, 64, 4)
    hi = h[..., 0] | (h[..., 1] << 2) | (h[..., 2] << 4) | (h[..., 3] << 6)
    return np.concatenate([lo, hi], axis=1)


S8_SHIFT = {3: 5, 4: 4, 5: 3}   # q' = 2^k (q - half) stays within [-127, 127]


def sk_to_s8_exact(raw: np.ndarray, code: int) -> np.ndarray:
    """Re-express an S3/S4/S5_K span as S8_K with identical decoded values: D' = D / 2^k (exact in
    fp16 while D' stays normal), u unchanged, q' = 2^k (q - half).  Raises if D' would be subnormal."""
    k = S8_SHIFT[code]
    sb = np.asarray(raw).reshape(-1, SK_SB_BYTES[code])
    d = sb[:, :2].copy().view("<f2")[:, 0]
    d2 = (d.astype(np.float32) / np.float32(2 ** k)).astype(np.float16)
    if not np.all(d2.astype(np.float32) * np.float32(2 ** k) == d.astype(np.float32)) or np.any(np.abs(d2) < np.float16(6.1035e-05)):
        raise ValueError("D / 2^k not exact in fp16")
    q = sk_unpack_q(sb[:, 6:], code).astype(np.int16)
    q2 = ((q * 2 - int(2 * SK_HALF[code])) * (2 ** (k - 1))).astype(np.int8)   # 2^k (q - half)
    hdr = np.concatenate([d2.reshape(-1, 1).view(np.uint8).reshape(-1, 2), sb[:, 2:6]], axis=1)
    return np.concatenate([hdr, sk_pack_q(q2, 8)], axis=1).reshape(-1)


def sk_split(raw: np.ndarray, code: int):
    sb = raw.reshape(-1, SK_SB_BYTES[code])
    d = sb[:, :2].copy().view("<f2")[:, 0].astype(np.float32)
    u = sb[:, 2:6].astype(np.float32)
    eff = d[:, None] * (u * INV255)                     # [n_sb, 4] f32, shader order
    return sb, eff


def decode_sk_span(raw, code: int, rows: int = HIDDEN, cols: int = INTER) -> np.ndarray:
    """Standalone paired-2-row S_K down-span decoder (raw space, shader formula)."""
    sb, eff = sk_split(np.asarray(raw), code)
    q = sk_unpack_q(sb[:, 6:], code)
    w = (q.astype(np.float32) - np.float32(SK_HALF[code])) * np.repeat(eff, 64, axis=1)
    return w.reshape(rows, cols).astype(np.float32)


def sk_eff_cpu(raw: np.ndarray, code: int) -> np.ndarray:
    """CPU dequant.cpp order: D * (u / 255.0f).  Only used to quantify the CPU/shader difference."""
    sb = raw.reshape(-1, SK_SB_BYTES[code])
    d = sb[:, :2].copy().view("<f2")[:, 0].astype(np.float32)
    return d[:, None] * (sb[:, 2:6].astype(np.float32) / np.float32(255.0))


# ----------------------------------------------------------------------------------------------
# Nearest-grid helpers.  Each returns (codes, values_f32) for target t and per-element scales.
# Exact-hit guarantee: if t equals a grid value bit-for-bit (incl. -0.0 for SF4), that code is chosen.
# ----------------------------------------------------------------------------------------------
def nearest_sk(t: np.ndarray, eff: np.ndarray, code: int):
    half = np.float32(SK_HALF[code])
    qmin, qmax = SK_QMIN[code], SK_QMAX[code]
    t64 = t.astype(np.float64)
    e64 = eff.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = t64 / e64 + float(half)
    x = np.where(np.isfinite(x), x, 0.0)
    qa = np.clip(np.floor(x), qmin, qmax)
    qb = np.clip(qa + 1, qmin, qmax)
    va = ((qa.astype(np.float32) - half) * eff)
    vb = ((qb.astype(np.float32) - half) * eff)
    pick_b = np.abs(vb.astype(np.float64) - t64) < np.abs(va.astype(np.float64) - t64)
    q = np.where(pick_b, qb, qa)
    v = np.where(pick_b, vb, va).astype(np.float32)
    dt = np.int8 if code == 8 else np.uint8
    return q.astype(dt), v


def nearest_sf4(t: np.ndarray, sc: np.ndarray):
    """sc: per-element f32 scale (= e4m3(s) * outer).  Returns nibbles uint8 and f32 values."""
    t = t.astype(np.float32)
    mag = np.abs(t).astype(np.float64)
    cand = (E2M1_MAG[:, None] * sc.reshape(1, -1).astype(np.float32)).astype(np.float64)  # [8, n]
    idx = np.argmin(np.abs(cand - mag.reshape(1, -1)), axis=0).astype(np.uint8)            # ties -> lower mag
    idx = idx.reshape(t.shape)
    neg = np.signbit(t)
    nib = idx | (neg.astype(np.uint8) << 3)
    v = E2M1[nib] * sc.astype(np.float32)
    return nib, v.astype(np.float32)


def nearest_q8(t: np.ndarray, sc: np.ndarray):
    t64 = t.astype(np.float64)
    s32 = sc.astype(np.float32)
    with np.errstate(divide="ignore", invalid="ignore"):
        x = t64 / s32.astype(np.float64)
    x = np.where(np.isfinite(x), x, 0.0)
    qa = np.clip(np.floor(x), -127, 127)
    qb = np.clip(qa + 1, -127, 127)
    va = qa.astype(np.float32) * s32
    vb = qb.astype(np.float32) * s32
    pick_b = np.abs(vb.astype(np.float64) - t64) < np.abs(va.astype(np.float64) - t64)
    return np.where(pick_b, qb, qa).astype(np.int8), np.where(pick_b, vb, va).astype(np.float32)


def f32_to_bf16(x: np.ndarray) -> np.ndarray:
    b = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    rnd = ((b >> 16) & 1) + np.uint32(0x7FFF)
    return ((b + rnd) >> 16).astype(np.uint16)


def bf16_to_f32(b: np.ndarray) -> np.ndarray:
    return (b.astype(np.uint32) << 16).view(np.float32)


# ----------------------------------------------------------------------------------------------
# Grids: per-element fixed-scale code books with nearest() and step() (move one grid point up/down).
# ----------------------------------------------------------------------------------------------
class SKGrid:
    def __init__(self, eff: np.ndarray, code: int):
        self.eff, self.code = eff, code
        self.dtype = np.int8 if code == 8 else np.uint8

    def nearest(self, T):
        return nearest_sk(T, self.eff, self.code)

    def step(self, q, d, T):
        q2 = np.clip(q.astype(np.int16) + d, SK_QMIN[self.code], SK_QMAX[self.code])
        v = (q2.astype(np.float32) - np.float32(SK_HALF[self.code])) * self.eff
        return q2.astype(self.dtype), v.astype(np.float32)

    def row(self, i):
        return SKGrid(self.eff[i], self.code)


class SF4Grid:
    """sc: per-element f32 scale (e4m3(s) * outer), assumed > 0 (true for every block in SQ-15)."""
    dtype = np.uint8

    def __init__(self, sc: np.ndarray):
        self.sc = sc

    def nearest(self, T):
        out_n = np.empty(T.shape, np.uint8)
        out_v = np.empty(T.shape, np.float32)
        if T.ndim == 2:
            for a in range(0, T.shape[0], 256):
                out_n[a:a + 256], out_v[a:a + 256] = nearest_sf4(T[a:a + 256], self.sc[a:a + 256])
            return out_n, out_v
        return nearest_sf4(T, self.sc)

    def step(self, nib, d, T):
        m = np.where(nib >= 8, -(nib & 7).astype(np.int16), (nib & 7).astype(np.int16))
        m2 = np.clip(m + d, -7, 7)
        nib2 = (np.abs(m2) | ((m2 < 0) << 3)).astype(np.uint8)
        nib2 = np.where(m2 == 0, np.signbit(T).astype(np.uint8) << 3, nib2).astype(np.uint8)
        return nib2, (E2M1[nib2] * self.sc).astype(np.float32)

    def row(self, i):
        return SF4Grid(self.sc[i])


class Q8Grid:
    dtype = np.int8

    def __init__(self, sc: np.ndarray):
        self.sc = sc

    def nearest(self, T):
        return nearest_q8(T, self.sc)

    def step(self, q, d, T):
        q2 = np.clip(q.astype(np.int16) + d, -127, 127)
        return q2.astype(np.int8), (q2.astype(np.float32) * self.sc.astype(np.float32)).astype(np.float32)

    def row(self, i):
        return Q8Grid(self.sc[i])


# greedy_round (direction-aware rounding) and ef_round (error-feedback rounding) are format-agnostic and live in
# patch/rounding.py; they are imported above and re-exported here.


# ----------------------------------------------------------------------------------------------
# Pack access
# ----------------------------------------------------------------------------------------------
DIR_DT = np.dtype([("eid", "<u2"), ("gu", "u1"), ("dn", "u1"), ("cluster", "<u2"), ("flags", "<u2"),
                   ("body", "<u8"), ("body_bytes", "<u4"), ("dn_off", "<u4"), ("aux0", "<u4"), ("aux1", "<u4")])


class StrataPack:
    def __init__(self, path: str, manifest_json: str | None = None):
        self.path = path
        self._f = open(path, "rb")
        self.mm = mmap.mmap(self._f.fileno(), 0, access=mmap.ACCESS_READ)
        self.buf = np.frombuffer(self.mm, np.uint8)
        h = bytes(self.mm[:4096])
        assert h[:8] == b"STRATA\x00\x01", "bad magic"
        self.hdr_flags, self.profile, nsec = struct.unpack_from("<IHH", h, 8)
        assert nsec == 6
        self.secs = []
        for i in range(nsec):
            off, ln, sha, sid, fl, _ = struct.unpack_from("<QQ32sIIQ", h, 0x10 + 64 * i)
            self.secs.append(dict(off=off, len=ln, sha=sha, sid=sid, flags=fl))
        self.sec1 = self.secs[1]["off"]
        self.sec4 = self.secs[4]["off"]
        self.manifest = self._load_manifest(manifest_json)
        self.trunk = {t["name"]: t for t in self.manifest["trunk_tensors"]}
        s3 = self.secs[3]
        raw = self.buf[s3["off"]: s3["off"] + s3["len"]]
        n_ent = s3["len"] // 34 if (s3["flags"] & 1) else s3["len"] // 32
        self.dir = np.frombuffer(raw[: n_ent * 32].tobytes(), DIR_DT)
        self.vram_order = np.frombuffer(raw[n_ent * 32: n_ent * 34].tobytes(), "<u2") if (s3["flags"] & 1) else None
        layer = np.arange(n_ent) // N_EXPERTS
        gid = layer * N_EXPERTS + self.dir["eid"].astype(np.int64)
        self.gid_to_entry = np.full(N_LAYERS * N_EXPERTS, -1, np.int64)
        self.gid_to_entry[gid] = np.arange(n_ent)
        assert (self.gid_to_entry >= 0).all(), "directory does not cover every (layer, expert)"

    def _load_manifest(self, manifest_json):
        s0 = self.secs[0]
        raw = bytes(self.mm[s0["off"]: s0["off"] + s0["len"]])
        try:
            import zstandard
            return json.loads(zstandard.ZstdDecompressor().decompress(raw, max_output_size=1 << 30))
        except ImportError:
            # Without the zstandard module: a manifest JSON file, by default tools/SQ-15-manifest.json next to the
            # pack (the pack's distribution ships one there).
            p = manifest_json or os.path.join(os.path.dirname(os.path.abspath(self.path)), "tools", "SQ-15-manifest.json")
            return json.load(open(p))

    # ---------------- experts ----------------
    def entry(self, gid: int):
        return self.dir[self.gid_to_entry[gid]]

    def codes(self, gid: int):
        e = self.entry(gid)
        return int(e["gu"]), int(e["dn"])

    def expert_body_span(self, gid: int):
        e = self.entry(gid)
        return self.sec4 + int(e["body"]), int(e["body_bytes"])

    def expert_down_span(self, gid: int):
        e = self.entry(gid)
        return self.sec4 + int(e["body"]) + int(e["dn_off"]), dn_bytes(int(e["dn"]))

    def _raw(self, off, n):
        return self.buf[off: off + n]

    def expert_sq(self, gid: int) -> np.ndarray:
        """fp16 sq[1152] as f32 (S_K down only)."""
        b, _ = self.expert_body_span(gid)
        return self._raw(b, SQ_BYTES).view("<f2").astype(np.float32)

    def expert_inv_sq(self, gid: int) -> np.ndarray:
        return (np.float32(1.0) / self.expert_sq(gid)).astype(np.float32)

    def sf4_dn_outer(self, gid: int) -> np.float32:
        b, _ = self.expert_body_span(gid)
        hdr = self._raw(b, 16).view("<f4")
        return np.float32(hdr[2] / hdr[3])                  # f32 IEEE division (MathModeSafe)

    def _sf4_dn_scale_matrix(self, gid):
        off, _ = self.expert_down_span(gid)
        s = self._raw(off + SF4_DN_P, SF4_DN_S).reshape(HIDDEN, INTER // 16)
        sc = E4M3[s] * self.sf4_dn_outer(gid)               # f32 [2304, 72]
        return s, sc

    def _sk_eff_matrix(self, gid, code):
        off, n = self.expert_down_span(gid)
        _, eff = sk_split(self._raw(off, n), code)          # [1152*9, 4]
        # pair p = 9 SBs = 2304 weights = rows 2p, 2p+1 back to back; each eff covers 64 weights
        return np.repeat(eff.reshape(-1), 64).reshape(HIDDEN, INTER)

    def decode_expert_down_codes(self, gid: int, raw: bytes | None = None):
        """Returns (code_id, codes [2304,1152]).  SF4 codes are nibbles 0..15.
        raw: optional replacement bytes for the down span (e.g. the output of encode_expert_down)."""
        _, dn = self.codes(gid)
        off, n = self.expert_down_span(gid)
        raw = self._raw(off, n) if raw is None else np.frombuffer(raw, np.uint8)
        assert raw.size == n
        if dn == 15:
            p = raw[:SF4_DN_P].reshape(HIDDEN, INTER // 2)
            nib = np.stack([p & 15, p >> 4], axis=-1).reshape(HIDDEN, INTER)   # low nibble = even column
            return dn, nib
        if dn in SK_SB_BYTES:
            sb = raw.reshape(-1, SK_SB_BYTES[dn])
            q = sk_unpack_q(sb[:, 6:], dn)                   # [1152*9, 256]
            return dn, q.reshape(HIDDEN, INTER)
        raise NotImplementedError(f"dn code {dn}")

    def decode_expert_down(self, gid: int, space: str = "eff", fp16: bool = False, raw: bytes | None = None) -> np.ndarray:
        """float32 [2304 out, 1152 in].  fp16=True rounds the per-weight value to half first, as the
        prefill (simdgroup/MPP) kernels do; eff scaling is then applied on top (as the engine applies
        inv_sq to the activation, not to the weight).  raw: decode these down-span bytes instead of the
        pack's (side information is taken from the bytes themselves, header/sq from the pack)."""
        dn, q = self.decode_expert_down_codes(gid, raw)
        if raw is not None and dn != 15:
            eff = np.repeat(sk_split(np.frombuffer(raw, np.uint8), dn)[1].reshape(-1), 64).reshape(HIDDEN, INTER)
            W = ((q.astype(np.float32) - np.float32(SK_HALF[dn])) * eff).astype(np.float32)
            if fp16:
                W = W.astype(np.float16).astype(np.float32)
            return W * self.expert_inv_sq(gid)[None, :] if space == "eff" else W
        if raw is not None and dn == 15:
            s = np.frombuffer(raw, np.uint8)[SF4_DN_P:].reshape(HIDDEN, INTER // 16)
            W = (E2M1[q] * np.repeat(E4M3[s] * self.sf4_dn_outer(gid), 16, axis=1)).astype(np.float32)
            return W.astype(np.float16).astype(np.float32) if fp16 else W
        if dn == 15:
            _, sc = self._sf4_dn_scale_matrix(gid)
            W = E2M1[q] * np.repeat(sc, 16, axis=1)
        else:
            eff = self._sk_eff_matrix(gid, dn)
            W = (q.astype(np.float32) - np.float32(SK_HALF[dn])) * eff
        W = W.astype(np.float32)
        if fp16:
            W = W.astype(np.float16).astype(np.float32)
        if space == "eff" and dn != 15:
            W = W * self.expert_inv_sq(gid)[None, :]
        return W

    def down_grid(self, gid: int):
        dn = self.codes(gid)[1]
        if dn == 15:
            return SF4Grid(np.repeat(self._sf4_dn_scale_matrix(gid)[1], 16, axis=1))
        return SKGrid(self._sk_eff_matrix(gid, dn), dn)

    def encode_expert_down(self, gid: int, W: np.ndarray, space: str = "eff", r: np.ndarray | None = None,
                           method: str = "greedy", keep_degenerate: bool = True, return_values: bool = False,
                           passes: int = 3):
        """Re-encode the down matrix with all side information fixed.  Returns the bytes of the down
        span (same length/offset as the original).  W is [2304, 1152] in the given space.
        r=None: nearest code per weight.  r given: ablation-aware rounding along r over the output
        axis, method 'greedy' (recommended, minimum added error) or 'ef' (strict sequential
        error feedback)."""
        dn = self.codes(gid)[1]
        off, n = self.expert_down_span(gid)
        raw = self._raw(off, n)
        W = np.asarray(W)
        if space == "eff" and dn != 15:
            T = W.astype(np.float64) * self.expert_sq(gid).astype(np.float64)[None, :]
        else:
            T = W
        grid = self.down_grid(gid)
        if r is None:
            q, vals = grid.nearest(T)
        elif method == "greedy":
            q, vals = greedy_round(T, r, grid, passes=passes)
        elif method == "ef":
            q, vals = ef_round(T, r, lambda i, t: grid.row(i).nearest(t), grid.dtype)
        else:
            raise ValueError(method)
        if dn == 15:
            nib = q
            if keep_degenerate:
                deg = grid.sc == 0
                if deg.any():
                    _, q0 = self.decode_expert_down_codes(gid)
                    nib = np.where(deg & (np.signbit(T) == (q0 >= 8)), q0, nib)
            p = (nib[:, 0::2] | (nib[:, 1::2] << 4)).astype(np.uint8)
            out = np.concatenate([p.reshape(-1), np.asarray(raw[SF4_DN_P:])])
        elif dn in SK_SB_BYTES:
            SB = SK_SB_BYTES[dn]
            sb = raw.reshape(-1, SB)
            pl = sk_pack_q(q.reshape(-1, 256), dn)
            out = np.concatenate([np.asarray(sb[:, :6]), pl], axis=1).reshape(-1)
        else:
            raise NotImplementedError(f"dn code {dn}")
        assert out.size == n
        if return_values:
            v = vals.astype(np.float32)
            if space == "eff" and dn != 15:
                v = v * self.expert_inv_sq(gid)[None, :]
            return out.tobytes(), v
        return out.tobytes()

    def down_grid_range(self, gid: int, space: str = "raw"):
        """Per-weight representable [lo, hi] of the fixed-scale grid (for clipping statistics)."""
        dn = self.codes(gid)[1]
        if dn == 15:
            _, sc = self._sf4_dn_scale_matrix(gid)
            hi = np.repeat(sc, 16, axis=1) * np.float32(6.0)
            lo = -hi
        else:
            eff = self._sk_eff_matrix(gid, dn)
            lo = (np.float32(SK_QMIN[dn]) - np.float32(SK_HALF[dn])) * eff
            hi = (np.float32(SK_QMAX[dn]) - np.float32(SK_HALF[dn])) * eff
        if space == "eff" and dn != 15:
            isq = self.expert_inv_sq(gid)[None, :]
            lo, hi = lo * isq, hi * isq
        return lo, hi

    # ---------------- trunk ----------------
    def trunk_span(self, name: str):
        t = self.trunk[name]
        return self.sec1 + int(t["offset"]), int(t["bytes"])

    def trunk_format(self, name: str) -> str:
        return self.trunk[name]["format"]

    def trunk_q8(self, name):
        t = self.trunk[name]
        N, K = t["shape"]
        off, n = self.trunk_span(name)
        raw = self._raw(off, n)
        sc = raw[: 4 * N].view("<f4")
        q = raw[4 * N:].view(np.int8).reshape(N, K)
        return sc, q

    def decode_trunk(self, name: str, fp16: bool = False, rows: slice | None = None) -> np.ndarray:
        t = self.trunk[name]
        N, K = t["shape"]
        off, n = self.trunk_span(name)
        fmt = t["format"]
        if fmt == "Q8_ROW":
            sc, q = self.trunk_q8(name)
            if rows is not None:
                sc, q = sc[rows], q[rows]
            W = q.astype(np.float32) * sc[:, None]
        elif fmt == "BF16":
            W = bf16_to_f32(self._raw(off, n).view("<u2").reshape(N, K) if rows is None
                            else self._raw(off, n).view("<u2").reshape(N, K)[rows])
        elif fmt == "F32":
            W = self._raw(off, n).view("<f4").reshape(N, K).copy()
        else:
            raise NotImplementedError(fmt)
        if fp16:
            W = W.astype(np.float16).astype(np.float32)
        return W

    def encode_trunk(self, name: str, W: np.ndarray, r: np.ndarray | None = None, return_values: bool = False,
                     chunk: int = 16384, method: str = "greedy", passes: int = 3):
        t = self.trunk[name]
        N, K = t["shape"]
        fmt = t["format"]
        if fmt == "BF16":
            # BF16 is a float format: ablation-aware rounding is not needed (error ~2^-9 relative).
            b = f32_to_bf16(np.asarray(W, np.float32))
            return (b.tobytes(), bf16_to_f32(b)) if return_values else b.tobytes()
        if fmt != "Q8_ROW":
            raise NotImplementedError(fmt)
        sc, _ = self.trunk_q8(name)
        sc = np.asarray(sc, np.float32)
        q = np.empty((N, K), np.int8)
        vals = np.empty((N, K), np.float32) if return_values else None
        if r is None:
            for a in range(0, N, chunk):
                b = min(N, a + chunk)
                qq, vv = nearest_q8(W[a:b], np.broadcast_to(sc[a:b, None], (b - a, K)))
                q[a:b] = qq
                if vals is not None:
                    vals[a:b] = vv
        elif len(r) == N:            # constraint over rows: r^T W per column (wo_ud: r length 2304 = N)
            grid = Q8Grid(np.broadcast_to(sc[:, None], (N, K)))
            if method == "greedy":
                qq, vv = greedy_round(np.asarray(W), r, grid, passes=passes)
            else:
                qq, vv = ef_round(np.asarray(W), r, lambda i, tt: nearest_q8(tt, np.full(K, sc[i], np.float32)), np.int8)
            q[:] = qq
            if vals is not None:
                vals[:] = vv
        elif len(r) == K:            # constraint over columns: W r per row (embed)
            for a in range(0, N, chunk):
                b = min(N, a + chunk)
                Tt = np.ascontiguousarray(np.asarray(W[a:b]).T)       # [K, rows]
                scc = sc[a:b]
                if method == "greedy":
                    qq, vv = greedy_round(Tt, r, Q8Grid(np.broadcast_to(scc[None, :], Tt.shape)), passes=passes)
                else:
                    qq, vv = ef_round(Tt, r, lambda i, tt: nearest_q8(tt, scc), np.int8)
                q[a:b] = qq.T
                if vals is not None:
                    vals[a:b] = vv.T
        else:
            raise ValueError("r length matches neither axis")
        out = sc.tobytes() + q.tobytes()
        assert len(out) == t["bytes"]
        return (out, vals) if return_values else out

    def trunk_grid_range(self, name):
        sc, _ = self.trunk_q8(name)
        return -127.0 * np.asarray(sc), 127.0 * np.asarray(sc)


# ----------------------------------------------------------------------------------------------
# Convenience: the recommended edit (re-derive the projection on the pack's own weights, then
# ablation-aware greedy rounding with all side information fixed).
# ----------------------------------------------------------------------------------------------
def abliterate_expert_down(pk: StrataPack, gid: int, r: np.ndarray, passes: int = 3) -> bytes:
    """W' = W - r r^T W on the effective down matrix, re-encoded in place.  r: unit [2304]."""
    r = np.asarray(r, np.float64)
    W = pk.decode_expert_down(gid, space="raw").astype(np.float64)   # sq is a column scale: same projection
    T = W - np.outer(r, r @ W)
    return pk.encode_expert_down(gid, T, space="raw", r=r, method="greedy", passes=passes)


def abliterate_trunk(pk: StrataPack, name: str, r: np.ndarray, passes: int = 3, chunk: int = 16384) -> bytes:
    """wo_ud [2304, 3072]: W' = W - r r^T W.  embed [201024, 2304]: E' = E - (E r) r^T (row-chunked)."""
    r = np.asarray(r, np.float64)
    t = pk.trunk[name]
    N, K = t["shape"]
    fmt = t["format"]
    if len(r) == N:
        W = pk.decode_trunk(name).astype(np.float64)
        T = W - np.outer(r, r @ W)
        return pk.encode_trunk(name, T, r=(r if fmt == "Q8_ROW" else None), passes=passes)
    assert len(r) == K and fmt == "Q8_ROW"
    sc_all, _ = pk.trunk_q8(name)
    sc_all = np.asarray(sc_all, np.float32)
    q = np.empty((N, K), np.int8)
    for a in range(0, N, chunk):
        b = min(N, a + chunk)
        W = pk.decode_trunk(name, rows=slice(a, b)).astype(np.float64)
        T = (W - np.outer(W @ r, r)).T.copy()                       # [K, rows]: constraint axis first
        qq, _ = greedy_round(T, r, Q8Grid(np.broadcast_to(sc_all[None, a:b], T.shape)), passes=passes)
        q[a:b] = qq.T
    out = sc_all.tobytes() + q.tobytes()
    assert len(out) == t["bytes"]
    return out


def wo_name(layer: int) -> str:
    return f"model.llm.layers.{layer}.attn.wo_ud.weight"


EMBED = "model.llm.embed.weight"
