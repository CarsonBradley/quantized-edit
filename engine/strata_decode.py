"""strata_decode.py - minimal CPU decoders for the Inkling-2-Mini SQ-15 .strata pack + AtomicChat LoRA reader.

Only what the ablation checks need: header/sections, manifest (zstd), trunk tensors in BF16 / Q8_ROW / F32,
the Section 3 expert directory (keyed by expert_id), and SF4 (NVFP4) expert down matrices.
Layouts follow the Metal shaders at commit 26350e4 (ink_metal_ops.metal ink_embed_q8row, ink_metal_codecs.metal
decode_sb_sf4_16 / ink_moe_dn_mm_gen); verified against the LoRA (a = -W^T r to 1e-3 on SF4, 3e-7 on BF16).
"""
import hashlib
import json
import struct

import numpy as np

SEC_FMT = "<QQ32sIIQ"
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float64)


def _e4m3(c):
    mag = c & 0x7F
    if mag == 0x7F:
        return 0.0
    e, m = mag >> 3, mag & 7
    v = m / 512.0 if e == 0 else struct.unpack("<f", struct.pack("<I", ((e + 120) << 23) | (m << 20)))[0]
    return -v if c & 0x80 else v


E4M3 = np.array([_e4m3(i) for i in range(256)], dtype=np.float64)


class Pack:
    def __init__(self, path, manifest_json=None):
        self.path = path
        self.f = open(path, "rb")
        hdr = self.f.read(4096)
        assert hdr[:8] == b"STRATA\x00\x01", "bad magic"
        self.secs = [struct.unpack_from(SEC_FMT, hdr, 0x10 + 64 * i) for i in range(6)]
        self.mm = np.memmap(path, dtype=np.uint8, mode="r")
        if manifest_json is None:
            import zstandard  # only needed when no manifest JSON file is given
            off, ln = self.secs[0][0], self.secs[0][1]
            self.man = json.loads(zstandard.ZstdDecompressor().decompress(bytes(self.mm[off:off + ln])))
        else:
            self.man = json.load(open(manifest_json))
        self.T = {t["name"]: t for t in self.man["trunk_tensors"]}
        s3off, s3len = self.secs[3][0], self.secs[3][1]
        dt = np.dtype([("eid", "<u2"), ("gu", "u1"), ("dn", "u1"), ("cl", "<u2"), ("fl", "<u2"),
                       ("body", "<u8"), ("bb", "<u4"), ("dno", "<u4"), ("aux", "<u4", 2)])
        n = 24 * 128
        self.ent = np.frombuffer(bytes(self.mm[s3off:s3off + n * 32]), dtype=dt)
        # (layer, expert_id) -> directory row (the directory is NOT in expert_id order)
        self.dir_of = {}
        for i in range(n):
            self.dir_of[(i // 128, int(self.ent[i]["eid"]))] = i

    def trunk(self, name, rows=None):
        t = self.T[name]
        off = self.secs[1][0] + t["offset"]
        N, K = t["shape"]
        fmt = t["format"]
        if fmt == "BF16":
            raw = self.mm[off:off + N * K * 2].view(np.uint16).reshape(N, K)
            if rows is not None:
                raw = raw[rows]
            return (raw.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
        if fmt == "Q8_ROW":
            sc = self.mm[off:off + N * 4].view(np.float32).astype(np.float64)
            q = self.mm[off + N * 4:off + N * 4 + N * K].view(np.int8).reshape(N, K)
            if rows is not None:
                return q[rows].astype(np.float64) * sc[rows][:, None]
            return q.astype(np.float64) * sc[:, None]
        if fmt == "F32":
            a = self.mm[off:off + N * K * 4].view(np.float32).reshape(N, K).astype(np.float64)
            return a if rows is None else a[rows]
        raise NotImplementedError(f"{name}: {fmt}")

    def expert_dn_code(self, layer, eid):
        return int(self.ent[self.dir_of[(layer, eid)]]["dn"])

    def expert_down(self, layer, eid):
        """[2304, 1152] float64 down matrix for SF4 (15) or S3_K/S4_K/S5_K (3/4/5) experts."""
        code = self.expert_dn_code(layer, eid)
        if code == 15:
            return self.expert_down_sf4(layer, eid)
        if code in (3, 4, 5):
            return self.expert_down_sk(layer, eid, code)
        raise NotImplementedError(f"dn code {code}")

    def expert_down_sk(self, layer, eid, code):
        # Row-paired S_K (codecs.metal decode_sb_sk_16 / sk_w8, ink_moe_dn_mm_gen): pair rp = n >> 1 holds
        # 9 superblocks of 256 = [row 2rp (1152) | row 2rp+1 (1152)]. Superblock: fp16 d, u8 g[4] (64-col
        # group scales /255), then the quants. Value = level * d * g/255.
        SB = {3: 102, 4: 134, 5: 166}[code]
        e = self.ent[self.dir_of[(layer, eid)]]
        o = self.secs[4][0] + int(e["body"]) + int(e["dno"])
        blk = np.asarray(self.mm[o:o + 1152 * 9 * SB]).reshape(1152 * 9, SB)
        d = blk[:, 0:2].copy().view(np.float16).astype(np.float64)[:, 0]
        g = blk[:, 2:6].astype(np.float64) / 255.0
        pl = blk[:, 6:]
        j = np.arange(256)
        if code == 3:
            lo = ((pl[:, j >> 2] >> (2 * (j & 3))) & 3).astype(np.float64) - 1.5
            hi = np.where((pl[:, 64 + (j >> 3)] >> (j & 7)) & 1, 2.0, -2.0)
            lev = lo + hi
        else:
            lo = ((pl[:, j >> 1] >> (4 * (j & 1))) & 15).astype(np.float64) - 7.5
            lev = lo
            if code == 5:
                lev = lo + np.where((pl[:, 128 + (j >> 3)] >> (j & 7)) & 1, 8.0, -8.0)
        vals = lev * d[:, None] * np.repeat(g, 64, axis=1)       # [1152*9, 256]
        pairs = vals.reshape(1152, 2304)
        W = np.empty((2304, 1152))
        W[0::2] = pairs[:, :1152]
        W[1::2] = pairs[:, 1152:]
        return W

    def expert_down_sf4(self, layer, eid):
        """[2304, 1152] float64 down matrix of an SF4 expert."""
        e = self.ent[self.dir_of[(layer, eid)]]
        assert int(e["dn"]) == 15, "not SF4"
        base = self.secs[4][0] + int(e["body"])
        hdr = self.mm[base:base + 16].view(np.float32)
        outer = float(hdr[2]) / float(hdr[3])
        o = base + int(e["dno"])
        p = np.asarray(self.mm[o:o + 2304 * 576]).reshape(2304, 576)
        sc = E4M3[np.asarray(self.mm[o + 1327104:o + 1327104 + 2304 * 72]).reshape(2304, 72)] * outer
        q = np.empty((2304, 1152))
        q[:, 0::2] = E2M1[p & 15]
        q[:, 1::2] = E2M1[p >> 4]
        return q * np.repeat(sc, 16, axis=1)


class Lora:
    def __init__(self, path):
        from gguf import GGUFReader
        self.path = path
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        self.sha256 = h.digest()
        r = GGUFReader(path)
        self.meta = {}
        for k, fld in r.fields.items():
            if k.startswith("GGUF."):
                continue
            try:
                self.meta[k] = fld.contents()
            except Exception:
                pass
        self.T = {t.name: np.array(t.data, dtype=np.float64) for t in r.tensors}
        self.ne = {t.name: [int(x) for x in t.shape] for t in r.tensors}

    def attn(self, L):
        return (self.T[f"blk.{L}.attn_output.weight.lora_a"].reshape(-1),
                self.T[f"blk.{L}.attn_output.weight.lora_b"].reshape(-1))

    def experts(self, L):
        return (self.T[f"blk.{L}.ffn_down_exps.weight.lora_a"].reshape(128, 1152),
                self.T[f"blk.{L}.ffn_down_exps.weight.lora_b"].reshape(128, 2304))

    def embd(self):
        return (self.T["token_embd.weight.lora_a"].reshape(-1), self.T["token_embd.weight.lora_b"].reshape(-1))
