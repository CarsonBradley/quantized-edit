# The `.strata` pack, as far as an edit needs it

This describes the parts of Strata's `Inkling-2-Mini-SQ-15.strata` pack (manifest `format: "strata-v1.1"`,
`build: "SQ-15"`, 14,348,316,672 bytes) that a weight edit reads or writes: the header, the section table, the
manifest's trunk-tensor index, the expert directory, and the bit layouts of the codecs used by the tensors the
abliteration edit touches. Everything here was read from Strata's published decoders at commit `26350e4` (the C++
reader, the CPU dequantizer and the Metal shaders) and checked against the pack's bytes with
`codecs/test_codecs.py`. It is not a complete specification: gate/up matrices, the DSpark drafter tensors, the
vision pack and sections 2 and 5 are not described.

Terms used below:
- **trunk tensor**: a weight outside the routed experts (embedding, attention, norms, ...), stored in section 1.
- **expert**: one of the 128 routed experts of a layer; its gate/up and down matrices are stored in one **body**
  in section 4.
- **down matrix**: an expert's output projection, shape [2304 out, 1152 in] (hidden size 2304, expert width 1152).
- **gid**: global expert id = layer × 128 + local expert id, the id the router produces.

## Header (bytes 0..4095)

| Offset | Type | Value in SQ-15 |
|---|---|---|
| 0x000 | char[8] | magic `STRATA\0\1` |
| 0x008 | u32 | flags = 1 |
| 0x00C | u16 | profile = 0 |
| 0x00E | u16 | number of sections = 6 |
| 0x010 | 6 × 64 B | section entries, `<QQ32sIIQ`: file offset, byte length, sha256, section id, flags, reserved |
| 0x190.. | | zero |

All integers are little-endian. Sections in the SQ-15 pack:

| Section | Offset | Length | Content |
|---|---|---|---|
| 1 | 4,096 | 2,068,299,712 | trunk tensors, back to back (offsets in the manifest) |
| 4 | 2,068,303,872 | 12,279,889,920 | expert bodies |
| 3 | 14,348,193,792 | 104,448 | expert directory (flags = 1, see below) |
| 5 | 14,348,300,288 | 154 | self-check data (not described here) |
| 0 | 14,348,304,384 | 10,738 | manifest: zstd-compressed JSON |
| 2 | 0 | 0 | empty |

Every section starts on a 4 KiB boundary. Note that the manifest (section 0) is the *last* thing in the file;
Strata's public `tools/squant/strata_format.py`, written for the earlier `.strata` v1.0 layout, puts it at 0x1000.
The magic is the same in both.

**Hashes.** Each section entry carries the sha256 of its bytes. The engine checks them only when asked:
`strata-inkling --selfcheck --native PACK --verify-hash`. A pack with stale hashes loads and runs normally, so an
edit that forgets to update them fails silently later, if at all. After editing, recompute the sha256 of every
section you changed (for this edit: sections 1 and 4) and write it at offset 0x10 + 64 × index + 16.
`eval/verify_strata_hash.py` checks all six without filling the page cache.

## Manifest (section 0)

zstd-compressed JSON. The keys an edit needs:
- `trunk_tensors`: a list of `{name, format, shape, offset, bytes}`. `offset` is relative to the start of
  section 1. Formats in SQ-15: F32 (500 tensors), BF16 (84), Q8_ROW (61), S6_K (1, the unembedding).
- `num_physical_layers` = 24, `num_experts_per_layer` = 128.
- `code_counts`: expert counts per (gate/up, down) codec pair: SF4+SF4 1,538; S3_K+S3_K 1,022; S4_K+S4_K 509;
  S5_K+S4_K 1; S5_K+S5_K 2. So the down matrices are SF4 1,538, S3_K 1,022, S4_K 510, S5_K 2.
- `allocation`: `"global_hessian_knapsack_v2"`, the name of the method that chose a codec per expert. That tool is
  not in the public Strata repository.

## Expert directory (section 3)

3,072 entries of 32 bytes, `<HBBHHQIIII`:

| Field | Type | Meaning |
|---|---|---|
| expert_id | u16 | local expert id within the layer, 0..127 |
| gu_code, dn_code | u8, u8 | codec of the gate/up and of the down matrix (2..6, 8 = S2_K..S6_K, S8_K; 15 = SF4; 16 = MX8) |
| cluster_id, flags | u16, u16 | 0 in SQ-15; not read by the Metal engine |
| body_offset | u64 | offset of the body, relative to section 4 |
| body_bytes | u32 | body length (a multiple of 4 KiB in SQ-15) |
| dn_offset | u32 | offset of the down matrix within the body |
| aux0, aux1 | u32, u32 | 0 in SQ-15 |

When the section's flags have bit 0 set (they do in SQ-15, so the section is 3,072 × 34 bytes), the 32-byte
entries are followed by 3,072 u16 values: a load order the engine does not need for decoding.

**Pitfall: the directory is not in expert order.** Entry *i* belongs to layer *i* // 128, but within a layer the
entries are in some other order (layer 0 starts with expert ids 21, 18, 32, 70, ...). Look experts up by
`(i // 128, expert_id)`, as the engine does (`ink_metal_common.h`), never by position.

## Expert bodies (section 4)

**S_K experts** (both codecs S2_K..S8_K): `fp16 sq[1152]` at body bytes 0..2303, then the gate/up super-blocks, then
the down matrix at `dn_offset` (= 2,304 + the gate/up size).

**SF4 experts**: a 16-byte header of four f32 (gate/up amax, gate/up denominator, down amax, down denominator), then
the gate/up nibbles and scales, then the down matrix at `dn_offset` (= 16 + 2,654,208 + 331,776 = 2,986,000).

### Pitfall: the per-unit scale on S_K down matrices

For S_K experts the stored down matrix is not the matrix the model multiplies by. The engine divides the
expert's intermediate activation by `sq[j]` before the down projection (`ink_metal_codecs.metal`, the `inv_sq`
factor near line 1695; `inkling2_loop.cpp` near line 387), so the effective matrix is

    W_eff[:, j] = W_stored[:, j] / sq[j]

and the stored matrix sits at roughly four times the scale of the original weights (the LoRA's factors give
W_stored ≈ 4 W_eff on S3_K, S4_K and S5_K experts). An output-side projection `(I − r rᵀ)` commutes with a column
scale, so the edit itself can be computed on either; but anything that compares against or adds a delta computed
from the original weights, such as a LoRA's stored factors, must use `W_eff`. `strata_codecs.py` exposes both
(`space="eff"` and `space="raw"`). SF4 experts have no such vector (`eff == raw`).

## Codecs

### S_K super-blocks (S2_K, S3_K, S4_K, S5_K, S6_K, S8_K)

256 weights per super-block, in four sub-groups of 64 consecutive weights:

| Bytes | Content |
|---|---|
| 0..1 | fp16 D |
| 2..5 | u8 u[4], one per sub-group |
| 6.. | the codes q |

Value of weight k in sub-group g = k // 64: `(q − half) × D × u[g] / 255`, with `half` = 1.5, 3.5, 7.5, 15.5,
31.5 for 2..6 bits (symmetric levels, no exact zero), and for S8_K `q` is a signed int8 in [−127, 127] with no
offset. Super-block sizes: S2_K 70, S3_K 102, S4_K 134, S5_K 166, S6_K 198, S8_K 262 bytes.

Payload layouts (bit order within a byte is least-significant first):
- S2_K: 64 bytes, four 2-bit codes per byte.
- S3_K: 64 bytes of the low 2 bits (four per byte), then 32 bytes of the high bit (eight per byte).
- S4_K: 128 bytes, two 4-bit codes per byte (low nibble first).
- S5_K: 128 bytes of low nibbles, then 32 bytes of the fifth bit (eight per byte).
- S6_K: 128 bytes of low nibbles, then 64 bytes of the top 2 bits (four per byte).
- S8_K: 256 signed bytes.

**Scale product.** The Metal shaders compute the sub-group scale as `D × (u × (1/255))` with `1/255` a float
constant; the CPU dequantizer (`dequant.cpp`) computes `D × (u / 255)`. The two differ in the last bit for 30–35%
of sub-groups in SQ-15. Our decoder follows the shader, since that is what runs on Apple GPUs; the test suite
checks that the difference never changes which code is nearest (it did not, on 746,496 sub-groups sampled).

**Down-matrix layout ("row-paired").** A [2304, 1152] down matrix is stored as 1,152 row pairs. Pair p holds rows
2p and 2p+1 back to back (2 × 1,152 = 2,304 weights = 9 super-blocks), so reading the super-blocks in order and
reshaping the 2,654,208 values to [2304, 1152] gives the matrix in row-major order.

**Cross-check against Strata's reference codec.** Strata's public repository contains a reference S_K encoder and
decoder (`tools/squant/codecs.py`, `encode_sk` / `decode_sk`), written for the earlier `.strata` v1.0 layout. Its
super-block layout is the same as above. On real down-matrix bytes from this pack (4 S3_K, 4 S4_K and the 2 S5_K
experts; 26.5 million values), its `decode_sk` is bit-identical to our decoder when ours uses the CPU's order of the
scale product, and differs from our shader-order decoder by at most 2.3e-7 relative (a couple of units in the
last place of a float32) on the 30–35% of values whose sub-group scale differs. Its `encode_sk`, given our decoded weights, reproduces 99.99% (S3_K),
99.5% (S4_K) and 96.9% (S5_K) of the super-blocks byte for byte. It chooses D and u from the weights (absmax with a
search over clipping ratios) and then rounds to nearest, so on *edited* weights it would compute new scales.
`test_codecs.py --strata-src` reruns this check.

### SF4 (4-bit float, NVFP4-style)

Down matrix: 2304 × 576 bytes of nibbles (two weights per byte, the low nibble is the even column), followed by
2304 × 72 bytes of E4M3 scales, one per 16 weights of a row.

Value = `E2M1[nibble] × E4M3[scale byte] × (hdr[2] / hdr[3])`, where the outer factor is an f32 division of the
body header's down amax and down denominator, and

    E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, −0, −0.5, −1, −1.5, −2, −3, −4, −6]     (bit 3 = sign)

E4M3 is the usual 8-bit float with exponent bias 7 and no infinities; the shader maps the NaN codes 0x7F and 0xFF
to 0.

**Pitfall: signed zero.** Nibbles 0 and 8 both decode to zero, and the packer uses both (in one SF4 expert:
91,066 and 90,868 times). A re-encoder that always writes +0 decodes to the same values but does not reproduce the
original bytes, which breaks the byte-identity test that makes an encoder trustworthy. Ours takes the sign from the
target's sign bit, so decoding and re-encoding returns the original nibbles. Where a block's scale is zero (every
value decodes to zero), the original nibble is kept whenever its sign agrees with the target's.

### Q8_ROW (trunk)

`f32 scale[N]` followed by `int8 q[N, K]`; value = q × scale[row]. Codes are in [−127, 127].

### BF16 (trunk)

Raw bfloat16, row-major. Encode with round-to-nearest-even; when starting from float64, round once (float64 →
float32 → bfloat16 can double-round; `patch/build_abl.py` handles the tie case).

## What `test_codecs.py` checks

1. A literal per-element transcription of the Metal down-projection kernels (lookup tables parsed from
   `ink_metal_codecs.metal`) agrees bit for bit with the vectorised decoder, including the fp16 rounding of the
   prefill path: 389,184 values, 0 mismatches.
2. decode → encode reproduces the original bytes for every sampled expert down matrix (all 3,072 with `--all`),
   in both spaces.
3. The same for the embedding and all 24 attention output matrices (Q8_ROW and BF16).
4. Pack/unpack is a bijection for S2_K..S8_K on random codes.
5. The CPU/shader scale-product difference never changes a nearest-code decision.
6. With `--lora`: the LoRA's A vectors against −(rᵀW) decoded from the pack (an end-to-end check of the decode
   semantics, including the per-unit scale, and of the LoRA-to-pack mapping), with a wrong-expert control.
7. With `--strata-src`: the cross-check against Strata's reference codec described above.
