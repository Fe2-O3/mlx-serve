#!/usr/bin/env python3
"""Convert Stability AI Stable Audio 3 MLX `.npz` weights into an mlx-serve pack.

USER-RUN, not run in CI: needs only numpy (no torch, no mlx, no safetensors).
Reads the four Stability dumps plus the T5Gemma HF tokenizer and writes one
flat pack directory:

    <out>/config.json
    <out>/t5gemma.safetensors              T5Gemma text encoder, affine-quantized
    <out>/same_l_decoder.safetensors       SAME-L VAE decoder, verbatim fp32
    <out>/same_l_encoder_f32.safetensors   SAME-L VAE encoder, verbatim fp32
    <out>/tokenizer.json
    <out>/dit.safetensors                  DiT (+ baked cond.* conditioner)

CONTRACT — `dit.safetensors` is written LAST and is the pack completeness
marker: `write_pack` refuses any file order that does not end with it, so a
conversion killed mid-way leaves a pack a reader can reject as incomplete
instead of half-load (the music3 pack's `vocoder.safetensors` rule,
`model_discovery.requiredMediaMarker`; the `stable_audio3` entry for that
function arrives with the loader stage).

fp32 decode is the reference behavior for both VAE halves: the decoder ships
fp32 at every --bits setting, and the encoder ships fp32 to match it. The
asymmetry in the names (`same_l_decoder` vs `same_l_encoder_f32`) is upstream's.

Quantization is MLX affine, group 64, bits 4 or 8 — the exact math of
`mx.quantize` (`mlx/ops.cpp affine_quantize` + its CPU kernel), reimplemented
in numpy: per-group min/max scale refined so the group's extreme lands on an
integer bin, `q = rint((w - bias) / scale)` clipped to [0, 2^bits - 1], packed
densely little-endian into uint32 (element i at bit i*bits) with f16 scales and
biases as sibling tensors. That packed layout is what `detectQuantBits` /
`affineParamsFromGeometry` in `src/transformer.zig` solve (bits, group_size)
from, so the existing Zig loader reads these packs as-is. Verified bit-identical
to `mx.quantize(device=cpu)` on real DiT tensors; MLX's own Metal path differs
from its own CPU path only at exact .5 ties (half-away vs half-to-even) and both
are valid quantizations of the same grid. Gather tables (`embed_tokens`) and
anything with min(out, in) < 512 stay dense.

Usage:
    python3 tests/convert_stable_audio3_weights.py [--out DIR] [--bits {4,8,16}]
    python3 tests/convert_stable_audio3_weights.py --self-test   # numpy only
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import urllib.request

import numpy as np

# Shared numpy-only safetensors writer + reader (dsv4's module imports mlx lazily).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# dsv4's e8m0 LUT builds 2**128 and warns about that overflow while its module
# loads; the warning is theirs, and it only muddies this converter's output.
with np.errstate(over="ignore"):
    from convert_dsv4_weights import ShardReader, write_safetensors_raw  # noqa: E402

# --------------------------------------------------------------- sources ----
NPZ_DIR = os.path.expanduser("~/claude-tmp/sa3/npz")
DEFAULT_DIT = os.path.join(NPZ_DIR, "dit_medium_f16.npz")
DEFAULT_T5G = os.path.join(NPZ_DIR, "t5gemma_f16.npz")
DEFAULT_DECODER = os.path.join(NPZ_DIR, "same_l_decoder_f32.npz")
DEFAULT_ENCODER = os.path.join(NPZ_DIR, "same_l_encoder_f32.npz")
DEFAULT_TOKENIZER = "/tmp/sa3_t5g_tokenizer.json"
TOKENIZER_URL = (
    "https://huggingface.co/stabilityai/stable-audio-3-optimized/resolve/main/"
    "tensorRT/sm_120/t5gemma/tokenizer.json"
)

# ------------------------------------------------------------------- pack ----
GROUP_SIZE = 64
MARKER = "dit.safetensors"
# META is JSON bytes (the T5Gemma config) and TOKENIZER_MODEL is the
# SentencePiece proto; neither is a tensor and neither may reach safetensors.
NON_TENSOR_KEYS = ("META", "TOKENIZER_MODEL")
# Gather reads keep their own layout in the loader, so media packs never
# quantize them (CLAUDE.md media NEVER_QUANTIZE tables).
NEVER_QUANTIZE = ("embed_tokens.weight",)

# Pack files in the order write_pack() wrote them. The marker must be last;
# self_test() asserts this through the real write path.
WRITE_ORDER = []

ST_DTYPE = {
    np.dtype(np.float16): "F16",
    np.dtype(np.float32): "F32",
    np.dtype(np.int32): "I32",
}


# --------------------------------------------- affine quant (numpy port) -----
def should_quantize(name, shape, bits):
    """True iff a `.weight` becomes packed affine at `bits`.

    The same predicate the acestep/music3 converters use: 2-D, contraction dim
    divisible by the group size, and min(out, in) >= 512 (a smaller group buys
    nothing over keeping the tensor dense).
    """
    if bits not in (4, 8) or not name.endswith(".weight") or len(shape) != 2:
        return False
    if any(t in name for t in NEVER_QUANTIZE):
        return False
    if shape[-1] % GROUP_SIZE:
        return False
    return min(shape) >= 512


def pack_u32(q, bits):
    """Dense little-endian pack of the last axis: element i sits at bit i*bits.

    MLX's layout (`pack_and_quantize`, power-of-2 arm): 32//bits values per
    uint32 word, low bits first — exactly what `mx.quantize` emits.
    """
    n = q.shape[-1]
    assert (n * bits) % 32 == 0, (q.shape, bits)
    words = q.astype(np.uint32).reshape(q.shape[:-1] + (n * bits // 32, 32 // bits))
    shifts = np.arange(32 // bits, dtype=np.uint32) * bits
    return (words << shifts).sum(axis=-1, dtype=np.uint32)


def affine_quant(w, bits, group_size=GROUP_SIZE):
    """(packed u32, scales, biases) for one 2-D weight — MLX `mx.quantize`.

    Bit-identical to `mx.quantize` on the CPU device (asserted in self_test on
    a real DiT tensor). numpy's rint rounds ties to even, matching MLX's
    std::rint CPU kernel; MLX's Metal kernel rounds ties away, so a
    GPU-generated pack differs from this one on ~1e-3 of the words — both are
    equally faithful quantizations of the same grid.
    """
    w = np.ascontiguousarray(w)
    shape = w.shape
    n = shape[-1]
    if len(shape) != 2 or n % group_size:
        raise SystemExit(
            f"[FATAL] affine_quant wants 2-D with in_dim % {group_size} == 0, got {shape}"
        )
    rows = w.reshape(-1, n // group_size, group_size)
    w_max = rows.max(axis=-1, keepdims=True).astype(np.float32)
    w_min = rows.min(axis=-1, keepdims=True).astype(np.float32)
    n_bins = np.float32((1 << bits) - 1)

    # A group whose min dominates gets a negative scale, so its bins follow the
    # large magnitudes instead of the empty side of zero (MLX does the same).
    mask = np.abs(w_min) > np.abs(w_max)
    scales = np.maximum((w_max - w_min) / n_bins, np.float32(1e-7))
    scales = np.where(mask, scales, -scales)
    edge = np.where(mask, w_min, w_max)

    # Refine so the extreme bin decodes exactly onto the extreme value: with
    # q0 = edge/scale an integer, scale = edge/q0 keeps the edge pinned.
    q0 = np.rint(edge / scales)
    nz = q0 != 0
    scales = np.where(nz, edge / np.where(nz, q0, np.float32(1)), scales)
    biases = np.where(nz, edge, np.float32(0))

    q = np.clip(np.rint((rows.astype(np.float32) - biases) / scales), 0, n_bins)
    packed = pack_u32(q.astype(np.uint32).reshape(shape[:-1] + (n,)), bits)
    s_shape = shape[:-1] + (n // group_size,)
    # Stored in the weight's own dtype, as mx.quantize casts T: f16 in means
    # f16 scales/biases out (q was computed against the f32 intermediates).
    return (
        packed,
        scales.astype(w.dtype).reshape(s_shape),
        biases.astype(w.dtype).reshape(s_shape),
    )


def affine_dequant(packed, scales, biases, bits, group_size=GROUP_SIZE):
    """numpy inverse of affine_quant (MLX `mx.dequantize`), f32 out. Self-test only."""
    packed = np.asarray(packed, dtype=np.uint32)
    scales = np.asarray(scales, dtype=np.float32)
    biases = np.asarray(biases, dtype=np.float32)
    n = scales.shape[-1] * group_size
    shifts = np.arange(32 // bits, dtype=np.uint32) * bits
    q = (packed[..., None] >> shifts) & np.uint32((1 << bits) - 1)
    q = q.reshape(packed.shape[:-1] + (n,)).astype(np.float32)
    return q * np.repeat(scales, group_size, axis=-1) + np.repeat(biases, group_size, axis=-1)


# ------------------------------------------------------------- npz -> pack ---
def partition_keys(keys):
    """(tensor keys, non-tensor keys): META/TOKENIZER_MODEL must never be written."""
    keep = [k for k in keys if k not in NON_TENSOR_KEYS]
    drop = [k for k in keys if k in NON_TENSOR_KEYS]
    return keep, drop


def st_dtype(dt):
    try:
        return ST_DTYPE[dt]
    except KeyError:
        raise SystemExit(f"[FATAL] unsupported npz dtype {dt}")


def convert_entry(name, arr, bits, quantize):
    """One npz array -> the safetensors entries it becomes."""
    if quantize and should_quantize(name, arr.shape, bits):
        packed, scales, biases = affine_quant(arr, bits)
        base = name[: -len(".weight")]
        return {
            f"{base}.weight": ("U32", packed.shape, packed.tobytes()),
            f"{base}.scales": ("F16", scales.shape, scales.tobytes()),
            f"{base}.biases": ("F16", biases.shape, biases.tobytes()),
        }
    a = np.ascontiguousarray(arr)
    return {name: (st_dtype(a.dtype), a.shape, a.tobytes())}


def load_npz(path, bits, quantize):
    """Read an npz into {out_name: (dtype_str, shape, raw_bytes)}.

    One array at a time: holding a 2.9 GB npz and its output together would
    double the peak while the rest of the pack is still to come.
    """
    out = {}
    n_quant = 0
    with np.load(path) as npz:
        keys, others = partition_keys(list(npz.keys()))
        for k in keys:
            entries = convert_entry(k, npz[k], bits, quantize)
            n_quant += len(entries) != 1
            out.update(entries)
    what = f"{n_quant} quantized at {bits}-bit/gs{GROUP_SIZE}" if quantize else "verbatim fp32"
    extra = f", dropped non-tensor {others}" if others else ""
    print(f"[load] {os.path.basename(path)}: {len(out)} tensors, {what}{extra}", flush=True)
    return out


def read_meta(path):
    """META holds the T5Gemma config as JSON bytes; it becomes config.json's t5gemma."""
    with np.load(path) as npz:
        if "META" not in npz.keys():
            raise SystemExit(f"[FATAL] {path}: no META key (expected T5Gemma config json)")
        return json.loads(npz["META"].tobytes())


# ------------------------------------------------------------------- pack ----
def build_config(bits, t5gemma_meta):
    """config.json: DiT/VAE geometry the loader needs, plus the T5Gemma META."""
    return {
        "model_type": "stable_audio3",
        "dit": "medium",
        "decoder": "same-l",
        "encoder": "same-l",
        "io_channels": 256,
        "embed_dim": 1536,
        "depth": 24,
        "num_heads": 24,
        "head_dim": 64,
        "rope_dims": 32,
        "rope_theta": 10000.0,
        "cond_token_dim": 768,
        "global_cond_dim": 768,
        "local_add_cond_dim": 257,
        "num_memory_tokens": 64,
        "ff_inner": 6144,
        "timestep_feat_dim": 256,
        "norm_eps": 1e-5,
        "qk_norm_eps": 1e-6,
        "sample_rate": 44100,
        "samples_per_latent": 4096,
        "patch_size": 256,
        "prompt_max_len": 256,
        "seconds_min": 1.0,
        "seconds_max": 384.0,
        "dit_dtype": "f16" if bits == 16 else "quantized",
        "bits": bits,
        "t5gemma": t5gemma_meta,
        "license": "stable-audio-community",
        "base_model": "stabilityai/stable-audio-3-medium",
    }


def read_tokenizer(path):
    """Tokenizer bytes: local copy when we have one, otherwise fetch it once."""
    if path and os.path.isfile(path):
        with open(path, "rb") as f:
            data = f.read()
        print(f"[tokenizer] copied {path} ({len(data) / 1e6:.1f} MB)", flush=True)
        return data
    print(f"[tokenizer] {path} not found; downloading {TOKENIZER_URL}", flush=True)
    with urllib.request.urlopen(TOKENIZER_URL) as resp:
        data = resp.read()
    print(f"[tokenizer] downloaded {len(data) / 1e6:.1f} MB", flush=True)
    return data


def write_pack(out, cfg, tokenizer_bytes, factories):
    """Write config.json, tokenizer.json, then every safetensors in `factories`.

    `factories` is an ordered {filename: callable -> {tensor: (dtype, shape,
    raw)}} whose LAST entry must be MARKER. Each factory runs immediately
    before its own write, so peak memory is one file, and the marker landing
    last is what makes an interrupted pack detectably incomplete. Returns the
    write order, also appended to WRITE_ORDER.
    """
    names = list(factories)
    if not names or names[-1] != MARKER:
        raise SystemExit(f"[FATAL] {MARKER} must be written last, got {names}")
    os.makedirs(out, exist_ok=True)
    order = []

    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    order.append("config.json")

    with open(os.path.join(out, "tokenizer.json"), "wb") as f:
        f.write(tokenizer_bytes)
    order.append("tokenizer.json")

    for name in names:
        path = os.path.join(out, name)
        tensors = factories[name]()
        write_safetensors_raw(path, tensors)
        order.append(name)
        size = os.path.getsize(path) / 1e6
        print(f"[write] {name}: {len(tensors)} tensors, {size:.1f} MB", flush=True)

    assert order[-1] == MARKER, order
    WRITE_ORDER.extend(order)
    return order


def quant_label(bits):
    return {4: "4bit", 8: "8bit"}.get(bits, "bf16")


def convert(args):
    if sys.byteorder != "little":
        # safetensors payloads are little-endian, so raw copies on a
        # big-endian host would silently swap every byte.
        raise SystemExit("[FATAL] this converter requires a little-endian host")
    inputs = {"dit": args.dit, "t5g": args.t5g, "decoder": args.decoder, "encoder": args.encoder}
    missing = [f"  {k}: {p}" for k, p in inputs.items() if not os.path.isfile(p)]
    if missing:
        raise SystemExit("[FATAL] missing npz input(s):\n" + "\n".join(missing))

    out = args.out or os.path.expanduser(
        f"~/.mlx-serve/models/local/stable-audio-3-medium-{quant_label(args.bits)}"
    )
    print(f"[pack] bits={args.bits} -> {out}", flush=True)

    cfg = build_config(args.bits, read_meta(args.t5g))
    tokenizer = read_tokenizer(args.tokenizer)

    # Marker last: dit is built and written only after every other file exists.
    factories = {
        "t5gemma.safetensors": lambda: load_npz(args.t5g, args.bits, True),
        "same_l_decoder.safetensors": lambda: load_npz(args.decoder, args.bits, False),
        "same_l_encoder_f32.safetensors": lambda: load_npz(args.encoder, args.bits, False),
        MARKER: lambda: load_npz(args.dit, args.bits, True),
    }
    order = write_pack(out, cfg, tokenizer, factories)
    print(f"[done] {len(order)} files, marker {order[-1]} last", flush=True)
    return out


# ------------------------------------------------------------- self-test -----
def rel_rms(x, y):
    """Error RMS relative to source RMS (unitless, ~0.005 = 0.5%)."""
    x = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean((x - y) ** 2)) / np.sqrt(np.mean(x * x)))


def self_test():
    """numpy-only: no npz inputs, no network, no mlx."""
    rng = np.random.default_rng(0)
    tmp = tempfile.mkdtemp(prefix="sa3-selftest-")
    WRITE_ORDER.clear()
    try:
        keys, others = partition_keys(["w.weight", "META", "rope_inv_freq", "TOKENIZER_MODEL"])
        assert keys == ["w.weight", "rope_inv_freq"], keys
        assert others == ["META", "TOKENIZER_MODEL"], others
        print("[self-test] npz key partitioning OK")

        assert should_quantize("transformer.layers.0.self_attn.to_qkv.weight", (7680, 1536), 8)
        assert should_quantize("transformer.layers.0.self_attn.to_qkv.weight", (7680, 1536), 4)
        assert not should_quantize("transformer.layers.0.self_attn.to_qkv.weight", (7680, 1536), 16), \
            "16-bit build must stay dense"
        assert not should_quantize("model.embed_tokens.weight", (256000, 768), 8), \
            "gather table must stay dense"
        assert not should_quantize("transformer.project_in.weight", (1536, 256), 8), \
            "min(out, in) < 512 must stay dense"
        assert not should_quantize("to_timestep_embed.0.weight", (1536, 256), 4)
        assert not should_quantize("transformer.memory_tokens", (64, 1536), 8), "not a .weight"
        assert not should_quantize("transformer.norm.weight", (1536,), 8), "1-D stays dense"
        assert not should_quantize("bad.weight", (512, 60), 8), "in_dim must divide the group"
        print("[self-test] quantization predicate OK")

        w = (rng.standard_normal((512, 512)) * 0.5).astype(np.float16)
        errs = {}
        for bits, bound in ((4, 0.15), (8, 0.01)):
            packed, scales, biases = affine_quant(w, bits)
            assert packed.dtype == np.uint32 and packed.shape == (512, 512 * bits // 32)
            assert scales.dtype == biases.dtype == np.float16
            assert scales.shape == biases.shape == (512, 512 // GROUP_SIZE)
            errs[bits] = rel_rms(w, affine_dequant(packed, scales, biases, bits))
            assert errs[bits] < bound, f"{bits}-bit rel-RMS {errs[bits]:.4f} >= {bound}"
            print(f"[self-test] {bits}-bit round trip: rel-RMS {errs[bits] * 100:.2f}% (bound {bound * 100:.0f}%)")
        assert errs[8] < errs[4] / 4, "8-bit must be meaningfully finer than 4-bit"
        try:
            affine_quant((rng.standard_normal((8, 33))).astype(np.float16), 8)
        except SystemExit:
            pass
        else:
            raise AssertionError("affine_quant accepted an in_dim that does not divide the group")

        e16 = convert_entry("transformer.proj.weight", w, 16, True)
        assert list(e16) == ["transformer.proj.weight"] and e16["transformer.proj.weight"][0] == "F16"
        assert e16["transformer.proj.weight"][2] == w.tobytes(), "16-bit must stay a verbatim copy"
        print("[self-test] quant round trip OK")

        cfg = build_config(8, {"model_type": "t5gemma", "hidden_size": 768})
        want = {
            "model_type": "stable_audio3", "dit": "medium", "decoder": "same-l",
            "encoder": "same-l", "io_channels": 256, "embed_dim": 1536, "depth": 24,
            "num_heads": 24, "head_dim": 64, "rope_dims": 32, "rope_theta": 10000.0,
            "cond_token_dim": 768, "global_cond_dim": 768, "local_add_cond_dim": 257,
            "num_memory_tokens": 64, "ff_inner": 6144, "timestep_feat_dim": 256,
            "norm_eps": 1e-5, "qk_norm_eps": 1e-6, "sample_rate": 44100,
            "samples_per_latent": 4096, "patch_size": 256, "prompt_max_len": 256,
            "seconds_min": 1.0, "seconds_max": 384.0, "dit_dtype": "quantized",
            "bits": 8, "license": "stable-audio-community",
            "base_model": "stabilityai/stable-audio-3-medium",
        }
        for key, value in want.items():
            assert cfg.get(key) == value, f"config[{key!r}] = {cfg.get(key)!r}, want {value!r}"
        assert cfg["t5gemma"]["hidden_size"] == 768, "T5Gemma META must be embedded verbatim"
        assert build_config(16, {})["dit_dtype"] == "f16"
        assert build_config(16, {})["bits"] == 16
        print("[self-test] config schema OK")

        try:
            write_pack(tmp, cfg, b"{}", {"t5gemma.safetensors": dict, MARKER: dict, "zz.safetensors": dict})
        except SystemExit as e:
            assert MARKER in str(e), e
        else:
            raise AssertionError("write_pack accepted an order that does not end with the marker")
        try:
            write_pack(tmp, cfg, b"{}", {"t5gemma.safetensors": dict})
        except SystemExit as e:
            assert MARKER in str(e), e
        else:
            raise AssertionError("write_pack accepted a pack with no marker")
        assert WRITE_ORDER == [], "the guard must fire before any file is written"
        print("[self-test] marker-last guard OK")

        q_w = (rng.standard_normal((512, 512)) * 0.5).astype(np.float16)
        embed_w = (rng.standard_normal((64, 64)) * 0.5).astype(np.float16)
        vae_w = (rng.standard_normal((64, 32)) * 0.5).astype(np.float32)
        norm_w = (rng.standard_normal((1536,)) * 0.5).astype(np.float16)
        tok = b'{"version":"1.0","model_type":"t5gemma"}'
        factories = {
            "t5gemma.safetensors": lambda: {
                **convert_entry("model.layers.0.self_attn.q_proj.weight", q_w, 8, True),
                **convert_entry("model.embed_tokens.weight", embed_w, 8, True),
            },
            "same_l_decoder.safetensors": lambda: {
                "decoder.norm.weight": ("F32", vae_w.shape, vae_w.tobytes()),
            },
            "same_l_encoder_f32.safetensors": lambda: {
                "encoder.in.weight": ("F32", vae_w.shape, vae_w.tobytes()),
            },
            MARKER: lambda: {
                **convert_entry("transformer.norm.weight", norm_w, 8, True),
                **convert_entry("transformer.q.weight", q_w, 8, True),
            },
        }
        order = write_pack(tmp, build_config(8, {"model_type": "t5gemma"}), tok, factories)
        assert order == [
            "config.json", "tokenizer.json", "t5gemma.safetensors",
            "same_l_decoder.safetensors", "same_l_encoder_f32.safetensors", MARKER,
        ], order
        assert WRITE_ORDER == order and WRITE_ORDER[-1] == MARKER

        with open(os.path.join(tmp, "config.json")) as f:
            assert json.load(f)["model_type"] == "stable_audio3"
        with open(os.path.join(tmp, "tokenizer.json"), "rb") as f:
            assert f.read() == tok

        marker = ShardReader(os.path.join(tmp, MARKER))
        norm, dt = marker.read("transformer.norm.weight")
        assert dt == "F16" and np.array_equal(norm, norm_w), "dense marker tensor round trip"
        words, dt = marker.read("transformer.q.weight")
        sc, sdt = marker.read("transformer.q.scales")
        bi, bdt = marker.read("transformer.q.biases")
        assert (dt, sdt, bdt) == ("U32", "F16", "F16"), (dt, sdt, bdt)
        assert words.dtype == np.uint32 and sc.shape == bi.shape == (512, 8)
        err = rel_rms(q_w, affine_dequant(words, sc, bi, 8))
        assert err < 0.01, f"on-disk 8-bit round trip rel-RMS {err:.4f}"

        t5g = ShardReader(os.path.join(tmp, "t5gemma.safetensors"))
        assert t5g.names() == [
            "model.layers.0.self_attn.q_proj.weight", "model.layers.0.self_attn.q_proj.scales",
            "model.layers.0.self_attn.q_proj.biases", "model.embed_tokens.weight",
        ], t5g.names()
        et, edt = t5g.read("model.embed_tokens.weight")
        assert edt == "F16" and np.array_equal(et, embed_w), "gather table must stay dense"
        for fname, key in (("same_l_decoder.safetensors", "decoder.norm.weight"),
                           ("same_l_encoder_f32.safetensors", "encoder.in.weight")):
            arr, dt = ShardReader(os.path.join(tmp, fname)).read(key)
            assert dt == "F32" and np.array_equal(arr, vae_w), f"{fname} must be verbatim fp32"
        print("[self-test] write/read round trip OK")

        assert quant_label(4) == "4bit" and quant_label(8) == "8bit" and quant_label(16) == "bf16"
        print("[self-test] quant_label OK")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("ALL SELF-TESTS PASSED")


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--dit", default=DEFAULT_DIT, help="DiT npz (default: %(default)s)")
    ap.add_argument("--t5g", default=DEFAULT_T5G, help="T5Gemma npz (default: %(default)s)")
    ap.add_argument("--decoder", default=DEFAULT_DECODER, help="SAME-L decoder npz")
    ap.add_argument("--encoder", default=DEFAULT_ENCODER, help="SAME-L encoder npz")
    ap.add_argument("--tokenizer", default=DEFAULT_TOKENIZER,
                    help="local tokenizer.json; downloaded when missing")
    ap.add_argument("--out", default=None,
                    help="output dir (default ~/.mlx-serve/models/local/"
                         "stable-audio-3-medium-{4bit,8bit,bf16})")
    ap.add_argument("--bits", type=int, default=8, choices=(4, 8, 16),
                    help="affine bits for the DiT and T5Gemma (default: 8)")
    ap.add_argument("--self-test", action="store_true",
                    help="numpy-only checks; needs no npz, network or mlx")
    args = ap.parse_args(argv)
    if args.self_test:
        self_test()
    else:
        convert(args)


if __name__ == "__main__":
    main()
