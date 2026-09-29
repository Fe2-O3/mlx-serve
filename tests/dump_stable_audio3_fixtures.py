#!/usr/bin/env python3
"""Dump Stable Audio 3 (medium, text-to-audio) parity fixtures for the Zig
oracle tests in `src/stable_audio3.zig`.

USER-RUN (needs mlx + numpy + sentencepiece + tokenizers). Run through uv so
the homebrew python stays untouched:

    uv run --with mlx --with numpy --with sentencepiece --with tokenizers \
        python tests/dump_stable_audio3_fixtures.py \
        [--pack DIR] [--npz DIR] [--ref DIR] [--out DIR] [--seconds S]

Inputs
  --pack  our converted pack (default ~/claude-tmp/sa3/pack-medium-8bit).
          Quantized tensors are DEQUANTIZED before the reference sees them,
          so the fixtures measure the Zig port rather than the quantizer —
          the `dump_music3_fixtures.py` convention.
  --npz   Stability's original MLX npz dumps (META + TOKENIZER_MODEL bytes
          for the T5Gemma loader).
  --ref   flat copy of `Stability-AI/stable-audio-3` `optimized/mlx`
          {scripts,models/defs} (sa3_pipeline.py, t5gemma_mlx.py,
          dit_mlx_medium.py, same_l_decoder.py).

Every tap is f32 little-endian raw unless the name says otherwise; shapes and
dtypes live in meta.json. Noise tensors are dumped as FIXTURE INPUTS: Zig
must not reproduce MLX's RNG, it consumes these.

Fixture taps (all under --out, default ~/claude-tmp/sa3/fixtures):

  ids_cond.i32.raw       [L]   prompt token ids (L <= 256), SentencePiece ground truth
  ids_empty.i32.raw      [0]   the unconditional prompt's ids (empty)
  mask_cond.i32.raw      [256] attention mask (1 = real token)
  mask_empty.i32.raw     [256] all zero
  t5_hidden_cond.f32.raw [1,256,768]  T5Gemma last hidden (cond)
  t5_hidden_empty.f32.raw[1,256,768]  T5Gemma last hidden (empty prompt)
  cross_attn.f32.raw     [1,257,768]  padded embeds + seconds token
  global_cond.f32.raw    [1,768]      the seconds embed (also the global cond)
  x0.f32.raw             [1,256,T]    seeded initial noise (the sampler input)
  v_t01/v_t05/v_t09.f32.raw [1,256,T] DiT velocity at t = 0.1 / 0.5 / 0.9
  sigmas.f32.raw         [steps+1]    ping-pong schedule (logsnr-shifted)
  noise_00..NN.f32.raw   [1,256,T]    per-step redraw (steps-1 of them)
  latents_stepNN.f32.raw [1,256,T]    latent after each step (01 .. final)
  patches_TNN.f32.raw    decoder output for the arms in meta.decode
  audio_TNN.f32.raw      [1,2,S]      patched + trimmed stereo (production arm)
  meta.json              shapes, seeds, dispatch, tokenizer cross-check

Then:  export SA3_TEST_MODEL=<pack> SA3_FIXTURES=<out>  and run the Zig
oracle tests once `src/stable_audio3.zig` exists.
"""

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# The dsv4 module builds 2**128 in its e8m0 LUT and warns about the overflow
# while loading; the warning is theirs and only muddies our output.
with np.errstate(over="ignore"):
    from convert_dsv4_weights import ShardReader  # noqa: E402
    from convert_stable_audio3_weights import affine_dequant, GROUP_SIZE  # noqa: E402

import mlx.core as mx  # noqa: E402

SAMPLE_RATE = 44100
SAMPLES_PER_LATENT = 4096
PROMPT = "A beautiful piano arpeggio grows into a cinematic climax"
STEPS = 8
SEED = 1234
# sa3_mlx.py taps for the three velocity probes (kept as f32 scalars; the
# sampler hands the model an f32 `t` because sigmas is f32).
VEL_TAPS = (0.1, 0.5, 0.9)


def log(msg):
    print(f"[sa3] {msg}", flush=True)


# ------------------------------------------------------------- pack reader ---
def dequantized_pack_arrays(path, bits):
    """{name: ndarray} for one pack file, quantized tensors dequantized to f32.

    Mirrors the converter's own naming: `X.weight` (U32) plus `X.scales` /
    `X.biases` (F16, group 64). Dense tensors come through untouched.
    """
    r = ShardReader(path)
    names = set(r.names())
    out = {}
    for n in sorted(names):
        if n.endswith(".scales") or n.endswith(".biases"):
            continue
        arr, dt = r.read(n)
        if dt == "U32":
            base = n[: -len(".weight")]
            scales, _ = r.read(base + ".scales")
            biases, _ = r.read(base + ".biases")
            out[n] = affine_dequant(arr, scales, biases, bits)
        else:
            out[n] = arr
    return out


def write_ref_npz(out_path, arrays):
    np.savez(out_path, **arrays)
    log(f"reference npz: {out_path} ({os.path.getsize(out_path) / 1e6:.0f} MB)")


def pack_f32_keys(path):
    """Names the pack stores as float32 — these must NOT be f16-rounded.

    The reference npz convention casts every weight to f16 (t5gemma_f16),
    but the F32 keys are not weights: `rope_inv_freq` (official loader keeps
    it f32 → rope phases in f32) and the DiT `cond.*` trio (official casts
    padding_embedding to f32; SecondsTotalEmbedder runs in fp32). Rounding
    them to f16 moves rope phases by up to 0.041 rad at position 255 — a
    real divergence the e2e oracle then measures as a port bug.
    """
    import json
    import struct

    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    return {k for k, e in hdr.items() if e["dtype"] == "F32"}


def prepare_dit_npz(pack, bits, scratch):
    """Dequantized DiT (+ baked `cond.*` conditioner) as npz for the loader."""
    path_in = os.path.join(pack, "dit.safetensors")
    arrays = dequantized_pack_arrays(path_in, bits)
    f32keys = pack_f32_keys(path_in)
    f16 = {
        k: (np.ascontiguousarray(v) if k in f32keys else np.ascontiguousarray(v, dtype=np.float16))
        for k, v in arrays.items()
    }
    del arrays
    path = os.path.join(scratch, "dit_pack.npz")
    write_ref_npz(path, f16)
    return path


def prepare_t5gemma_npz(pack, npz_dir, bits, scratch):
    """Dequantized T5Gemma + META/TOKENIZER_MODEL (not tensors: loader input)."""
    path_in = os.path.join(pack, "t5gemma.safetensors")
    arrays = dequantized_pack_arrays(path_in, bits)
    f32keys = pack_f32_keys(path_in)
    out = {
        k: (np.ascontiguousarray(v) if k in f32keys else np.ascontiguousarray(v, dtype=np.float16))
        for k, v in arrays.items()
    }
    with np.load(os.path.join(npz_dir, "t5gemma_f16.npz")) as z:
        out["META"] = z["META"]
        out["TOKENIZER_MODEL"] = z["TOKENIZER_MODEL"]
    path = os.path.join(scratch, "t5gemma_pack.npz")
    write_ref_npz(path, out)
    return path


def prepare_decoder_npz(pack, scratch):
    """SAME-L decoder ships fp32 verbatim in the pack — copy, no dequant."""
    arrays = dequantized_pack_arrays(os.path.join(pack, "same_l_decoder.safetensors"), 8)
    path = os.path.join(scratch, "same_l_decoder_pack.npz")
    write_ref_npz(path, arrays)
    return path


# ------------------------------------------------------------------ sampler ---
def sample_loop(model_fn, x, sigmas, seed):
    """`sa3_pipeline.sample_flow_pingpong` with the redraws exposed.

    Byte-for-byte the reference update (the equivalence check below runs the
    reference function against this one), but it RETURNS every intermediate
    latent and noise draw so they can be dumped as fixtures.
    """
    key = mx.random.key(seed)
    num_steps = sigmas.shape[0] - 1
    latents = []
    noises = []
    for i in range(num_steps):
        t_curr, t_next = sigmas[i], sigmas[i + 1]
        t_tensor = t_curr * mx.ones((x.shape[0],), dtype=x.dtype)
        v = model_fn(x, t_tensor)
        denoised = x - t_curr.astype(x.dtype) * v
        if i < num_steps - 1 and float(t_next) > 0.0:
            key, sub = mx.random.split(key)
            noise = mx.random.normal(x.shape, dtype=x.dtype, key=sub)
            x = (1.0 - t_next).astype(x.dtype) * denoised + t_next.astype(x.dtype) * noise
            noises.append(noise)
        else:
            x = denoised
        mx.eval(x)
        latents.append(x)
    return latents, noises


# ------------------------------------------------------------------ dumping ---
def dump(out, name, arr, meta, dtype=None):
    a = np.asarray(arr if isinstance(arr, np.ndarray) else np.array(arr))
    a = np.ascontiguousarray(a.astype(np.float32 if a.dtype == np.float16 else a.dtype))
    path = os.path.join(out, name)
    a.tofile(path)
    meta[name] = {"shape": list(a.shape), "dtype": dtype or str(a.dtype)}
    log(f"  {name:<26} {str(list(a.shape)):<18} {a.dtype}  {a.nbytes / 1e6:.1f} MB")


def main(argv=None):
    here = os.path.expanduser
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pack", default=here("~/claude-tmp/sa3/pack-medium-8bit"))
    ap.add_argument("--npz", default=here("~/claude-tmp/sa3/npz"))
    ap.add_argument("--ref", default="/tmp/sa3ref")
    ap.add_argument("--out", default=here("~/claude-tmp/sa3/fixtures"))
    ap.add_argument("--scratch", default=here("~/claude-tmp/sa3/refnpz"))
    ap.add_argument("--seconds", type=float, default=15.0)
    args = ap.parse_args(argv)

    pack_cfg = json.load(open(os.path.join(args.pack, "config.json")))
    bits = int(pack_cfg["bits"])
    if pack_cfg["dit"] != "medium" or pack_cfg["decoder"] != "same-l":
        raise SystemExit(f"[FATAL] this oracle covers medium/same-l, pack says "
                         f"{pack_cfg['dit']}/{pack_cfg['decoder']}")
    for p in (args.pack, args.npz, args.ref):
        if not os.path.isdir(p):
            raise SystemExit(f"[FATAL] missing: {p}")
    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.scratch, exist_ok=True)
    sys.path.insert(0, args.ref)

    import sentencepiece as spm
    from tokenizers import Tokenizer
    import sa3_pipeline as pipe
    import t5gemma_mlx as t5g
    import dit_mlx_medium as dit_mod
    import same_l_decoder as dec_mod

    seconds = args.seconds
    steps = STEPS
    seed = SEED
    # Decoder-independent natural ceil, exactly as sa3_mlx.py resolves T_lat.
    T_lat = max(1, math.ceil(seconds * SAMPLE_RATE / SAMPLES_PER_LATENT))
    meta = {
        "prompt": PROMPT,
        "seconds": seconds,
        "steps": steps,
        "seed": seed,
        "cfg": 1.0,
        "t_lat": T_lat,
        "sample_rate": SAMPLE_RATE,
        "samples_per_latent": SAMPLES_PER_LATENT,
        "pack": args.pack,
        "pack_bits": bits,
        "pack_dit_dtype": pack_cfg["dit_dtype"],
        "ref": args.ref,
        "mlx": getattr(mx, "__version__", "unknown"),
        "weights_source": "dequantized pack (fixtures measure the port, not the quantizer)",
    }
    log(f"pack bits={bits}  seconds={seconds}  T_lat={T_lat}  steps={steps}  seed={seed}")

    # ── 1. tokenize — SentencePiece proto is ground truth; assert the pack's
    # tokenizer.json agrees, because that is the file Zig reads.
    log("stage 1/6  tokenize")
    with np.load(os.path.join(args.npz, "t5gemma_f16.npz")) as z:
        sp = spm.SentencePieceProcessor()
        sp.LoadFromSerializedProto(z["TOKENIZER_MODEL"].tobytes())
    hf = Tokenizer.from_file(os.path.join(args.pack, "tokenizer.json"))
    for p in (PROMPT, ""):
        proto, js = sp.Encode(p), hf.encode(p).ids
        if proto != js:
            raise SystemExit(
                f"[FATAL] tokenizer.json disagrees with the SentencePiece proto on {p!r}:\n"
                f"  proto {proto[:16]}\n  json  {js[:16]}\n"
                "Zig reads tokenizer.json, so the pack would never match the reference.")
    ids_cond = np.asarray(sp.Encode(PROMPT), dtype=np.int32)
    ids_empty = np.asarray(sp.Encode(""), dtype=np.int32)
    mask_cond = np.zeros(256, dtype=np.int32)
    mask_cond[: len(ids_cond)] = 1
    mask_empty = np.zeros(256, dtype=np.int32)
    log("  tokenizer.json == SentencePiece proto (no BOS is added)")

    # ── 2. T5Gemma forward (empty prompt takes the reference's own special
    # path: one visible position during the forward, mask restored after).
    log("stage 2/6  T5Gemma")
    t5_npz = prepare_t5gemma_npz(args.pack, args.npz, bits, args.scratch)
    enc = t5g.T5Gemma.from_npz(t5_npz)
    hidden_cond, _ = enc.encode([PROMPT], max_len=256)
    hidden_empty, mask_empty_out = enc.encode([""], max_len=256)
    mx.eval(hidden_cond, hidden_empty)

    # ── 3. conditioning: padding + seconds token.
    log("stage 3/6  conditioning")
    dit_npz = prepare_dit_npz(args.pack, bits, args.scratch)
    padding_emb, secs = pipe.load_conditioner_from_npz(dit_npz, prefix="cond.")
    dtype = mx.float16  # sa3_mlx.py --dit-dtype fp16 default
    embeds, mask = enc.encode([PROMPT], max_len=256)
    embeds = embeds.astype(dtype)
    embeds_padded = pipe.apply_prompt_padding(embeds, mask, padding_emb.astype(dtype))
    seconds_embed = secs(seconds).astype(dtype)
    cross = mx.concatenate([embeds_padded, seconds_embed], axis=1)
    global_cond = seconds_embed[:, 0, :]
    mx.eval(cross, global_cond)

    # ── 4. DiT velocity probes + sampler inputs.
    log("stage 4/6  DiT")
    dit = dit_mod.load_dit(dit_npz, T_lat=T_lat, dtype=dtype, compile_=False)
    key = mx.random.key(seed)
    x0 = mx.random.normal((1, 256, T_lat), dtype=dtype, key=key)
    mx.eval(x0)
    t_taps = {}
    for tv in VEL_TAPS:
        t = mx.array(tv, dtype=mx.float32) * mx.ones((1,), dtype=x0.dtype)
        v = dit(x0, t, cross, global_cond, local_add_cond=None)
        mx.eval(v)
        t_taps[tv] = v
    del dit
    mx.clear_cache() if hasattr(mx, "clear_cache") else None

    # ── 5. sampler: 8 ping-pong steps, cross-checked against the reference
    # function (my exposed loop must land exactly where theirs does).
    log("stage 5/6  sampler")
    sigmas = pipe.build_pingpong_schedule(steps, sigma_max=1.0, use_logsnr_shift=True)

    def model_fn(x, t):
        return dit(x, t, cross, global_cond, local_add_cond=None)

    # reload — the probes above freed the model to keep peak RAM honest
    dit = dit_mod.load_dit(dit_npz, T_lat=T_lat, dtype=dtype, compile_=False)
    latents, noises = sample_loop(model_fn, x0, sigmas, seed + 1)
    ref_final = pipe.sample_flow_pingpong(model_fn, x0, sigmas, seed=seed + 1)
    delta = float(mx.abs(ref_final - latents[-1]).max())
    if delta > 1e-6:
        raise SystemExit(f"[FATAL] sampler reimplementation diverges from the "
                         f"reference: max|diff| = {delta}")
    log(f"  sampler matches sample_flow_pingpong (max|diff| = {delta:g})")
    del dit
    mx.clear_cache() if hasattr(mx, "clear_cache") else None

    # ── 6. decode: production arm (chunked when T_lat > kernel 144) plus the
    # two short/odd arms of sa3_mlx.py's dispatch.
    log("stage 6/6  decode")
    dec_npz = prepare_decoder_npz(args.pack, args.scratch)
    decoder = dec_mod.load_model(weights_path=dec_npz, dtype=mx.float32)
    latents_final = latents[-1].astype(mx.float32)
    decode_arms = {}
    if T_lat > 128 + 2 * 8:
        patches = dec_mod.decode_chunked(decoder, latents_final, 128, 8)
        decode_arms["production"] = f"chunked(chunk=128, ovl=8), T_lat={T_lat}"
        mx.eval(patches)
        audio = pipe.patched_decode(patches, patch_size=256, channels=2)
        want = int(round(seconds * SAMPLE_RATE))
        audio = audio[..., :want]
        mx.eval(audio)
    else:
        patches = decoder(latents_final)
        decode_arms["production"] = f"un-chunked, T_lat={T_lat}"
        mx.eval(patches)
        audio = pipe.patched_decode(patches, patch_size=256, channels=2)
        audio = audio[..., : int(round(seconds * SAMPLE_RATE))]
        mx.eval(audio)
    # Short-arm fixtures: the dispatch also hands the decoder even-length
    # direct calls and odd T <= kernel through chunk(2, ovl 2).
    patches_t8 = decoder(latents_final[..., :8])
    mx.eval(patches_t8)
    decode_arms["t8_direct"] = "un-chunked, T_lat=8"
    patches_t7 = dec_mod.decode_chunked(decoder, latents_final[..., :7], 2, 2)
    mx.eval(patches_t7)
    decode_arms["t7_chunk2"] = "chunked(chunk=2, ovl=2), T_lat=7"
    meta["decode"] = decode_arms

    # ── dump
    log(f"dump -> {args.out}")
    f32 = lambda a: np.asarray(np.array(a), dtype=np.float32)  # noqa: E731
    dump(args.out, "ids_cond.i32.raw", ids_cond, meta)
    dump(args.out, "ids_empty.i32.raw", ids_empty, meta)
    dump(args.out, "mask_cond.i32.raw", mask_cond, meta)
    dump(args.out, "mask_empty.i32.raw", mask_empty, meta)
    dump(args.out, "t5_hidden_cond.f32.raw", f32(hidden_cond), meta, dtype="f32")
    dump(args.out, "t5_hidden_empty.f32.raw", f32(hidden_empty), meta, dtype="f32")
    dump(args.out, "mask_empty_out.i32.raw", np.array(mask_empty_out), meta)
    dump(args.out, "cross_attn.f32.raw", f32(cross), meta, dtype="f32")
    dump(args.out, "global_cond.f32.raw", f32(global_cond), meta, dtype="f32")
    dump(args.out, "x0.f32.raw", f32(x0), meta, dtype="f32")
    for tv in VEL_TAPS:
        dump(args.out, f"v_t{int(tv * 100):02d}.f32.raw", f32(t_taps[tv]), meta, dtype="f32")
    dump(args.out, "sigmas.f32.raw", f32(sigmas), meta, dtype="f32")
    for i, n in enumerate(noises):
        dump(args.out, f"noise_{i:02d}.f32.raw", f32(n), meta, dtype="f32")
    for i, l in enumerate(latents, start=1):
        dump(args.out, f"latents_step{i:02d}.f32.raw", f32(l), meta, dtype="f32")
    dump(args.out, f"patches_T{T_lat}.f32.raw", f32(patches), meta, dtype="f32")
    dump(args.out, f"audio_T{T_lat}.f32.raw", f32(audio), meta, dtype="f32")
    dump(args.out, "patches_T8_direct.f32.raw", f32(patches_t8), meta, dtype="f32")
    dump(args.out, "patches_T7_chunk2.f32.raw", f32(patches_t7), meta, dtype="f32")

    meta_path = os.path.join(args.out, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")
    log(f"done — {len([k for k in meta if k.endswith('raw')])} fixtures + meta.json")


if __name__ == "__main__":
    main()
