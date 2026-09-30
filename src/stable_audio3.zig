//! Stability AI Stable Audio 3 — text-to-music engine (`stable_audio3` arm
//! of gen.AudioBackend; the `stable_audio3` model_discovery / pack-marker
//! entries arrive with the loader stage).
//!
//! Reference = Stability-AI/stable-audio-3 `optimized/mlx` (sa3_mlx.py
//! pipeline): prompt → T5Gemma encoder (SentencePiece-style BPE, 12 layers,
//! 768 wide, RoPE + softcap 50) → condition padding (learned padding_embedding
//! fills masked positions) + a `seconds_total` Fourier embedder token
//! appended as [1, 257, 768] cross-attn cond (its [1,768] row also serves as
//! the global cond) → medium DiT (1.4B, DIFFERENTIAL attention, 24 layers,
//! embed 1536, patch 256 over a [1,256,T_lat] latent) denoised by a
//! ping-pong rectified-flow sampler (LogSNR-shifted linear schedule, 8 steps,
//! CFG batches cond/uncond) → SAME-L codec decoder (fp32, uniform-kernel
//! chunked decode: chunk 128 / overlap 8 when T_lat > 144) → patched
//! [B,512,T_lat*16] → [B,2,T_lat*4096] stereo at 44.1 kHz.
//!
//! Pack: tests/convert_stable_audio3_weights.py — npz → flat pack
//! {config.json, tokenizer.json, dit.safetensors (affine group-64 quantized
//! + baked cond.*), t5gemma.safetensors (partially quantized),
//! same_l_decoder.safetensors / same_l_encoder_f32.safetensors (verbatim
//! fp32)}. dit.safetensors is the completeness marker (written last).
//!
//! T_lat = ceil(seconds * 44100 / 4096), decoder-independent — same prompt +
//! seed + seconds must yield the same latent whatever decoder is paired.
//!
//! Parity: env-gated SA3_* oracles fed by tests/dump_stable_audio3_fixtures.py
//! (SA3_TEST_MODEL = pack dir, SA3_FIXTURES = dump dir). Fixtures are dumped
//! from the DEQUANTIZED pack, so they measure this port, not the quantizer.

const std = @import("std");
const testing = std.testing;
const tok_mod = @import("tokenizer.zig");
const mlx = @import("mlx.zig");
const model_mod = @import("model.zig");
const sse = @import("gen_sse.zig");
const wav_mod = @import("wav.zig");
const log = @import("log.zig");

const S = mlx.mlx_stream;
const Weights = model_mod.Weights;

// ── env-gated oracle plumbing ───────────────────────────────────────────────

fn sa3ModelDir() ![]const u8 {
    return std.mem.span(std.c.getenv("SA3_TEST_MODEL") orelse return error.SkipZigTest);
}

fn fixturesDir() ![]const u8 {
    return std.mem.span(std.c.getenv("SA3_FIXTURES") orelse return error.SkipZigTest);
}

fn readRawI32(io: std.Io, a: std.mem.Allocator, dir: []const u8, name: []const u8) ![]i32 {
    const raw = try readRawF32(io, a, dir, name);
    return @as([]i32, @ptrCast(raw));
}

fn readRawF32(io: std.Io, a: std.mem.Allocator, dir: []const u8, name: []const u8) ![]f32 {
    const path = try std.fmt.allocPrint(a, "{s}/{s}", .{ dir, name });
    defer a.free(path);
    const f = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var rs = f.reader(io, &rb);
    const bytes = try rs.interface.allocRemaining(a, .limited(1024 * 1024 * 1024));
    defer a.free(bytes);
    const n = bytes.len / 4;
    const out = try a.alloc(f32, n);
    @memcpy(std.mem.sliceAsBytes(out), bytes[0 .. n * 4]);
    return out;
}

// ── mlx test helpers ──

fn astype(x: mlx.mlx_array, dt: mlx.mlx_dtype, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_astype(&o, x, dt, s));
    return o;
}

fn evalA(x: mlx.mlx_array) void {
    const v = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(v);
    _ = mlx.mlx_vector_array_append_value(v, x);
    _ = mlx.mlx_eval(v);
}

fn cosineF64(x: []const f32, y: []const f32) f64 {
    var dot: f64 = 0;
    var nx: f64 = 0;
    var ny: f64 = 0;
    for (x, y) |a, b| {
        dot += @as(f64, a) * b;
        nx += @as(f64, a) * a;
        ny += @as(f64, b) * b;
    }
    return dot / (@sqrt(nx) * @sqrt(ny) + 1e-30);
}

fn rmsRatio(x: []const f32, y: []const f32) f64 {
    var sx: f64 = 0;
    var sy: f64 = 0;
    for (x) |a| sx += @as(f64, a) * a;
    for (y) |b| sy += @as(f64, b) * b;
    return @sqrt(sx / @as(f64, @floatFromInt(x.len))) / (@sqrt(sy / @as(f64, @floatFromInt(y.len))) + 1e-30);
}

/// cos AND rms_ratio vs a fixture — a cosine alone cannot see a scale error.
fn assertParity(arr: mlx.mlx_array, ref: []const f32, label: []const u8, min_cos: f64, rms_tol: f64, s: S) !void {
    const f = try astype(arr, .float32, s);
    defer _ = mlx.mlx_array_free(f);
    var c = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(c);
    try mlx.check(mlx.mlx_contiguous(&c, f, false, s));
    evalA(c);
    const n: usize = @intCast(mlx.mlx_array_size(c));
    try testing.expectEqual(ref.len, n);
    const d = mlx.mlx_array_data_float32(c) orelse return error.NoData;
    const cos = cosineF64(d[0..n], ref);
    const rr = rmsRatio(d[0..n], ref);
    std.debug.print("[sa3-{s}] cos={d:.6} rms_ratio={d:.4}\n", .{ label, cos, rr });
    try testing.expect(cos > min_cos);
    try testing.expect(rr > 1.0 - rms_tol and rr < 1.0 + rms_tol);
}

// ── tests ───────────────────────────────────────────────────────────────────

test "stable_audio3 oracle: prompt token ids byte-exact" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = try sa3ModelDir();
    const fix = try fixturesDir();

    const ref = try readRawI32(io, a, fix, "ids_cond.i32.raw");
    defer a.free(ref);

    var t = try tok_mod.loadTokenizer(io, a, dir);
    defer t.deinit();
    const ids = try t.encode(a, "A beautiful piano arpeggio grows into a cinematic climax");
    defer a.free(ids);

    const want = try a.alloc(u32, ref.len);
    defer a.free(want);
    for (ref, want) |r, *w| w.* = @intCast(r);
    try testing.expectEqualSlices(u32, want, ids);
    std.debug.print("[sa3] tokenizer oracle: {d} ids byte-exact vs fixture\n", .{ids.len});
}

test "stable_audio3 oracle: empty prompt encodes zero tokens" {
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = try sa3ModelDir();

    var t = try tok_mod.loadTokenizer(io, a, dir);
    defer t.deinit();
    const ids = try t.encode(a, "");
    defer a.free(ids);
    try testing.expectEqual(@as(usize, 0), ids.len);
    std.debug.print("[sa3] tokenizer oracle: empty prompt -> 0 tokens\n", .{});
}

// ── T5Gemma encoder (12-layer f16, reference = stable-audio-3 t5gemma_mlx.py)
//
// Pack contract: `t5gemma.safetensors` — all 7 linears per layer affine
// group-64 8-bit (`.scales`/`.biases` are quant params, NOT linear biases;
// T5Gemma linears are bias-free), norms + embed_tokens F16, rope_inv_freq F32.
// Forward matches the reference op-for-op, incl. f16 weak-scalar semantics:
// every constant is materialized in the ARRAY dtype (MLX weak-scalar rule).

const T5Json = struct {
    hidden_size: u32 = 768,
    num_hidden_layers: u32 = 12,
    num_attention_heads: u32 = 12,
    num_key_value_heads: u32 = 12,
    head_dim: u32 = 64,
    intermediate_size: u32 = 2048,
    vocab_size: u32 = 256000,
    rope_theta: f32 = 10000.0,
    rms_norm_eps: f32 = 1e-6,
    attn_logit_softcapping: f32 = 50.0,
    query_pre_attn_scalar: i32 = 64,
    pad_token_id: i32 = 0,
};

const T5ConfigFile = struct { t5gemma: T5Json };

fn readT5Cfg(io: std.Io, a: std.mem.Allocator, model_dir: []const u8) !T5Json {
    const path = try std.fmt.allocPrint(a, "{s}/config.json", .{model_dir});
    defer a.free(path);
    const f = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var rs = f.reader(io, &rb);
    const content = try rs.interface.allocRemaining(a, .limited(16 * 1024 * 1024));
    defer a.free(content);
    const parsed = try std.json.parseFromSlice(T5ConfigFile, a, content, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    return parsed.value.t5gemma;
}

/// Load ONE safetensors file into a Weights map (music3's loadFileWeights
/// pattern: CPU stream; iterator +1 transferred into the map).
fn loadFileWeights(allocator: std.mem.Allocator, model_dir: []const u8, file: []const u8) !Weights {
    var w = Weights.init(allocator);
    errdefer w.deinit();
    const cpu_s = mlx.mlx_default_cpu_stream_new();
    const path = try std.fmt.allocPrintSentinel(allocator, "{s}/{s}", .{ model_dir, file }, 0);
    defer allocator.free(path);

    var tensor_map = mlx.mlx_map_string_to_array_new();
    defer _ = mlx.mlx_map_string_to_array_free(tensor_map);
    var meta_map = mlx.mlx_map_string_to_string_new();
    defer _ = mlx.mlx_map_string_to_string_free(meta_map);
    try mlx.check(mlx.mlx_load_safetensors(&tensor_map, &meta_map, path, cpu_s));

    const iter = mlx.mlx_map_string_to_array_iterator_new(tensor_map);
    defer _ = mlx.mlx_map_string_to_array_iterator_free(iter);
    while (true) {
        var key: ?[*:0]const u8 = null;
        var value = mlx.mlx_array_new();
        const rc = mlx.mlx_map_string_to_array_iterator_next(&key, &value, iter);
        if (rc != 0 or key == null) {
            _ = mlx.mlx_array_free(value);
            break;
        }
        const owned_key = try allocator.dupe(u8, std.mem.span(key.?));
        errdefer allocator.free(owned_key);
        try w.map.put(owned_key, value);
    }
    return w;
}

// ── mlx micro-helpers (file-local) ──

fn getW(w: *const Weights, name: []const u8) !mlx.mlx_array {
    return w.get(name) orelse error.MissingWeight;
}

fn reshape(x: mlx.mlx_array, shape: []const c_int, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&o, x, shape.ptr, shape.len, s));
    return o;
}

fn transposeA(x: mlx.mlx_array, axes: []const c_int, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_transpose_axes(&o, x, axes.ptr, axes.len, s));
    return o;
}

fn addA(x: mlx.mlx_array, y: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_add(&o, x, y, s));
    return o;
}

fn subA(x: mlx.mlx_array, y: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_subtract(&o, x, y, s));
    return o;
}

fn mulA(x: mlx.mlx_array, y: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_multiply(&o, x, y, s));
    return o;
}

fn divA(x: mlx.mlx_array, y: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_divide(&o, x, y, s));
    return o;
}

/// Scalar in x's OWN dtype — the MLX weak-scalar rule (a bare f32 scalar
/// would promote the f16 operands the reference keeps in f16).
fn scalarLike(x: mlx.mlx_array, v: f32, s: S) !mlx.mlx_array {
    const c = mlx.mlx_array_new_float(v);
    defer _ = mlx.mlx_array_free(c);
    return astype(c, mlx.mlx_array_dtype(x), s);
}

fn mulScalar(x: mlx.mlx_array, v: f32, s: S) !mlx.mlx_array {
    const c = try scalarLike(x, v, s);
    defer _ = mlx.mlx_array_free(c);
    return mulA(x, c, s);
}

fn addScalar(x: mlx.mlx_array, v: f32, s: S) !mlx.mlx_array {
    const c = try scalarLike(x, v, s);
    defer _ = mlx.mlx_array_free(c);
    return addA(x, c, s);
}

/// x / scalar — a DIVISION, not a reciprocal-multiply: `x * f16(1/c)` and
/// `x / c` differ by up to ~1 ulp (f16(1/c) carries a +2e-4 bias), and the
/// reference softcap path really divides (`qk / softcap`).
fn divScalar(x: mlx.mlx_array, v: f32, s: S) !mlx.mlx_array {
    const c = try scalarLike(x, v, s);
    defer _ = mlx.mlx_array_free(c);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_divide(&o, x, c, s));
    return o;
}

fn tanhA(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_tanh(&o, x, s));
    return o;
}

fn rsqrtA(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_rsqrt(&o, x, s));
    return o;
}

/// Gemma-style RMSNorm: fp32 normalize, scale by (1 + weight), cast back.
fn rmsNorm(x: mlx.mlx_array, w: mlx.mlx_array, eps: f32, s: S) !mlx.mlx_array {
    const x32 = try astype(x, .float32, s);
    defer _ = mlx.mlx_array_free(x32);
    const sq = try mulA(x32, x32, s);
    defer _ = mlx.mlx_array_free(sq);
    const nd: c_int = @intCast(mlx.getShape(sq).len);
    var varr = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(varr);
    try mlx.check(mlx.mlx_mean_axis(&varr, sq, nd - 1, true, s));
    const e = mlx.mlx_array_new_float(eps);
    defer _ = mlx.mlx_array_free(e);
    const ve = try addA(varr, e, s);
    defer _ = mlx.mlx_array_free(ve);
    const inv = try rsqrtA(ve, s);
    defer _ = mlx.mlx_array_free(inv);
    const n = try mulA(x32, inv, s);
    defer _ = mlx.mlx_array_free(n);
    const w32 = try astype(w, .float32, s);
    defer _ = mlx.mlx_array_free(w32);
    const one = mlx.mlx_array_new_float(1.0);
    defer _ = mlx.mlx_array_free(one);
    const wp = try addA(w32, one, s);
    defer _ = mlx.mlx_array_free(wp);
    const scaled = try mulA(n, wp, s);
    defer _ = mlx.mlx_array_free(scaled);
    return astype(scaled, mlx.mlx_array_dtype(x), s);
}

/// Linear over the Weights map: affine-quantized (bits/group solved from
/// packed geometry — the MixedLinear rule) or dense (lazy-transpose matmul).
/// Output narrowed to f16: the reference computes entirely in f16.
fn lin(w: *const Weights, a: std.mem.Allocator, x: mlx.mlx_array, prefix: []const u8, s: S) !mlx.mlx_array {
    const wk = try std.fmt.allocPrint(a, "{s}.weight", .{prefix});
    defer a.free(wk);
    const sk = try std.fmt.allocPrint(a, "{s}.scales", .{prefix});
    defer a.free(sk);
    const bk = try std.fmt.allocPrint(a, "{s}.biases", .{prefix});
    defer a.free(bk);
    const ak = try std.fmt.allocPrint(a, "{s}.bias", .{prefix});
    defer a.free(ak);

    const xsh = mlx.getShape(x);
    const in_features: u32 = @intCast(xsh[xsh.len - 1]);

    var o = mlx.mlx_array_new();
    if (w.get(sk)) |scales| {
        const wq = try getW(w, wk);
        const qb = try getW(w, bk);
        const w_cols: u32 = @intCast(mlx.getShape(wq)[1]);
        const s_cols: u32 = @intCast(mlx.getShape(scales)[1]);
        const bits: u32 = @divExact(32 * w_cols, in_features);
        const gs: u32 = @divExact(in_features, s_cols);
        try mlx.check(mlx.mlx_quantized_matmul(&o, x, wq, scales, qb, true, mlx.mlx_optional_int.some(@intCast(gs)), mlx.mlx_optional_int.some(@intCast(bits)), "affine", s));
    } else {
        const wd = try getW(w, wk);
        var wt = mlx.mlx_array_new();
        defer _ = mlx.mlx_array_free(wt);
        const axes = [_]c_int{ 1, 0 };
        try mlx.check(mlx.mlx_transpose_axes(&wt, wd, &axes, 2, s));
        try mlx.check(mlx.mlx_matmul(&o, x, wt, s));
    }
    if (w.get(ak)) |bias| {
        const r = try addA(o, bias, s);
        _ = mlx.mlx_array_free(o);
        o = r;
    }
    return o;
}

/// gelu_approx: 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715 x^3))) — every
/// constant materialized in x's dtype (weak-scalar rule), f16 like MLX.
fn geluApprox(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    // x³ via mlx power — the reference is Python `0.044715 * x ** 3`, and
    // MLX computes integer pow in f32, rounding ONCE. Hand-rolled x*x*x
    // rounds twice in f16 and disagrees on ~24% of elements (up to 2 ulps of
    // x³) — enough for the 12-layer stack to drift to mean 0.002.
    const three = mlx.mlx_array_new_int(3);
    defer _ = mlx.mlx_array_free(three);
    var x3 = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_power(&x3, x, three, s));
    defer _ = mlx.mlx_array_free(x3);
    const t = try mulScalar(x3, 0.044715, s);
    defer _ = mlx.mlx_array_free(t);
    const inner = try addA(x, t, s);
    defer _ = mlx.mlx_array_free(inner);
    const c = try mulScalar(inner, @sqrt(2.0 / 3.141592653589793), s);
    defer _ = mlx.mlx_array_free(c);
    const th = try tanhA(c, s);
    defer _ = mlx.mlx_array_free(th);
    const th1 = try addScalar(th, 1.0, s);
    defer _ = mlx.mlx_array_free(th1);
    const hx = try mulScalar(x, 0.5, s);
    defer _ = mlx.mlx_array_free(hx);
    return mulA(hx, th1, s);
}

/// rotate_half: concat([-x[..., half:], x[..., :half]], -1) over a 4-D array.
fn rotateHalf(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const last: c_int = sh[sh.len - 1];
    const half = @divExact(last, 2);
    const starts_r = [_]c_int{ 0, 0, 0, half };
    var right = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(right);
    {
        const stop = [_]c_int{ sh[0], sh[1], sh[2], last };
        const strides = [_]c_int{ 1, 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&right, x, &starts_r, 4, &stop, 4, &strides, 4, s));
    }
    const starts_l = [_]c_int{ 0, 0, 0, 0 };
    var left = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(left);
    {
        const stop = [_]c_int{ sh[0], sh[1], sh[2], half };
        const strides = [_]c_int{ 1, 1, 1, 1 };
        try mlx.check(mlx.mlx_slice(&left, x, &starts_l, 4, &stop, 4, &strides, 4, s));
    }
    var neg_r = mlx.mlx_array_new();
    defer _ = mlx.mlx_array_free(neg_r);
    try mlx.check(mlx.mlx_negative(&neg_r, right, s));

    const v = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(v);
    _ = mlx.mlx_vector_array_append_value(v, neg_r);
    _ = mlx.mlx_vector_array_append_value(v, left);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&o, v, 3, s));
    return o;
}

fn applyRope(q: mlx.mlx_array, k: mlx.mlx_array, cos16: mlx.mlx_array, sin16: mlx.mlx_array, s: S) !struct { mlx.mlx_array, mlx.mlx_array } {
    const qc = try mulA(q, cos16, s);
    const qr = try rotateHalf(q, s);
    defer _ = mlx.mlx_array_free(qr);
    const qs = try mulA(qr, sin16, s);
    defer _ = mlx.mlx_array_free(qs);
    const qn = try addA(qc, qs, s);
    _ = mlx.mlx_array_free(qc);

    const kc = try mulA(k, cos16, s);
    const kr = try rotateHalf(k, s);
    defer _ = mlx.mlx_array_free(kr);
    const ks = try mulA(kr, sin16, s);
    defer _ = mlx.mlx_array_free(ks);
    const kn = try addA(kc, ks, s);
    _ = mlx.mlx_array_free(kc);

    return .{ qn, kn };
}

const AttnCtx = struct {
    cos: mlx.mlx_array,
    sin: mlx.mlx_array,
    add_mask: mlx.mlx_array,
};

/// Gemma-style self-attention: qk × scaling → tanh softcap → +mask →
/// softmax in f32 (narrowed to v.dtype) → @v → o_proj.
/// Free discipline: every handle freed exactly once, eagerly after its last
/// consumer — no defers on reassigned variables (Zig defers read the FINAL
/// value of a var, which would free undefined / double-free).
fn selfAttention(
    x: mlx.mlx_array,
    ctx: *const AttnCtx,
    w: *const Weights,
    cfg: *const T5Json,
    layer: usize,
    a: std.mem.Allocator,
    s: S,
) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const B: c_int = sh[0];
    const Sl: c_int = sh[1];
    const H: c_int = @intCast(cfg.num_attention_heads);
    const kvH: c_int = @intCast(cfg.num_key_value_heads);
    const D: c_int = @intCast(cfg.head_dim);

    const qp = try std.fmt.allocPrint(a, "layers.{d}.self_attn.q_proj", .{layer});
    defer a.free(qp);
    const kp = try std.fmt.allocPrint(a, "layers.{d}.self_attn.k_proj", .{layer});
    defer a.free(kp);
    const vp = try std.fmt.allocPrint(a, "layers.{d}.self_attn.v_proj", .{layer});
    defer a.free(vp);
    const op = try std.fmt.allocPrint(a, "layers.{d}.self_attn.o_proj", .{layer});
    defer a.free(op);

    const q0 = try lin(w, a, x, qp, s);
    const k0 = try lin(w, a, x, kp, s);
    const v0 = try lin(w, a, x, vp, s);

    const qsh = [_]c_int{ B, Sl, H, D };
    const ksh = [_]c_int{ B, Sl, kvH, D };
    const vsh = [_]c_int{ B, Sl, kvH, D };
    const q4 = try reshape(q0, &qsh, s);
    _ = mlx.mlx_array_free(q0);
    const k4 = try reshape(k0, &ksh, s);
    _ = mlx.mlx_array_free(k0);
    const v4 = try reshape(v0, &vsh, s);
    _ = mlx.mlx_array_free(v0);

    const t01 = [_]c_int{ 0, 2, 1, 3 };
    const qt = try transposeA(q4, &t01, s);
    _ = mlx.mlx_array_free(q4);
    const kt = try transposeA(k4, &t01, s);
    _ = mlx.mlx_array_free(k4);
    const vt = try transposeA(v4, &t01, s);
    _ = mlx.mlx_array_free(v4);

    const qr, const kr = try applyRope(qt, kt, ctx.cos, ctx.sin, s);
    _ = mlx.mlx_array_free(qt);
    _ = mlx.mlx_array_free(kt);

    const t0132 = [_]c_int{ 0, 1, 3, 2 };
    const ktt = try transposeA(kr, &t0132, s);
    _ = mlx.mlx_array_free(kr);

    var qk = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&qk, qr, ktt, s));
    _ = mlx.mlx_array_free(qr);
    _ = mlx.mlx_array_free(ktt);

    { // qk × scaling, then tanh softcap (both in x's dtype, like MLX weak scalars)
        const t = try mulScalar(qk, std.math.pow(f32, @as(f32, @floatFromInt(cfg.query_pre_attn_scalar)), -0.5), s);
        _ = mlx.mlx_array_free(qk);
        qk = t;
    }
    {
        const cap = cfg.attn_logit_softcapping;
        const d = try divScalar(qk, cap, s);
        _ = mlx.mlx_array_free(qk);
        const th = try tanhA(d, s);
        _ = mlx.mlx_array_free(d);
        const r = try mulScalar(th, cap, s);
        _ = mlx.mlx_array_free(th);
        qk = r;
    }
    {
        const r = try addA(qk, ctx.add_mask, s);
        _ = mlx.mlx_array_free(qk);
        qk = r;
    }

    const qk32 = try astype(qk, .float32, s);
    _ = mlx.mlx_array_free(qk);
    var p32 = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_softmax_axis(&p32, qk32, 3, false, s));
    _ = mlx.mlx_array_free(qk32);
    const p = try astype(p32, mlx.mlx_array_dtype(vt), s);
    _ = mlx.mlx_array_free(p32);

    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&o, p, vt, s));
    _ = mlx.mlx_array_free(p);
    _ = mlx.mlx_array_free(vt);

    const ot = try transposeA(o, &t01, s);
    _ = mlx.mlx_array_free(o);
    const osh = [_]c_int{ B, Sl, H * D };
    const or_ = try reshape(ot, &osh, s);
    _ = mlx.mlx_array_free(ot);
    const out = try lin(w, a, or_, op, s);
    _ = mlx.mlx_array_free(or_);
    return out;
}

fn encoderLayer(
    x: mlx.mlx_array,
    ctx: *const AttnCtx,
    w: *const Weights,
    cfg: *const T5Json,
    layer: usize,
    a: std.mem.Allocator,
    s: S,
) !mlx.mlx_array {
    const eps = cfg.rms_norm_eps;

    const pre_a = try std.fmt.allocPrint(a, "layers.{d}.pre_self_attn_layernorm.weight", .{layer});
    defer a.free(pre_a);
    const post_a = try std.fmt.allocPrint(a, "layers.{d}.post_self_attn_layernorm.weight", .{layer});
    defer a.free(post_a);
    const pre_f = try std.fmt.allocPrint(a, "layers.{d}.pre_feedforward_layernorm.weight", .{layer});
    defer a.free(pre_f);
    const post_f = try std.fmt.allocPrint(a, "layers.{d}.post_feedforward_layernorm.weight", .{layer});
    defer a.free(post_f);

    const h0 = try rmsNorm(x, try getW(w, pre_a), eps, s);
    defer _ = mlx.mlx_array_free(h0);
    const ha = try selfAttention(h0, ctx, w, cfg, layer, a, s);
    defer _ = mlx.mlx_array_free(ha);
    const h1 = try rmsNorm(ha, try getW(w, post_a), eps, s);
    defer _ = mlx.mlx_array_free(h1);
    const x1 = try addA(x, h1, s);
    defer _ = mlx.mlx_array_free(x1);

    const h2 = try rmsNorm(x1, try getW(w, pre_f), eps, s);
    defer _ = mlx.mlx_array_free(h2);
    const g = try std.fmt.allocPrint(a, "layers.{d}.mlp.gate_proj", .{layer});
    defer a.free(g);
    const u = try std.fmt.allocPrint(a, "layers.{d}.mlp.up_proj", .{layer});
    defer a.free(u);
    const d = try std.fmt.allocPrint(a, "layers.{d}.mlp.down_proj", .{layer});
    defer a.free(d);
    const gate = try lin(w, a, h2, g, s);
    defer _ = mlx.mlx_array_free(gate);
    const up = try lin(w, a, h2, u, s);
    defer _ = mlx.mlx_array_free(up);
    const gg = try geluApprox(gate, s);
    defer _ = mlx.mlx_array_free(gg);
    const mul = try mulA(gg, up, s);
    defer _ = mlx.mlx_array_free(mul);
    const mlp_out = try lin(w, a, mul, d, s);
    defer _ = mlx.mlx_array_free(mlp_out);
    const h3 = try rmsNorm(mlp_out, try getW(w, post_f), eps, s);
    defer _ = mlx.mlx_array_free(h3);
    return addA(x1, h3, s);
}

pub const T5Gemma = struct {
    cfg: T5Json,
    w: Weights,
    s: S,

    pub fn load(io: std.Io, a: std.mem.Allocator, model_dir: []const u8) !T5Gemma {
        var self: T5Gemma = undefined;
        self.cfg = try readT5Cfg(io, a, model_dir);
        self.w = try loadFileWeights(a, model_dir, "t5gemma.safetensors");
        self.s = mlx.mlx_default_gpu_stream_new();
        return self;
    }

    pub fn deinit(self: T5Gemma) void {
        var w = self.w;
        w.deinit();
        _ = mlx.mlx_stream_free(self.s);
    }

    /// `ids` = real tokens (any length ≤ mask.len), `mask` = the full
    /// [max_len] reference mask. An all-zero mask gets one visible position
    /// for the forward (the reference's NaN guard), then no mask is returned
    /// — the caller keeps its own zeros. Returns [1, max_len, 768] f16.
    pub fn encode(self: *const T5Gemma, a: std.mem.Allocator, ids: []const i32, mask: []const i32, s: S) !mlx.mlx_array {
        const cfg = &self.cfg;
        const max_len: usize = mask.len;
        if (ids.len > max_len) return error.PromptTooLong;
        const w = &self.w;

        const pad_ids = try a.alloc(i32, max_len);
        defer a.free(pad_ids);
        @memset(pad_ids, cfg.pad_token_id);
        @memcpy(pad_ids[0..ids.len], ids);

        var mask_fwd = try a.alloc(i32, max_len);
        defer a.free(mask_fwd);
        @memcpy(mask_fwd, mask);
        var nz: usize = 0;
        for (mask) |m| nz += @intCast(@max(m, 0));
        if (nz == 0 and max_len > 0) mask_fwd[0] = 1;

        const S_len: c_int = @intCast(max_len);
        const idshape = [_]c_int{ 1, S_len };
        const ids_arr = mlx.mlx_array_new_data(pad_ids.ptr, &idshape, 2, .int32);
        defer _ = mlx.mlx_array_free(ids_arr);
        const mask_arr = mlx.mlx_array_new_data(mask_fwd.ptr, &idshape, 2, .int32);
        defer _ = mlx.mlx_array_free(mask_arr);

        // embed × sqrt(hidden) — f16 like the reference.
        var x = blk: {
            const table = try getW(w, "embed_tokens.weight");
            var e = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_take_axis(&e, table, ids_arr, 0, s));
            const nrm = try scalarLike(e, @sqrt(@as(f32, @floatFromInt(cfg.hidden_size))), s);
            const r = try mulA(e, nrm, s);
            _ = mlx.mlx_array_free(e);
            _ = mlx.mlx_array_free(nrm);
            break :blk r;
        };
        defer _ = mlx.mlx_array_free(x);

        // RoPE cos/sin from the pack's rope_inv_freq (f32), broadcast to
        // [1,1,max_len,head_dim] and narrowed to x's dtype.
        var cos16: mlx.mlx_array = undefined;
        var sin16: mlx.mlx_array = undefined;
        {
            const inv = try getW(w, "rope_inv_freq");
            const hd2: c_int = @intCast(mlx.getShape(inv)[0]);
            var pos = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(pos);
            try mlx.check(mlx.mlx_arange(&pos, 0, @floatFromInt(max_len), 1, .float32, s));
            const ps = [_]c_int{ S_len, 1 };
            const posr = try reshape(pos, &ps, s);
            defer _ = mlx.mlx_array_free(posr);
            const is = [_]c_int{ 1, hd2 };
            const invr = try reshape(inv, &is, s);
            defer _ = mlx.mlx_array_free(invr);
            const freqs = try mulA(posr, invr, s);
            defer _ = mlx.mlx_array_free(freqs);
            const v = mlx.mlx_vector_array_new();
            defer _ = mlx.mlx_vector_array_free(v);
            _ = mlx.mlx_vector_array_append_value(v, freqs);
            _ = mlx.mlx_vector_array_append_value(v, freqs);
            var emb = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(emb);
            try mlx.check(mlx.mlx_concatenate_axis(&emb, v, 1, s));
            var cs = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(cs);
            try mlx.check(mlx.mlx_cos(&cs, emb, s));
            var sn = mlx.mlx_array_new();
            defer _ = mlx.mlx_array_free(sn);
            try mlx.check(mlx.mlx_sin(&sn, emb, s));
            // [S, hd] → [1,1,S,hd]
            cos16 = blk: {
                var e1 = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_expand_dims(&e1, cs, 0, s));
                var e2 = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(e2);
                try mlx.check(mlx.mlx_expand_dims(&e2, e1, 0, s));
                _ = mlx.mlx_array_free(e1);
                break :blk try astype(e2, mlx.mlx_array_dtype(x), s);
            };
            sin16 = blk: {
                var e1 = mlx.mlx_array_new();
                try mlx.check(mlx.mlx_expand_dims(&e1, sn, 0, s));
                var e2 = mlx.mlx_array_new();
                defer _ = mlx.mlx_array_free(e2);
                try mlx.check(mlx.mlx_expand_dims(&e2, e1, 0, s));
                _ = mlx.mlx_array_free(e1);
                break :blk try astype(e2, mlx.mlx_array_dtype(x), s);
            };
        }
        defer _ = mlx.mlx_array_free(cos16);
        defer _ = mlx.mlx_array_free(sin16);

        // add_mask = ((1 - mask) * -1e9)[:, None, None, :] in x's dtype.
        const add_mask = blk: {
            const keep = try astype(mask_arr, .float32, s);
            defer _ = mlx.mlx_array_free(keep);
            const one = mlx.mlx_array_new_float(1.0);
            defer _ = mlx.mlx_array_free(one);
            const invk = try subA(one, keep, s);
            defer _ = mlx.mlx_array_free(invk);
            const neg = mlx.mlx_array_new_float(-1e9);
            defer _ = mlx.mlx_array_free(neg);
            const scaled = try mulA(invk, neg, s);
            defer _ = mlx.mlx_array_free(scaled);
            var e1 = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_expand_dims(&e1, scaled, 1, s));
            defer _ = mlx.mlx_array_free(e1);
            var e2 = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_expand_dims(&e2, e1, 2, s));
            defer _ = mlx.mlx_array_free(e2);
            break :blk try astype(e2, mlx.mlx_array_dtype(x), s);
        };
        defer _ = mlx.mlx_array_free(add_mask);

        const ctx = AttnCtx{ .cos = cos16, .sin = sin16, .add_mask = add_mask };
        for (0..cfg.num_hidden_layers) |i| {
            const nx = try encoderLayer(x, &ctx, w, cfg, i, a, s);
            _ = mlx.mlx_array_free(x);
            x = nx;
        }

        return rmsNorm(x, try getW(w, "norm.weight"), cfg.rms_norm_eps, s);
    }
};

// ── conditioner: prompt padding + seconds_total token ──
//
// reference = sa3_pipeline.py apply_prompt_padding + SecondsTotalEmbedder
// (weights baked into dit.safetensors as cond.* by the converter). Op order
// mirrors the reference: f16 padding math, f32 fourier + linear, narrow to
// f16 before the concat.

pub const CondResult = struct {
    cross: mlx.mlx_array,       // [1, S+1, 768] f16 — padded embeds + seconds token
    global_cond: mlx.mlx_array, // [1, 768] f16 — the seconds embed row

    pub fn deinit(self: CondResult) void {
        _ = mlx.mlx_array_free(self.cross);
        _ = mlx.mlx_array_free(self.global_cond);
    }
};

pub fn conditionPrompt(
    w: *const Weights,
    embeds: mlx.mlx_array,
    mask: []const i32,
    seconds: f32,
    s: S,
) !CondResult {
    const esh = mlx.getShape(embeds);
    const B: c_int = esh[0];
    const Sl: c_int = esh[1];
    const dim: c_int = esh[2];

    // embeds * m + pe * (1 - m)  — all in embeds' dtype (the reference rule)
    const padded = blk: {
        const msh = [_]c_int{ B, Sl };
        const m_i = mlx.mlx_array_new_data(mask.ptr, &msh, 2, .int32);
        defer _ = mlx.mlx_array_free(m_i);
        const m = try astype(m_i, mlx.mlx_array_dtype(embeds), s);
        defer _ = mlx.mlx_array_free(m);
        var m1 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_expand_dims(&m1, m, 2, s));
        defer _ = mlx.mlx_array_free(m1);

        const pe = try getW(w, "cond.padding_embedding");
        const pe_d = try astype(pe, mlx.mlx_array_dtype(embeds), s);
        defer _ = mlx.mlx_array_free(pe_d);
        const psh = [_]c_int{ 1, 1, dim };
        const pe3 = try reshape(pe_d, &psh, s);
        defer _ = mlx.mlx_array_free(pe3);

        const one = try scalarLike(m1, 1.0, s);
        defer _ = mlx.mlx_array_free(one);
        const invm = try subA(one, m1, s);
        defer _ = mlx.mlx_array_free(invm);
        const left = try mulA(embeds, m1, s);
        defer _ = mlx.mlx_array_free(left);
        const right = try mulA(pe3, invm, s);
        defer _ = mlx.mlx_array_free(right);
        break :blk try addA(left, right, s);
    };
    defer _ = mlx.mlx_array_free(padded);

    // seconds_total → [1,1,768] f16 (NumberConditioner: clip → /384 → expo
    // fourier 256 → linear 768, all f32, narrowed after)
    const secs_embed = blk: {
        const raw = mlx.mlx_array_new_float(seconds);
        defer _ = mlx.mlx_array_free(raw);
        const lo = mlx.mlx_array_new_float(0.0);
        defer _ = mlx.mlx_array_free(lo);
        const hi = mlx.mlx_array_new_float(384.0);
        defer _ = mlx.mlx_array_free(hi);
        var clipped = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_clip(&clipped, raw, lo, hi, s));
        defer _ = mlx.mlx_array_free(clipped);
        const d384 = mlx.mlx_array_new_float(384.0);
        defer _ = mlx.mlx_array_free(d384);
        var norm = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_divide(&norm, clipped, d384, s));
        defer _ = mlx.mlx_array_free(norm);
        const tsh = [_]c_int{ 1, 1 };
        const t = try reshape(norm, &tsh, s);
        defer _ = mlx.mlx_array_free(t);

        var ramp0 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_arange(&ramp0, 0, 128, 1, .float32, s));
        defer _ = mlx.mlx_array_free(ramp0);
        const d127 = mlx.mlx_array_new_float(127.0);
        defer _ = mlx.mlx_array_free(d127);
        const ramp = try divA(ramp0, d127, s);
        defer _ = mlx.mlx_array_free(ramp);
        const span = mlx.mlx_array_new_float(@floatCast(@log(@as(f64, 10000.0)) - @log(@as(f64, 0.5))));
        defer _ = mlx.mlx_array_free(span);
        const rm = try mulA(ramp, span, s);
        defer _ = mlx.mlx_array_free(rm);
        const lmin = mlx.mlx_array_new_float(@floatCast(@log(@as(f64, 0.5))));
        defer _ = mlx.mlx_array_free(lmin);
        const ra = try addA(rm, lmin, s);
        defer _ = mlx.mlx_array_free(ra);
        var freqs = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_exp(&freqs, ra, s));
        defer _ = mlx.mlx_array_free(freqs);

        // args = t * freqs * 2 * pi  → [1,128]
        const tf = try mulA(t, freqs, s);
        defer _ = mlx.mlx_array_free(tf);
        const t2 = try mulScalar(tf, 2.0, s);
        defer _ = mlx.mlx_array_free(t2);
        const args = try mulScalar(t2, 3.141592653589793, s);
        defer _ = mlx.mlx_array_free(args);

        var cs = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_cos(&cs, args, s));
        defer _ = mlx.mlx_array_free(cs);
        var sn = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_sin(&sn, args, s));
        defer _ = mlx.mlx_array_free(sn);
        const vv = mlx.mlx_vector_array_new();
        defer _ = mlx.mlx_vector_array_free(vv);
        _ = mlx.mlx_vector_array_append_value(vv, cs);
        _ = mlx.mlx_vector_array_append_value(vv, sn);
        var ff = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_concatenate_axis(&ff, vv, 1, s));
        defer _ = mlx.mlx_array_free(ff);

        // ff @ W.T + b → [1,768] f32 → [1,1,768] f16
        const W = try getW(w, "cond.seconds_total_weight");
        const wt = [_]c_int{ 1, 0 };
        const Wt = try transposeA(W, &wt, s);
        defer _ = mlx.mlx_array_free(Wt);
        var proj = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_matmul(&proj, ff, Wt, s));
        defer _ = mlx.mlx_array_free(proj);
        const b = try getW(w, "cond.seconds_total_bias");
        const lin_o = try addA(proj, b, s);
        defer _ = mlx.mlx_array_free(lin_o);
        var e1 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_expand_dims(&e1, lin_o, 1, s));
        defer _ = mlx.mlx_array_free(e1);
        break :blk try astype(e1, .float16, s);
    };
    defer _ = mlx.mlx_array_free(secs_embed);

    // cross = concat(padded, seconds_embed, axis=1) → [1, S+1, 768]
    const cv = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(cv);
    _ = mlx.mlx_vector_array_append_value(cv, padded);
    _ = mlx.mlx_vector_array_append_value(cv, secs_embed);
    var cross = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&cross, cv, 1, s));
    errdefer _ = mlx.mlx_array_free(cross);

    // global_cond = seconds_embed[:, 0, :] — reshape IS that integer-index
    // view (dim removed, same buffer).
    const gsh = [_]c_int{ 1, dim };
    const global = try reshape(secs_embed, &gsh, s);

    return .{ .cross = cross, .global_cond = global };
}

// ── DiT (medium; reference = dit_mlx_medium.py) ──
//
// Contract: `dit.safetensors` — linears 8-bit affine (or dense f16), norms /
// biases / convs / memory f16, baked cond.* shared with conditionPrompt.
// Forward mirrors the reference op-for-op and lets MLX promote dtypes the way
// the reference does: the f32 timestep path widens the stream after block 0,
// so most of the net runs f32 and v comes back f32.

const DitJson = struct {
    io_channels: u32 = 256,
    embed_dim: u32 = 1536,
    depth: u32 = 24,
    num_heads: u32 = 24,
    head_dim: u32 = 64,
    rope_dims: u32 = 32,
    rope_theta: f32 = 10000.0,
    cond_token_dim: u32 = 768,
    global_cond_dim: u32 = 768,
    local_add_cond_dim: u32 = 257,
    num_memory_tokens: u32 = 64,
    ff_inner: u32 = 6144,
    timestep_feat_dim: u32 = 256,
    norm_eps: f32 = 1e-5,
    qk_norm_eps: f32 = 1e-6,
};

fn readDitCfg(io: std.Io, a: std.mem.Allocator, model_dir: []const u8) !DitJson {
    const path = try std.fmt.allocPrint(a, "{s}/config.json", .{model_dir});
    defer a.free(path);
    const f = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var rs = f.reader(io, &rb);
    const content = try rs.interface.allocRemaining(a, .limited(16 * 1024 * 1024));
    defer a.free(content);
    const parsed = try std.json.parseFromSlice(DitJson, a, content, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    return parsed.value;
}

// ── DiT micro-helpers (file-local) ──

fn sigmoidA(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_sigmoid(&o, x, s));
    return o;
}

/// nn.silu = x * sigmoid(x).
fn silu(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const sg = try sigmoidA(x, s);
    defer _ = mlx.mlx_array_free(sg);
    return mulA(x, sg, s);
}

/// nn.RMSNorm == mx.fast.rms_norm (mean accumulated in f32, out in x's dtype).
fn rmsFast(x: mlx.mlx_array, w: mlx.mlx_array, eps: f32, s: S) !mlx.mlx_array {
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_fast_rms_norm(&o, x, w, eps, s));
    return o;
}

/// Split into `n` equal parts on `axis`, owned outputs (caller frees).
fn splitEqual(x: mlx.mlx_array, n: usize, axis: c_int, out: []mlx.mlx_array, s: S) !void {
    var vec = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(vec);
    try mlx.check(mlx.mlx_split(&vec, x, @intCast(n), axis, s));
    for (0..n) |i| {
        var o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_vector_array_get(&o, vec, i));
        out[i] = o;
    }
}

fn concatA(x: mlx.mlx_array, y: mlx.mlx_array, axis: c_int, s: S) !mlx.mlx_array {
    const v = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(v);
    _ = mlx.mlx_vector_array_append_value(v, x);
    _ = mlx.mlx_vector_array_append_value(v, y);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&o, v, axis, s));
    return o;
}

/// mx.fast.scaled_dot_product_attention semantics (fast.cpp): promote q/k/v to
/// their result type, scale q, scores = q @ kᵀ, softmax precise, p @ v.
/// No mask — the reference passes none.
fn sdpa(q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32, mask: ?mlx.mlx_array, s: S) !mlx.mlx_array {
    const final: mlx.mlx_dtype = blk: {
        const dqs = mlx.mlx_array_dtype(q);
        if (dqs == .float32 or mlx.mlx_array_dtype(k) == .float32 or mlx.mlx_array_dtype(v) == .float32) break :blk .float32;
        break :blk dqs;
    };
    const q1 = if (mlx.mlx_array_dtype(q) == final) q else try astype(q, final, s);
    const q1_owned = mlx.mlx_array_dtype(q) != final;
    defer if (q1_owned) { _ = mlx.mlx_array_free(q1); };
    const k1 = if (mlx.mlx_array_dtype(k) == final) k else try astype(k, final, s);
    const k1_owned = mlx.mlx_array_dtype(k) != final;
    defer if (k1_owned) { _ = mlx.mlx_array_free(k1); };
    const v1 = if (mlx.mlx_array_dtype(v) == final) v else try astype(v, final, s);
    const v1_owned = mlx.mlx_array_dtype(v) != final;
    defer if (v1_owned) { _ = mlx.mlx_array_free(v1); };

    const qs = try mulScalar(q1, scale, s);
    defer _ = mlx.mlx_array_free(qs);
    const t0132 = [_]c_int{ 0, 1, 3, 2 };
    const kt = try transposeA(k1, &t0132, s);
    defer _ = mlx.mlx_array_free(kt);
    var scores = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&scores, qs, kt, s));
    defer _ = mlx.mlx_array_free(scores);
    if (mask) |m| {
        var masked = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_add(&masked, scores, m, s));
        _ = mlx.mlx_array_free(scores);
        scores = masked; // defer frees the reassigned handle — deliberate
    }
    var p = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_softmax_axis(&p, scores, 3, true, s));
    defer _ = mlx.mlx_array_free(p);
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_matmul(&out, p, v1, s));
    return out;
}

// ── DiT blocks ──

/// Differential self-attention: fused 5-way QKV split, shared qk RMS norms,
/// partial RoPE (first rope_dims of head_dim), two SDPAs subtracted.
fn ditSelfAttn(a: std.mem.Allocator, w: *const Weights, cfg: *const DitJson, x: mlx.mlx_array, layer: usize, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const B: c_int = sh[0];
    const Sl: c_int = sh[1];
    const E: c_int = @intCast(cfg.embed_dim);
    const H: c_int = @intCast(cfg.num_heads);
    const D: c_int = @intCast(cfg.head_dim);
    const scale: f32 = std.math.pow(f32, @as(f32, @floatFromInt(cfg.head_dim)), -0.5);

    const qkvp = try std.fmt.allocPrint(a, "transformer.layers.{d}.self_attn.to_qkv", .{layer});
    defer a.free(qkvp);
    const qkv = try lin(w, a, x, qkvp, s);
    defer _ = mlx.mlx_array_free(qkv);
    var parts: [5]mlx.mlx_array = undefined;
    try splitEqual(qkv, 5, -1, &parts, s);
    defer { for (parts) |p| _ = mlx.mlx_array_free(p); }

    // [B, Sl, E] → [B, H, Sl, D]
    const hsh = [_]c_int{ B, Sl, H, D };
    const t0213 = [_]c_int{ 0, 2, 1, 3 };
    var heads: [5]mlx.mlx_array = undefined;
    for (parts, 0..) |p, i| {
        const r4 = try reshape(p, &hsh, s);
        defer _ = mlx.mlx_array_free(r4);
        heads[i] = try transposeA(r4, &t0213, s);
    }
    defer { for (heads) |h| _ = mlx.mlx_array_free(h); }

    // q_norm/k_norm on {q,k} and {q_diff,k_diff} (same weights), then RoPE.
    const qn = try std.fmt.allocPrint(a, "transformer.layers.{d}.self_attn.q_norm.weight", .{layer});
    defer a.free(qn);
    const kn = try std.fmt.allocPrint(a, "transformer.layers.{d}.self_attn.k_norm.weight", .{layer});
    defer a.free(kn);
    const qw = try getW(w, qn);
    const kw = try getW(w, kn);
    var normed: [4]mlx.mlx_array = undefined;
    for ([_]usize{ 0, 1, 3, 4 }, 0..) |pi, i| {
        const uses_q = pi == 0 or pi == 3;
        normed[i] = try rmsFast(heads[pi], if (uses_q) qw else kw, cfg.qk_norm_eps, s);
    }
    defer { for (normed) |n| _ = mlx.mlx_array_free(n); }
    // RoPE on all four (fused mx.fast.rope — same primitive as the reference)
    for (normed, 0..) |n, i| {
        var r = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_fast_rope(&r, n, @intCast(cfg.rope_dims), false, mlx.mlx_optional_float.some(cfg.rope_theta), 1.0, 0, .{ .ctx = null }, s));
        _ = mlx.mlx_array_free(n);
        normed[i] = r;
    }

    // out = SDPA(q,k,v) − SDPA(q_diff,k_diff,v)
    const main = try sdpa(normed[0], normed[1], heads[2], scale, null, s);
    defer _ = mlx.mlx_array_free(main);
    const diff = try sdpa(normed[2], normed[3], heads[2], scale, null, s);
    defer _ = mlx.mlx_array_free(diff);
    const out = try subA(main, diff, s);
    defer _ = mlx.mlx_array_free(out);

    const ot = try transposeA(out, &t0213, s);
    defer _ = mlx.mlx_array_free(ot);
    const osh = [_]c_int{ B, Sl, E };
    const or_ = try reshape(ot, &osh, s);
    defer _ = mlx.mlx_array_free(or_);

    const opp = try std.fmt.allocPrint(a, "transformer.layers.{d}.self_attn.to_out", .{layer});
    defer a.free(opp);
    return lin(w, a, or_, opp, s);
}

/// Differential cross-attention: separate to_q (2×) / to_kv (3×), no RoPE,
/// two SDPAs subtracted. qk norms shared with self-attn's per layer.
fn ditCrossAttn(a: std.mem.Allocator, w: *const Weights, cfg: *const DitJson, x: mlx.mlx_array, context: mlx.mlx_array, layer: usize, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const B: c_int = sh[0];
    const Tx: c_int = sh[1];
    const Tc: c_int = mlx.getShape(context)[1];
    const E: c_int = @intCast(cfg.embed_dim);
    const H: c_int = @intCast(cfg.num_heads);
    const D: c_int = @intCast(cfg.head_dim);
    const scale: f32 = std.math.pow(f32, @as(f32, @floatFromInt(cfg.head_dim)), -0.5);

    const qp = try std.fmt.allocPrint(a, "transformer.layers.{d}.cross_attn.to_q", .{layer});
    defer a.free(qp);
    const kvp = try std.fmt.allocPrint(a, "transformer.layers.{d}.cross_attn.to_kv", .{layer});
    defer a.free(kvp);
    const opp = try std.fmt.allocPrint(a, "transformer.layers.{d}.cross_attn.to_out", .{layer});
    defer a.free(opp);

    const q_all = try lin(w, a, x, qp, s);
    defer _ = mlx.mlx_array_free(q_all);
    var qps: [2]mlx.mlx_array = undefined;
    try splitEqual(q_all, 2, -1, &qps, s);
    defer { for (qps) |p| _ = mlx.mlx_array_free(p); }

    const kv_all = try lin(w, a, context, kvp, s);
    defer _ = mlx.mlx_array_free(kv_all);
    var kvs: [3]mlx.mlx_array = undefined;
    try splitEqual(kv_all, 3, -1, &kvs, s);
    defer { for (kvs) |p| _ = mlx.mlx_array_free(p); }

    // [B, T, E] → [B, H, T, D]
    const qsh = [_]c_int{ B, Tx, H, D };
    const csh = [_]c_int{ B, Tc, H, D };
    const t0213 = [_]c_int{ 0, 2, 1, 3 };
    var qh: [2]mlx.mlx_array = undefined;
    for (qps, 0..) |p, i| {
        const r4 = try reshape(p, &qsh, s);
        defer _ = mlx.mlx_array_free(r4);
        qh[i] = try transposeA(r4, &t0213, s);
    }
    defer { for (qh) |h| _ = mlx.mlx_array_free(h); }
    var ch: [3]mlx.mlx_array = undefined;
    for (kvs, 0..) |p, i| {
        const r4 = try reshape(p, &csh, s);
        defer _ = mlx.mlx_array_free(r4);
        ch[i] = try transposeA(r4, &t0213, s);
    }
    defer { for (ch) |h| _ = mlx.mlx_array_free(h); }

    const qn = try std.fmt.allocPrint(a, "transformer.layers.{d}.cross_attn.q_norm.weight", .{layer});
    defer a.free(qn);
    const kn = try std.fmt.allocPrint(a, "transformer.layers.{d}.cross_attn.k_norm.weight", .{layer});
    defer a.free(kn);
    const qw = try getW(w, qn);
    const kw = try getW(w, kn);

    const q0 = try rmsFast(qh[0], qw, cfg.qk_norm_eps, s);
    defer _ = mlx.mlx_array_free(q0);
    const q1 = try rmsFast(qh[1], qw, cfg.qk_norm_eps, s);
    defer _ = mlx.mlx_array_free(q1);
    const k0 = try rmsFast(ch[0], kw, cfg.qk_norm_eps, s);
    defer _ = mlx.mlx_array_free(k0);
    const k1 = try rmsFast(ch[1], kw, cfg.qk_norm_eps, s);
    defer _ = mlx.mlx_array_free(k1);

    const main = try sdpa(q0, k0, ch[2], scale, null, s);
    defer _ = mlx.mlx_array_free(main);
    const diff = try sdpa(q1, k1, ch[2], scale, null, s);
    defer _ = mlx.mlx_array_free(diff);
    const out = try subA(main, diff, s);
    defer _ = mlx.mlx_array_free(out);

    const ot = try transposeA(out, &t0213, s);
    defer _ = mlx.mlx_array_free(ot);
    const osh = [_]c_int{ B, Tx, E };
    const or_ = try reshape(ot, &osh, s);
    defer _ = mlx.mlx_array_free(or_);
    return lin(w, a, or_, opp, s);
}

/// GLU feed-forward: proj → (x, gate) split → x * silu(gate) → down.
fn ditFF(a: std.mem.Allocator, w: *const Weights, layer: usize, x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const pp = try std.fmt.allocPrint(a, "transformer.layers.{d}.ff.ff.0.proj", .{layer});
    defer a.free(pp);
    const dp = try std.fmt.allocPrint(a, "transformer.layers.{d}.ff.ff.2", .{layer});
    defer a.free(dp);
    const proj = try lin(w, a, x, pp, s);
    defer _ = mlx.mlx_array_free(proj);
    var halves: [2]mlx.mlx_array = undefined;
    try splitEqual(proj, 2, -1, &halves, s);
    defer { for (halves) |h| _ = mlx.mlx_array_free(h); }
    const g = try silu(halves[1], s);
    defer _ = mlx.mlx_array_free(g);
    const y = try mulA(halves[0], g, s);
    defer _ = mlx.mlx_array_free(y);
    return lin(w, a, y, dp, s);
}

/// Per-block local embed MLP: seq.0 → silu → seq.2.
fn ditLocalEmbed(a: std.mem.Allocator, w: *const Weights, layer: usize, x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const p0 = try std.fmt.allocPrint(a, "transformer.layers.{d}.to_local_embed.seq.0", .{layer});
    defer a.free(p0);
    const p2 = try std.fmt.allocPrint(a, "transformer.layers.{d}.to_local_embed.seq.2", .{layer});
    defer a.free(p2);
    const h = try lin(w, a, x, p0, s);
    defer _ = mlx.mlx_array_free(h);
    const hs = try silu(h, s);
    defer _ = mlx.mlx_array_free(hs);
    return lin(w, a, hs, p2, s);
}

/// One TransformerBlock (reference order: scale/shift/gate self-attn →
/// cross → +local embed → scale/shift/gate ff).
fn ditBlock(
    a: std.mem.Allocator,
    w: *const Weights,
    cfg: *const DitJson,
    x: mlx.mlx_array,
    context: mlx.mlx_array,
    global_cond: mlx.mlx_array,
    local_padded: mlx.mlx_array,
    layer: usize,
    s: S,
) !mlx.mlx_array {
    // ss = (to_scale_shift_gate + global_cond)[:, None, :] → 6 splits
    const parts: [6]mlx.mlx_array = blk: {
        const gk = try std.fmt.allocPrint(a, "transformer.layers.{d}.to_scale_shift_gate", .{layer});
        defer a.free(gk);
        const g = try getW(w, gk);
        const sum = try addA(g, global_cond, s);
        defer _ = mlx.mlx_array_free(sum);
        var e = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_expand_dims(&e, sum, 1, s));
        defer _ = mlx.mlx_array_free(e);
        var out: [6]mlx.mlx_array = undefined;
        try splitEqual(e, 6, -1, &out, s);
        break :blk out;
    };
    defer { for (parts) |p| _ = mlx.mlx_array_free(p); }
    const scale_self = parts[0];
    const shift_self = parts[1];
    const gate_self = parts[2];
    const scale_ff = parts[3];
    const shift_ff = parts[4];
    const gate_ff = parts[5];

    const pre_n = try std.fmt.allocPrint(a, "transformer.layers.{d}.pre_norm.weight", .{layer});
    defer a.free(pre_n);
    const post_n = try std.fmt.allocPrint(a, "transformer.layers.{d}.cross_attend_norm.weight", .{layer});
    defer a.free(post_n);
    const ff_n = try std.fmt.allocPrint(a, "transformer.layers.{d}.ff_norm.weight", .{layer});
    defer a.free(ff_n);

    // ── self-attn branch ──
    const h0 = try rmsFast(x, try getW(w, pre_n), cfg.norm_eps, s);
    defer _ = mlx.mlx_array_free(h0);
    const s1 = try addScalar(scale_self, 1.0, s); // h * (1 + scale) + shift
    defer _ = mlx.mlx_array_free(s1);
    const h1 = try mulA(h0, s1, s);
    defer _ = mlx.mlx_array_free(h1);
    const h2 = try addA(h1, shift_self, s);
    defer _ = mlx.mlx_array_free(h2);
    const sa = try ditSelfAttn(a, w, cfg, h2, layer, s);
    defer _ = mlx.mlx_array_free(sa);
    const gs0 = try scalarLike(gate_self, 1.0, s);
    defer _ = mlx.mlx_array_free(gs0);
    const gi = try subA(gs0, gate_self, s);
    defer _ = mlx.mlx_array_free(gi);
    const sg = try sigmoidA(gi, s);
    defer _ = mlx.mlx_array_free(sg);
    const hg = try mulA(sa, sg, s);
    defer _ = mlx.mlx_array_free(hg);
    const x1 = try addA(hg, x, s);
    defer _ = mlx.mlx_array_free(x1);

    // ── cross-attn branch ──
    const cn = try rmsFast(x1, try getW(w, post_n), cfg.norm_eps, s);
    defer _ = mlx.mlx_array_free(cn);
    const ca = try ditCrossAttn(a, w, cfg, cn, context, layer, s);
    defer _ = mlx.mlx_array_free(ca);
    const x2 = try addA(x1, ca, s);
    defer _ = mlx.mlx_array_free(x2);

    // ── + local embed ──
    const x3 = try addA(x2, local_padded, s);
    defer _ = mlx.mlx_array_free(x3);

    // ── ff branch ──
    const f0 = try rmsFast(x3, try getW(w, ff_n), cfg.norm_eps, s);
    defer _ = mlx.mlx_array_free(f0);
    const f1s = try addScalar(scale_ff, 1.0, s);
    defer _ = mlx.mlx_array_free(f1s);
    const f1 = try mulA(f0, f1s, s);
    defer _ = mlx.mlx_array_free(f1);
    const f2 = try addA(f1, shift_ff, s);
    defer _ = mlx.mlx_array_free(f2);
    const ff_out = try ditFF(a, w, layer, f2, s);
    defer _ = mlx.mlx_array_free(ff_out);
    const fgs0 = try scalarLike(gate_ff, 1.0, s);
    defer _ = mlx.mlx_array_free(fgs0);
    const fgi = try subA(fgs0, gate_ff, s);
    defer _ = mlx.mlx_array_free(fgi);
    const fsg = try sigmoidA(fgi, s);
    defer _ = mlx.mlx_array_free(fsg);
    const fhg = try mulA(ff_out, fsg, s);
    defer _ = mlx.mlx_array_free(fhg);
    return addA(fhg, x3, s);
}

pub const Dit = struct {
    cfg: DitJson,
    w: Weights,
    s: S,

    pub fn load(io: std.Io, a: std.mem.Allocator, model_dir: []const u8) !Dit {
        var self: Dit = undefined;
        self.cfg = try readDitCfg(io, a, model_dir);
        self.w = try loadFileWeights(a, model_dir, "dit.safetensors");
        self.s = mlx.mlx_default_gpu_stream_new();
        return self;
    }

    pub fn deinit(self: Dit) void {
        var w = self.w;
        w.deinit();
        _ = mlx.mlx_stream_free(self.s);
    }

    /// Reference DiT.__call__: x [1, io, T] f16, t [1] f32, cross [1, S+1, cond]
    /// f16, global_cond [1, cond] f16, local [1, T, local_dim] or null for the
    /// zero inpaint buffer. Returns v [1, io, T] (f32: the timestep path
    /// widens the stream — matches the reference's own promotion).
    pub fn forward(
        self: *const Dit,
        a: std.mem.Allocator,
        x: mlx.mlx_array,
        t: mlx.mlx_array,
        cross: mlx.mlx_array,
        global_cond_raw: mlx.mlx_array,
        local: ?mlx.mlx_array,
        s: S,
    ) !mlx.mlx_array {
        const cfg = &self.cfg;
        const w = &self.w;
        const E: c_int = @intCast(cfg.embed_dim);
        const mem: c_int = @intCast(cfg.num_memory_tokens);
        const ld: c_int = @intCast(cfg.local_add_cond_dim);
        // B > 1 only happens on the CFG path (batched cond+uncond); every
        // shape below derives from it, mirroring dit_mlx_medium's `B = x.shape[0]`.
        const B: c_int = mlx.getShape(x)[0];
        const T: c_int = mlx.getShape(x)[2];

        // ── cond projections ──
        const context = blk: {
            const h = try lin(w, a, cross, "to_cond_embed.0", s);
            defer _ = mlx.mlx_array_free(h);
            const hs = try silu(h, s);
            defer _ = mlx.mlx_array_free(hs);
            break :blk try lin(w, a, hs, "to_cond_embed.2", s);
        };
        defer _ = mlx.mlx_array_free(context);

        const global_pre = blk: {
            const h = try lin(w, a, global_cond_raw, "to_global_embed.0", s);
            defer _ = mlx.mlx_array_free(h);
            const hs = try silu(h, s);
            defer _ = mlx.mlx_array_free(hs);
            break :blk try lin(w, a, hs, "to_global_embed.2", s);
        };
        defer _ = mlx.mlx_array_free(global_pre);

        // ── timestep: expo fourier (linspace == arange/127 exactly) + MLP ──
        const t_embed = blk: {
            const F: c_int = @intCast(cfg.timestep_feat_dim);
            const half: c_int = @divExact(F, 2);
            var ramp0 = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_arange(&ramp0, 0, @floatFromInt(half), 1, .float32, s));
            defer _ = mlx.mlx_array_free(ramp0);
            const den = mlx.mlx_array_new_float(@floatFromInt(half - 1));
            defer _ = mlx.mlx_array_free(den);
            const ramp = try divA(ramp0, den, s);
            defer _ = mlx.mlx_array_free(ramp);
            const span = mlx.mlx_array_new_float(@floatCast(@log(@as(f64, 10000.0)) - @log(@as(f64, 0.5))));
            defer _ = mlx.mlx_array_free(span);
            const rm = try mulA(ramp, span, s);
            defer _ = mlx.mlx_array_free(rm);
            const lmin = mlx.mlx_array_new_float(@floatCast(@log(@as(f64, 0.5))));
            defer _ = mlx.mlx_array_free(lmin);
            const ra = try addA(rm, lmin, s);
            defer _ = mlx.mlx_array_free(ra);
            var f1 = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_exp(&f1, ra, s));
            defer _ = mlx.mlx_array_free(f1);
            const f2 = try mulScalar(f1, 2.0, s);
            defer _ = mlx.mlx_array_free(f2);
            const freqs = try mulScalar(f2, 3.141592653589793, s);
            defer _ = mlx.mlx_array_free(freqs);
            const tsh = [_]c_int{ B, 1 };
            const t2 = try reshape(t, &tsh, s);
            defer _ = mlx.mlx_array_free(t2);
            const args = try mulA(t2, freqs, s);
            defer _ = mlx.mlx_array_free(args);
            var cs = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_cos(&cs, args, s));
            defer _ = mlx.mlx_array_free(cs);
            var sn = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_sin(&sn, args, s));
            defer _ = mlx.mlx_array_free(sn);
            const tf = try concatA(cs, sn, 1, s);
            defer _ = mlx.mlx_array_free(tf);
            const h0 = try lin(w, a, tf, "to_timestep_embed.0", s);
            defer _ = mlx.mlx_array_free(h0);
            const h1 = try silu(h0, s);
            defer _ = mlx.mlx_array_free(h1);
            break :blk try lin(w, a, h1, "to_timestep_embed.2", s);
        };
        defer _ = mlx.mlx_array_free(t_embed);

        const global_embed = try addA(global_pre, t_embed, s); // f16 + f32 → f32
        defer _ = mlx.mlx_array_free(global_embed);

        // ── preprocess conv (1×1, NLC) + residual ──
        const x_lc_t = [_]c_int{ 0, 2, 1 };
        const x_lc = try transposeA(x, &x_lc_t, s);
        defer _ = mlx.mlx_array_free(x_lc);
        var conv_o = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_conv1d(&conv_o, x_lc, try getW(w, "preprocess_conv.weight"), 1, 0, 1, 1, s));
        defer _ = mlx.mlx_array_free(conv_o);
        const x_pp = try addA(conv_o, x_lc, s);
        defer _ = mlx.mlx_array_free(x_pp);

        // ── local inpaint buffer: zeros at INPUT length (reference rule) ──
        const local_buf = if (local) |l| l else blk: {
            const lsh = [_]c_int{ B, T, ld };
            var z = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&z, &lsh, 3, .float32, s));
            break :blk z;
        };
        defer if (local == null) { _ = mlx.mlx_array_free(local_buf); };

        // ── ContinuousTransformer ──
        var seq = blk: {
            const p = try lin(w, a, x_pp, "transformer.project_in", s);
            defer _ = mlx.mlx_array_free(p);
            var m = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_expand_dims(&m, try getW(w, "transformer.memory_tokens"), 0, s));
            defer _ = mlx.mlx_array_free(m);
            // reference: broadcast_to(memory_tokens[None], (B, mem, E))
            const msh = [_]c_int{ B, mem, E };
            var mb = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_broadcast_to(&mb, m, &msh, 3, s));
            defer _ = mlx.mlx_array_free(mb);
            break :blk try concatA(mb, p, 1, s);
        };
        defer _ = mlx.mlx_array_free(seq);

        const gcond = blk: {
            const h = try lin(w, a, global_embed, "transformer.global_cond_embedder.0", s);
            defer _ = mlx.mlx_array_free(h);
            const hs = try silu(h, s);
            defer _ = mlx.mlx_array_free(hs);
            break :blk try lin(w, a, hs, "transformer.global_cond_embedder.2", s);
        };
        defer _ = mlx.mlx_array_free(gcond);

        for (0..cfg.depth) |i| {
            const lem = try ditLocalEmbed(a, w, i, local_buf, s);
            defer _ = mlx.mlx_array_free(lem);
            const psh = [_]c_int{ B, mem, E };
            var pad = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_zeros(&pad, &psh, 3, mlx.mlx_array_dtype(lem), s));
            defer _ = mlx.mlx_array_free(pad);
            const lp = try concatA(pad, lem, 1, s);
            defer _ = mlx.mlx_array_free(lp);
            const nx = try ditBlock(a, w, cfg, seq, context, gcond, lp, i, s);
            _ = mlx.mlx_array_free(seq);
            seq = nx;
        }

        // strip memory tokens, project_out
        const stripped = blk: {
            const seq_len: c_int = mlx.getShape(seq)[1];
            const lo = [_]c_int{ 0, mem, 0 };
            const hi = [_]c_int{ B, seq_len, E };
            const stp = [_]c_int{ 1, 1, 1 };
            var o = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&o, seq, &lo, 3, &hi, 3, &stp, 3, s));
            break :blk o;
        };
        defer _ = mlx.mlx_array_free(stripped);
        const h_out = try lin(w, a, stripped, "transformer.project_out", s);
        defer _ = mlx.mlx_array_free(h_out);

        // ── postprocess conv + residual → [1, io, T] ──
        var pconv = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_conv1d(&pconv, h_out, try getW(w, "postprocess_conv.weight"), 1, 0, 1, 1, s));
        defer _ = mlx.mlx_array_free(pconv);
        const hb = try addA(pconv, h_out, s);
        defer _ = mlx.mlx_array_free(hb);
        const back_t = [_]c_int{ 0, 2, 1 };
        return transposeA(hb, &back_t, s);
    }
};

// ── T5Gemma oracle ──────────────────────────────────────────────────────────


test "stable_audio3 oracle: T5Gemma hidden matches reference" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = try sa3ModelDir();
    const fix = try fixturesDir();

    const ids = try readRawI32(io, a, fix, "ids_cond.i32.raw");
    defer a.free(ids);
    const mask = try readRawI32(io, a, fix, "mask_cond.i32.raw");
    defer a.free(mask);
    const ref = try readRawF32(io, a, fix, "t5_hidden_cond.f32.raw");
    defer a.free(ref);

    const enc = try T5Gemma.load(io, a, dir);
    defer enc.deinit();
    const hidden = try enc.encode(a, ids, mask, enc.s);
    defer _ = mlx.mlx_array_free(hidden);
    try assertParity(hidden, ref, "t5 cond", 0.999, 0.01, enc.s);
}

test "stable_audio3 oracle: T5Gemma empty prompt matches reference" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = try sa3ModelDir();
    const fix = try fixturesDir();

    const ids = try readRawI32(io, a, fix, "ids_empty.i32.raw");
    defer a.free(ids);
    const mask = try readRawI32(io, a, fix, "mask_empty.i32.raw");
    defer a.free(mask);
    const ref = try readRawF32(io, a, fix, "t5_hidden_empty.f32.raw");
    defer a.free(ref);

    const enc = try T5Gemma.load(io, a, dir);
    defer enc.deinit();
    const hidden = try enc.encode(a, ids, mask, enc.s);
    defer _ = mlx.mlx_array_free(hidden);
    try assertParity(hidden, ref, "t5 empty", 0.999, 0.01, enc.s);
}

// ── conditioner oracle ──────────────────────────────────────────────────────

fn readMetaSeconds(io: std.Io, a: std.mem.Allocator, fix: []const u8) !f32 {
    const path = try std.fmt.allocPrint(a, "{s}/meta.json", .{fix});
    defer a.free(path);
    const f = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var rs = f.reader(io, &rb);
    const content = try rs.interface.allocRemaining(a, .limited(4 * 1024 * 1024));
    defer a.free(content);
    const parsed = try std.json.parseFromSlice(std.json.Value, a, content, .{});
    defer parsed.deinit();
    const v = parsed.value.object.get("seconds") orelse return error.MetaMissingSeconds;
    return switch (v) {
        .float => @floatCast(v.float),
        .integer => @floatFromInt(v.integer),
        else => error.MetaBadSeconds,
    };
}

test "stable_audio3 oracle: conditioner cross_attn + global_cond match reference" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = try sa3ModelDir();
    const fix = try fixturesDir();
    const st = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(st);

    const embeds_raw = try readRawF32(io, a, fix, "t5_hidden_cond.f32.raw");
    defer a.free(embeds_raw);
    const mask = try readRawI32(io, a, fix, "mask_cond.i32.raw");
    defer a.free(mask);
    const ref_cross = try readRawF32(io, a, fix, "cross_attn.f32.raw");
    defer a.free(ref_cross);
    const ref_glob = try readRawF32(io, a, fix, "global_cond.f32.raw");
    defer a.free(ref_glob);
    const seconds = try readMetaSeconds(io, a, fix);

    var w = try loadFileWeights(a, dir, "dit.safetensors");
    defer w.deinit();

    // fixture hidden is f16-valued stored as f32 — back to the f16 the
    // reference conditioner consumed (exact: values are f16-representable).
    const msh = [_]c_int{ 1, 256, 768 };
    const e32 = mlx.mlx_array_new_data(embeds_raw.ptr, &msh, 3, .float32);
    defer _ = mlx.mlx_array_free(e32);
    const embeds = try astype(e32, .float16, st);
    defer _ = mlx.mlx_array_free(embeds);

    const out = try conditionPrompt(&w, embeds, mask, seconds, st);
    defer out.deinit();

    try assertParity(out.cross, ref_cross, "cond cross", 0.999, 0.01, st);
    try assertParity(out.global_cond, ref_glob, "cond global", 0.999, 0.01, st);
}

// ── CFG (stage 2; reference = sa3_mlx.py model_fn CFG branch, ~line 736) ────

test "stable_audio3 oracle: negative-prompt cross_attn matches reference" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = try sa3ModelDir();
    const fix = try fixturesDir();
    const st = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(st);

    const ids = try readRawI32(io, a, fix, "ids_neg.i32.raw");
    defer a.free(ids);
    const mask = try readRawI32(io, a, fix, "mask_neg.i32.raw");
    defer a.free(mask);
    const ref_cross = try readRawF32(io, a, fix, "cross_attn_neg.f32.raw");
    defer a.free(ref_cross);
    const seconds = try readMetaSeconds(io, a, fix);

    // full conditioning path for the uncond branch: T5 on NEG_PROMPT, then
    // the SAME padding + seconds-token concat the positive prompt gets.
    var enc = try T5Gemma.load(io, a, dir);
    defer enc.deinit();
    const hidden = try enc.encode(a, ids, mask, enc.s);
    defer _ = mlx.mlx_array_free(hidden);
    const h16 = try astype(hidden, .float16, st);
    defer _ = mlx.mlx_array_free(h16);

    var w = try loadFileWeights(a, dir, "dit.safetensors");
    defer w.deinit();
    const out = try conditionPrompt(&w, h16, mask, seconds, st);
    defer out.deinit();
    try assertParity(out.cross, ref_cross, "neg cross", 0.999, 0.01, st);
}

// ── DiT oracle ──────────────────────────────────────────────────────────────

test "stable_audio3 oracle: DiT velocity taps match reference" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = try sa3ModelDir();
    const fix = try fixturesDir();
    const st = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(st);

    const x0_raw = try readRawF32(io, a, fix, "x0.f32.raw");
    defer a.free(x0_raw);
    const cross_raw = try readRawF32(io, a, fix, "cross_attn.f32.raw");
    defer a.free(cross_raw);
    const glob_raw = try readRawF32(io, a, fix, "global_cond.f32.raw");
    defer a.free(glob_raw);

    var dit = try Dit.load(io, a, dir);
    defer dit.deinit();

    const C: u32 = dit.cfg.io_channels;
    const T: u32 = @intCast(x0_raw.len / C);
    try testing.expectEqual(@as(usize, 197376), cross_raw.len);

    // fixture values are f16 stored as f32 — back to f16 (exact).
    const xsh = [_]c_int{ 1, @intCast(C), @intCast(T) };
    const x0_32 = mlx.mlx_array_new_data(x0_raw.ptr, &xsh, 3, .float32);
    defer _ = mlx.mlx_array_free(x0_32);
    const x0 = try astype(x0_32, .float16, st);
    defer _ = mlx.mlx_array_free(x0);

    const csh = [_]c_int{ 1, 257, 768 };
    const cross_32 = mlx.mlx_array_new_data(cross_raw.ptr, &csh, 3, .float32);
    defer _ = mlx.mlx_array_free(cross_32);
    const cross = try astype(cross_32, .float16, st);
    defer _ = mlx.mlx_array_free(cross);

    const gsh = [_]c_int{ 1, 768 };
    const glob_32 = mlx.mlx_array_new_data(glob_raw.ptr, &gsh, 2, .float32);
    defer _ = mlx.mlx_array_free(glob_32);
    const glob = try astype(glob_32, .float16, st);
    defer _ = mlx.mlx_array_free(glob);

    const taps = [_]struct { tv: f32, label: []const u8, file: []const u8 }{
        .{ .tv = 0.1, .label = "t10", .file = "v_t10.f32.raw" },
        .{ .tv = 0.5, .label = "t50", .file = "v_t50.f32.raw" },
        .{ .tv = 0.9, .label = "t90", .file = "v_t90.f32.raw" },
    };
    for (taps) |tap| {
        const ref = try readRawF32(io, a, fix, tap.file);
        defer a.free(ref);
        const tv = tap.tv;
        const tsh = [_]c_int{1};
        const t_arr = mlx.mlx_array_new_data(&tv, &tsh, 1, .float32);
        defer _ = mlx.mlx_array_free(t_arr);

        const v = try dit.forward(a, x0, t_arr, cross, glob, null, st);
        defer _ = mlx.mlx_array_free(v);
        try assertParity(v, ref, tap.label, 0.999, 0.01, st);
    }
}

test "stable_audio3 oracle: CFG velocity taps match reference (apg branches + zero uncond)" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = try sa3ModelDir();
    const fix = try fixturesDir();
    const st = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(st);

    const x0_raw = try readRawF32(io, a, fix, "x0.f32.raw");
    defer a.free(x0_raw);
    const cross_raw = try readRawF32(io, a, fix, "cross_attn.f32.raw");
    defer a.free(cross_raw);
    const neg_raw = try readRawF32(io, a, fix, "cross_attn_neg.f32.raw");
    defer a.free(neg_raw);
    const glob_raw = try readRawF32(io, a, fix, "global_cond.f32.raw");
    defer a.free(glob_raw);

    var dit = try Dit.load(io, a, dir);
    defer dit.deinit();

    const T: u32 = @intCast(x0_raw.len / 256);
    const x0_32 = mlx.mlx_array_new_data(x0_raw.ptr, &[_]c_int{ 1, 256, @intCast(T) }, 3, .float32);
    defer _ = mlx.mlx_array_free(x0_32);
    const x0 = try astype(x0_32, .float16, st);
    defer _ = mlx.mlx_array_free(x0);
    const cross_32 = mlx.mlx_array_new_data(cross_raw.ptr, &[_]c_int{ 1, 257, 768 }, 3, .float32);
    defer _ = mlx.mlx_array_free(cross_32);
    const cross = try astype(cross_32, .float16, st);
    defer _ = mlx.mlx_array_free(cross);
    const neg_32 = mlx.mlx_array_new_data(neg_raw.ptr, &[_]c_int{ 1, 257, 768 }, 3, .float32);
    defer _ = mlx.mlx_array_free(neg_32);
    const cross_neg = try astype(neg_32, .float16, st);
    defer _ = mlx.mlx_array_free(cross_neg);
    const glob_32 = mlx.mlx_array_new_data(glob_raw.ptr, &[_]c_int{ 1, 768 }, 2, .float32);
    defer _ = mlx.mlx_array_free(glob_32);
    const glob = try astype(glob_32, .float16, st);
    defer _ = mlx.mlx_array_free(glob);

    // zeros uncond — sa3_mlx.py's mx.zeros_like(cross_attn) arm.
    const zsh = [_]c_int{ 1, 257, 768 };
    var zeros_cross = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_zeros(&zeros_cross, &zsh, 3, .float16, st));
    defer _ = mlx.mlx_array_free(zeros_cross);

    // (file, label, cfg, apg, negative prompt, t) — one fixture per formula
    // branch: full APG at two t's, vanilla, the intermediate blend, zeros.
    const taps = [_]struct { file: []const u8, label: []const u8, cfg: f32, apg: f32, neg: bool, tv: f32 }{
        .{ .file = "cfg_v_apg1_t01.f32.raw", .label = "cfg apg=1 t=0.1", .cfg = 3.0, .apg = 1.0, .neg = true, .tv = 0.1 },
        .{ .file = "cfg_v_apg1_t09.f32.raw", .label = "cfg apg=1 t=0.9", .cfg = 3.0, .apg = 1.0, .neg = true, .tv = 0.9 },
        .{ .file = "cfg_v_apg0_t05.f32.raw", .label = "cfg apg=0 t=0.5", .cfg = 3.0, .apg = 0.0, .neg = true, .tv = 0.5 },
        .{ .file = "cfg_v_apg05_t05.f32.raw", .label = "cfg apg=0.5 t=0.5", .cfg = 3.0, .apg = 0.5, .neg = true, .tv = 0.5 },
        .{ .file = "cfg_v_zerouncond_t05.f32.raw", .label = "cfg zero-uncond t=0.5", .cfg = 3.0, .apg = 1.0, .neg = false, .tv = 0.5 },
    };
    for (taps) |tap| {
        const ref = try readRawF32(io, a, fix, tap.file);
        defer a.free(ref);
        const tv = tap.tv;
        const t_arr = mlx.mlx_array_new_data(&tv, &[_]c_int{1}, 1, .float32);
        defer _ = mlx.mlx_array_free(t_arr);
        const v = try cfgVelocity(a, &dit, x0, t_arr, cross, glob, .{
            .cfg = tap.cfg,
            .apg = tap.apg,
            .null_cross = if (tap.neg) cross_neg else zeros_cross,
        }, st);
        defer _ = mlx.mlx_array_free(v);
        try assertParity(v, ref, tap.label, 0.999, 0.01, st);
    }
}

// ── Ping-pong sampler (reference = sa3_pipeline.sample_flow_pingpong) ──
//
// t math runs on the host in f32 — the same IEEE ops the reference does on
// f32 arrays (linspace, logsnr, casts); only sigmoid is host @exp, within
// 1 ulp of MLX's kernel. The model itself always receives f32 t.

/// linspace(sigma_max, 0, steps+1) → LogSNRShift (endpoints preserved) →
/// re-anchor start to sigma_max. Returns host f32, len steps+1.
pub fn buildPingpongSchedule(a: std.mem.Allocator, steps: usize, sigma_max: f32) ![]f32 {
    const out = try a.alloc(f32, steps + 1);
    errdefer a.free(out);
    for (0..steps + 1) |i| {
        const frac = @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(steps));
        const v = (1.0 - frac) * sigma_max;
        if (v <= 0.0) {
            out[i] = 0.0;
            continue;
        }
        if (v >= 1.0) {
            out[i] = 1.0;
            continue;
        }
        const logsnr = 2.0 - v * 8.2; // logsnr_end − t·(logsnr_end − anchor)
        out[i] = 1.0 / (1.0 + @exp(logsnr)); // sigmoid(−logsnr)
    }
    out[0] = sigma_max;
    return out;
}

/// [1] array of `v` in `dtype`. f64 in, one rounding out — matches the
/// reference's python float → array.astype(dtype) path.
fn scalarIn(dtype: mlx.mlx_dtype, v: f64, s: S) !mlx.mlx_array {
    const sh = [1]c_int{1};
    switch (dtype) {
        .float16 => {
            const h: f16 = @floatCast(v);
            return mlx.mlx_array_new_data(&h, &sh, 1, .float16);
        },
        .float32 => {
            const f: f32 = @floatCast(v);
            return mlx.mlx_array_new_data(&f, &sh, 1, .float32);
        },
        else => return error.UnsupportedSamplerDtype,
    }
    _ = s;
}

/// CFG inputs for the sampler (sa3_mlx.py model_fn CFG branch).
pub const Guidance = struct {
    cfg: f32,
    apg: f32,
    /// Uncond-branch cross_attn [1, S+1, 768] — the negative prompt's
    /// conditioning, or zeros_like(cross) when no negative prompt is set.
    /// Borrowed: the caller owns it and frees it after sampling.
    null_cross: mlx.mlx_array,
};

/// zeros_like(cross) — the uncond branch when no negative prompt is set
/// (sa3_mlx.py: `null_cross_attn = mx.zeros_like(cross_attn)`).
fn zerosLikeCross(cross: mlx.mlx_array, s: S) !mlx.mlx_array {
    const csh = mlx.getShape(cross);
    var z = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_zeros(&z, csh.ptr, csh.len, mlx.mlx_array_dtype(cross), s));
    return z;
}

/// sa3_mlx.py ~736-762, verbatim math: one batched cond+uncond forward over
/// cat([x, x]); the blend happens in DENOISED space (RF: d = x − σ·v with
/// σ = t), optionally projecting the CFG difference orthogonal to cond_d
/// (APG), then converts back to velocity in x's dtype. `t` is the sampler's
/// [1] f32 sigma. Returns an owned velocity for the batch-1 `x`.
fn cfgVelocity(
    a: std.mem.Allocator,
    dit: *const Dit,
    x: mlx.mlx_array,
    t: mlx.mlx_array,
    cross: mlx.mlx_array,
    global_cond: mlx.mlx_array,
    g: Guidance,
    s: S,
) !mlx.mlx_array {
    // ── batched forward: x2 [2,256,T], t2 [2], cross2 [2,S+1,768], g2 [2,768]
    const x2 = try concatA(x, x, 0, s);
    defer _ = mlx.mlx_array_free(x2);
    const t2 = try concatA(t, t, 0, s);
    defer _ = mlx.mlx_array_free(t2);
    const cross2 = try concatA(cross, g.null_cross, 0, s);
    defer _ = mlx.mlx_array_free(cross2);
    const global2 = try concatA(global_cond, global_cond, 0, s);
    defer _ = mlx.mlx_array_free(global2);
    const vb = try dit.forward(a, x2, t2, cross2, global2, null, s);
    defer _ = mlx.mlx_array_free(vb);
    var parts = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(parts);
    try mlx.check(mlx.mlx_split(&parts, vb, 2, 0, s));
    var cond_v = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&cond_v, parts, 0));
    defer _ = mlx.mlx_array_free(cond_v);
    var uncond_v = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_vector_array_get(&uncond_v, parts, 1));
    defer _ = mlx.mlx_array_free(uncond_v);

    // ── denoised space, f32 (reference: `x.astype(f32) − v.astype(f32)·σ`)
    const t32 = try astype(t, .float32, s);
    defer _ = mlx.mlx_array_free(t32);
    const sigma = try reshape(t32, &[_]c_int{ 1, 1, 1 }, s);
    defer _ = mlx.mlx_array_free(sigma);
    const x32 = try astype(x, .float32, s);
    defer _ = mlx.mlx_array_free(x32);
    const cv32 = try astype(cond_v, .float32, s);
    defer _ = mlx.mlx_array_free(cv32);
    const uv32 = try astype(uncond_v, .float32, s);
    defer _ = mlx.mlx_array_free(uv32);
    const cvm = try mulA(cv32, sigma, s);
    defer _ = mlx.mlx_array_free(cvm);
    const cond_d = try subA(x32, cvm, s);
    defer _ = mlx.mlx_array_free(cond_d);
    const uvm = try mulA(uv32, sigma, s);
    defer _ = mlx.mlx_array_free(uvm);
    const uncond_d = try subA(x32, uvm, s);
    defer _ = mlx.mlx_array_free(uncond_d);
    const diff = try subA(cond_d, uncond_d, s);
    defer _ = mlx.mlx_array_free(diff);

    // ── cfg_d = cond_d + (cfg − 1)·cfg_diff, per APG branch
    const cfg_d = blk: {
        if (g.apg <= 0.0) {
            // vanilla CFG: cfg_diff = diff
            const scaled = try mulScalar(diff, g.cfg - 1.0, s);
            defer _ = mlx.mlx_array_free(scaled);
            break :blk try addA(cond_d, scaled, s);
        }
        // APG: project diff onto the direction orthogonal to cond_d,
        // per-sample over (C, T) — fp32 throughout (reference: same).
        const sq = try mulA(cond_d, cond_d, s);
        defer _ = mlx.mlx_array_free(sq);
        var sum1 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_sum_axis(&sum1, sq, 1, true, s));
        defer _ = mlx.mlx_array_free(sum1);
        var sum2 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_sum_axis(&sum2, sum1, 2, true, s));
        defer _ = mlx.mlx_array_free(sum2);
        var norm = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_sqrt(&norm, sum2, s));
        defer _ = mlx.mlx_array_free(norm);
        const floor = try scalarIn(.float32, 1e-8, s);
        defer _ = mlx.mlx_array_free(floor);
        var denom = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_maximum(&denom, norm, floor, s));
        defer _ = mlx.mlx_array_free(denom);
        const unit = try divA(cond_d, denom, s);
        defer _ = mlx.mlx_array_free(unit);
        const du = try mulA(diff, unit, s);
        defer _ = mlx.mlx_array_free(du);
        var du1 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_sum_axis(&du1, du, 1, true, s));
        defer _ = mlx.mlx_array_free(du1);
        var par0 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_sum_axis(&par0, du1, 2, true, s));
        defer _ = mlx.mlx_array_free(par0);
        const parallel = try mulA(par0, unit, s);
        defer _ = mlx.mlx_array_free(parallel);
        const diff_orth = try subA(diff, parallel, s);
        defer _ = mlx.mlx_array_free(diff_orth);
        if (g.apg >= 1.0) {
            const scaled = try mulScalar(diff_orth, g.cfg - 1.0, s);
            defer _ = mlx.mlx_array_free(scaled);
            break :blk try addA(cond_d, scaled, s);
        }
        // 0 < apg < 1: cfg_diff = apg·diff_orth + (1 − apg)·diff
        const p1 = try mulScalar(diff_orth, g.apg, s);
        defer _ = mlx.mlx_array_free(p1);
        const p0 = try mulScalar(diff, 1.0 - g.apg, s);
        defer _ = mlx.mlx_array_free(p0);
        const blend = try addA(p1, p0, s);
        defer _ = mlx.mlx_array_free(blend);
        const scaled = try mulScalar(blend, g.cfg - 1.0, s);
        defer _ = mlx.mlx_array_free(scaled);
        break :blk try addA(cond_d, scaled, s);
    };
    defer _ = mlx.mlx_array_free(cfg_d);

    // ── back to velocity: cfg_v = (x − cfg_d)/σ in x's dtype
    const num = try subA(x32, cfg_d, s);
    defer _ = mlx.mlx_array_free(num);
    const v32 = try divA(num, sigma, s);
    defer _ = mlx.mlx_array_free(v32);
    return astype(v32, mlx.mlx_array_dtype(x), s);
}

/// One rf_denoiser ping-pong loop. `sigmas` is the host schedule (len
/// steps+1); `noises[k]` is the k-th redraw — drawn only while
/// i < steps−1 && t_next > 0, so callers size it steps−1 for a healthy
/// schedule (the engine passes MLX-random draws, the oracle fixtures).
/// `guidance` non-null swaps the per-step forward for the batched CFG
/// denoiser (cfgVelocity). Returns `steps` latents; caller frees each plus
/// the slice.
pub fn samplePingPong(
    a: std.mem.Allocator,
    dit: *const Dit,
    x0: mlx.mlx_array,
    sigmas: []const f32,
    noises: []const mlx.mlx_array,
    cross: mlx.mlx_array,
    global_cond: mlx.mlx_array,
    guidance: ?Guidance,
    progress: ?sse.Progress,
    s: S,
) ![]mlx.mlx_array {
    const steps = sigmas.len - 1;
    const outs = try a.alloc(mlx.mlx_array, steps);
    errdefer a.free(outs);
    var cur = x0;
    var draw: usize = 0;
    for (0..steps) |i| {
        // A hung-up client latches `cancelled`; stop at the next step
        // boundary (music3 does the same per chunk) instead of burning GPU
        // on a response nobody will read.
        if (progress) |p| if (p.cancelled()) return error.Cancelled;
        const t_curr = sigmas[i];
        const t_next = sigmas[i + 1];
        const cur_dtype = mlx.mlx_array_dtype(cur);

        // t_tensor: f32 [1] — sigmas are f32, so the model always sees f32 t.
        const t_ten = try scalarIn(.float32, t_curr, s);
        defer _ = mlx.mlx_array_free(t_ten);
        const v = if (guidance) |g|
            try cfgVelocity(a, dit, cur, t_ten, cross, global_cond, g, s)
        else
            try dit.forward(a, cur, t_ten, cross, global_cond, null, s);
        defer _ = mlx.mlx_array_free(v);

        // denoised = x − t_curr.astype(x.dtype) * v
        const tc = try scalarIn(cur_dtype, t_curr, s);
        defer _ = mlx.mlx_array_free(tc);
        const tv = try mulA(tc, v, s);
        defer _ = mlx.mlx_array_free(tv);
        const den = try subA(cur, tv, s);
        var den_owned = true;
        defer if (den_owned) {
            _ = mlx.mlx_array_free(den);
        };

        var next: mlx.mlx_array = undefined;
        if (i < steps - 1 and t_next > 0.0) {
            if (draw >= noises.len) return error.NoiseDraw;
            const noise = try astype(noises[draw], cur_dtype, s);
            defer _ = mlx.mlx_array_free(noise);
            draw += 1;
            // (1 − t_next).astype(x.dtype)·denoised + t_next.astype(x.dtype)·noise
            const om = try scalarIn(cur_dtype, 1.0 - @as(f64, t_next), s);
            defer _ = mlx.mlx_array_free(om);
            const t1 = try mulA(om, den, s);
            defer _ = mlx.mlx_array_free(t1);
            const tn = try scalarIn(cur_dtype, t_next, s);
            defer _ = mlx.mlx_array_free(tn);
            const t2 = try mulA(tn, noise, s);
            defer _ = mlx.mlx_array_free(t2);
            next = try addA(t1, t2, s);
        } else {
            next = den;
            den_owned = false;
        }
        evalA(next); // reference mx.eval(x) per step
        outs[i] = next;
        cur = next;
        if (progress) |p| p.emit("sample", @intCast(i + 1), @intCast(steps));
    }
    return outs;
}

// ── Ping-pong sampler oracle ────────────────────────────────────────────────

test "stable_audio3 oracle: ping-pong sampler latents match reference" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = try sa3ModelDir();
    const fix = try fixturesDir();
    const st = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(st);

    // ── schedule: built vs fixture ──
    const built = try buildPingpongSchedule(a, 8, 1.0);
    defer a.free(built);
    const sig_raw = try readRawF32(io, a, fix, "sigmas.f32.raw");
    defer a.free(sig_raw);
    try testing.expectEqual(built.len, sig_raw.len);
    const ssh = [1]c_int{@intCast(built.len)};
    const sig_arr = mlx.mlx_array_new_data(built.ptr, &ssh, 1, .float32);
    defer _ = mlx.mlx_array_free(sig_arr);
    try assertParity(sig_arr, sig_raw, "sigmas", 0.9999, 0.001, st);

    // ── loop driven by the FIXTURE schedule (isolates loop math) ──
    const x0_raw = try readRawF32(io, a, fix, "x0.f32.raw");
    defer a.free(x0_raw);
    const cross_raw = try readRawF32(io, a, fix, "cross_attn.f32.raw");
    defer a.free(cross_raw);
    const glob_raw = try readRawF32(io, a, fix, "global_cond.f32.raw");
    defer a.free(glob_raw);

    var dit = try Dit.load(io, a, dir);
    defer dit.deinit();

    const xsh = [1]c_int{@intCast(x0_raw.len / 256)};
    const x0_32 = mlx.mlx_array_new_data(x0_raw.ptr, &[_]c_int{ 1, 256, xsh[0] }, 3, .float32);
    defer _ = mlx.mlx_array_free(x0_32);
    const x0 = try astype(x0_32, .float16, st);
    defer _ = mlx.mlx_array_free(x0);

    const cross_32 = mlx.mlx_array_new_data(cross_raw.ptr, &[_]c_int{ 1, 257, 768 }, 3, .float32);
    defer _ = mlx.mlx_array_free(cross_32);
    const cross = try astype(cross_32, .float16, st);
    defer _ = mlx.mlx_array_free(cross);

    const glob_32 = mlx.mlx_array_new_data(glob_raw.ptr, &[_]c_int{ 1, 768 }, 2, .float32);
    defer _ = mlx.mlx_array_free(glob_32);
    const glob = try astype(glob_32, .float16, st);
    defer _ = mlx.mlx_array_free(glob);

    // 7 redraws for 8 steps (dump appends one per i < steps-1).
    var noises: [7]mlx.mlx_array = undefined;
    var noise_owned: [7]bool = undefined;
    for (0..7) |k| {
        const buf = try allocPrint_noise(a, k);
        defer a.free(buf);
        const raw = try readRawF32(io, a, fix, buf);
        defer a.free(raw);
        const nsh = [3]c_int{ 1, 256, xsh[0] };
        const n32 = mlx.mlx_array_new_data(raw.ptr, &nsh, 3, .float32);
        const arr = try astype(n32, .float16, st); // step 0 draws f16; steps 1+ cast back up
        _ = mlx.mlx_array_free(n32);
        noises[k] = arr;
        noise_owned[k] = true;
    }
    defer for (noises, 0..) |n, k| {
        if (noise_owned[k]) {
            _ = mlx.mlx_array_free(n);
        }
    };

    const lats = try samplePingPong(a, &dit, x0, sig_raw, noises[0..], cross, glob, null, null, st);
    defer {
        for (lats) |l| _ = mlx.mlx_array_free(l);
        a.free(lats);
    }
    try testing.expectEqual(@as(usize, 8), lats.len);

    for (lats, 1..) |lat, si| {
        const name = try std.fmt.allocPrint(a, "latents_step{d:0>2}.f32.raw", .{si});
        defer a.free(name);
        const ref = try readRawF32(io, a, fix, name);
        defer a.free(ref);
        const label = try std.fmt.allocPrint(a, "step{d:0>2}", .{si});
        defer a.free(label);
        try assertParity(lat, ref, label, 0.999, 0.01, st);
    }
}

test "stable_audio3 oracle: ping-pong + CFG guidance final latent matches reference" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = try sa3ModelDir();
    const fix = try fixturesDir();
    const st = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(st);

    const x0_raw = try readRawF32(io, a, fix, "x0.f32.raw");
    defer a.free(x0_raw);
    const cross_raw = try readRawF32(io, a, fix, "cross_attn.f32.raw");
    defer a.free(cross_raw);
    const neg_raw = try readRawF32(io, a, fix, "cross_attn_neg.f32.raw");
    defer a.free(neg_raw);
    const glob_raw = try readRawF32(io, a, fix, "global_cond.f32.raw");
    defer a.free(glob_raw);
    const sig_raw = try readRawF32(io, a, fix, "sigmas.f32.raw");
    defer a.free(sig_raw);
    const ref_final = try readRawF32(io, a, fix, "latents_cfg_final.f32.raw");
    defer a.free(ref_final);

    var dit = try Dit.load(io, a, dir);
    defer dit.deinit();

    const T: u32 = @intCast(x0_raw.len / 256);
    const x0_32 = mlx.mlx_array_new_data(x0_raw.ptr, &[_]c_int{ 1, 256, @intCast(T) }, 3, .float32);
    defer _ = mlx.mlx_array_free(x0_32);
    const x0 = try astype(x0_32, .float16, st);
    defer _ = mlx.mlx_array_free(x0);
    const cross_32 = mlx.mlx_array_new_data(cross_raw.ptr, &[_]c_int{ 1, 257, 768 }, 3, .float32);
    defer _ = mlx.mlx_array_free(cross_32);
    const cross = try astype(cross_32, .float16, st);
    defer _ = mlx.mlx_array_free(cross);
    const neg_32 = mlx.mlx_array_new_data(neg_raw.ptr, &[_]c_int{ 1, 257, 768 }, 3, .float32);
    defer _ = mlx.mlx_array_free(neg_32);
    const cross_neg = try astype(neg_32, .float16, st);
    defer _ = mlx.mlx_array_free(cross_neg);
    const glob_32 = mlx.mlx_array_new_data(glob_raw.ptr, &[_]c_int{ 1, 768 }, 2, .float32);
    defer _ = mlx.mlx_array_free(glob_32);
    const glob = try astype(glob_32, .float16, st);
    defer _ = mlx.mlx_array_free(glob);

    // 7 redraws for 8 steps — same fixture chain as the plain sampler oracle
    // (guidance changes v, not the schedule or the noise).
    var noises: [7]mlx.mlx_array = undefined;
    var noise_owned: [7]bool = undefined;
    for (0..7) |k| {
        const buf = try allocPrint_noise(a, k);
        defer a.free(buf);
        const raw = try readRawF32(io, a, fix, buf);
        defer a.free(raw);
        const n32 = mlx.mlx_array_new_data(raw.ptr, &[_]c_int{ 1, 256, @intCast(T) }, 3, .float32);
        const arr = try astype(n32, .float16, st);
        _ = mlx.mlx_array_free(n32);
        noises[k] = arr;
        noise_owned[k] = true;
    }
    defer for (noises, 0..) |n, k| {
        if (noise_owned[k]) {
            _ = mlx.mlx_array_free(n);
        }
    };

    const lats = try samplePingPong(a, &dit, x0, sig_raw, noises[0..], cross, glob, .{
        .cfg = 3.0,
        .apg = 1.0,
        .null_cross = cross_neg,
    }, null, st);
    defer {
        for (lats) |l| _ = mlx.mlx_array_free(l);
        a.free(lats);
    }
    try testing.expectEqual(@as(usize, 8), lats.len);
    try assertParity(lats[7], ref_final, "cfg final latent", 0.999, 0.01, st);
}

fn allocPrint_noise(a: std.mem.Allocator, k: usize) ![]u8 {
    return std.fmt.allocPrint(a, "noise_{d:0>2}.f32.raw", .{k});
}

// ── SAME-L decoder (reference = same_l_decoder.py) ──────────────────────────
//
// `same_l_decoder.safetensors` ships dense f32 verbatim (no quant keys).
// Latents f32 [B, 256, T] → patches f32 [B, 512, T*16]. Constants are
// architectural to SAME-L (small variants use SAME-S — its own port).

const SameL = struct {
    const DIM: c_int = 1536;
    const HEADS: c_int = 24;
    const HEAD_DIM: c_int = 64;
    const ROPE_DIMS: c_int = 32;
    const ROPE_BASE: f32 = 10000.0;
    const BLOCKS: usize = 12;
    const FF_INNER: c_int = 4608;
    const SIN_START: usize = 5;
    const OUT_CH: c_int = 512;
    const SUB_CHUNK: c_int = 17; // query positions per latent
    const W: c_int = 51; // 3 * SUB_CHUNK KV positions per window
};

fn concatMany(x: []const mlx.mlx_array, axis: c_int, s: S) !mlx.mlx_array {
    const v = mlx.mlx_vector_array_new();
    defer _ = mlx.mlx_vector_array_free(v);
    for (x) |e| _ = mlx.mlx_vector_array_append_value(v, e);
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_concatenate_axis(&o, v, axis, s));
    return o;
}

/// DyT: gamma * tanh(alpha * x) + beta (alpha [1], gamma/beta [dim], last axis).
fn dyt(a: std.mem.Allocator, w: *const Weights, prefix: []const u8, x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const ak = try std.fmt.allocPrint(a, "{s}.alpha", .{prefix});
    defer a.free(ak);
    const gk = try std.fmt.allocPrint(a, "{s}.gamma", .{prefix});
    defer a.free(gk);
    const bk = try std.fmt.allocPrint(a, "{s}.beta", .{prefix});
    defer a.free(bk);
    const alpha = w.get(ak) orelse return error.MissingWeight;
    const gamma = w.get(gk) orelse return error.MissingWeight;
    const beta = w.get(bk) orelse return error.MissingWeight;
    const ax = try mulA(x, alpha, s);
    defer _ = mlx.mlx_array_free(ax);
    const th = try tanhA(ax, s);
    defer _ = mlx.mlx_array_free(th);
    const gt = try mulA(th, gamma, s);
    defer _ = mlx.mlx_array_free(gt);
    return addA(gt, beta, s);
}

/// Static SWA bias [G, 1, 51]: padded group position g*17+w is valid iff
/// 17 <= pos < T+17 (T = expanded sequence length), else -1e9.
fn swaBoundary(a: std.mem.Allocator, g_count: usize, t_exp: usize, s: S) !mlx.mlx_array {
    _ = s;
    const w: usize = @intCast(SameL.W);
    const buf = try a.alloc(f32, g_count * w);
    defer a.free(buf);
    for (0..g_count) |g| {
        for (0..w) |wi| {
            const pos = g * @as(usize, @intCast(SameL.SUB_CHUNK)) + wi;
            buf[g * w + wi] = if (pos >= @as(usize, @intCast(SameL.SUB_CHUNK)) and pos < t_exp + @as(usize, @intCast(SameL.SUB_CHUNK))) 0.0 else -1e9;
        }
    }
    const sh = [3]c_int{ @intCast(g_count), 1, @intCast(w) };
    return mlx.mlx_array_new_data(buf.ptr, &sh, 3, .float32);
}

/// Sliding windows over the group axis: gp [B, H, G+2, 17, D] (K/V padded by
/// one group each side) → [B, H, G, 51, D] where window g = groups [g, g+3).
fn swaWindows(a: std.mem.Allocator, gp: mlx.mlx_array, g_count: usize, s: S) !mlx.mlx_array {
    const shp = mlx.getShape(gp);
    const b = shp[0];
    const h = shp[1];
    const d = shp[4];
    const pieces = try a.alloc(mlx.mlx_array, g_count);
    defer a.free(pieces);
    for (0..g_count) |gi| {
        const lo = [5]c_int{ 0, 0, @intCast(gi), 0, 0 };
        const hi = [5]c_int{ b, h, @intCast(gi + 3), @intCast(SameL.SUB_CHUNK), d };
        const stp = [5]c_int{ 1, 1, 1, 1, 1 };
        var sl = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(&sl, gp, &lo, 5, &hi, 5, &stp, 5, s));
        defer _ = mlx.mlx_array_free(sl);
        var ct = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_contiguous(&ct, sl, false, s));
        defer _ = mlx.mlx_array_free(ct);
        var r = mlx.mlx_array_new();
        const rsh = [5]c_int{ b, h, 1, SameL.W, d };
        try mlx.check(mlx.mlx_reshape(&r, ct, &rsh, 5, s));
        pieces[gi] = r;
    }
    const out = try concatMany(pieces, 2, s);
    for (pieces) |pi| _ = mlx.mlx_array_free(pi);
    return out;
}

fn headsView(a: std.mem.Allocator, x: mlx.mlx_array, g_count: usize, s: S) !mlx.mlx_array {
    // [B, H, T, D] → [B, H, G, 17, D] → [B*G, H, 17, D]
    _ = a;
    const shp = mlx.getShape(x);
    const b = shp[0];
    const h = shp[1];
    const d = shp[3];
    const gsh = [5]c_int{ b, h, @intCast(g_count), SameL.SUB_CHUNK, d };
    var g4 = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&g4, x, &gsh, 5, s));
    defer _ = mlx.mlx_array_free(g4);
    var tr = mlx.mlx_array_new();
    const axes5 = [5]c_int{ 0, 2, 1, 3, 4 };
    try mlx.check(mlx.mlx_transpose_axes(&tr, g4, &axes5, 5, s));
    defer _ = mlx.mlx_array_free(tr);
    const tsh = [4]c_int{ b * @as(c_int, @intCast(g_count)), h, SameL.SUB_CHUNK, d };
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, tr, &tsh, 4, s));
    return out;
}

/// Windowed [B, H, G, 51, D] → [B*G, H, 51, D] (group axis folds into batch).
fn windowsFlat(x5: mlx.mlx_array, s: S) !mlx.mlx_array {
    var tr = mlx.mlx_array_new();
    const axes5 = [5]c_int{ 0, 2, 1, 3, 4 };
    try mlx.check(mlx.mlx_transpose_axes(&tr, x5, &axes5, 5, s));
    defer _ = mlx.mlx_array_free(tr);
    const shp = mlx.getShape(tr);
    const fsh = [4]c_int{ shp[0] * shp[1], shp[2], shp[3], shp[4] };
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, tr, &fsh, 4, s));
    return out;
}

/// [B*G, H, G-ish] flat diff output → [B, H, T, D] (inverse of headsView).
fn headsRestore(a: std.mem.Allocator, flat: mlx.mlx_array, b: c_int, h: c_int, g_count: usize, s: S) !mlx.mlx_array {
    _ = a;
    const d = mlx.getShape(flat)[3];
    const gsh = [5]c_int{ b, @intCast(g_count), h, SameL.SUB_CHUNK, d };
    var g4 = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&g4, flat, &gsh, 5, s));
    defer _ = mlx.mlx_array_free(g4);
    var tr = mlx.mlx_array_new();
    const axes5 = [5]c_int{ 0, 2, 1, 3, 4 };
    try mlx.check(mlx.mlx_transpose_axes(&tr, g4, &axes5, 5, s));
    defer _ = mlx.mlx_array_free(tr);
    const tsh = [4]c_int{ b, h, @intCast(g_count * @as(usize, @intCast(SameL.SUB_CHUNK))), d };
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, tr, &tsh, 4, s));
    return out;
}

/// Differential self-attention for one decoder block: x [B, T, 1536] → [B, T, 1536].
/// T here is the EXPANDED sequence (T_lat * 17). T <= 17 takes the batched
/// full-attention arm; otherwise SWA with the static 17x51 mask + boundary.
fn decAttn(a: std.mem.Allocator, w: *const Weights, x: mlx.mlx_array, blk: usize, swa_mask: mlx.mlx_array, s: S) !mlx.mlx_array {
    const b = mlx.getShape(x)[0];
    const t = mlx.getShape(x)[1];
    const h = SameL.HEADS;
    const d = SameL.HEAD_DIM;

    const qkp = try std.fmt.allocPrint(a, "blocks.{d}.attn.to_qkv", .{blk});
    defer a.free(qkp);
    const qkv = try lin(w, a, x, qkp, s);
    defer _ = mlx.mlx_array_free(qkv);
    var parts: [5]mlx.mlx_array = undefined;
    try splitEqual(qkv, 5, -1, &parts, s);
    defer {
        for (parts) |p| _ = mlx.mlx_array_free(p);
    }

    var hd: [5]mlx.mlx_array = undefined;
    for (0..5) |i| {
        const rsh = [4]c_int{ b, t, h, d };
        var r = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&r, parts[i], &rsh, 4, s));
        defer _ = mlx.mlx_array_free(r);
        var tr = mlx.mlx_array_new();
        const axes = [4]c_int{ 0, 2, 1, 3 };
        try mlx.check(mlx.mlx_transpose_axes(&tr, r, &axes, 4, s));
        hd[i] = tr;
    }
    defer {
        for (hd) |hh| _ = mlx.mlx_array_free(hh);
    }

    const qnp = try std.fmt.allocPrint(a, "blocks.{d}.attn.q_norm", .{blk});
    defer a.free(qnp);
    const knp = try std.fmt.allocPrint(a, "blocks.{d}.attn.k_norm", .{blk});
    defer a.free(knp);
    var q1 = try dyt(a, w, qnp, hd[0], s);
    defer _ = mlx.mlx_array_free(q1);
    var k1 = try dyt(a, w, knp, hd[1], s);
    defer _ = mlx.mlx_array_free(k1);
    var q2 = try dyt(a, w, qnp, hd[3], s);
    defer _ = mlx.mlx_array_free(q2);
    var k2 = try dyt(a, w, knp, hd[4], s);
    defer _ = mlx.mlx_array_free(k2);

    inline for (.{ &q1, &k1, &q2, &k2 }) |pt| {
        var rp = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_fast_rope(&rp, pt.*, SameL.ROPE_DIMS, false, mlx.mlx_optional_float.some(SameL.ROPE_BASE), 1.0, 0, .{ .ctx = null }, s));
        _ = mlx.mlx_array_free(pt.*);
        pt.* = rp;
    }

    var diff: mlx.mlx_array = undefined;
    if (t <= SameL.SUB_CHUNK) {
        // batched arm: Q = [q1, q2] heads-concat, K = [k1, k2], V = [v, v]
        const Q = try concatMany(&.{ q1, q2 }, 1, s);
        defer _ = mlx.mlx_array_free(Q);
        const K = try concatMany(&.{ k1, k2 }, 1, s);
        defer _ = mlx.mlx_array_free(K);
        const V = try concatMany(&.{ hd[2], hd[2] }, 1, s);
        defer _ = mlx.mlx_array_free(V);
        const out = try sdpa(Q, K, V, 0.125, null, s);
        defer _ = mlx.mlx_array_free(out);
        var o: [2]mlx.mlx_array = undefined;
        try splitEqual(out, 2, 1, &o, s);
        defer {
            for (o) |oo| _ = mlx.mlx_array_free(oo);
        }
        diff = try subA(o[0], o[1], s);
    } else {
        const g_count: usize = @intCast(@divExact(t, SameL.SUB_CHUNK));
        // pad K/V by one SUB_CHUNK on both sides
        const psh = [4]c_int{ b, h, SameL.SUB_CHUNK, d };
        var z = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_zeros(&z, &psh, 4, .float32, s));
        defer _ = mlx.mlx_array_free(z);
        const k1p = try concatMany(&.{ z, k1, z }, 2, s);
        defer _ = mlx.mlx_array_free(k1p);
        const k2p = try concatMany(&.{ z, k2, z }, 2, s);
        defer _ = mlx.mlx_array_free(k2p);
        const vp = try concatMany(&.{ z, hd[2], z }, 2, s);
        defer _ = mlx.mlx_array_free(vp);

        // group views [B, H, G+2, 17, D]
        const gsh = [5]c_int{ b, h, @intCast(g_count + 2), SameL.SUB_CHUNK, d };
        var k1g = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&k1g, k1p, &gsh, 5, s));
        defer _ = mlx.mlx_array_free(k1g);
        var k2g = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&k2g, k2p, &gsh, 5, s));
        defer _ = mlx.mlx_array_free(k2g);
        var vg = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&vg, vp, &gsh, 5, s));
        defer _ = mlx.mlx_array_free(vg);

        const k1w = try swaWindows(a, k1g, g_count, s);
        defer _ = mlx.mlx_array_free(k1w);
        const k2w = try swaWindows(a, k2g, g_count, s);
        defer _ = mlx.mlx_array_free(k2w);
        const vw = try swaWindows(a, vg, g_count, s);
        defer _ = mlx.mlx_array_free(vw);

        // boundary bias → [B*G, 1, 17, 51]
        const bd = try swaBoundary(a, g_count, @intCast(t), s);
        defer _ = mlx.mlx_array_free(bd);
        const comb = try addA(swa_mask, bd, s); // [G,17,51] (mask [17,51] + [G,1,51])
        defer _ = mlx.mlx_array_free(comb);
        var comb1 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_expand_dims(&comb1, comb, 0, s));
        defer _ = mlx.mlx_array_free(comb1);
        const bsh = [4]c_int{ b, @intCast(g_count), SameL.SUB_CHUNK, SameL.W };
        var bcb = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_broadcast_to(&bcb, comb1, &bsh, 4, s));
        defer _ = mlx.mlx_array_free(bcb);
        const fsh = [4]c_int{ b * @as(c_int, @intCast(g_count)), 1, SameL.SUB_CHUNK, SameL.W };
        var msk = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&msk, bcb, &fsh, 4, s));
        defer _ = mlx.mlx_array_free(msk);

        // flat heads
        const q1f = try headsView(a, q1, g_count, s);
        defer _ = mlx.mlx_array_free(q1f);
        const q2f = try headsView(a, q2, g_count, s);
        defer _ = mlx.mlx_array_free(q2f);
        const k1f = try windowsFlat(k1w, s);
        defer _ = mlx.mlx_array_free(k1f);
        const k2f = try windowsFlat(k2w, s);
        defer _ = mlx.mlx_array_free(k2f);
        const vf = try windowsFlat(vw, s);
        defer _ = mlx.mlx_array_free(vf);

        const Q = try concatMany(&.{ q1f, q2f }, 1, s);
        defer _ = mlx.mlx_array_free(Q);
        const K = try concatMany(&.{ k1f, k2f }, 1, s);
        defer _ = mlx.mlx_array_free(K);
        const V = try concatMany(&.{ vf, vf }, 1, s);
        defer _ = mlx.mlx_array_free(V);

        const out = try sdpa(Q, K, V, 0.125, msk, s);
        defer _ = mlx.mlx_array_free(out);
        var o: [2]mlx.mlx_array = undefined;
        try splitEqual(out, 2, 1, &o, s);
        defer {
            for (o) |oo| _ = mlx.mlx_array_free(oo);
        }
        const dflat = try subA(o[0], o[1], s);
        defer _ = mlx.mlx_array_free(dflat);
        diff = try headsRestore(a, dflat, b, h, g_count, s);
    }

    var merged = mlx.mlx_array_new();
    const maxes = [4]c_int{ 0, 2, 1, 3 };
    try mlx.check(mlx.mlx_transpose_axes(&merged, diff, &maxes, 4, s));
    defer _ = mlx.mlx_array_free(merged);
    _ = mlx.mlx_array_free(diff);
    const msh = [3]c_int{ b, t, SameL.DIM };
    var flat = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&flat, merged, &msh, 3, s));
    defer _ = mlx.mlx_array_free(flat);

    const op = try std.fmt.allocPrint(a, "blocks.{d}.attn.to_out", .{blk});
    defer a.free(op);
    return lin(w, a, flat, op, s);
}

/// Feed-forward: glu_proj → (value ⊗ gate), sin(·π) gate for blocks >= 5 else SiLU.
fn decFF(a: std.mem.Allocator, w: *const Weights, x: mlx.mlx_array, blk: usize, s: S) !mlx.mlx_array {
    const gp = try std.fmt.allocPrint(a, "blocks.{d}.ff.glu_proj", .{blk});
    defer a.free(gp);
    const g = try lin(w, a, x, gp, s);
    defer _ = mlx.mlx_array_free(g);
    var halves: [2]mlx.mlx_array = undefined;
    try splitEqual(g, 2, -1, &halves, s);
    defer {
        for (halves) |hh| _ = mlx.mlx_array_free(hh);
    }
    var act: mlx.mlx_array = undefined;
    if (blk >= SameL.SIN_START) {
        const ang = try mulScalar(halves[1], @as(f32, std.math.pi), s);
        defer _ = mlx.mlx_array_free(ang);
        var sn = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_sin(&sn, ang, s));
        defer _ = mlx.mlx_array_free(sn);
        act = try mulA(halves[0], sn, s);
    } else {
        const gl = try silu(halves[1], s);
        defer _ = mlx.mlx_array_free(gl);
        act = try mulA(halves[0], gl, s);
    }
    defer _ = mlx.mlx_array_free(act);
    const dp = try std.fmt.allocPrint(a, "blocks.{d}.ff.proj_out", .{blk});
    defer a.free(dp);
    return lin(w, a, act, dp, s);
}

/// `rearrange("b (c h) l -> b c (l h)", h=256)` — patches [B, 512, L] → [B, 2, L*256].
pub fn patchedDecode(a: std.mem.Allocator, patches: mlx.mlx_array, s: S) !mlx.mlx_array {
    _ = a;
    const shp = mlx.getShape(patches);
    const b = shp[0];
    const l = shp[2];
    const rsh = [4]c_int{ b, 2, 256, l };
    var r = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&r, patches, &rsh, 4, s));
    defer _ = mlx.mlx_array_free(r);
    var tr = mlx.mlx_array_new();
    const axes = [4]c_int{ 0, 1, 3, 2 };
    try mlx.check(mlx.mlx_transpose_axes(&tr, r, &axes, 4, s));
    defer _ = mlx.mlx_array_free(tr);
    const osh = [3]c_int{ b, 2, l * 256 };
    var out = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_reshape(&out, tr, &osh, 3, s));
    return out;
}

pub const SameLDecoder = struct {
    w: Weights,
    swa_mask: mlx.mlx_array, // [17, 51] f32, 0 / -1e9

    pub fn load(io: std.Io, a: std.mem.Allocator, model_dir: []const u8) !SameLDecoder {
        var w = try loadFileWeights(a, model_dir, "same_l_decoder.safetensors");
        errdefer w.deinit();
        // static mask: valid iff kv >= q and kv <= q + 34 (2*BLOCK_SIZE)
        const w_n: usize = @intCast(SameL.W);
        const q_n: usize = @intCast(SameL.SUB_CHUNK);
        const buf = try a.alloc(f32, q_n * w_n);
        defer a.free(buf);
        for (0..q_n) |q| {
            for (0..w_n) |kv| {
                buf[q * w_n + kv] = if (kv >= q and kv <= q + 2 * q_n) 0.0 else -1e9;
            }
        }
        const msh = [2]c_int{ SameL.SUB_CHUNK, SameL.W };
        const m = mlx.mlx_array_new_data(buf.ptr, &msh, 2, .float32);
        _ = io;
        return .{ .w = w, .swa_mask = m };
    }

    pub fn deinit(self: *SameLDecoder) void {
        _ = mlx.mlx_array_free(self.swa_mask);
        self.w.deinit();
    }

    /// One un-chunked forward: latents [B, 256, T] → patches [B, 512, T*16].
    pub fn decode(self: *SameLDecoder, alloc: std.mem.Allocator, latents: mlx.mlx_array, s: S) !mlx.mlx_array {
        const w = &self.w;
        const b = mlx.getShape(latents)[0];
        const t_lat = mlx.getShape(latents)[2];

        const rs = w.get("running_std") orelse return error.MissingWeight;
        const xs = try mulA(latents, rs, s);
        defer _ = mlx.mlx_array_free(xs);
        const xt = try transposeA(xs, &[_]c_int{ 0, 2, 1 }, s);
        defer _ = mlx.mlx_array_free(xt);
        const xp = try lin(w, alloc, xt, "project_in", s);
        defer _ = mlx.mlx_array_free(xp);

        // latent slot + 16 broadcast learnable tokens → [B, T, 17, E]
        var xe = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_expand_dims(&xe, xp, 2, s));
        defer _ = mlx.mlx_array_free(xe);
        const nt = w.get("new_tokens") orelse return error.MissingWeight;
        var nt0 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_expand_dims(&nt0, nt, 0, s));
        defer _ = mlx.mlx_array_free(nt0);
        const nsh = [4]c_int{ b, t_lat, 16, SameL.DIM };
        var ntb = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_broadcast_to(&ntb, nt0, &nsh, 4, s));
        defer _ = mlx.mlx_array_free(ntb);
        const cat = try concatMany(&.{ xe, ntb }, 2, s);
        defer _ = mlx.mlx_array_free(cat);
        const esh = [3]c_int{ b, t_lat * SameL.SUB_CHUNK, SameL.DIM };
        var cur = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&cur, cat, &esh, 3, s));

        // 12 residual blocks (explicit frees: cur is reassigned each step)
        for (0..SameL.BLOCKS) |i| {
            const pnp = try std.fmt.allocPrint(alloc, "blocks.{d}.pre_norm", .{i});
            defer alloc.free(pnp);
            const hn = try dyt(alloc, w, pnp, cur, s);
            defer _ = mlx.mlx_array_free(hn);
            const att = try decAttn(alloc, w, hn, i, self.swa_mask, s);
            defer _ = mlx.mlx_array_free(att);
            const after_att = try addA(cur, att, s);
            _ = mlx.mlx_array_free(cur);
            cur = after_att;

            const fnp = try std.fmt.allocPrint(alloc, "blocks.{d}.ff_norm", .{i});
            defer alloc.free(fnp);
            const fnorm = try dyt(alloc, w, fnp, cur, s);
            defer _ = mlx.mlx_array_free(fnorm);
            const ff = try decFF(alloc, w, fnorm, i, s);
            defer _ = mlx.mlx_array_free(ff);
            const after_ff = try addA(cur, ff, s);
            _ = mlx.mlx_array_free(cur);
            cur = after_ff;
        }

        // drop index 0 of each 17-group → [B, T, 16, E] → [B, T*16, E]
        const gsh = [4]c_int{ b, t_lat, SameL.SUB_CHUNK, SameL.DIM };
        var r4 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&r4, cur, &gsh, 4, s));
        _ = mlx.mlx_array_free(cur);
        defer _ = mlx.mlx_array_free(r4);
        const lo = [4]c_int{ 0, 0, 1, 0 };
        const hi = [4]c_int{ b, t_lat, SameL.SUB_CHUNK, SameL.DIM };
        const stp = [4]c_int{ 1, 1, 1, 1 };
        var sl = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_slice(&sl, r4, &lo, 4, &hi, 4, &stp, 4, s));
        defer _ = mlx.mlx_array_free(sl);
        const fsh = [3]c_int{ b, t_lat * (SameL.SUB_CHUNK - 1), SameL.DIM };
        var flat = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&flat, sl, &fsh, 3, s));
        defer _ = mlx.mlx_array_free(flat);

        // mapping: Conv1d k=1 stored [512, 1536, 1] → Linear [512, 1536]
        const mw = w.get("mapping.weight") orelse return error.MissingWeight;
        const mwrsh = [2]c_int{ SameL.OUT_CH, SameL.DIM };
        var mwr = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_reshape(&mwr, mw, &mwrsh, 2, s));
        defer _ = mlx.mlx_array_free(mwr);
        const mwt = try transposeA(mwr, &[_]c_int{ 1, 0 }, s);
        defer _ = mlx.mlx_array_free(mwt);
        var mo = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_matmul(&mo, flat, mwt, s));
        defer _ = mlx.mlx_array_free(mo);
        const mb = w.get("mapping.bias") orelse return error.MissingWeight;
        const mob = try addA(mo, mb, s);
        defer _ = mlx.mlx_array_free(mob);
        return transposeA(mob, &[_]c_int{ 0, 2, 1 }, s);
    }

    /// Reference `decode_chunked`: windows of `chunk + 2*ovl` latents with
    /// `ovl` valid on each side (edges take only their in-bounds valid part),
    /// results concatenated on the patch axis. `T <= kernel` → one direct call.
    pub fn decodeChunked(self: *SameLDecoder, alloc: std.mem.Allocator, latents: mlx.mlx_array, chunk: usize, ovl: usize, s: S) !mlx.mlx_array {
        const t: usize = @intCast(mlx.getShape(latents)[2]);
        const kernel = chunk + 2 * ovl;
        if (t <= kernel) return self.decode(alloc, latents, s);

        // count pieces: first + interiors + possible last
        var i = chunk + ovl;
        var interiors: usize = 0;
        while (i + chunk + ovl <= t) : (i += chunk) {
            interiors += 1;
        }
        const remaining = t - i;
        const count = 1 + interiors + (if (remaining > 0) @as(usize, 1) else 0);
        const pieces = try alloc.alloc(mlx.mlx_array, count);
        defer alloc.free(pieces);
        var n: usize = 0;

        // first window: [0, kernel) → first (chunk+ovl)*16 patches
        {
            const kw = try sliceAxis2(latents, 0, @intCast(kernel), s);
            defer _ = mlx.mlx_array_free(kw);
            const o = try self.decode(alloc, kw, s);
            defer _ = mlx.mlx_array_free(o);
            pieces[n] = try sliceAxis2(o, 0, @intCast((chunk + ovl) * 16), s);
            n += 1;
        }
        // interiors: valid [ovl, chunk+ovl)*16 per window
        i = chunk + ovl;
        while (i + chunk + ovl <= t) : (i += chunk) {
            const kw = try sliceAxis2(latents, @intCast(i - ovl), @intCast(i + chunk + ovl), s);
            defer _ = mlx.mlx_array_free(kw);
            const o = try self.decode(alloc, kw, s);
            defer _ = mlx.mlx_array_free(o);
            pieces[n] = try sliceAxis2(o, @intCast(ovl * 16), @intCast((ovl + chunk) * 16), s);
            n += 1;
        }
        // last: [t-kernel, t) → last remaining*16 patches
        if (remaining > 0) {
            const kw = try sliceAxis2(latents, @intCast(t - kernel), @intCast(t), s);
            defer _ = mlx.mlx_array_free(kw);
            const o = try self.decode(alloc, kw, s);
            defer _ = mlx.mlx_array_free(o);
            const start: c_int = @intCast((kernel - remaining) * 16);
            const end: c_int = @intCast(kernel * 16);
            pieces[n] = try sliceAxis2(o, start, end, s);
            n += 1;
        }
        const out = try concatMany(pieces[0..n], 2, s);
        for (pieces[0..n]) |pi| _ = mlx.mlx_array_free(pi);
        return out;
    }
};

// ── SAME-L decoder oracle ───────────────────────────────────────────────────

test "stable_audio3 oracle: SAME-L decoder arms match reference" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = try sa3ModelDir();
    const fix = try fixturesDir();
    const st = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(st);

    const lat_raw = try readRawF32(io, a, fix, "latents_step08.f32.raw");
    defer a.free(lat_raw);
    const lat_32 = mlx.mlx_array_new_data(lat_raw.ptr, &[_]c_int{ 1, 256, 162 }, 3, .float32);
    defer _ = mlx.mlx_array_free(lat_32);

    var dec = try SameLDecoder.load(io, a, dir);
    defer dec.deinit();

    // ── production arm: chunked(128, 8) at T_lat=162 ──
    const patches = try dec.decodeChunked(a, lat_32, 128, 8, st);
    defer _ = mlx.mlx_array_free(patches);
    const ref_p = try readRawF32(io, a, fix, "patches_T162.f32.raw");
    defer a.free(ref_p);
    try assertParity(patches, ref_p, "patches T162", 0.999, 0.01, st);

    // ── patched_decode → audio, trimmed to 15 s ──
    const audio_full = try patchedDecode(a, patches, st);
    defer _ = mlx.mlx_array_free(audio_full);
    const lo = [_]c_int{ 0, 0, 0 };
    const hi = [_]c_int{ 1, 2, 661500 };
    const stp = [_]c_int{ 1, 1, 1 };
    var audio = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&audio, audio_full, &lo, 3, &hi, 3, &stp, 3, st));
    defer _ = mlx.mlx_array_free(audio);
    const ref_a = try readRawF32(io, a, fix, "audio_T162.f32.raw");
    defer a.free(ref_a);
    try assertParity(audio, ref_a, "audio T162", 0.999, 0.01, st);

    // ── short arms: direct even T=8, chunked(2, 2) odd T=7 ──
    const l8 = try sliceAxis2(lat_32, 0, 8, st);
    defer _ = mlx.mlx_array_free(l8);
    const p8 = try dec.decode(a, l8, st);
    defer _ = mlx.mlx_array_free(p8);
    const ref8 = try readRawF32(io, a, fix, "patches_T8_direct.f32.raw");
    defer a.free(ref8);
    try assertParity(p8, ref8, "patches T8", 0.999, 0.01, st);

    const l7 = try sliceAxis2(lat_32, 0, 7, st);
    defer _ = mlx.mlx_array_free(l7);
    const p7 = try dec.decodeChunked(a, l7, 2, 2, st);
    defer _ = mlx.mlx_array_free(p7);
    const ref7 = try readRawF32(io, a, fix, "patches_T7_chunk2.f32.raw");
    defer a.free(ref7);
    try assertParity(p7, ref7, "patches T7", 0.999, 0.01, st);
}

/// `x[..., lo:hi]` on axis 2 (latent/time); input `[1, C, T]`.
fn sliceAxis2(x: mlx.mlx_array, lo: c_int, hi: c_int, s: S) !mlx.mlx_array {
    const sh = mlx.getShape(x);
    const l = [_]c_int{ 0, 0, lo };
    const h = [_]c_int{ sh[0], sh[1], hi };
    const stp = [_]c_int{ 1, 1, 1 };
    var o = mlx.mlx_array_new();
    try mlx.check(mlx.mlx_slice(&o, x, &l, 3, &h, 3, &stp, 3, s));
    return o;
}

// ── engine: pack → WAV (reference = sa3_mlx.py generate) ────────────────────

/// API-facing generation bounds — the medium pack's `seconds_min` /
/// `seconds_max` (generate() clamps independently as a backstop) and the
/// reference's ping-pong step default (sa3_mlx `--steps`, min 1).
pub const MIN_DURATION_S: u32 = 1;
pub const MAX_DURATION_S: u32 = 384;
pub const DEFAULT_STEPS: u32 = 8;
pub const MAX_STEPS: u32 = 100;

pub const GenerateRequest = struct {
    prompt: []const u8,
    seconds: f32 = 30.0,
    steps: u32 = 8,
    seed: u64 = 42,
    /// CFG scale (sa3_mlx `--cfg`). 1.0 = off, the reference default: one
    /// forward per step. Any other finite value runs the batched cond+uncond
    /// pass (~2x per step) and blends in denoised space.
    cfg_scale: f32 = 1.0,
    /// CFG unconditional branch (sa3_mlx `--negative-prompt`). null (or "")
    /// → the upstream default zeros_like(cross). Only read when
    /// cfg_scale != 1.0 (no uncond branch exists otherwise).
    negative_prompt: ?[]const u8 = null,
    /// APG scale (sa3_mlx `--apg`) in [0,1] — only matters when
    /// cfg_scale != 1.0. 1.0 = full projection (reference default),
    /// 0.0 = vanilla CFG, in between blends.
    apg: f32 = 1.0,
};

pub const Generated = struct {
    /// Planar [ch0 | ch1] f32 in [-1, 1], trimmed to round(seconds * 44100).
    samples: []f32,
    channels: u16 = 2,
    sample_rate: u32 = 44100,
};

pub const Engine = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    s: S,
    tok: tok_mod.Tokenizer,
    t5: T5Gemma,
    dit: Dit,
    dec: SameLDecoder,

    pub fn load(io: std.Io, a: std.mem.Allocator, model_dir: []const u8) !*Engine {
        const self = try a.create(Engine);
        errdefer a.destroy(self);
        self.allocator = a;
        self.io = io;
        self.s = mlx.mlx_default_gpu_stream_new();
        errdefer _ = mlx.mlx_stream_free(self.s);
        self.tok = try tok_mod.loadTokenizer(io, a, model_dir);
        errdefer self.tok.deinit();
        self.t5 = try T5Gemma.load(io, a, model_dir);
        errdefer self.t5.deinit();
        self.dit = try Dit.load(io, a, model_dir);
        errdefer self.dit.deinit();
        self.dec = try SameLDecoder.load(io, a, model_dir);
        errdefer self.dec.deinit();
        log.info("[sa3] engine ready (t5 {d} + dit {d} + decoder {d} tensors)\n", .{
            self.t5.w.count(),
            self.dit.w.count(),
            self.dec.w.count(),
        });
        return self;
    }

    pub fn deinit(self: *Engine) void {
        self.dec.deinit();
        self.dit.deinit();
        self.t5.deinit();
        self.tok.deinit();
        _ = mlx.mlx_stream_free(self.s);
        self.allocator.destroy(self);
    }

    /// Full pipeline: tokenize → T5Gemma → condition → seeded noise →
    /// ping-pong sampler → SAME-L decode → trim. Noise comes from the same
    /// MLX RNG as the reference (`key(seed)`, then `split` per redraw), so a
    /// prompt/seconds/steps/seed tuple reproduces the reference latent.
    pub fn generate(self: *Engine, allocator: std.mem.Allocator, req: GenerateRequest, progress: ?sse.Progress) !Generated {
        const seconds = std.math.clamp(@as(f64, req.seconds), 1.0, 384.0);
        const steps: usize = if (req.steps == 0) 8 else @intCast(req.steps);
        // guidance: NaN/inf would silently engage the uncond branch (NaN != 1.0)
        // and emit garbage audio — fail loud instead. apg clamps like seconds:
        // the reference formula already treats <0 as 0 and >1 as 1.
        if (!std.math.isFinite(req.cfg_scale)) return error.InvalidCfgScale;
        const apg = std.math.clamp(req.apg, 0.0, 1.0);

        // tokenize; the reference truncates at prompt_max_len (256)
        const ids_full = try self.tok.encode(allocator, req.prompt);
        defer allocator.free(ids_full);
        const n_ids = @min(ids_full.len, 256);
        const ids = try allocator.alloc(i32, n_ids);
        defer allocator.free(ids);
        for (ids_full[0..n_ids], ids) |u, *i| i.* = @intCast(u);
        var mask: [256]i32 = @splat(0);
        for (0..n_ids) |i| mask[i] = 1;

        if (progress) |p| p.emit("condition", 0, 3);
        const hidden = try self.t5.encode(allocator, ids, mask[0..], self.t5.s);
        defer _ = mlx.mlx_array_free(hidden);
        const h16 = try astype(hidden, .float16, self.s);
        defer _ = mlx.mlx_array_free(h16);
        const cond = try conditionPrompt(&self.dit.w, h16, mask[0..], @floatCast(seconds), self.s);
        defer cond.deinit();

        // T_lat = ceil(seconds * 44100 / 4096) — decoder-independent
        const t_lat_raw: usize = @intFromFloat(@ceil(seconds * 44100.0 / 4096.0));
        const t_lat: usize = @max(1, t_lat_raw);

        // initial noise: mx.random.normal((1,256,T_lat), f16, key(seed))
        if (progress) |p| p.emit("sample", 0, @intCast(steps));
        var k0 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_random_key(&k0, req.seed));
        defer _ = mlx.mlx_array_free(k0);
        const nsh = [3]c_int{ 1, 256, @intCast(t_lat) };
        var x0 = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_random_normal(&x0, &nsh, 3, .float16, 0.0, 1.0, k0, self.s));
        defer _ = mlx.mlx_array_free(x0);

        const sigmas = try buildPingpongSchedule(allocator, steps, 1.0);
        defer allocator.free(sigmas);

        // redraws: (key, sub) = split(key); draw sub at every step with
        // t_next > 0 before the last — dtype follows x (f16 at step 0, f32
        // after the first update widens it).
        var n_draw: usize = 0;
        for (0..steps) |i| {
            if (i < steps - 1 and sigmas[i + 1] > 0.0) n_draw += 1;
        }
        const noises = try allocator.alloc(mlx.mlx_array, n_draw);
        defer {
            for (noises) |nz| _ = mlx.mlx_array_free(nz);
            allocator.free(noises);
        }
        // the reference seeds the redraw chain with seed + 1 (sa3_mlx:
        // `sample_flow_pingpong(..., seed=args.seed + 1)`) while x0 above
        // uses `seed` directly.
        // ks ownership rides key_owned: the first split frees it, or the
        // end-of-loop free does when no redraw happened.
        var ks = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_random_key(&ks, req.seed + 1));
        var key_cur = ks;
        var key_owned = true;
        var di: usize = 0;
        for (0..steps) |i| {
            if (!(i < steps - 1 and sigmas[i + 1] > 0.0)) continue;
            var kn = mlx.mlx_array_new();
            var sub = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_random_split(&kn, &sub, key_cur, self.s));
            if (key_owned) _ = mlx.mlx_array_free(key_cur);
            key_cur = kn;
            key_owned = true;
            const dt: mlx.mlx_dtype = if (i == 0) .float16 else .float32;
            var nz = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_random_normal(&nz, &nsh, 3, dt, 0.0, 1.0, sub, self.s));
            _ = mlx.mlx_array_free(sub);
            noises[di] = nz;
            di += 1;
        }
        if (key_owned) _ = mlx.mlx_array_free(key_cur);

        // CFG uncond branch — built only when cfg != 1.0 (reference rule;
        // no uncond pass exists otherwise). The negative prompt conditions
        // exactly like the positive one (T5 → padding → same seconds token);
        // absent/empty → zeros_like(cross), sa3_mlx.py's upstream default.
        var guidance: ?Guidance = null;
        var null_cross: ?mlx.mlx_array = null;
        defer if (null_cross) |nc| {
            _ = mlx.mlx_array_free(nc);
        };
        if (req.cfg_scale != 1.0) {
            const nc: mlx.mlx_array = if (req.negative_prompt) |np| blk: {
                if (np.len == 0) break :blk try zerosLikeCross(cond.cross, self.s);
                const neg_ids_full = try self.tok.encode(allocator, np);
                defer allocator.free(neg_ids_full);
                const neg_n = @min(neg_ids_full.len, 256);
                const neg_ids = try allocator.alloc(i32, neg_n);
                defer allocator.free(neg_ids);
                for (neg_ids_full[0..neg_n], neg_ids) |u, *i| i.* = @intCast(u);
                var nmask: [256]i32 = @splat(0);
                for (0..neg_n) |i| nmask[i] = 1;
                const nh = try self.t5.encode(allocator, neg_ids, nmask[0..], self.t5.s);
                defer _ = mlx.mlx_array_free(nh);
                const nh16 = try astype(nh, .float16, self.s);
                defer _ = mlx.mlx_array_free(nh16);
                const ncond = try conditionPrompt(&self.dit.w, nh16, nmask[0..], @floatCast(seconds), self.s);
                // seconds token is identical to the positive branch's;
                // cond.global_cond stays the one the sampler uses.
                _ = mlx.mlx_array_free(ncond.global_cond);
                break :blk ncond.cross;
            } else try zerosLikeCross(cond.cross, self.s);
            null_cross = nc;
            guidance = .{ .cfg = req.cfg_scale, .apg = apg, .null_cross = nc };
        }

        const lats = try samplePingPong(allocator, &self.dit, x0, sigmas, noises, cond.cross, cond.global_cond, guidance, progress, self.s);
        defer {
            for (lats) |l| _ = mlx.mlx_array_free(l);
            allocator.free(lats);
        }
        if (progress) |p| p.emit("sample", @intCast(steps), @intCast(steps));

        // decode dispatch (sa3_mlx): >144 chunked(128,8); even direct;
        // odd >6 chunked(2,2); tiny odd reflect-pad one latent then trim.
        if (progress) |p| p.emit("decode", 0, 1);
        const lat = lats[steps - 1];
        var patches: mlx.mlx_array = undefined;
        if (t_lat > 144) {
            patches = try self.dec.decodeChunked(allocator, lat, 128, 8, self.s);
        } else if (@mod(t_lat, 2) == 0) {
            patches = try self.dec.decode(allocator, lat, self.s);
        } else if (t_lat > 6) {
            patches = try self.dec.decodeChunked(allocator, lat, 2, 2, self.s);
        } else {
            const llo = [_]c_int{ 0, 0, @intCast(t_lat - 1) };
            const lhi = [_]c_int{ 1, 256, @intCast(t_lat) };
            const lst = [_]c_int{ 1, 1, 1 };
            var last = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&last, lat, &llo, 3, &lhi, 3, &lst, 3, self.s));
            defer _ = mlx.mlx_array_free(last);
            const pe = try concatA(lat, last, 2, self.s);
            defer _ = mlx.mlx_array_free(pe);
            const pd = try self.dec.decode(allocator, pe, self.s);
            defer _ = mlx.mlx_array_free(pd);
            const thi = [_]c_int{ 1, 512, @intCast(t_lat * 16) };
            const tlo = [_]c_int{ 0, 0, 0 };
            const tst = [_]c_int{ 1, 1, 1 };
            patches = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&patches, pd, &tlo, 3, &thi, 3, &tst, 3, self.s));
        }
        defer _ = mlx.mlx_array_free(patches);

        const audio_full = try patchedDecode(allocator, patches, self.s);
        defer _ = mlx.mlx_array_free(audio_full);
        const want: usize = @intFromFloat(@round(seconds * 44100.0));
        const full: usize = t_lat * 4096;
        var trimmed = audio_full;
        if (want < full) {
            const alo = [_]c_int{ 0, 0, 0 };
            const ahi = [_]c_int{ 1, 2, @intCast(want) };
            const ast = [_]c_int{ 1, 1, 1 };
            trimmed = mlx.mlx_array_new();
            try mlx.check(mlx.mlx_slice(&trimmed, audio_full, &alo, 3, &ahi, 3, &ast, 3, self.s));
        }
        defer if (trimmed.ctx != audio_full.ctx) {
            _ = mlx.mlx_array_free(trimmed);
        };

        var ct = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_contiguous(&ct, trimmed, false, self.s));
        defer _ = mlx.mlx_array_free(ct);
        evalA(ct);
        const data = mlx.mlx_array_data_float32(ct) orelse return error.NoAudioData;
        const out = try allocator.alloc(f32, want * 2);
        @memcpy(out, data[0 .. want * 2]);
        if (progress) |p| p.emit("decode", 1, 1);
        return .{ .samples = out };
    }

    /// `generate` + WAV bytes (PCM16, stereo interleaved) for the HTTP route.
    pub fn generateWav(self: *Engine, allocator: std.mem.Allocator, req: GenerateRequest, progress: ?sse.Progress) ![]u8 {
        const g = try self.generate(allocator, req, progress);
        defer allocator.free(g.samples);
        const n = g.samples.len / 2;
        const inter = try allocator.alloc(f32, g.samples.len);
        defer allocator.free(inter);
        for (0..n) |i| {
            inter[2 * i] = g.samples[i];
            inter[2 * i + 1] = g.samples[n + i];
        }
        return wav_mod.encodePcm16(allocator, inter, g.sample_rate, g.channels);
    }
};

// ── end-to-end oracle ────────────────────────────────────────────────────────

/// Test-only: dequantize every affine-quantized tensor of `w` in place, so an
/// oracle compares against the dequantized-fixture reference WITHOUT the
/// quantizer in the loop (the dump's own rule: fixtures measure the port).
/// Weights are cast to f16 exactly like the fixture npz; `.scales`/`.biases`
/// are dropped so `lin()` takes its dense branch. Production stays quantized.
fn dequantizeForTest(w: *Weights, s: S) !void {
    const a = w.allocator;
    var deq_count: u32 = 0;
    var prefixes: std.ArrayListUnmanaged([]const u8) = .empty;
    defer {
        for (prefixes.items) |p| a.free(p);
        prefixes.deinit(a);
    }
    {
        var it = w.map.keyIterator();
        while (it.next()) |k| {
            const key = k.*;
            const suffix = ".scales";
            if (std.mem.endsWith(u8, key, suffix)) {
                try prefixes.append(a, try a.dupe(u8, key[0 .. key.len - suffix.len]));
            }
        }
    }
    for (prefixes.items) |prefix| {
        deq_count += 1;
        const wk = try std.fmt.allocPrint(a, "{s}.weight", .{prefix});
        defer a.free(wk);
        const sk = try std.fmt.allocPrint(a, "{s}.scales", .{prefix});
        defer a.free(sk);
        const bk = try std.fmt.allocPrint(a, "{s}.biases", .{prefix});
        defer a.free(bk);
        const wq = try getW(w, wk);
        const sc = try getW(w, sk);
        const bi = try getW(w, bk);
        // group-64 8-bit affine (converter GROUP_SIZE, config bits) — solved
        // from geometry like lin(), but asserted: a different pack contract
        // must fail loud, not dequantize silently wrong.
        const s_cols: u32 = @intCast(mlx.getShape(sc)[1]);
        const in_features: u32 = 64 * s_cols;
        const w_cols: u32 = @intCast(mlx.getShape(wq)[1]);
        const bits: u32 = @divExact(32 * w_cols, in_features);
        if (bits != 8) return error.UnexpectedQuantBits;
        var dense = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_dequantize(&dense, wq, sc, bi, mlx.mlx_optional_int.some(64), mlx.mlx_optional_int.some(8), "affine", .{}, mlx.mlx_optional_dtype{}, s));
        var f16a = mlx.mlx_array_new();
        try mlx.check(mlx.mlx_astype(&f16a, dense, .float16, s));
        _ = mlx.mlx_array_free(dense);
        const cell = w.map.getPtr(wk) orelse return error.MissingWeight;
        _ = mlx.mlx_array_free(cell.*);
        cell.* = f16a;
        if (w.map.fetchRemove(sk)) |kv| {
            _ = mlx.mlx_array_free(kv.value);
            a.free(kv.key);
        }
        if (w.map.fetchRemove(bk)) |kv| {
            _ = mlx.mlx_array_free(kv.value);
            a.free(kv.key);
        }
    }
    std.debug.print("[sa3-dequant] {d} tensors\n", .{deq_count});
}


test "stable_audio3 oracle: end-to-end generation matches reference audio" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = try sa3ModelDir();
    const fix = try fixturesDir();
    const st = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(st);

    var eng = try Engine.load(io, a, dir);
    defer eng.deinit();
    // Fixtures are dumped from DEQUANTIZED pack weights ("measure the port,
    // not the quantizer"), so this oracle dequantizes in place as well: the
    // 8 sampler steps amplify per-stage noise (quantizer or otherwise) roughly
    // 100×, which would swamp this oracle's signal. The stage oracles above
    // keep exercising the quantized production path.
    try dequantizeForTest(&eng.t5.w, st);
    try dequantizeForTest(&eng.dit.w, st);

    const out = try eng.generate(a, .{
        .prompt = "A beautiful piano arpeggio grows into a cinematic climax",
        .seconds = 15.0,
        .steps = 8,
        .seed = 1234,
    }, null);
    defer a.free(out.samples);

    const ref = try readRawF32(io, a, fix, "audio_T162.f32.raw");
    defer a.free(ref);
    const want: usize = 661500; // 15 s * 44100, same trim the reference applies
    try testing.expectEqual(@as(usize, want * 2), out.samples.len);

    // out.samples is planar [ch0 | ch1], the fixture's own layout.
    const sh = [_]c_int{ 1, 2, @intCast(want) };
    const arr = mlx.mlx_array_new_data(out.samples.ptr, &sh, 3, .float32);
    defer _ = mlx.mlx_array_free(arr);
    try assertParity(arr, ref, "e2e audio", 0.99, 0.05, st);
}

test "stable_audio3 oracle: end-to-end CFG generation matches reference audio" {
    if (mlx.noGpuBackend()) return error.SkipZigTest;
    const a = testing.allocator;
    const io = std.Io.Threaded.global_single_threaded.io();
    const dir = try sa3ModelDir();
    const fix = try fixturesDir();
    const st = mlx.mlx_default_gpu_stream_new();
    defer _ = mlx.mlx_stream_free(st);

    var eng = try Engine.load(io, a, dir);
    defer eng.deinit();
    // same dequantize-for-test rule as the plain e2e (see above).
    try dequantizeForTest(&eng.t5.w, st);
    try dequantizeForTest(&eng.dit.w, st);

    const out = try eng.generate(a, .{
        .prompt = "A beautiful piano arpeggio grows into a cinematic climax",
        .seconds = 15.0,
        .steps = 8,
        .seed = 1234,
        .cfg_scale = 3.0,
        // dump_stable_audio3_fixtures.NEG_PROMPT — must stay in sync.
        .negative_prompt = "muffled, distorted, low quality, background noise",
        .apg = 1.0,
    }, null);
    defer a.free(out.samples);

    const ref = try readRawF32(io, a, fix, "audio_cfg.f32.raw");
    defer a.free(ref);
    const want: usize = 661500; // 15 s * 44100, same trim the reference applies
    try testing.expectEqual(@as(usize, want * 2), out.samples.len);

    const sh = [_]c_int{ 1, 2, @intCast(want) };
    const arr = mlx.mlx_array_new_data(out.samples.ptr, &sh, 3, .float32);
    defer _ = mlx.mlx_array_free(arr);
    try assertParity(arr, ref, "e2e cfg audio", 0.99, 0.05, st);
}
