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
    if (mlx.mlx_array_dtype(o) == .float16) return o;
    const narrowed = try astype(o, .float16, s);
    _ = mlx.mlx_array_free(o);
    return narrowed;
}

/// gelu_approx: 0.5x(1 + tanh(sqrt(2/pi)(x + 0.044715 x^3))) — every
/// constant materialized in x's dtype (weak-scalar rule), f16 like MLX.
fn geluApprox(x: mlx.mlx_array, s: S) !mlx.mlx_array {
    const x2 = try mulA(x, x, s);
    defer _ = mlx.mlx_array_free(x2);
    const x3 = try mulA(x2, x, s);
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
        const d = try mulScalar(qk, 1.0 / cap, s);
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
