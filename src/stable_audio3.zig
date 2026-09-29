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
fn sdpa(q: mlx.mlx_array, k: mlx.mlx_array, v: mlx.mlx_array, scale: f32, s: S) !mlx.mlx_array {
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
    const main = try sdpa(normed[0], normed[1], heads[2], scale, s);
    defer _ = mlx.mlx_array_free(main);
    const diff = try sdpa(normed[2], normed[3], heads[2], scale, s);
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

    const main = try sdpa(q0, k0, ch[2], scale, s);
    defer _ = mlx.mlx_array_free(main);
    const diff = try sdpa(q1, k1, ch[2], scale, s);
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
            const tsh = [_]c_int{ 1, 1 };
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
            const lsh = [_]c_int{ 1, T, ld };
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
            break :blk try concatA(m, p, 1, s);
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
            const psh = [_]c_int{ 1, mem, E };
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
            const hi = [_]c_int{ 1, seq_len, E };
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
