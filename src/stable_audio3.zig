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

// ── env-gated oracle plumbing ───────────────────────────────────────────────

fn sa3ModelDir() ![]const u8 {
    return std.mem.span(std.c.getenv("SA3_TEST_MODEL") orelse return error.SkipZigTest);
}

fn fixturesDir() ![]const u8 {
    return std.mem.span(std.c.getenv("SA3_FIXTURES") orelse return error.SkipZigTest);
}

fn readRawI32(io: std.Io, a: std.mem.Allocator, dir: []const u8, name: []const u8) ![]i32 {
    const path = try std.fmt.allocPrint(a, "{s}/{s}", .{ dir, name });
    defer a.free(path);
    const f = try std.Io.Dir.openFileAbsolute(io, path, .{});
    defer f.close(io);
    var rb: [4096]u8 = undefined;
    var rs = f.reader(io, &rb);
    const bytes = try rs.interface.allocRemaining(a, .limited(64 * 1024 * 1024));
    errdefer a.free(bytes);
    const n = bytes.len / 4;
    const out = try a.alloc(i32, n);
    @memcpy(std.mem.sliceAsBytes(out), bytes[0 .. n * 4]);
    a.free(bytes);
    return out;
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
