#!/usr/bin/env bash
# Stable Audio 3 text-to-audio on the ONE main server: headless boot -> load
# the pack by absolute path -> /v1/models shows "audio"+"music" capabilities ->
# POST /v1/audio/music-generations -> assert a valid 44.1 kHz stereo PCM16 WAV
# -> the 400 family (missing prompt, bad duration/steps, ACE-Step / MiniMax
# fields refused BY NAME, TTS endpoint mismatch) -> SSE streaming with
# condition/sample progress + base64 complete -> chat coexistence -> unload.
# Proves the third music backend routes end to end.
#
# Skips gracefully when no converted pack is present. Convert with:
#   python3 tests/convert_stable_audio3_weights.py ... (see its header)
#
# Usage: SA3_MODEL=<dir> CHAT_MODEL=<dir> ./tests/test_stable_audio3_gen.sh [port]
set -uo pipefail
PORT="${1:-11438}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="$ROOT/zig-out/bin/mlx-serve"
[ -x "$BIN" ] || { echo "FAIL: build first (zig build -Doptimize=ReleaseFast)"; exit 1; }

SA3="${SA3_MODEL:-$(ls -d ~/.mlx-serve/models/*stable-audio* 2>/dev/null | head -1)}"
CHAT="${CHAT_MODEL:-$(ls -d ~/.mlx-serve/models/mlx-community/Qwen3.5-0.8B-MLX-4bit 2>/dev/null | head -1)}"
[ -n "$SA3" ] || { echo "SKIP: no Stable Audio 3 pack (set SA3_MODEL to a converted dir)"; exit 0; }
[ -f "$SA3/config.json" ] || { echo "SKIP: $SA3 has no config.json"; exit 0; }
[ -f "$SA3/dit.safetensors" ] || { echo "SKIP: $SA3 is incomplete (no dit.safetensors marker)"; exit 0; }

# Headless: the empty HF hub discovers 0 models (load-by-path case).
HUB=~/.cache/huggingface/hub
"$BIN" --serve --model-dir "$HUB" --port "$PORT" >/tmp/test_stable_audio3_server.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
for i in $(seq 1 60); do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "FAIL: headless server did not start"; tail -5 /tmp/test_stable_audio3_server.log; exit 1; }
  sleep 1
done

api() { curl -s -m 3600 "http://127.0.0.1:$PORT$1" "${@:2}"; }
SA3_ID="$(basename "$SA3")"

# 1. Load by absolute path -> ready with "audio" + "music" capabilities.
api /v1/load-model -X POST -H 'Content-Type: application/json' -d "{\"model\":\"$SA3\"}" >/dev/null
api /v1/models | python3 -c "
import sys,json
d=json.load(sys.stdin)['data']
m=[x for x in d if x['id']=='$SA3_ID' and x['state']=='ready']
assert m, 'SA3 not ready: '+json.dumps(d)
caps=m[0].get('capabilities',[])
assert 'audio' in caps and 'music' in caps, f'want audio+music caps, got {caps}'
print('PASS: load-model by path -> music model ready, capabilities', caps)
" || { echo "FAIL: ready music model missing audio/music capability"; exit 1; }

# 2. The 400 family — cheap, BEFORE the expensive generation.
b400() { # label body
  local code
  code=$(api /v1/audio/music-generations -X POST -H 'Content-Type: application/json' \
    -d "$2" -o /tmp/test_stable_audio3_err.txt -w "%{http_code}")
  [ "$code" = "400" ] || { echo "FAIL: $1 returned $code (want 400)"; cat /tmp/test_stable_audio3_err.txt; exit 1; }
  echo "PASS: $1 -> 400"
}
b400 "missing prompt" "{\"model\":\"$SA3_ID\",\"duration_seconds\":4}"
b400 "empty prompt" "{\"model\":\"$SA3_ID\",\"prompt\":\"\",\"duration_seconds\":4}"
b400 "duration 0" "{\"model\":\"$SA3_ID\",\"prompt\":\"jazz\",\"duration_seconds\":0}"
b400 "duration 999" "{\"model\":\"$SA3_ID\",\"prompt\":\"jazz\",\"duration_seconds\":999}"
b400 "steps 0" "{\"model\":\"$SA3_ID\",\"prompt\":\"jazz\",\"steps\":0}"
b400 "steps 101" "{\"model\":\"$SA3_ID\",\"prompt\":\"jazz\",\"steps\":101}"
# Fields the OTHER music engines condition on: SA3 has no conditioning path
# for them, so each is refused BY NAME pointing at 'prompt', never ignored.
for f in '"lyrics":"[verse]\nla la"' '"instrumental":true' '"bpm":120' '"keyscale":"C major"' '"ref_audio":"AAAA"' '"src_audio":"AAAA"' '"task":"cover"' '"timesignature":"4/4"' '"vocal_language":"en"'; do
  b400 "unsupported field ${f%%:*}" "{\"model\":\"$SA3_ID\",\"prompt\":\"jazz\",$f}"
  grep -q 'Stable Audio 3 is text-to-audio' /tmp/test_stable_audio3_err.txt \
    || { echo "FAIL: unsupported-field 400 does not NAME the SA3 contract"; cat /tmp/test_stable_audio3_err.txt; exit 1; }
done
# The TTS endpoint against a music model is an explicit 400.
code=$(api /v1/audio/speech -X POST -H 'Content-Type: application/json' \
  -d "{\"model\":\"$SA3_ID\",\"input\":\"hello\"}" -o /dev/null -w "%{http_code}")
[ "$code" = "400" ] || { echo "FAIL: /v1/audio/speech on music model returned $code (want 400)"; exit 1; }
echo "PASS: /v1/audio/speech on a music model -> 400"

# 3. Generate (short duration + few steps -> smoke, not quality) -> valid WAV.
cat > /tmp/test_stable_audio3_req.json <<EOF
{"model":"$SA3_ID","prompt":"a gentle piano arpeggio fading into warm pads","duration_seconds":4,"steps":8,"seed":7}
EOF
code=$(api /v1/audio/music-generations -X POST -H 'Content-Type: application/json' \
  -d @/tmp/test_stable_audio3_req.json -o /tmp/test_stable_audio3_out.wav -w "%{http_code}")
[ "$code" = "200" ] || { echo "FAIL: music gen http $code"; head -c 300 /tmp/test_stable_audio3_out.wav; tail -20 /tmp/test_stable_audio3_server.log; exit 1; }
python3 - /tmp/test_stable_audio3_out.wav <<'PY'
import sys, struct
b = open(sys.argv[1], "rb").read()
assert b[:4] == b"RIFF" and b[8:12] == b"WAVE", f"not a WAV: {b[:12]!r}"
fmt, channels, rate = struct.unpack("<HHI", b[20:28])
bits = struct.unpack("<H", b[34:36])[0]
assert fmt == 1 and bits == 16, (fmt, bits)
assert channels == 2, f"want stereo, got {channels}"
assert rate == 44100, f"want 44.1 kHz, got {rate}"
n_samples = (len(b) - 44) // (2 * channels)
dur = n_samples / rate
assert 0.5 <= dur <= 4.5, f"want (0.5, 4.5] s, got {dur:.2f} s"
assert any(b[44:44 + 4 * 44100]), "output is all-zero audio"
print(f"PASS: /v1/audio/music-generations -> {len(b)} byte WAV, {dur:.2f} s 44.1 kHz stereo")
PY
[ $? -eq 0 ] || exit 1
grep -q '\[sa3\] engine ready' /tmp/test_stable_audio3_server.log || { echo "FAIL: no [sa3] engine engagement in log"; exit 1; }
grep -q '\[sa3\] generating' /tmp/test_stable_audio3_server.log || { echo "FAIL: no [sa3] generation in log"; exit 1; }
echo "PASS: [sa3] engine + generation logged"

# 4. Server survives the gen.
curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || { echo "FAIL: server died after music gen"; exit 1; }

# 5. Streaming: SSE progress (condition/sample/decode) + base64 complete.
cat > /tmp/test_stable_audio3_stream_req.json <<EOF
{"model":"$SA3_ID","prompt":"soft ambient drone with distant bells","duration_seconds":4,"steps":8,"seed":7,"stream":true}
EOF
code=$(api /v1/audio/music-generations -X POST -H 'Content-Type: application/json' \
  -d @/tmp/test_stable_audio3_stream_req.json -o /tmp/test_stable_audio3_stream.txt -w "%{http_code}")
[ "$code" = "200" ] || { echo "FAIL: stream music gen http $code"; exit 1; }
grep -q '"stage":"condition"' /tmp/test_stable_audio3_stream.txt || { echo "FAIL: no condition progress in stream"; exit 1; }
grep -q '"stage":"sample"' /tmp/test_stable_audio3_stream.txt || { echo "FAIL: no sample progress in stream"; exit 1; }
grep -q '"stage":"decode"' /tmp/test_stable_audio3_stream.txt || { echo "FAIL: no decode progress in stream"; exit 1; }
grep -q '"type":"complete"' /tmp/test_stable_audio3_stream.txt || { echo "FAIL: no complete event in stream"; exit 1; }
echo "PASS: streaming -> SSE condition/sample/decode progress + complete event"

# 6. Coexistence with a chat model.
if [ -n "$CHAT" ]; then
  CHAT_ID="$(basename "$CHAT")"
  api /v1/load-model -X POST -H 'Content-Type: application/json' -d "{\"model\":\"$CHAT\"}" >/dev/null
  TOK=$(curl -s -m 120 -N -X POST "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$CHAT_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi in 3 words.\"}],\"max_tokens\":16,\"stream\":true}" \
    | grep -c '"content":')
  [ "$TOK" -ge 1 ] || { echo "FAIL: chat did not stream while music model resident"; exit 1; }
  echo "PASS: chat streams ($TOK content deltas) with music model also resident"
fi

# 7. Unload -> stub returns to unloaded.
api /v1/unload-model -X POST -H 'Content-Type: application/json' -d "{\"model\":\"$SA3_ID\"}" >/dev/null
api /v1/models | python3 -c "
import sys,json
d=json.load(sys.stdin)['data']
m=[x for x in d if x['id']=='$SA3_ID']
assert m and m[0]['state']=='unloaded', 'SA3 model should be unloaded: '+json.dumps(d)
print('PASS: unload-model -> music model unloaded (stub retained)')
"
echo "ALL PASS"
