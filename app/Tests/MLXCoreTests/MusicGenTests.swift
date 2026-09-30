import XCTest
@testable import MLXCore

/// Music tab (ACE-Step text2music): preset catalog, the
/// `/v1/audio/music-generations` wire contract, output-path slugging,
/// sticky-settings round-trip, and bundle readiness.
final class MusicGenTests: XCTestCase {

    // MARK: - Preset catalog

    func testMusicPresetCatalogIsWellFormed() {
        XCTAssertFalse(MusicModelPreset.all.isEmpty)
        // Default (first) is the XL Turbo 8-bit build.
        XCTAssertEqual(MusicModelPreset.all.first?.id, MusicModelPreset.acestepXLTurbo8bit.id)
        for p in MusicModelPreset.all {
            XCTAssertFalse(p.id.isEmpty)
            XCTAssertFalse(p.repo.isEmpty)
            XCTAssertGreaterThan(p.approxRAMGB, 0)
            // Steps are checkpoint facts: ACE Turbo is distillation-fixed at
            // 8; Music 3 runs the reference 30-step flow-match schedule;
            // Stable Audio 3 the reference ping-pong default of 8.
            switch p.family {
            case .acestep: XCTAssertEqual(p.fixedSteps, 8, p.id)
            case .minimaxMusic3: XCTAssertEqual(p.fixedSteps, 30, p.id)
            case .stableAudio3: XCTAssertEqual(p.fixedSteps, 8, p.id)
            }
        }
        // Published converted repo → the pane offers a one-click download
        // (a `local/` prefix would show the convert-locally hint instead).
        XCTAssertFalse(MusicModelPreset.acestepXLTurbo8bit.isLocalOnly)
        XCTAssertEqual(MusicModelPreset.acestepXLTurbo8bit.repo,
                       "ddalcu/ACE-Step-1.5-XL-Turbo-MLX-Serve-8bit")
    }

    /// Stable Audio 3 is the third engine: text-to-audio from prompt text
    /// (plus its own audio-to-audio, `testStableAudio3InitAudioIsGatedAndLandsOnTheWire`).
    /// The server names EVERY field the other two condition on a 400, so the
    /// preset's flags must gate the pane AND the wire — a value lingering in
    /// sticky settings across a model switch must not reach a server that
    /// refuses it, and must not be claimed in the sidecar either.
    func testStableAudio3IsTextToAudioAndGatesEveryOtherEngineField() {
        let p = MusicModelPreset.stableAudio3Medium
        XCTAssertEqual(p.family, .stableAudio3)
        XCTAssertTrue(MusicModelPreset.all.contains(p), "in the picker catalog")
        XCTAssertFalse(p.supportsLyrics, "no lyric conditioning")
        XCTAssertFalse(p.requiresLyrics, "never demands lyrics")
        XCTAssertFalse(p.supportsMusicalMeta, "vocal_language/timesignature are named 400s")
        XCTAssertFalse(p.supportsTempoAndKey, "bpm/keyscale are named 400s")
        XCTAssertFalse(p.supportsReferenceAudio, "ref_audio is a named 400")
        XCTAssertFalse(p.supportsSourceAudio, "task/src_audio are named 400s")
        XCTAssertTrue(p.supportsSteps, "steps are user-editable")
        XCTAssertEqual(p.stepsRange, 1...100, "server range [1,100], reference default 1 forward pass")
        XCTAssertEqual(p.durationRange, 5...384, "server range [1,384], floored at 5 for a usable slider")
        XCTAssertEqual(p.fixedSteps, 8, "the reference ping-pong default")
    }

    func testStableAudio3RequestBodySendsOnlyWhatTheServerReads() {
        let req = MusicGenRequest(model: .stableAudio3Medium, prompt: "rain on a tin roof",
                                  lyrics: "[Verse]\nla la la", instrumental: true,
                                  vocalLanguage: "ja", bpm: 96, keyscale: "C major",
                                  timesignature: "4/4", durationSeconds: 600,
                                  seed: 7, steps: 50, refAudioPath: "/tmp/ref.wav")
        let body = MusicGenService.requestBody(req, modelName: "sa3", refAudioB64: "UklGRg==")
        XCTAssertNil(body["instrumental"], "SA3 has no instrumental path — the server 400s the field")
        XCTAssertNil(body["lyrics"])
        XCTAssertNil(body["bpm"])
        XCTAssertNil(body["keyscale"])
        XCTAssertNil(body["vocal_language"])
        XCTAssertNil(body["timesignature"])
        XCTAssertNil(body["ref_audio"])
        XCTAssertEqual(body["duration_seconds"] as? Int, 384, "sticky 600 clamps into the server range")
        XCTAssertEqual(body["steps"] as? Int, 50)
        XCTAssertEqual(body["seed"] as? Int, 7, "resolved seed rides the body for reproducibility")
        // The sidecar is the reproducibility record: it may claim only what
        // the body actually carried.
        let txt = MusicGenService.settingsText(req, resolvedSeed: 7, modelName: "sa3")
        XCTAssertFalse(txt.contains("instrumental"), "an omitted field must not be recorded as sent")
        XCTAssertFalse(txt.contains("# Lyrics"), "lyrics never travel to this engine")
        XCTAssertTrue(txt.contains("duration_seconds: 384"), "records the clamped value actually sent")
    }

    /// Stage 2 of the SA3 port: CFG scale + negative prompt + APG travel ONLY
    /// to the one engine that reads them, and the pairing the server enforces
    /// (negative_prompt REQUIRES cfg_scale != 1.0) holds on the wire — sticky
    /// negative text while the slider sits at 1.0 would be a server 400.
    func testStableAudio3GuidanceFieldsAreGatedAndPaired() throws {
        XCTAssertTrue(MusicModelPreset.stableAudio3Medium.supportsGuidance)
        XCTAssertFalse(MusicModelPreset.acestepXLTurbo8bit.supportsGuidance)
        XCTAssertFalse(MusicModelPreset.miniMaxMusic3_8bit.supportsGuidance)
        for p in MusicModelPreset.all {
            XCTAssertEqual(p.supportsGuidance, p.family == .stableAudio3, p.id)
        }

        // Guidance on: all three fields ride, exactly as set, and the sidecar
        // claims the same run the body asked for.
        var on = MusicGenRequest(model: .stableAudio3Medium, prompt: "piano")
        on.cfgScale = 3.0
        on.negativePrompt = "muffled, distorted"
        on.apg = 0.5
        let body = MusicGenService.requestBody(on, modelName: "sa3")
        XCTAssertEqual(body["cfg_scale"] as? Double, 3.0)
        XCTAssertEqual(body["negative_prompt"] as? String, "muffled, distorted")
        XCTAssertEqual(body["apg"] as? Double, 0.5)
        let txt = MusicGenService.settingsText(on, resolvedSeed: 7, modelName: "sa3")
        XCTAssertTrue(txt.contains("cfg_scale: 3.0"))
        XCTAssertTrue(txt.contains("negative_prompt: muffled, distorted"))
        XCTAssertTrue(txt.contains("apg: 0.5"))

        // Guidance off: negative text may linger in sticky settings but must
        // NOT travel (the server names the pair a 400) nor be claimed sent.
        var off = MusicGenRequest(model: .stableAudio3Medium, prompt: "piano")
        off.cfgScale = 1.0
        off.negativePrompt = "sticky leftover"
        off.apg = 0.5
        let offBody = MusicGenService.requestBody(off, modelName: "sa3")
        XCTAssertEqual(offBody["cfg_scale"] as? Double, 1.0, "cfg rides even at 1.0 (explicit off)")
        XCTAssertNil(offBody["negative_prompt"])
        XCTAssertNil(offBody["apg"])
        let offTxt = MusicGenService.settingsText(off, resolvedSeed: 7, modelName: "sa3")
        XCTAssertTrue(offTxt.contains("cfg_scale: 1.0"))
        XCTAssertFalse(offTxt.contains("negative_prompt"),
                       "an omitted field must not be recorded as sent")
        XCTAssertFalse(offTxt.contains("apg"))

        // The other engines name every one of these fields a 400 — sticky
        // values across a model switch must not reach them.
        var ace = MusicGenRequest(model: .acestepXLTurbo8bit, prompt: "piano")
        ace.cfgScale = 3.0
        ace.negativePrompt = "no drums"
        ace.apg = 0.5
        let aceBody = MusicGenService.requestBody(ace, modelName: "ace")
        XCTAssertNil(aceBody["cfg_scale"])
        XCTAssertNil(aceBody["negative_prompt"])
        XCTAssertNil(aceBody["apg"])
        XCTAssertFalse(MusicGenService.settingsText(ace, resolvedSeed: 1, modelName: "ace")
            .contains("cfg_scale"))

        // The server's pairing rule is the contract the wire-shape above
        // exists to satisfy — pin the message so a server change surfaces
        // here rather than in a live 400.
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        let gen = try String(contentsOf: root.appendingPathComponent("src/gen.zig"), encoding: .utf8)
        XCTAssertTrue(gen.contains("'negative_prompt' needs an uncond branch"),
                      "src/gen.zig no longer refuses negative_prompt beside cfg_scale == 1.0")
    }

    /// The three guidance knobs are sticky like every other music setting:
    /// they survive a pane unmount, and a blob written by a build that
    /// predates them still decodes (the decodeIfPresent migration rule — a
    /// throwing or key-requiring decode would reset every existing install).
    func testMusicGuidanceSettingsRoundTripAndDecodeLegacyBlobs() throws {
        var s = MusicGenSettings()
        s.cfgScale = 7.5
        s.negativePrompt = "live audience noise"
        s.apg = 0.25
        let data = try JSONEncoder().encode(s)
        let back = try JSONDecoder().decode(MusicGenSettings.self, from: data)
        XCTAssertEqual(back.cfgScale, 7.5)
        XCTAssertEqual(back.negativePrompt, "live audience noise")
        XCTAssertEqual(back.apg, 0.25)

        var legacy = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        legacy.removeValue(forKey: "cfgScale")
        legacy.removeValue(forKey: "negativePrompt")
        legacy.removeValue(forKey: "apg")
        let old = try JSONSerialization.data(withJSONObject: legacy)
        let fromOld = try JSONDecoder().decode(MusicGenSettings.self, from: old)
        XCTAssertEqual(fromOld.cfgScale, 1.0, "guidance off — the reference default")
        XCTAssertEqual(fromOld.negativePrompt, "")
        XCTAssertEqual(fromOld.apg, 1.0, "full APG — the reference default")
    }

    /// Stage 3 of the SA3 port: audio-to-audio. `init_audio` (base64 WAV) +
    /// `init_noise_level` travel ONLY to the one engine that reads them —
    /// ACE-Step and Music 3 both name them a 400 — and each stands alone:
    /// there is no pairing rule between them (a σmax below 1.0 with no init
    /// clip is a legal schedule effect the reference allows).
    func testStableAudio3InitAudioIsGatedAndLandsOnTheWire() throws {
        XCTAssertTrue(MusicModelPreset.stableAudio3Medium.supportsInitAudio)
        for p in MusicModelPreset.all {
            XCTAssertEqual(p.supportsInitAudio, p.family == .stableAudio3, p.id)
        }

        // Both set: both ride, exactly as set, and the sidecar claims the
        // same run the body asked for.
        var on = MusicGenRequest(model: .stableAudio3Medium, prompt: "piano")
        on.initAudioPath = "/tmp/init.wav"
        on.initNoiseLevel = 0.5
        let body = MusicGenService.requestBody(on, modelName: "sa3", initAudioB64: "UklGRg==")
        XCTAssertEqual(body["init_audio"] as? String, "UklGRg==")
        XCTAssertEqual(body["init_noise_level"] as? Double, 0.5)
        let txt = MusicGenService.settingsText(on, resolvedSeed: 7, modelName: "sa3")
        XCTAssertTrue(txt.contains("init_noise_level: 0.5"))
        XCTAssertTrue(txt.contains("init_audio: init.wav"), "records the clip actually used")

        // σmax alone (no clip) still rides: it is a schedule effect, not a
        // pairing, and the server's default is 1.0 — never sent as 0.0 here.
        var sched = MusicGenRequest(model: .stableAudio3Medium, prompt: "piano")
        sched.initNoiseLevel = 0.8
        let schedBody = MusicGenService.requestBody(sched, modelName: "sa3")
        XCTAssertNil(schedBody["init_audio"], "no clip, no field")
        XCTAssertEqual(schedBody["init_noise_level"] as? Double, 0.8)

        // A clip alone rides too (σmax defaults to 1.0 server-side).
        var clipOnly = MusicGenRequest(model: .stableAudio3Medium, prompt: "piano")
        clipOnly.initAudioPath = "/tmp/init.wav"
        let clipBody = MusicGenService.requestBody(clipOnly, modelName: "sa3", initAudioB64: "UklGRg==")
        XCTAssertEqual(clipBody["init_audio"] as? String, "UklGRg==")

        // Sticky values across a model switch must not reach the engines
        // that refuse them by name.
        var ace = MusicGenRequest(model: .acestepXLTurbo8bit, prompt: "piano")
        ace.initAudioPath = "/tmp/init.wav"
        ace.initNoiseLevel = 0.5
        let aceBody = MusicGenService.requestBody(ace, modelName: "ace", initAudioB64: "UklGRg==")
        XCTAssertNil(aceBody["init_audio"])
        XCTAssertNil(aceBody["init_noise_level"])
        XCTAssertFalse(MusicGenService.settingsText(ace, resolvedSeed: 1, modelName: "ace")
            .contains("init_noise_level"))

        // The server's refusal messages are the contract the gate exists to
        // satisfy — pin both spellings so a server change surfaces here
        // rather than in a live 400.
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        let gen = try String(contentsOf: root.appendingPathComponent("src/gen.zig"), encoding: .utf8)
        XCTAssertTrue(gen.contains("are Stable Audio 3 fields"),
                      "src/gen.zig no longer refuses init_audio on ACE-Step / Music 3")
        XCTAssertTrue(gen.contains("'init_noise_level' must be ≥ 0.01"),
                      "src/gen.zig no longer enforces the σmax floor")
    }

    /// The init clip + σmax are sticky like every other music setting, and a
    /// blob written by a build that predates them still decodes (the
    /// decodeIfPresent migration rule).
    func testMusicInitAudioSettingsRoundTripAndDecodeLegacyBlobs() throws {
        var s = MusicGenSettings()
        s.initAudioPath = "/tmp/init.wav"
        s.initNoiseLevel = 0.4
        let data = try JSONEncoder().encode(s)
        let back = try JSONDecoder().decode(MusicGenSettings.self, from: data)
        XCTAssertEqual(back.initAudioPath, "/tmp/init.wav")
        XCTAssertEqual(back.initNoiseLevel, 0.4)

        var legacy = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        legacy.removeValue(forKey: "initAudioPath")
        legacy.removeValue(forKey: "initNoiseLevel")
        let old = try JSONSerialization.data(withJSONObject: legacy)
        let fromOld = try JSONDecoder().decode(MusicGenSettings.self, from: old)
        XCTAssertNil(fromOld.initAudioPath)
        XCTAssertEqual(fromOld.initNoiseLevel, 1.0, "σmax 1.0 = pure text-to-audio, the reference default")
    }

    /// Stage 4 of the SA3 port: inpainting. `inpaint_range: [start, end]` in
    /// seconds regenerates that slice of the seed clip and keeps the rest. It
    /// rides only to the one family that reads it (the others name it a 400
    /// beside `init_audio`), and — unlike every other optional field —
    /// `requestBody` must NOT drop it when the clip is missing: a run that
    /// ignored the range is worse than a 400, because the 400 says so and the
    /// ignored run does not.
    func testStableAudio3InpaintRangeIsGatedAndOnTheWire() throws {
        XCTAssertTrue(MusicModelPreset.stableAudio3Medium.supportsInpaint)
        for p in MusicModelPreset.all {
            XCTAssertEqual(p.supportsInpaint, p.family == .stableAudio3, p.id)
        }

        var on = MusicGenRequest(model: .stableAudio3Medium, prompt: "piano")
        on.initAudioPath = "/tmp/init.wav"
        on.inpaintRange = (start: 0.5, end: 2.0)
        let body = MusicGenService.requestBody(on, modelName: "sa3", initAudioB64: "UklGRg==")
        XCTAssertEqual(try XCTUnwrap(body["inpaint_range"] as? [Double]), [0.5, 2.0])
        XCTAssertEqual(body["init_audio"] as? String, "UklGRg==")
        let txt = MusicGenService.settingsText(on, resolvedSeed: 7, modelName: "sa3")
        XCTAssertTrue(txt.contains("inpaint_range: 0.5 - 2.0"),
                      "the sidecar must record the range that ran")

        // A range with no clip still reaches the server, so the server's own
        // pairing 400 names it. `generate` refuses first with a sentence (see
        // testInpaintRefusalNamesWhyItCannotRun) — this is the backstop for
        // the agent path and for a clip that became unreadable mid-flight.
        var noclip = MusicGenRequest(model: .stableAudio3Medium, prompt: "piano")
        noclip.inpaintRange = (start: 0.5, end: 2.0)
        let bare = MusicGenService.requestBody(noclip, modelName: "sa3")
        XCTAssertNotNil(bare["inpaint_range"],
                        "silently dropping the range would hand back audio that ignored it")
        XCTAssertNil(bare["init_audio"])

        // Sticky across a model switch: the other families refuse it by name.
        var ace = MusicGenRequest(model: .acestepXLTurbo8bit, prompt: "piano")
        ace.initAudioPath = "/tmp/init.wav"
        ace.inpaintRange = (start: 0.5, end: 2.0)
        XCTAssertNil(MusicGenService.requestBody(ace, modelName: "ace")["inpaint_range"])
        XCTAssertFalse(MusicGenService.settingsText(ace, resolvedSeed: 1, modelName: "ace")
            .contains("inpaint_range"))

        // The server's own refusals are the contract these gates exist to
        // satisfy — pin both spellings so a server change surfaces here
        // rather than in a live 400.
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        let gen = try String(contentsOf: root.appendingPathComponent("src/gen.zig"), encoding: .utf8)
        XCTAssertTrue(gen.contains("'inpaint_range' must satisfy 0 <= start < end <= duration_seconds"),
                      "src/gen.zig no longer states the inpaint range constraint")
        XCTAssertTrue(gen.contains("'inpaint_range' needs a non-empty 'init_audio'"),
                      "src/gen.zig no longer pairs inpaint_range with init_audio")
    }

    /// The pane shows a SENTENCE when inpainting cannot run, not a 400's JSON.
    /// The refusal runs only where the field is actually sent — a sticky
    /// range on ACE-Step is dropped by the gate and must not block that
    /// engine's run.
    func testInpaintRefusalNamesWhyItCannotRun() {
        XCTAssertNil(MusicGenRequest.inpaintRefusal(model: .stableAudio3Medium, inpaintRange: nil,
                                                    initAudioPath: "/tmp/init.wav", durationSeconds: 60))
        XCTAssertNil(MusicGenRequest.inpaintRefusal(model: .acestepXLTurbo8bit,
                                                    inpaintRange: (start: 0.5, end: 2.0),
                                                    initAudioPath: nil, durationSeconds: 60),
                     "a sticky value on another family must not block that family's run")

        // The pairing: there is nothing to inpaint INTO.
        let noclip = MusicGenRequest.inpaintRefusal(model: .stableAudio3Medium,
                                                    inpaintRange: (start: 0.5, end: 2.0),
                                                    initAudioPath: nil, durationSeconds: 60)
        XCTAssertNotNil(noclip)
        XCTAssertTrue(noclip!.contains("seed audio file"), noclip!)
        XCTAssertNotNil(MusicGenRequest.inpaintRefusal(model: .stableAudio3Medium,
                                                       inpaintRange: (start: 0.5, end: 2.0),
                                                       initAudioPath: "", durationSeconds: 60))

        // `0 <= start < end <= duration_seconds`, against the SAME clamped
        // duration the body sends (sticky values outlive a model switch).
        let inverted = MusicGenRequest.inpaintRefusal(model: .stableAudio3Medium,
                                                      inpaintRange: (start: 2.0, end: 0.5),
                                                      initAudioPath: "/tmp/init.wav", durationSeconds: 60)
        XCTAssertNotNil(inverted)
        XCTAssertNotNil(MusicGenRequest.inpaintRefusal(model: .stableAudio3Medium,
                                                       inpaintRange: (start: -1.0, end: 2.0),
                                                       initAudioPath: "/tmp/init.wav", durationSeconds: 60))
        XCTAssertNotNil(MusicGenRequest.inpaintRefusal(model: .stableAudio3Medium,
                                                       inpaintRange: (start: 0.5, end: Double.nan),
                                                       initAudioPath: "/tmp/init.wav", durationSeconds: 60))
        let past = MusicGenRequest.inpaintRefusal(model: .stableAudio3Medium,
                                                  inpaintRange: (start: 0.5, end: 61.0),
                                                  initAudioPath: "/tmp/init.wav", durationSeconds: 60)
        XCTAssertNotNil(past)
        XCTAssertTrue(past!.contains("60 s"), past!)

        // The happy path: the range the fixture dump used.
        XCTAssertNil(MusicGenRequest.inpaintRefusal(model: .stableAudio3Medium,
                                                    inpaintRange: (start: 4.0, end: 9.0),
                                                    initAudioPath: "/tmp/init.wav", durationSeconds: 15))
    }

    func testMusicInpaintSettingsRoundTripAndDecodeLegacyBlobs() throws {
        var s = MusicGenSettings()
        s.inpaintEnabled = true
        s.inpaintStart = 4.0
        s.inpaintEnd = 9.0
        let data = try JSONEncoder().encode(s)
        let back = try JSONDecoder().decode(MusicGenSettings.self, from: data)
        XCTAssertEqual(back.inpaintEnabled, true)
        XCTAssertEqual(back.inpaintStart, 4.0)
        XCTAssertEqual(back.inpaintEnd, 9.0)

        var legacy = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        legacy.removeValue(forKey: "inpaintEnabled")
        legacy.removeValue(forKey: "inpaintStart")
        legacy.removeValue(forKey: "inpaintEnd")
        let old = try JSONSerialization.data(withJSONObject: legacy)
        let fromOld = try JSONDecoder().decode(MusicGenSettings.self, from: old)
        XCTAssertFalse(fromOld.inpaintEnabled, "a blob written before inpainting decodes with the feature off")
        XCTAssertEqual(fromOld.inpaintStart, 0.0)
        XCTAssertEqual(fromOld.inpaintEnd, 9.0)
    }

    func testReferenceAudioIsDeclaredPerFamilyAndSentOnlyThere() {
        // ACE-Step has a timbre slot; Music 3 names `ref_audio` a 400. The
        // preset flag gates the control AND the field, so a clip left behind
        // by a model switch never reaches the server that refuses it.
        XCTAssertTrue(MusicModelPreset.acestepXLTurbo8bit.supportsReferenceAudio)
        XCTAssertFalse(MusicModelPreset.miniMaxMusic3_8bit.supportsReferenceAudio)
        for p in MusicModelPreset.all {
            XCTAssertEqual(p.supportsReferenceAudio, p.family == .acestep, p.id)
        }
        let ace = MusicGenRequest(model: .acestepXLTurbo8bit, prompt: "lo-fi", refAudioPath: "/tmp/ref.wav")
        XCTAssertEqual(MusicGenService.requestBody(ace, modelName: "ace", refAudioB64: "UklGRg==")["ref_audio"] as? String, "UklGRg==")
        XCTAssertNil(MusicGenService.requestBody(ace, modelName: "ace")["ref_audio"], "no clip, no field")
        let m3 = MusicGenRequest(model: .miniMaxMusic3_8bit, prompt: "lo-fi", lyrics: "la", refAudioPath: "/tmp/ref.wav")
        XCTAssertNil(MusicGenService.requestBody(m3, modelName: "m3", refAudioB64: "UklGRg==")["ref_audio"])
        XCTAssertNil(MusicGenService.referenceB64(m3))
        // The sidecar names the clip where it was used, so a track stays reproducible.
        XCTAssertTrue(MusicGenService.settingsText(ace, resolvedSeed: 1, modelName: "ace").contains("ref_audio: ref.wav"))
        XCTAssertFalse(MusicGenService.settingsText(m3, resolvedSeed: 1, modelName: "m3").contains("ref_audio"))
    }

    func testSourceAudioTasksGateTheirFieldsOnModelAndTask() throws {
        XCTAssertTrue(MusicModelPreset.acestepXLTurbo8bit.supportsSourceAudio)
        XCTAssertFalse(MusicModelPreset.miniMaxMusic3_8bit.supportsSourceAudio)
        // Cover: task + source + its two strengths; never the instrument list.
        var cover = MusicGenRequest(model: .acestepXLTurbo8bit, prompt: "orchestral", task: .cover,
                                    srcAudioPath: "/tmp/src.wav", coverStrength: 0.7, coverNoiseStrength: 1.4,
                                    trackClasses: ["bass"])
        var body = MusicGenService.requestBody(cover, modelName: "ace", srcAudioB64: "UklGRg==")
        XCTAssertEqual(body["task"] as? String, "cover")
        XCTAssertEqual(body["src_audio"] as? String, "UklGRg==")
        XCTAssertEqual(body["cover_strength"] as? Double, 0.7)
        XCTAssertEqual(body["cover_noise_strength"] as? Double, 1.0, "clamped into the server's [0,1]")
        XCTAssertNil(body["track_classes"])
        // Complete: task + source + the (vocabulary-filtered) list; no cover knobs.
        cover.task = .complete
        cover.trackClasses = ["bass", "kazoo", "drums"]
        body = MusicGenService.requestBody(cover, modelName: "ace", srcAudioB64: "UklGRg==")
        XCTAssertEqual(body["task"] as? String, "complete")
        XCTAssertEqual(body["track_classes"] as? [String], ["bass", "drums"])
        XCTAssertNil(body["cover_strength"]); XCTAssertNil(body["cover_noise_strength"])
        // No source clip → plain text2music, whatever the task says (the
        // server would 400 a source-less cover).
        body = MusicGenService.requestBody(cover, modelName: "ace")
        XCTAssertNil(body["task"]); XCTAssertNil(body["src_audio"]); XCTAssertNil(body["track_classes"])
        // text2music never carries a source even with one attached.
        cover.task = .text2music
        body = MusicGenService.requestBody(cover, modelName: "ace", srcAudioB64: "UklGRg==")
        XCTAssertNil(body["task"]); XCTAssertNil(body["src_audio"])
        XCTAssertNil(MusicGenService.sourceB64(cover))
        // Music 3 names every one of these a 400 — the FIELDS are gated.
        let m3 = MusicGenRequest(model: .miniMaxMusic3_8bit, prompt: "p", lyrics: "la", task: .cover,
                                 srcAudioPath: "/tmp/src.wav")
        body = MusicGenService.requestBody(m3, modelName: "m3", srcAudioB64: "UklGRg==")
        XCTAssertNil(body["task"]); XCTAssertNil(body["src_audio"]); XCTAssertNil(body["cover_strength"])
        XCTAssertNil(MusicGenService.sourceB64(m3))
        // The sidecar documents the task, so a cover stays reproducible.
        cover.task = .cover
        let txt = MusicGenService.settingsText(cover, resolvedSeed: 1, modelName: "ace")
        XCTAssertTrue(txt.contains("task: cover") && txt.contains("src_audio: src.wav") && txt.contains("cover_strength: 0.7"))
        XCTAssertFalse(MusicGenService.settingsText(m3, resolvedSeed: 1, modelName: "m3").contains("task:"))
    }

    func testCoverWeightsMissingIsAStatOnThePack() throws {
        // The migration fetch keys on this one file; a pack that has it must
        // never trigger a download, a pack without it must.
        let dir = (NSTemporaryDirectory() as NSString).appendingPathComponent("mlx-serve-cover-\(UUID().uuidString)")
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(atPath: dir) }
        XCTAssertTrue(MusicGenService.coverWeightsMissing(packDir: dir))
        try Data("x".utf8).write(to: URL(fileURLWithPath: (dir as NSString).appendingPathComponent("fsq.safetensors")))
        XCTAssertFalse(MusicGenService.coverWeightsMissing(packDir: dir))
        XCTAssertEqual(MusicGenService.coverWeightsFile, "fsq.safetensors")
    }

    /// The unreachable-settings class (#243): every source-task field the
    /// request carries has a control bound to it in the pane.
    func testSourceTaskControlsExistInThePane() throws {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/MLXServe/Views/MusicGenView.swift")
        let src = try String(contentsOf: url, encoding: .utf8)
        for needle in ["selection: $task", "value: $coverStrength", "value: $coverNoiseStrength",
                       "MusicTask.trackClasses", "acceptSource(", "maxSeconds: 600"] {
            XCTAssertTrue(src.contains(needle), "MusicGenView has no control for \(needle)")
        }
        // The migration fetch is reachable from the pane (service gets the manager).
        XCTAssertTrue(src.contains("service.generate(req, server: server, downloads: downloads)"))
    }

    /// The unreachable-settings class again: the audio-to-audio well and its
    /// σmax slider must be RENDERED (a gated control that is never drawn is
    /// a control that does nothing), so a source scan pins them.
    func testInitAudioControlsExistInThePane() throws {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/MLXServe/Views/MusicGenView.swift")
        let src = try String(contentsOf: url, encoding: .utf8)
        for needle in ["model.supportsInitAudio { initAudioSection }",
                       "value: $initNoiseLevel",
                       "acceptInitAudio(", "clearInitAudio()",
                       "chooseInitAudioFile()"] {
            XCTAssertTrue(src.contains(needle), "MusicGenView has no control for \(needle)")
        }
        // The service is what reads the clip — the pane alone never sends it.
        XCTAssertTrue(src.contains("initAudioPath: initAudioURL?.path"))
    }

    /// The unreachable-affordance class, stage 4: the inpaint toggle and its
    /// two sliders must be RENDERED, the pair must reach the request, and the
    /// SERVICE must be what refuses a range that cannot run — a control never
    /// drawn, or drawn but never read, does nothing.
    func testInpaintControlsExistInThePane() throws {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let view = try String(contentsOf: root.appendingPathComponent("Sources/MLXServe/Views/MusicGenView.swift"),
                              encoding: .utf8)
        for needle in ["value: $inpaintStart", "value: $inpaintEnd",
                       "inpaintRange: inpaintEnabled",
                       "inpaintEnabled = s.inpaintEnabled",
                       "s.inpaintEnabled = inpaintEnabled"] {
            XCTAssertTrue(view.contains(needle), "MusicGenView has no control for \(needle)")
        }
        let svc = try String(contentsOf: root.appendingPathComponent("Sources/MLXServe/Services/MusicGenService.swift"),
                             encoding: .utf8)
        XCTAssertTrue(svc.contains("inpaintRefusal"),
                      "MusicGenService never refuses a range that cannot run — the server's 400 JSON is not a message")
        XCTAssertTrue(svc.contains("\"inpaint_range\""),
                      "MusicGenService never puts the range on the wire")
    }

    // MARK: - Request wire contract
    func testRequestBodyOmitsEmptyOptionalFields() {
        let req = MusicGenRequest(
            model: .acestepXLTurbo8bit,
            prompt: "upbeat synthwave",
            lyrics: "  ",
            vocalLanguage: "en",
            durationSeconds: 45,
            seed: 7
        )
        let body = MusicGenService.requestBody(req, modelName: "acestep-v15-xl-turbo-8bit")
        XCTAssertEqual(body["model"] as? String, "acestep-v15-xl-turbo-8bit")
        XCTAssertEqual(body["prompt"] as? String, "upbeat synthwave")
        XCTAssertEqual(body["duration_seconds"] as? Int, 45)
        XCTAssertEqual(body["seed"] as? Int, 7)
        XCTAssertEqual(body["stream"] as? Bool, true)
        // Blank/absent optionals never ride the wire (server defaults apply).
        XCTAssertNil(body["lyrics"], "whitespace-only lyrics must be omitted")
        XCTAssertNil(body["bpm"])
        XCTAssertNil(body["keyscale"])
        XCTAssertNil(body["timesignature"])
    }

    func testRequestBodyCarriesLyricsAndMetadataWhenSet() {
        let req = MusicGenRequest(
            model: .acestepXLTurbo8bit,
            prompt: "power ballad",
            lyrics: "la la la",
            vocalLanguage: "en",
            bpm: 128,
            keyscale: "F# minor",
            timesignature: "4",
            durationSeconds: 120,
            seed: 42
        )
        let body = MusicGenService.requestBody(req, modelName: "m")
        XCTAssertEqual(body["lyrics"] as? String, "la la la")
        XCTAssertEqual(body["vocal_language"] as? String, "en")
        XCTAssertEqual(body["bpm"] as? Int, 128)
        XCTAssertEqual(body["keyscale"] as? String, "F# minor")
        XCTAssertEqual(body["timesignature"] as? String, "4")
    }

    func testRequestBodyResolvesRandomSeed() {
        let req = MusicGenRequest(model: .acestepXLTurbo8bit, prompt: "jazz", seed: -1)
        let body = MusicGenService.requestBody(req, modelName: "m")
        // -1 resolves to a concrete non-negative seed (so the run is loggable).
        let seed = body["seed"] as? Int
        XCTAssertNotNil(seed)
        XCTAssertGreaterThanOrEqual(seed ?? -1, 0)
    }

    // MARK: - Output path

    func testMakeOutputPathSlugsAndCaps() {
        let path = MusicGenService.makeOutputPath(prompt: "Upbeat SYNTH-wave!!  with pads & bass, a very long prompt that keeps going and going")
        XCTAssertTrue(path.hasSuffix(".wav"))
        XCTAssertTrue(path.contains(MediaStorage.musicRoot))
        let file = (path as NSString).lastPathComponent
        // slug: lowercase, non-alphanumerics collapsed to '-', capped at 40.
        XCTAssertTrue(file.contains("upbeat-synth-wave-with-pads-bass"), file)
        let slug = file.split(separator: "_").last.map(String.init) ?? ""
        XCTAssertLessThanOrEqual(slug.replacingOccurrences(of: ".wav", with: "").count, 40)
    }

    // MARK: - Sticky settings

    func testMusicSettingsRoundTripAndLegacyDecode() throws {
        var s = MusicGenSettings()
        s.modelId = "acestep-v15-xl-turbo-8bit"
        s.durationSeconds = 90
        s.vocalLanguage = "ja"
        s.keepResident = true
        // The pane UNMOUNTS on every navigation, so what the user TYPED must
        // ride the sticky blob too — or a trip to Chat for a copy-paste
        // wipes the prompt, the lyrics and the attached reference clip.
        s.prompt = "upbeat synthwave"
        s.lyrics = "[verse]\nla la"
        s.refAudioPath = "/tmp/ref.wav"
        s.task = .cover
        s.srcAudioPath = "/tmp/src.wav"
        s.coverStrength = 0.8
        s.coverNoiseStrength = 0.3
        s.trackClasses = ["bass", "drums"]
        let data = try JSONEncoder().encode(s)
        let back = try JSONDecoder().decode(MusicGenSettings.self, from: data)
        XCTAssertEqual(back, s)
        XCTAssertEqual(back.prompt, "upbeat synthwave")
        XCTAssertEqual(back.refAudioPath, "/tmp/ref.wav")
        XCTAssertEqual(back.task, .cover)
        XCTAssertEqual(back.srcAudioPath, "/tmp/src.wav")
        XCTAssertEqual(back.trackClasses, ["bass", "drums"])
        var a = AudioGenSettings()
        a.text = "hello there"
        a.refAudioPath = "/tmp/voice.wav"
        a.refText = "hello"
        let aback = try JSONDecoder().decode(AudioGenSettings.self, from: JSONEncoder().encode(a))
        XCTAssertEqual(aback, a)
        XCTAssertEqual(try JSONDecoder().decode(AudioGenSettings.self, from: Data("{}".utf8)), AudioGenSettings())
        XCTAssertEqual(back.resolvedModel.id, MusicModelPreset.acestepXLTurbo8bit.id)

        // Migration-safe: an old/partial payload decodes to defaults.
        let legacy = try JSONDecoder().decode(MusicGenSettings.self, from: Data("{}".utf8))
        XCTAssertEqual(legacy, MusicGenSettings())
        XCTAssertEqual(legacy.durationSeconds, 60)

        // Unknown model id falls back to the catalog default.
        var stale = MusicGenSettings()
        stale.modelId = "acestep-v99-does-not-exist"
        XCTAssertEqual(stale.resolvedModel.id, MusicModelPreset.acestepXLTurbo8bit.id)
    }

    // MARK: - Advanced-option dropdown catalogs

    /// Every dropdown value must be accepted by the server verbatim — the
    /// whole point of dropdowns is that users can't type an invalid value.
    func testMusicOptionCatalogsAreServerValid() {
        // Language codes ⊆ the reference pipeline's VALID_LANGUAGES.
        let validLanguages: Set<String> = [
            "ar", "az", "bg", "bn", "ca", "cs", "da", "de", "el", "en",
            "es", "fa", "fi", "fr", "he", "hi", "hr", "ht", "hu", "id",
            "is", "it", "ja", "ko", "la", "lt", "ms", "ne", "nl", "no",
            "pa", "pl", "pt", "ro", "ru", "sa", "sk", "sr", "sv", "sw",
            "ta", "te", "th", "tl", "tr", "uk", "ur", "vi", "yue", "zh",
            "unknown",
        ]
        XCTAssertFalse(MusicOptions.languages.isEmpty)
        XCTAssertEqual(MusicOptions.languages.first?.code, "unknown", "Auto first")
        for (label, code) in MusicOptions.languages {
            XCTAssertFalse(label.isEmpty)
            XCTAssertTrue(validLanguages.contains(code), "invalid language code \(code)")
        }
        // BPM within the server's [30,300] gate, ascending.
        for (label, bpm) in MusicOptions.bpms {
            XCTAssertTrue((30...300).contains(bpm), "bpm \(bpm) outside server range")
            XCTAssertTrue(label.hasPrefix("\(bpm)"), "label leads with the number: \(label)")
        }
        XCTAssertEqual(MusicOptions.bpms.map(\.bpm), MusicOptions.bpms.map(\.bpm).sorted())
        // Keyscales: 24 entries in "<note>[#|b] major|minor" form.
        XCTAssertEqual(MusicOptions.keyscales.count, 24)
        for key in MusicOptions.keyscales {
            XCTAssertNotNil(key.range(of: #"^[A-G][#b]? (major|minor)$"#, options: .regularExpression), key)
        }
        // Time signatures: the server's VALID_TIME_SIGNATURES.
        XCTAssertEqual(MusicOptions.timeSignatures.map(\.value).sorted(), ["2", "3", "4", "6"])
    }

    // MARK: - Built-in example / template catalogs

    func testBuiltinStylesAndLyricsAreWellFormed() {
        XCTAssertGreaterThanOrEqual(MusicPrompt.builtinLyrics.count, 4)
        // Both engine families have their own starters — Music 3 has no
        // bpm/key/meter controls, so its examples carry that in the caption.
        for family in [MusicEngineFamily.acestep, .minimaxMusic3] {
            let styles = MusicPrompt.builtinStyles(for: family)
            XCTAssertGreaterThanOrEqual(styles.count, 6)
            for p in styles {
                XCTAssertFalse(p.title.isEmpty)
                XCTAssertGreaterThan(p.body.count, 40, "style descriptions should be descriptive: \(p.title)")
            }
            XCTAssertEqual(Set(styles.map(\.title)).count, styles.count)
        }
        for p in MusicPrompt.builtinLyrics {
            XCTAssertFalse(p.title.isEmpty)
            // Original lyric templates model the structured [Verse]/[Chorus] convention.
            XCTAssertTrue(p.body.contains("[Verse]") && p.body.contains("[Chorus]"),
                          "lyric template needs section tags: \(p.title)")
        }
        // Titles are unique (they're the dedup + display key).
        XCTAssertEqual(Set(MusicPrompt.builtinLyrics.map(\.title)).count, MusicPrompt.builtinLyrics.count)
    }

    // MARK: - Saved-prompt store (pure)

    func testAutoTitleSkipsSectionTagsAndCaps() {
        XCTAssertEqual(MusicPromptStore.autoTitle(from: "Bright modern pop with punchy drums"),
                       "Bright modern pop with punchy drums")
        // First non-tag line of structured lyrics, not the [Verse] tag.
        XCTAssertEqual(MusicPromptStore.autoTitle(from: "[Verse]\nWoke up with the sunlight"),
                       "Woke up with the sunlight")
        XCTAssertEqual(MusicPromptStore.autoTitle(from: ""), "Untitled")
        // Long lines cap at 40 chars + ellipsis.
        let long = MusicPromptStore.autoTitle(from: String(repeating: "a", count: 80))
        XCTAssertTrue(long.hasSuffix("…"))
        XCTAssertLessThanOrEqual(long.count, 41)
    }

    func testStoreAddDedupesByTitleNewestFirst() {
        var list: [MusicPrompt] = []
        list = MusicPromptStore.adding(MusicPrompt(title: "A", body: "one"), to: list)
        list = MusicPromptStore.adding(MusicPrompt(title: "B", body: "two"), to: list)
        list = MusicPromptStore.adding(MusicPrompt(title: "A", body: "one-updated"), to: list)
        XCTAssertEqual(list.map(\.title), ["A", "B"])
        XCTAssertEqual(list.first?.body, "one-updated", "same title replaces + moves to front")
        list = MusicPromptStore.removing(title: "A", from: list)
        XCTAssertEqual(list.map(\.title), ["B"])
    }

    // MARK: - Saved-prompt library (persistence)

    @MainActor
    func testLibrarySavesLoadsAndDeletesAcrossInstances() {
        let suite = "music-lib-test-\(UUID().uuidString)"
        let ud = UserDefaults(suiteName: suite)!
        defer { ud.removePersistentDomain(forName: suite) }

        let lib = MusicPromptLibrary(defaults: ud, key: "k")
        lib.saveStyle(title: "My synth", body: "warm analog synths")
        lib.saveLyrics(title: "My hook", body: "[Verse]\nla la\n[Chorus]\noh oh")
        lib.saveStyle(title: "  ", body: "blank-title auto-names")  // auto-title fallback
        lib.saveStyle(title: "empty body", body: "   ")             // ignored

        XCTAssertEqual(lib.savedStyles.count, 2, "empty-body save ignored")
        XCTAssertEqual(lib.savedLyrics.map(\.title), ["My hook"])
        XCTAssertTrue(lib.savedStyles.contains { $0.title == "blank-title auto-names" })

        // A fresh instance on the same store reloads everything.
        let reloaded = MusicPromptLibrary(defaults: ud, key: "k")
        XCTAssertEqual(reloaded.savedStyles.map(\.title), lib.savedStyles.map(\.title))
        XCTAssertEqual(reloaded.savedLyrics.first?.body, "[Verse]\nla la\n[Chorus]\noh oh")

        reloaded.deleteStyle(title: "My synth")
        let after = MusicPromptLibrary(defaults: ud, key: "k")
        XCTAssertFalse(after.savedStyles.contains { $0.title == "My synth" })
    }

    // MARK: - Settings sidecar

    func testSettingsSidecarDocumentsPromptLyricsAndParams() {
        let req = MusicGenRequest(
            model: .acestepXLTurbo8bit,
            prompt: "  upbeat synthwave  ",
            lyrics: "[Verse]\nla la la",
            vocalLanguage: "en",
            bpm: 120,
            keyscale: "F# minor",
            timesignature: "4",
            durationSeconds: 30,
            seed: -1
        )
        let txt = MusicGenService.settingsText(req, resolvedSeed: 777, modelName: "acestep-8bit")
        XCTAssertTrue(txt.contains("model: acestep-8bit"))
        XCTAssertTrue(txt.contains("seed: 777"), "the RESOLVED seed, never -1")
        XCTAssertTrue(txt.contains("duration_seconds: 30"))
        XCTAssertTrue(txt.contains("bpm: 120"))
        XCTAssertTrue(txt.contains("keyscale: F# minor"))
        XCTAssertTrue(txt.contains("timesignature: 4"))
        XCTAssertTrue(txt.contains("# Style prompt\nupbeat synthwave"), "prompt trimmed")
        XCTAssertTrue(txt.contains("# Lyrics\n[Verse]\nla la la"))
        XCTAssertEqual(MusicGenService.sidecarPath(forWav: "/a/b/track.wav"), "/a/b/track.txt")
    }

    func testSettingsSidecarOmitsAutoFieldsAndMarksInstrumental() {
        let req = MusicGenRequest(model: .acestepXLTurbo8bit, prompt: "jazz trio",
                                  lyrics: "", vocalLanguage: "unknown", durationSeconds: 60, seed: 5)
        let txt = MusicGenService.settingsText(req, resolvedSeed: 5, modelName: "m")
        XCTAssertFalse(txt.contains("bpm:"))
        XCTAssertFalse(txt.contains("keyscale:"))
        XCTAssertFalse(txt.contains("vocal_language:"), "'unknown' language omitted")
        XCTAssertTrue(txt.contains("# Lyrics\n[Instrumental]"))
    }

        // MARK: - Bundle file selection (real repo listing)

    /// The PUBLISHED repo's actual file listing (huggingface.co API,
    /// 2026-07-05) through the music bundle's selection: exactly the engine
    /// files download; the model-card assets (README/LICENSE/screenshot) and
    /// .gitattributes never do.
    func testMusicSelectionAgainstPublishedRepoListing() {
        let entries: [[String: Any]] = [
            ["path": ".gitattributes", "type": "file", "size": 1600],
            ["path": "LICENSE", "type": "file", "size": 1546],
            ["path": "README.md", "type": "file", "size": 5900],
            ["path": "config.json", "type": "file", "size": 908],
            ["path": "model.safetensors", "type": "file", "size": 5_081_359_469],
            ["path": "music-tab.png", "type": "file", "size": 342_599],
            ["path": "text_encoder/added_tokens.json", "type": "file", "size": 700],
            ["path": "text_encoder/config.json", "type": "file", "size": 1500],
            ["path": "text_encoder/model.safetensors", "type": "file", "size": 1_191_600_000],
            ["path": "text_encoder/special_tokens_map.json", "type": "file", "size": 700],
            ["path": "text_encoder/tokenizer.json", "type": "file", "size": 11_400_000],
            ["path": "text_encoder/tokenizer_config.json", "type": "file", "size": 5000],
            ["path": "vae.safetensors", "type": "file", "size": 337_483_555],
        ]
        let sel = MusicModelPreset.acestepXLTurbo8bit.bundle.components[0].selection
        let picked = DownloadManager.selectNeededFiles(from: entries, selection: sel).map(\.0)
        XCTAssertEqual(Set(picked), [
            "config.json", "model.safetensors", "vae.safetensors",
            "text_encoder/added_tokens.json", "text_encoder/config.json",
            "text_encoder/model.safetensors", "text_encoder/special_tokens_map.json",
            "text_encoder/tokenizer.json", "text_encoder/tokenizer_config.json",
        ])
        // Every readiness marker must be in the downloaded set — otherwise a
        // fresh pull could never turn "ready".
        for marker in MusicModelPreset.acestepXLTurbo8bit.bundle.components[0].readyMarkers
        where marker.hasSuffix(".json") || marker.hasSuffix(".safetensors") {
            XCTAssertTrue(picked.contains(marker), "readiness marker \(marker) not downloaded")
        }
    }

    // MARK: - Bundle readiness

    func testMusicBundleRequiresAllComponents() throws {
        let b = MusicModelPreset.acestepXLTurbo8bit.bundle
        XCTAssertEqual(b.components.count, 1, "self-contained repo — no dependency components")
        let comp = b.components[0]

        let fm = FileManager.default
        let root = NSTemporaryDirectory() + "music-bundle-test-\(UUID().uuidString)"
        defer { try? fm.removeItem(atPath: root) }
        let modelDir = (root as NSString).appendingPathComponent(comp.repo)
        try fm.createDirectory(atPath: modelDir, withIntermediateDirectories: true)

        XCTAssertFalse(DownloadManager.componentReady(comp, modelsRoot: root))
        // Top-level weights alone → still not ready (text_encoder markers missing).
        fm.createFile(atPath: (modelDir as NSString).appendingPathComponent("config.json"), contents: Data("{}".utf8))
        fm.createFile(atPath: (modelDir as NSString).appendingPathComponent("model.safetensors"), contents: Data([0, 1]))
        fm.createFile(atPath: (modelDir as NSString).appendingPathComponent("vae.safetensors"), contents: Data([0, 1]))
        XCTAssertFalse(DownloadManager.componentReady(comp, modelsRoot: root))
        // Full text_encoder subdir → ready.
        let te = (modelDir as NSString).appendingPathComponent("text_encoder")
        try fm.createDirectory(atPath: te, withIntermediateDirectories: true)
        for f in ["config.json", "model.safetensors", "tokenizer.json"] {
            fm.createFile(atPath: (te as NSString).appendingPathComponent(f), contents: Data([0]))
        }
        XCTAssertTrue(DownloadManager.componentReady(comp, modelsRoot: root))
    }

    // MARK: - MiniMax Music 3 (second music family — the field-gating tests)

    func testMusic3PresetDeclaresItsOwnKnobSet() {
        let p = MusicModelPreset.miniMaxMusic3_8bit
        XCTAssertEqual(p.family, .minimaxMusic3)
        XCTAssertFalse(p.supportsMusicalMeta, "bpm/keyscale/timesignature/vocal_language are ACE-Step-only")
        XCTAssertTrue(p.requiresLyrics, "the model is lyric-conditioned; the server 400s empty lyrics")
        XCTAssertEqual(p.durationRange, 5...360)
        XCTAssertFalse(p.isLocalOnly)
        XCTAssertEqual(p.repo, "ddalcu/MiniMax-Music3-MLX-Serve-8bit")
        XCTAssertTrue(MusicModelPreset.all.contains(p), "catalog must offer it")
        // ACE keeps its full knob set.
        let ace = MusicModelPreset.acestepXLTurbo8bit
        XCTAssertTrue(ace.supportsMusicalMeta)
        XCTAssertFalse(ace.requiresLyrics)
        XCTAssertEqual(ace.durationRange, 10...600)
    }

    func testRequestBodyDropsOnlyTheUndocumentedAcestepFieldsForMusic3() {
        // Only the two the server still names a 400 are dropped. `bpm` and
        // `keyscale` used to be in that list, wrongly: MiniMax's card lists BPM
        // and key under Global Metadata, so the server folds them into the
        // caption instead of refusing them. Meter and vocal language have no
        // documented equivalent, so they stay gated at the FIELD level — values
        // linger in @State across a model switch.
        let req = MusicGenRequest(
            model: .miniMaxMusic3_8bit,
            prompt: "upbeat synthwave",
            lyrics: "[verse]\nneon lights",
            vocalLanguage: "en",
            bpm: 128,
            keyscale: "F# minor",
            timesignature: "4",
            durationSeconds: 60,
            seed: 7
        )
        let body = MusicGenService.requestBody(req, modelName: "m")
        XCTAssertEqual(body["bpm"] as? Int, 128)
        XCTAssertEqual(body["keyscale"] as? String, "F# minor")
        XCTAssertNil(body["timesignature"])
        XCTAssertNil(body["vocal_language"])
        XCTAssertEqual(body["lyrics"] as? String, "[verse]\nneon lights")
        XCTAssertEqual(body["prompt"] as? String, "upbeat synthwave")
    }

    func testRequestBodyClampsDurationToTheModelsRange() {
        // Sticky settings persist across model switches: ACE's 600 s against
        // music3's 360 cap (and ACE's own 10 s floor) must clamp, not 400.
        let long = MusicGenRequest(model: .miniMaxMusic3_8bit, prompt: "p", lyrics: "l", durationSeconds: 600)
        XCTAssertEqual(MusicGenService.requestBody(long, modelName: "m")["duration_seconds"] as? Int, 360)
        let short = MusicGenRequest(model: .acestepXLTurbo8bit, prompt: "p", durationSeconds: 5)
        XCTAssertEqual(MusicGenService.requestBody(short, modelName: "m")["duration_seconds"] as? Int, 10)
    }

    func testSettingsSidecarOmitsOnlyTheUndocumentedAcestepFieldsForMusic3() {
        let req = MusicGenRequest(
            model: .miniMaxMusic3_8bit,
            prompt: "synthwave",
            lyrics: "[verse]\nla",
            vocalLanguage: "en",
            bpm: 128,
            keyscale: "F# minor",
            timesignature: "4",
            durationSeconds: 30,
            seed: 3
        )
        let text = MusicGenService.settingsText(req, resolvedSeed: 3, modelName: "m")
        XCTAssertTrue(text.contains("bpm: 128"))
        XCTAssertTrue(text.contains("keyscale: F# minor"))
        XCTAssertFalse(text.contains("timesignature:"))
        XCTAssertFalse(text.contains("vocal_language:"))
        XCTAssertTrue(text.contains("# Lyrics\n[verse]"))
    }

    func testMusic3FamilyResolvesFromArchitecture() {
        // A downloaded pack (any dir name) surfaces in the pane with music3's
        // gated knob set — resolution keys on the server-reported arch.
        let models = [ModelInfo(name: "ddalcu/MiniMax-Music3-MLX-Serve-8bit",
                                quantBits: 8, layers: 0, hiddenSize: 0, vocabSize: 0,
                                contextLength: 0, modelMaxTokens: 0,
                                architecture: "minimax_music3",
                                capabilities: ["audio", "music"])]
        let p = CustomMediaModels.musicPreset(for: "ddalcu/MiniMax-Music3-MLX-Serve-8bit", from: models)
        XCTAssertEqual(p?.family, .minimaxMusic3)
        XCTAssertEqual(p?.repo, "ddalcu/MiniMax-Music3-MLX-Serve-8bit")
        XCTAssertFalse(p?.supportsMusicalMeta ?? true)
    }

    func testMusic3SelectionAgainstPackListing() {
        // Simulated HF tree of the published pack: the selection must pull the
        // five safetensors + config + both tokenizer dirs and skip the junk;
        // every readiness marker must be in the downloaded set.
        let entries: [[String: Any]] = [
            ["path": ".gitattributes", "type": "file", "size": 1600],
            ["path": "LICENSE", "type": "file", "size": 9000],
            ["path": "README.md", "type": "file", "size": 4000],
            ["path": "config.json", "type": "file", "size": 1400],
            ["path": "language_model.safetensors", "type": "file", "size": 9_890_000_000],
            ["path": "rvq_depth_decoder.safetensors", "type": "file", "size": 710_000_000],
            ["path": "transformer.safetensors", "type": "file", "size": 2_600_000_000],
            ["path": "condition_encoder.safetensors", "type": "file", "size": 100_000_000],
            ["path": "vocoder.safetensors", "type": "file", "size": 220_000_000],
            ["path": "tokenizer/tokenizer.json", "type": "file", "size": 11_400_000],
            ["path": "tokenizer/tokenizer_config.json", "type": "file", "size": 377],
            ["path": "tokenizer/chat_template.jinja", "type": "file", "size": 4168],
            ["path": "music_tokenizer/tokenizer.json", "type": "file", "size": 11_400_000],
            ["path": "music_tokenizer/vocab.json", "type": "file", "size": 2_700_000],
            ["path": "music_tokenizer/merges.txt", "type": "file", "size": 1_670_000],
        ]
        let comp = MusicModelPreset.miniMaxMusic3_8bit.bundle.components[0]
        let picked = DownloadManager.selectNeededFiles(from: entries, selection: comp.selection).map(\.0)
        for f in ["config.json", "language_model.safetensors", "rvq_depth_decoder.safetensors",
                  "transformer.safetensors", "condition_encoder.safetensors", "vocoder.safetensors",
                  "tokenizer/tokenizer.json"] {
            XCTAssertTrue(picked.contains(f), "\(f) must be downloaded")
        }
        XCTAssertFalse(picked.contains("README.md"))
        for marker in comp.readyMarkers {
            XCTAssertTrue(picked.contains(marker), "readiness marker \(marker) not downloaded")
        }
    }
}

// MARK: - Cover weights (fsq.safetensors) declared at the point of use

/// #269: a pack downloaded before cover mode lacks `fsq.safetensors`, and the
/// only signal was a 400 after the user had already attached a source track.
/// The Music tab must say so by name, offer the ONE file, and never offer a
/// button that cannot work.
final class CoverWeightsFetchTests: XCTestCase {

    private func tempDir() throws -> String {
        let dir = (NSTemporaryDirectory() as NSString)
            .appendingPathComponent("mlx-serve-fsq-\(UUID().uuidString)")
        try FileManager.default.createDirectory(atPath: dir, withIntermediateDirectories: true)
        return dir
    }

    func testFileNameIsTheOneConstant() {
        // One spelling, or the app fetches a file the server never looks for.
        XCTAssertEqual(CoverWeightsFetch.fileName, "fsq.safetensors")
        XCTAssertEqual(MusicGenService.coverWeightsFile, CoverWeightsFetch.fileName)
        // The size quoted in the UI is the real file (420,148,465 bytes).
        XCTAssertEqual(CoverWeightsFetch.approxMB, 420)
    }

    func testDecisionOnlyAppliesToCoverOnASourceAudioModel() throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(atPath: dir) }
        // Vocal to BGM does NOT read the FSQ tokenizer — the server only gates
        // task "cover" on it, and `complete` builds its context straight from
        // the source latents. It must never be disabled by this.
        for task in [MusicTask.text2music, .complete] {
            XCTAssertEqual(
                CoverWeightsFetch.decide(task: task, modelSupportsSourceAudio: true,
                                         isRemote: false, packDir: dir, fetching: false),
                .notApplicable, "\(task) must not be gated on \(CoverWeightsFetch.fileName)")
        }
        // A model with no source-audio tasks has no cover mode at all.
        XCTAssertEqual(
            CoverWeightsFetch.decide(task: .cover, modelSupportsSourceAudio: false,
                                     isRemote: false, packDir: dir, fetching: false),
            .notApplicable)
    }

    func testMissingWritablePackOffersTheFetchAndPresentSaysReady() throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(atPath: dir) }
        XCTAssertEqual(
            CoverWeightsFetch.decide(task: .cover, modelSupportsSourceAudio: true,
                                     isRemote: false, packDir: dir, fetching: false),
            .fetch)
        try Data("x".utf8).write(to: URL(fileURLWithPath:
            (dir as NSString).appendingPathComponent(CoverWeightsFetch.fileName)))
        XCTAssertEqual(
            CoverWeightsFetch.decide(task: .cover, modelSupportsSourceAudio: true,
                                     isRemote: false, packDir: dir, fetching: false),
            .ready)
    }

    func testInFlightFetchOutranksMissing() throws {
        let dir = try tempDir()
        defer { try? FileManager.default.removeItem(atPath: dir) }
        XCTAssertEqual(
            CoverWeightsFetch.decide(task: .cover, modelSupportsSourceAudio: true,
                                     isRemote: false, packDir: dir, fetching: true),
            .downloading)
    }

    func testUnwritablePackSaysWhereRatherThanOfferingADeadButton() throws {
        let dir = try tempDir()
        defer {
            _ = try? FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: dir)
            try? FileManager.default.removeItem(atPath: dir)
        }
        try FileManager.default.setAttributes([.posixPermissions: 0o555], ofItemAtPath: dir)
        XCTAssertEqual(
            CoverWeightsFetch.decide(task: .cover, modelSupportsSourceAudio: true,
                                     isRemote: false, packDir: dir, fetching: false),
            .missingUnwritable(dir: dir))
    }

    func testRemoteModelIsNotOursToComplete() {
        XCTAssertEqual(
            CoverWeightsFetch.decide(task: .cover, modelSupportsSourceAudio: true,
                                     isRemote: true, packDir: nil, fetching: false),
            .unavailableRemotely)
    }

    func testAbsentPackDirIsNotAnOffer() {
        // Not installed at all — the model row's own Download button covers it;
        // offering one file into a pack that isn't there would fail.
        XCTAssertEqual(
            CoverWeightsFetch.decide(task: .cover, modelSupportsSourceAudio: true,
                                     isRemote: false, packDir: nil, fetching: false),
            .notApplicable)
    }

    func testEveryDeclaringStateNamesTheFileAndTheSize() {
        // "The user must be able to read what is wrong without clicking."
        for decision: CoverWeightsFetch.Decision in [.fetch, .missingUnwritable(dir: "/tmp/pack"),
                                                     .unavailableRemotely] {
            let text = CoverWeightsFetch.notice(decision)
            XCTAssertNotNil(text, "\(decision) must say something")
            XCTAssertTrue(text?.contains(CoverWeightsFetch.fileName) ?? false,
                          "\(decision) must name the file: \(text ?? "nil")")
        }
        // The offer states the size AND that it is not the whole model again.
        let offer = CoverWeightsFetch.notice(.fetch) ?? ""
        XCTAssertTrue(offer.contains("420 MB"), offer)
        XCTAssertTrue(offer.lowercased().contains("whole model"), offer)
        // The unwritable state says WHERE to put it.
        XCTAssertTrue(CoverWeightsFetch.notice(.missingUnwritable(dir: "/tmp/pack"))?
            .contains("/tmp/pack") ?? false)
        // Nothing to say when there is nothing wrong.
        XCTAssertNil(CoverWeightsFetch.notice(.ready))
        XCTAssertNil(CoverWeightsFetch.notice(.notApplicable))
    }

    func testTheModeLabelItselfDeclaresTheMissingFile() {
        // Requirement 1: the Cover option declares it in its OWN label, so the
        // reason is readable before the mode is even selected.
        XCTAssertEqual(CoverWeightsFetch.modeLabel(MusicTask.cover, decision: .ready), "Cover")
        XCTAssertEqual(CoverWeightsFetch.modeLabel(MusicTask.cover, decision: .notApplicable), "Cover")
        for decision: CoverWeightsFetch.Decision in [.fetch, .downloading,
                                                     .missingUnwritable(dir: "/d"), .unavailableRemotely] {
            XCTAssertTrue(CoverWeightsFetch.modeLabel(.cover, decision: decision)
                            .contains(CoverWeightsFetch.fileName),
                          "\(decision) label: \(CoverWeightsFetch.modeLabel(.cover, decision: decision))")
        }
        // Never decorate the modes this has nothing to do with.
        XCTAssertEqual(CoverWeightsFetch.modeLabel(.complete, decision: .fetch), "Vocal to BGM")
        XCTAssertEqual(CoverWeightsFetch.modeLabel(.text2music, decision: .fetch), "Text to music")
    }

    /// The wiring a unit test cannot reach: the pane must actually render the
    /// notice, offer the fetch, and allow cancel (the source-scan idiom).
    func testMusicPaneWiresTheCoverWeightsNotice() throws {
        let url = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/MLXServe/Views/MusicGenView.swift")
        let src = try String(contentsOf: url, encoding: .utf8)
        for needle in ["CoverWeightsFetch.decide(", "CoverWeightsFetch.notice(",
                       "CoverWeightsFetch.modeLabel(",
                       "downloads.startPackFile(", "downloads.cancelPackFile("] {
            XCTAssertTrue(src.contains(needle), "MusicGenView does not wire \(needle)")
        }
    }

    /// The server's own gate must keep spelling the file the app fetches, and
    /// must keep gating ONLY cover on it.
    func testServerGatesOnlyCoverOnTheSameFileName() throws {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        let gen = try String(contentsOf: root.appendingPathComponent("src/gen.zig"), encoding: .utf8)
        XCTAssertTrue(gen.contains("task == .cover and !music.fsqAvailable()"),
                      "src/gen.zig no longer gates cover on the FSQ tokenizer")
        let ace = try String(contentsOf: root.appendingPathComponent("src/acestep.zig"), encoding: .utf8)
        XCTAssertTrue(ace.contains("FSQ_FILE = \"\(CoverWeightsFetch.fileName)\""),
                      "src/acestep.zig no longer reads \(CoverWeightsFetch.fileName)")
    }
}
