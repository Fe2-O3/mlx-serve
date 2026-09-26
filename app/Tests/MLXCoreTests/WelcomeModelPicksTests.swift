import XCTest
@testable import MLXCore

/// The welcome screen lists the best model of each type that fits this Mac. It
/// must (a) pick the largest fitting model per family, (b) drop families where
/// nothing fits, and (c) carry a one-line strength.
final class WelcomeModelPicksTests: XCTestCase {
    private let gib: UInt64 = 1_073_741_824
    private func mac(total: UInt64, usable: UInt64) -> SystemMemoryInfo {
        SystemMemoryInfo(totalBytes: total * gib, usableBytes: usable * gib)
    }

    func testTwentyFourGBMacGetsGemma12BAndMimo9B() {
        let picks = WelcomeModelPicks.forMemory(mac(total: 24, usable: 16))
        // General → Gemma 4 12B (Muse needs ~25.7 GB, exceeds 16 usable).
        XCTAssertEqual(picks.first { $0.category == "General" }?.pick.id, "gemma-4-12b")
        // Coding & agents → MiMo distill 9B (27B needs ~21.8 GB, exceeds).
        XCTAssertEqual(picks.first { $0.category == "Coding & agents" }?.pick.id, "mimo-distill-9b")
        XCTAssertEqual(picks.count, 2)
    }

    func testLargeMacGetsTheBiggestOfEachType() {
        let picks = WelcomeModelPicks.forMemory(mac(total: 256, usable: 200))
        XCTAssertEqual(picks.first { $0.category == "General" }?.pick.id, "muse-glimmer-30b")
        XCTAssertEqual(picks.first { $0.category == "Coding & agents" }?.pick.id, "qwen38-27b")
        XCTAssertNil(picks.first { $0.pick.id == "qwen38-flash-next" }, "Largest is a browser-only tier, not a welcome category")
        XCTAssertNil(picks.first { $0.pick.id == "gemma-4-26b-a4b-8bit" }, "the 8-bit build is browser-only")
        XCTAssertNil(picks.first { $0.pick.id == "qwen36-35b-a3b" }, "the 3.6 MoE is browser-only")
        XCTAssertEqual(picks.count, 2)
    }

    /// A 32 GB Mac (usable ~27): Gemma 4 31B and Qwen 3.8 27B are the
    /// biggest COMFORTABLE fits. Muse would land as a tight fit there, and
    /// the welcome leads with comfort — a tight fit is what fails under real
    /// memory pressure.
    func testThirtyTwoGBMacGetsGemma31BAndQwen27B() {
        let picks = WelcomeModelPicks.forMemory(mac(total: 32, usable: 27))
        XCTAssertEqual(picks.first { $0.category == "General" }?.pick.id, "gemma-4-31b")
        XCTAssertEqual(picks.first { $0.category == "Coding & agents" }?.pick.id, "qwen38-27b")
        XCTAssertEqual(picks.count, 2)
    }

    func testEveryPickHasAOneLineStrength() {
        for p in WelcomeModelPicks.forMemory(mac(total: 256, usable: 200)) {
            XCTAssertFalse(p.strength.isEmpty)
            XCTAssertFalse(p.strength.contains("\n"), "strength must be a single short line")
        }
    }

    func testTinyMacStillGetsAtLeastAGeneralModel() {
        // 8 GB: usable ~6. Only the smallest Gemma fits; coding families drop.
        let picks = WelcomeModelPicks.forMemory(mac(total: 8, usable: 6))
        XCTAssertEqual(picks.first { $0.category == "General" }?.pick.id, "gemma-4-e4b")
    }
}
