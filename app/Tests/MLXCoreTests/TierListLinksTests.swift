import XCTest
@testable import MLXCore

/// Deep links from the app into the website's LLM tier list. The page gives
/// every model row the DOM id `tier-<seed-id>`, so the contract under test is
/// just: each pane repo names a row that exists, and the URL lands on it.
final class TierListLinksTests: XCTestCase {
    func testEveryPaneRepoNamesATierListRow() {
        for pick in RecommendedModelPick.allCatalogs {
            XCTAssertNotNil(TierListLinks.anchor(repoId: pick.repoId),
                            "\(pick.id) (\(pick.repoId)) has no tier-list row — add one to SEED_MODELS or a nil case with a reason")
        }
    }

    func testAnchorsAreThePageSeedIdsVerbatim() {
        XCTAssertEqual(TierListLinks.anchor(repoId: "ddalcu/Qwen3.8-27B-MLX-Serve-4bit"), "qwen3.8-27b")
        XCTAssertEqual(TierListLinks.anchor(repoId: "ddalcu/MiMo-V2.6-Distill-Qwen-9B-MLX-Serve-4bit"), "mimo-v2.6-9b")
        XCTAssertEqual(TierListLinks.anchor(repoId: "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit"), "ternary-bonsai-2-27b")
        XCTAssertEqual(TierListLinks.anchor(repoId: "mlx-community/gemma-4-e4b-it-4bit"), "gemma-4-e4b")
        XCTAssertEqual(TierListLinks.anchor(repoId: "ddalcu/DeepSeek-V4-Flash-0731-iQ-MLX-3.3bpw"), "deepseek-v4-flash")
    }

    func testURLsLandOnTheTierListPageAtTheRow() {
        let url = TierListLinks.url(repoId: "ddalcu/Qwen3.8-27B-MLX-Serve-4bit")
        XCTAssertEqual(url?.absoluteString, "https://mlxserve.com/llm-tier-list/#tier-qwen3.8-27b")
    }

    func testUnknownReposLinkNowhere() {
        XCTAssertNil(TierListLinks.anchor(repoId: "someone/else-1B-MLX-4bit"))
        XCTAssertNil(TierListLinks.url(repoId: "someone/else-1B-MLX-4bit"))
    }
}
