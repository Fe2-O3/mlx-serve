import Foundation

/// Deep links from the app into the website's LLM tier list
/// (`https://mlxserve.com/llm-tier-list/`). The page gives every model row the
/// DOM id `tier-<seed-id>` and scrolls to `#tier-<seed-id>` on load, so a repo
/// only has to name its row. Mirror: the page's `SEED_MODELS` ids in
/// `website/llm-tier-list/index.html` — a row renamed there breaks the link,
/// which `TierListLinksTests` pins from this side.
enum TierListLinks {
    static let pagePath = "https://mlxserve.com/llm-tier-list/"

    /// The tier-list row id for a repo, or nil when the page has no such row.
    static func anchor(repoId: String) -> String? {
        switch repoId {
        case "mlx-community/gemma-4-e4b-it-4bit": "gemma-4-e4b"
        case "mlx-community/gemma-4-12b-it-4bit": "gemma-4-12b"
        case "mlx-community/gemma-4-26b-a4b-it-4bit",
             "mlx-community/gemma-4-26b-a4b-it-8bit": "gemma-4-26b-a4b"
        case "mlx-community/gemma-4-31b-it-4bit": "gemma-4-31b"
        case "ddalcu/MiMo-V2.6-Distill-Qwen-9B-MLX-Serve-4bit": "mimo-v2.6-9b"
        case "ddalcu/Qwen3.8-27B-MLX-Serve-4bit": "qwen3.8-27b"
        case "prism-ml/Ternary-Bonsai-2-27B-mlx-2bit": "ternary-bonsai-2-27b"
        case "ddalcu/DeepSeek-V4-Flash-0731-iQ-MLX-3.3bpw": "deepseek-v4-flash"
        case "ddalcu/Qwen3.6-35B-A3B-MLX-Serve-4bit": "qwen3.6-35b-a3b"
        case "ddalcu/Qwen3.8-Flash-Next-MLX-Serve-mixed-4-8bit": "qwen3.8-flash-next"
        default: nil
        }
    }

    /// The full deep link, or nil when the page has no such row.
    static func url(repoId: String) -> URL? {
        guard let anchor = anchor(repoId: repoId) else { return nil }
        return URL(string: pagePath + "#tier-" + anchor)
    }
}
