# 🧠 Permanent Notes

Zettelkastenの本体。永続的な知識ノートを保管する場所です。

## Zettelkasten ワークフロー

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': { 'primaryColor': '#1a1a2e', 'primaryTextColor': '#eaeaea', 'primaryBorderColor': '#4a4a6a', 'lineColor': '#8b8bab', 'secondaryColor': '#16213e', 'tertiaryColor': '#0f3460'}}}%%
flowchart TD
    subgraph Input["📥 インプット"]
        A[("💡 アイデア・思考")]
        B[("📚 本・記事・動画")]
    end

    subgraph Capture["🗂️ キャプチャ"]
        C["00_Inbox<br/>未処理の情報"]
    end

    subgraph Processing["⚙️ 処理"]
        D["10_Fleeting Notes<br/>一時的なメモ・着想"]
        E["20_Literature Notes<br/>文献からの要約・引用"]
    end

    subgraph Knowledge["🧠 知識構築"]
        F["30_Permanent Notes<br/>自分の言葉で書いた<br/>原子的なノート"]
    end

    subgraph Organization["🗺️ 整理・構造化"]
        G["40_MOCs<br/>Map of Content<br/>テーマ別インデックス"]
        H["50_Projects<br/>プロジェクト成果物"]
    end

    subgraph Archive["📦 アーカイブ"]
        I["90_Archives<br/>完了・非アクティブ"]
    end

    A --> C
    B --> C
    C --> D
    C --> E
    D --> F
    E --> F
    F <-->|"🔗 リンク"| F
    F --> G
    G --> H
    H --> I

    style C fill:#b8860b,stroke:#ffd700,color:#fff
    style D fill:#1e90ff,stroke:#87ceeb,color:#fff
    style E fill:#228b22,stroke:#90ee90,color:#fff
    style F fill:#ff4500,stroke:#ff6347,color:#fff,stroke-width:3px
    style G fill:#9932cc,stroke:#da70d6,color:#fff
    style H fill:#4169e1,stroke:#6495ed,color:#fff
    style I fill:#2f4f4f,stroke:#708090,color:#fff
```

> **📍 現在地: Permanent Notes** - Zettelkastenの心臓部、知識が蓄積される場所です

---

## 使い方
- 1つのノートに1つのアイデア（Atomic Notes）
- 他のノートへのリンクを積極的に作る
- 自分の言葉で、未来の自分に説明するように書く

## ルール
- 完全な文章で書く（箇条書きではなく）
- 他のノートとの関連性を意識する
- タイトルは内容を表す主張・質問にする

## ファイル命名規則（推奨）
- `YYYYMMDDHHMMSS タイトル.md`（タイムスタンプ）
- または意味のあるタイトルのみ
