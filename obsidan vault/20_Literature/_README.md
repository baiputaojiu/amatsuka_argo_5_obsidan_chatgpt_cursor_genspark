# 📚 Literature Notes

**外部ソースから得た知識**を自分の言葉でまとめる場所。

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
    style E fill:#32cd32,stroke:#7cfc00,color:#000,stroke-width:3px
    style F fill:#dc143c,stroke:#ff6b6b,color:#fff
    style G fill:#9932cc,stroke:#da70d6,color:#fff
    style H fill:#4169e1,stroke:#6495ed,color:#fff
    style I fill:#2f4f4f,stroke:#708090,color:#fff
```

> **📍 現在地: Literature Notes** - 外部知識を自分の言葉で整理します

---

## 何を入れる？
- 本・記事の要約・引用
- YouTube動画のメモ・学び
- ChatGPTから得た知識の整理
- ポッドキャスト・講義のメモ

## ルール
- 必ず出典を明記する
- コピペではなく自分の言葉で書く
- 重要な箇所はPermanent Notesへ発展
