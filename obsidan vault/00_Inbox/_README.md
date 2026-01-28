---
date created: 2026-01-12T01:23:15 (月)
date modified: 2026-01-12T12:07:37 (月)
---
# 📥 Inbox

**外部から入ってくる情報**の一時置き場。未処理・未分類のもの。

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

    style C fill:#ffd700,stroke:#ffed4a,color:#000,stroke-width:3px
    style D fill:#1e90ff,stroke:#87ceeb,color:#fff
    style E fill:#228b22,stroke:#90ee90,color:#fff
    style F fill:#dc143c,stroke:#ff6b6b,color:#fff
    style G fill:#9932cc,stroke:#da70d6,color:#fff
    style H fill:#4169e1,stroke:#6495ed,color:#fff
    style I fill:#2f4f4f,stroke:#708090,color:#fff
```

> **📍 現在地: Inbox** - 外部情報の入り口です

---

## 何を入れる？
- 後で読むWebクリップ・記事
- 後で見るYouTube動画のURL
- ChatGPT会話ログ（とりあえず保存）
- メールや会話から得た情報
- 他アプリからの転送メモ

## Fleeting Notesとの違い
- **Inbox** = 外部から来た「未読ボックス」
- **Fleeting** = 自分の頭から出た「アイデアメモ」

## ルール
- 週1回は整理（理想は毎日）
- 処理したら適切なフォルダへ移動
