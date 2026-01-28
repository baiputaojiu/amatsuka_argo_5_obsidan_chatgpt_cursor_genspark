# 🗺️ Maps of Content (MOCs)

トピックごとのインデックスページを保管する場所です。

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
    style F fill:#dc143c,stroke:#ff6b6b,color:#fff
    style G fill:#ba55d3,stroke:#ee82ee,color:#fff,stroke-width:3px
    style H fill:#4169e1,stroke:#6495ed,color:#fff
    style I fill:#2f4f4f,stroke:#708090,color:#fff
```

> **📍 現在地: MOCs** - Permanent Notesをテーマ別に整理するマップです

---

## 使い方
- 関連するノートをテーマ別にまとめる
- ノート間のナビゲーションを容易にする
- 新しいアイデアの発見を促す

## ルール
- トピックごとに1つのMOCを作る
- 定期的に更新する
- リンク切れをチェックする

## 例
- `プログラミング MOC.md`
- `読書 MOC.md`
- `プロジェクト管理 MOC.md`
