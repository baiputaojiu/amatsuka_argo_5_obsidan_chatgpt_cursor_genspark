# ワークフロー・構成図

## 1. 全体フロー

```mermaid
flowchart TD
    A["案件作成"] --> B["情報源・原文収集"]
    B --> C["AIで整理・予想抽出"]
    C --> D["対象解決・AI検証"]
    D --> E["Pythonで市場評価"]
    E --> F["能力・現在予想を集計"]
    F --> G["Obsidianレポート"]
```

## 2. AIとPython

```mermaid
flowchart LR
    U["ユーザー"] --> T["ターミナル"]
    T --> A["Cursor／ChatGPT"]
    A --> I["AI出力Inbox"]
    I --> P["Python検証・計算"]
    P --> O["SQLite・Obsidian"]
    P --> T
```

## 3. YouTube処理

```mermaid
flowchart TD
    A["変更禁止の文字起こし"] --> B["段落・発言分割"]
    B --> C["話者推定"]
    C --> D{"確信度・重要度"}
    D -->|要確認| E["別AIレビュー"]
    D -->|十分| F["話者付き原文"]
    E --> F
    F --> G["予想抽出"]
```

## 4. 予測対象解決

```mermaid
flowchart TD
    A["原文の予測対象"] --> B{"具体的銘柄等あり"}
    B -->|あり| C["原文を優先"]
    B -->|なし| D["AIが時点候補を提案"]
    C --> E["別AIで検証"]
    D --> E
    E --> F{"一致"}
    F -->|一致| G["マッピング固定"]
    F -->|不一致| H["AI裁定"]
    H --> G
    G --> I["Python市場評価"]
```

## 5. 統一収集と派生表示

```mermaid
flowchart TD
    A["指定期間の予想を一括収集"] --> B["統一予想台帳"]
    B --> C["評価基準日時点の状態計算"]
    C --> D["現在有効ビュー"]
    C --> E["評価完了ビュー"]
    C --> F["条件待ち・評価不能ビュー"]
```

過去実績用と現在予想用に収集期間を分けない。

## 6. 状態遷移の概略

```mermaid
stateDiagram-v2
    [*] --> not_started
    not_started --> active_on_track
    not_started --> active_off_track
    not_started --> condition_pending
    active_on_track --> fulfilled_early
    active_on_track --> expired_hit
    active_on_track --> expired_miss
    active_off_track --> active_on_track
    active_off_track --> expired_hit
    active_off_track --> expired_miss
    condition_pending --> active_indeterminate
    active_indeterminate --> expired_hit
    active_indeterminate --> expired_miss
```

撤回、置換、評価不能は各評価中状態から遷移可能とする。

## 7. 次行動案内

```mermaid
flowchart TD
    A["DB・ファイル・タスク依存を確認"] --> B["実行可能タスクを列挙"]
    B --> C["推奨順位を決定"]
    C --> D["担当・理由・入出力を表示"]
    D --> E["NEXT_ACTIONS.mdを更新"]
    E --> F["Cursorも同じ状態を参照"]
```

## 8. 現在予想から上昇候補

```mermaid
flowchart TD
    A["現在有効な予想"] --> B["対象・期間を統一"]
    B --> C["分野・期間別の過去成績"]
    C --> D["サンプル数・具体性を反映"]
    D --> E["上昇・下落双方を集計"]
    E --> F["根拠付き候補一覧"]
```

## 9. 可視化

### タイムライン

- 横軸：時期。
- 縦軸：分析対象者。
- フィルター：予測対象。
- 緑＋上矢印：上昇。
- 赤＋下矢印：下落。
- 点の大きさ：具体性。
- 線の長さ：予想期間。

### ヒートマップ

- 横軸：時期。
- 縦軸：予測対象。
- 分析対象者：フィルター。
- 色と記号を併用する。

3次元グラフは採用せず、2次元図とフィルターに分解する。

