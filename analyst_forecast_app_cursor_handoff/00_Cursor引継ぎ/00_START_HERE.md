# Cursor実装引継ぎ — 最初に読むファイル

## 1. このパッケージの目的

このパッケージは、「アナリスト予想検証アプリ」をCursor上で実装開始するための引継ぎ資料です。仕様書だけでなく、最初の依頼文、仕様の優先順位、受入確認、Cursor常設ルールを含みます。

## 2. Cursorへ渡すもの

このZIPを展開し、内容を新しいGitリポジトリのルートへ置いてください。主な構成は次のとおりです。

```text
project-root/
├─ .cursor/rules/analyst-forecast-app.mdc
├─ 00_Cursor引継ぎ/
├─ docs/
├─ reference/
└─ README.md
```

会話全文も渡す場合は、リポジトリ直下の実装ファイルと混ぜず、例えば次へ保存します。

```text
reference/chat_history.md
```

詳細は [CHAT_HISTORY_GUIDE.md](../reference/CHAT_HISTORY_GUIDE.md) を参照してください。

## 3. 最初の操作

1. Cursorで展開先フォルダを開く。
2. Gitリポジトリを初期化する。既にGit管理されている場合は不要。
3. [01_INITIAL_IMPLEMENTATION_REQUEST.md](01_INITIAL_IMPLEMENTATION_REQUEST.md) の「コピペ用依頼文」だけをCursorチャットへ貼る。
4. Cursorが作成した計画と、未決事項の扱いを確認する。
5. 重大な仕様変更がなければ、そのまま実装とテストを続けさせる。

## 4. 正式仕様と参考資料

- `docs/`：正式な仕様・計画の正本。
- `00_Cursor引継ぎ/`：実装開始方法、優先順位、受入確認。
- `.cursor/rules/`：Cursorが継続して守る常設ルール。
- `reference/chat_history.md`：背景確認用の参考資料。正式仕様ではない。

仕様と会話が矛盾した場合、会話を根拠に仕様を変更してはいけません。[02_SPEC_AUTHORITY.md](02_SPEC_AUTHORITY.md) に従ってください。

## 5. 最初の実装範囲

最初から全機能を完成させません。まず次を実装します。

- Pythonプロジェクトの基盤。
- 設定、SQLite、Schema、マイグレーション。
- 案件作成とフォルダ生成。
- 原文取込み、ハッシュ、重複防止。
- AI用JSONの形式・引用検査と取込み。
- 無料市場データの技術検証。
- 小さな固定サンプルによる方向評価。
- `status.yaml`、`WORKFLOW_STATE.json`、`NEXT_ACTIONS.md`、`OPEN_ISSUES.md`。
- ターミナルでの次行動案内。

上記の縦断経路がテストで動いてから、媒体探索、複雑予想、複数アナリスト比較へ拡張します。

## 6. 実装中の基本姿勢

- 未決事項を勝手に恒久仕様へ変えない。
- 実装を止めない安全な暫定値は、設定可能にして仮定として記録する。
- 原文を絶対に上書きしない。
- 意味判断はAI、検証可能な処理はPythonという境界を守る。
- AIが返したJSONを未検証でDBへ入れない。
- 価格を取得できない場合は推測せず、CSV取込み待ちにする。
- 大規模なループエンジニアリングは初期開発へ導入しない。

## 7. セッションを再開するとき

初回以後は [03_CONTINUATION_REQUEST.md](03_CONTINUATION_REQUEST.md) を使用します。Cursorはテスト結果、作業状態、Git差分、MVP計画から次の実装を判断します。

