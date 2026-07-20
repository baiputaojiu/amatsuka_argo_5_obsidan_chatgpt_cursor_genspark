# アナリスト予想検証アプリ — Cursor修正指示セット

## 1. 対象

- リポジトリ：`amatsuka_argo_5_obsidan_chatgpt_cursor_genspark`
- 対象ブランチの基準コミット：`cc9e8db6a5efcd5e79db343480319dcd4c27c99c`
- アプリルート：`analyst_forecast_app_cursor_handoff/analyst_forecast_app_cursor_handoff/`

基準コミット以後に変更がある場合、Cursorは最新コードとGit差分を先に確認し、完了済み修正を重複実装しない。

## 2. 確定事項

`reference/CHAT_HISTORY.pdf` が公開リポジトリへ含まれていることを、ユーザーは確認したうえで許容している。このファイルは削除、履歴除去、内容変更の対象にしない。

ただし、この判断は将来のAPIキー、認証情報、個人パス、調査原文、SQLite、秘密設定まで公開してよいという意味ではない。既存の秘密情報除外規則は維持する。

## 3. 使用方法

同じ高性能モデルを固定し、次の順で実行する。

1. `01_PREFLIGHT_AND_REMEDIATION_PLAN.md`
2. `02_AI_PIPELINE_AND_SCHEMA_FIX.md`
3. `03_SOURCE_IDENTITY_AND_DEDUP_FIX.md`
4. `04_WORKFLOW_AND_RESULTS_FIX.md`
5. `05_MARKET_EVALUATION_FIX.md`
6. `06_OBSIDIAN_AND_INTERACTIVE_CLI_FIX.md`
7. `07_FINAL_AUDIT.md`

一つの長いCursorセッションへ全依頼文を投入せず、原則として1ファイルにつき1セッションで実行する。各段階がテスト成功・文書更新まで終わってから次へ進む。

## 4. 共通ルール

- `docs/`、`00_Cursor引継ぎ/02_SPEC_AUTHORITY.md`、`.cursor/rules/analyst-forecast-app.mdc`を正本とする。
- 意味判断はAI、再現可能な検査・計算・状態管理はPythonという境界を維持する。
- raw原文を上書きしない。
- AI出力を検証前に正式テーブルへ入れない。
- 人間承認を必須工程へ追加しない。必要なダブルチェックは別AI工程とする。
- AI工程ではすべて高性能モデルを使用する。
- 市場結果を予測対象解決へ逆流させない。
- 取得不能値を推測しない。
- 初期MVPへループエンジニアリング基盤を導入しない。
- 無関係な既存プロジェクトやユーザー変更を編集しない。
- 自動でcommit、push、force-pushを行わない。変更とテスト結果をユーザーへ提示する。
- 仕様変更が必要な場合は、`IMPLEMENTATION_ASSUMPTIONS.md`へ隠れて追加せず、変更案と影響を報告する。

## 5. 各段階の終了報告

Cursorは毎回、次を報告する。

1. 修正した問題。
2. 変更した主要ファイル。
3. Alembic migrationの有無と互換性。
4. 追加したテスト。
5. formatter、lint、mypy、pytestの結果。
6. 残る未解決事項。
7. 次に使用する指示書。

