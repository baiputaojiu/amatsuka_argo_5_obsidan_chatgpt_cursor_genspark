# 第3次修正指示 — START HERE

## 対象

- Repository: baiputaojiu/amatsuka_argo_5_obsidan_chatgpt_cursor_genspark
- Review branch: fetch2_1_アナリスト調査の計画を追加
- Review commit: e165a6c8142fc6085caa6e73fb2da6e303abe3e9
- Python project root: repository直下の analyst_forecast_app_cursor_handoff/
- Review verdict: NEEDS_CHANGES_BEFORE_REAL_SAMPLE

Round2では、基本状態機械、P07、P06/P09、対象解決、同一通貨バスケット、migrationが大きく改善されました。一方、独立レビューでは、話者誤認、後発情報の逆流、P09修正版の検証迂回、複数情報源の取り残し、バスケットcache汚染、Vault説明書の要約化を再現しました。

今回の目的は機能追加ではありません。実原文を入れる前に、予想件数・的中率・市場評価を誤らせる経路を閉じることです。

## 配置場所

このフォルダを次の位置に配置してください。

    analyst_forecast_app_cursor_handoff/
    ├─ pyproject.toml
    ├─ src/
    ├─ tests/
    ├─ docs/
    └─ Fix_instructions/
       └─ analyst_forecast_cursor_fix_instructions_round3/

現在のproject rootを再び二重階層にしないでください。

## Cursorでの実行順序

原則として1ファイルを1セッションで実行してください。

1. 01_PREFLIGHT_AND_REPRODUCTION.md
2. 02_SPEAKER_ATTRIBUTION_AND_EVIDENCE_FIX.md
3. 03_TIME_AND_KNOWLEDGE_CUTOFF_FIX.md
4. 04_AI_CORRECTION_AND_REVIEW_FIX.md
5. 05_MULTI_SOURCE_WORKFLOW_AND_REUSE_FIX.md
6. 06_BASKET_CACHE_AND_CSV_FIX.md
7. 07_VAULT_DOCS_AND_PROMPTS_FIX.md
8. 08_MIGRATION_PACKAGE_AND_QUALITY_GATE.md
9. 09_FINAL_AUDIT.md
10. 10_ACCEPTANCE_MATRIX.mdで最終照合

各ファイルの「Cursorへ渡す依頼文」だけでも実行できますが、本フォルダ全体、既存docs、Round2 FINAL_REVIEWを参照可能にしてください。

## 全工程の拘束条件

- 意味判断を伴うAI処理は、ユーザーが設定した高性能モデルだけを使う。廉価モデルへ自動切替しない。
- 人間承認を必須にしない。曖昧な判断は別AIレビュー、必要時は別AI裁定で解決する。
- raw原文は変更禁止。正規化、segment、予想、修正版は別artifactとして保存する。
- 話者推定、予想抽出、対象候補、対象レビュー、裁定へ市場結果を渡さない。
- 対象解決には発言時点より後の情報を使わない。取得日やAI実行日を発言日として扱わない。
- SQLiteを機械処理の正本とし、Markdown、CSV、HTMLは再生成可能な表示物とする。
- 既存Schema、既存DB、既存案件を削除しない。DB変更は新しいAlembic revisionで行う。
- legacyデータを本人確認済みに昇格させない。由来不明は由来不明のまま保存し、正式集計から除外する。
- reference/CHAT_HISTORY.pdfは削除・改変しない。
- API key、token、実Vault絶対パス、raw、DB、市場cache、AI実データをGitへ追加しない。
- ループエンジニアリングは導入しない。
- commit、push、branch作成、PR作成を行わない。
- 指示外の大規模リファクタリングをしない。必要なら着手前に理由、代替案、影響範囲を報告する。

## Round3で変更してよいもの

- P05、P06、P07、P08、P09のSchemaと検証
- 話者・著者・引用元のDB lineage
- source、forecast issuance、evidence、AI artifactの状態
- workflowのsource単位判定
- 市場series cacheとbasket audit
- CSV providerのbasket対応
- packaged resources/docsとprompt
- Alembic revision、unit test、integration test、説明書

## 今回の範囲外

- 総合点・能力ランキング
- 複数アナリストの現在予想統合
- 上昇候補ランキング
- 時期・程度・早期実現の完全採点
- 期間不明予想の1・3・6・12か月自動観測
- PNG、ヒートマップ
- 情報源の完全自動収集
- exe化

これらを便乗実装せず、既存FUTURE_ROADMAPの詳細記載を維持してください。

## 各セッションの報告形式

    対象指示:
    修正前に再現した問題:
    変更ファイル:
    DB migration:
    Schema変更:
    追加したnegative test:
    追加したpositive test:
    実行した品質ゲート:
    未完了・限定:
    次の指示へ進めるか:
    commit / push: 未実施

## 完了判定

- NOT_READY: Round3受入項目に1件でもfailまたは未検証がある。
- READY_FOR_REAL_SAMPLE_DIRECTIONAL_SLICE: 10_ACCEPTANCE_MATRIXの全項目がpassし、実原文1件の手動投入へ進める。
- FULL_MVP_READY: 今回は原則使用しない。正式仕様の全MVP要件を満たす場合だけ使用する。

