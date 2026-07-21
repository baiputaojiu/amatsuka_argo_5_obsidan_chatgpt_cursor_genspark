# 第2次修正指示 — START HERE

## 対象

- Repository: `baiputaojiu/amatsuka_argo_5_obsidan_chatgpt_cursor_genspark`
- Review commit: `38d6d7e22d8b49b51f81bb03a2851bc5d62debfb`
- Python project root: `analyst_forecast_app_cursor_handoff/`
- Review verdict: `NEEDS_CHANGES_BEFORE_REAL_SAMPLE`

この指示セットは、前回修正で改善された機能を維持しながら、実際の案件を止める状態遷移、媒体経路、AIレビュー、話者帰属、対象解決、バスケット評価、migration品質を修正するためのものです。

## 配置場所

このフォルダをPython project root直下へ次の名前で配置してください。

```text
analyst_forecast_app_cursor_handoff/
├─ pyproject.toml
├─ src/
├─ tests/
├─ docs/
└─ fix_instructions_round2/
```

## 実行順序

Cursorでは、原則として1ファイルを1セッションで順番に実行してください。

1. `01_PREFLIGHT_AND_STATE_MACHINE_FIX.md`
2. `02_NON_YOUTUBE_AND_REUSE_FIX.md`
3. `03_AI_REVIEW_RESOLUTION_FIX.md`
4. `04_SPEAKER_AND_TIME_INTEGRITY_FIX.md`
5. `05_TARGET_CORRECTION_AND_BASKET_EVALUATION_FIX.md`
6. `06_PROMPT_AND_USER_GUIDE_QUALITY_FIX.md`
7. `07_MIGRATION_AND_QUALITY_GATE_FIX.md`
8. `08_FINAL_AUDIT.md`
9. `09_ACCEPTANCE_MATRIX.md` で最終照合

各ファイルの「コピペ用依頼文」だけをCursorへ渡しても実行できる構成です。ただし、Cursorには本フォルダ全体と既存`docs/`を読める状態にしてください。

## 全作業に共通する拘束条件

- 意味判断を伴うAI処理は、すべてユーザーが選んだ高性能モデルで実行する。廉価モデルへ自動切替しない。
- 人間承認を必須にしない。曖昧・重要な判断は、別AIレビューと必要時のAI裁定で解決する。
- raw原文を上書き・整形・統合しない。加工結果は別artifactとして保存する。
- 市場結果を、話者推定、予想抽出、対象候補選定、対象レビュー、裁定へ逆流させない。
- SQLiteを機械処理上の正本とし、Markdown・CSVは再生成可能な表示物とする。
- 既存Schema 1.0.0、既存DB、既存案件を消さない。変更はAlembic migrationと後方互換読込みで行う。
- `reference/CHAT_HISTORY.pdf`は今回削除・改変しない。
- APIキー、実Vault絶対パス、raw、DB、AI実データをGit管理へ追加しない。
- ループエンジニアリングは今回導入しない。
- commit、push、branch作成、PR作成は行わない。
- 指示外の大規模リファクタリングを行わない。必要な場合は理由と影響範囲を先に報告する。

## Cursorからの各回の報告形式

```text
対象指示:
変更ファイル:
DB migration:
追加・変更したSchema:
追加テスト:
実行した検査と結果:
未完了:
次の指示へ進めるか:
commit / push: 未実施
```

## 今回の範囲外

次は重要ですが、この修正セットで全完成を要求しません。未実装であれば、完成済みと書かず実装状況と将来計画へ明記してください。

- 総合点・能力ランキング
- 複数アナリストの現在予想統合
- 上昇候補ランキング
- 時期・程度・早期実現の完全採点
- 期間不明予想の1・3・6・12か月自動観測
- PNGグラフ・ヒートマップ
- 情報源の完全自動収集
- exe化

今回の完了目標は、`YouTube／ブログ／X／Webの原文 → AI処理 → 対象解決 → 単一銘柄または同一通貨バスケットのCSV方向評価 → 結果確認`を、ターミナルの案内どおりに実行できる状態です。
