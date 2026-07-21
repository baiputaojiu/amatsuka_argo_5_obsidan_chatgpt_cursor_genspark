# 実装状況

## 最終更新

- 日時：2026-07-21
- 状態：Round2 Fix 01～08 を実装。通常pytest **83 passed**、ruff、mypy、alembic check差分0を確認
- MVP判定：`READY_FOR_REAL_SAMPLE_DIRECTIONAL_SLICE`（詳細は `FINAL_REVIEW.md`）
- 重大ブロッカー：なし
- 推奨する次の一作業：実Vaultで原文1件の縦断目視確認
- commit / push：未実施

## Round2 完了状況

### Fix 01 — 状態機械（完了済み・維持）

- P11→awaiting_review→RUN_P12、P12 disagreed→awaiting_adjudication→RUN_P13
- `domain/resolution.py` + `test_state_machine_r2.py`

### Fix 02 — 非YouTubeと再利用

- P07（blog/X/web）Schema・取込み・workflow
- P08 `upstream_artifact_id` / `upstream_prompt_id`（`p05_artifact_id`後方互換）
- `can_reuse_processed_artifact` を raw import へ接続

### Fix 03 — AIレビュー解決

- P06/P09 Schema・取込み
- accept/correct/reject/unresolved、lineage、冪等materialize

### Fix 04 — 話者・時間整合

- forecastへsegment参照・attribution・statement_kind・made_at_source
- not_target / third_party_summary は正式成績へ入れない

### Fix 05 — 修正候補とバスケット

- `candidate_origin`（p11_proposal / p12_correction）でlock
- 同一通貨weight合成、mixed currencyは unevaluable_mixed_currency

### Fix 06 — prompt / 案内品質

- P05〜P13・P06/P09 template拡充、catalog更新
- `analyst-forecast config set-model` / init `--cursor-model`

### Fix 07 — migration / 品質ゲート

- Alembic `0006`（evaluation複合index維持）
- ruff / mypy / pytest / alembic check = 0 diffs

### Fix 08 — 最終監査

- 本ファイルと `FINAL_REVIEW.md` を実測更新

## 検査結果（Round2後）

- pytest unit：83 passed
- ruff format / check：pass
- mypy src（strict）：pass
- Alembic empty→head / 0001→head：pass
- alembic check：差分0
- ネットワーク integration：external_blocked
- wheel clean venv：未実施

## 未完了・限定

- 実原文の精度実証
- wheelクリーン導入確認
- 総合点・能力ランキング・PNG・1/3/6/12観測など仕様上の将来項目
