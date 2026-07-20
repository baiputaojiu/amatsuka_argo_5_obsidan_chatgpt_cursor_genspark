# 実装状況

## 最終更新

- 日時：2026-07-20
- 状態：修正指示 02～07 を実装し、通常pytest 64件・ruff・mypy・空DB/0001 Alembic upgrade を確認
- MVP判定：`READY_WITH_LIMITATIONS`（詳細は `FINAL_REVIEW.md`）
- 重大ブロッカー：なし
- 推奨する次の一作業：実Vaultで原文1件の縦断目視確認
- 代替案：CSV市場評価を前提に匿名fixture拡張

## 完了（02～07）

### Fix 02 — AIパイプライン・Schema分離

- P05/P08/P11/P12/P13 を Schema 2.0.0 で分離。Alembic `0002`
- Schema 1.0.0 は読込み互換。inline `review_result` は `legacy_inline_review` とし lock しない
- 案件promptに P13 を含め、中央テンプレートから snapshot 生成

### Fix 03 — raw artifact / source occurrence

- `raw_artifacts` と source occurrence（既存 `sources`）を分離。Alembic `0003`
- 同一bytesは artifact 共有、別アナリスト／URL／媒体は別 occurrence
- 案件内に変更禁止コピー＋artifact_ref manifest（symlink非必須）
- P05再利用は analyst / prompt / model / hash 一致時のみ
- 必須ケースの unit テストを追加

### Fix 04 — workflow / results

- workflow 正本を SQLite `workflow_tasks`（Alembic `0004`）へ
- needs_review はDB未解決件数、評価完了は component 単位の最新 as_of
- NEXT_ACTIONS に実 component ID と完全 evaluate コマンド
- `04_results` の5成果物を SQLite から生成（`application/results.py`）
- run/component 所属検証

### Fix 05 — market evaluation

- `direction-v2.0.0` で方向対応 MFE/MAE（v1意味は変更せず並存）
- 親 issuance 状態を単一 component で上書きしない
- yfinance: rate_limit / network / invalid_symbol / no_data 分類、retry、cache優先
- provider 監査列（Alembic `0005`）

### Fix 06 — Obsidian / interactive CLI

- `obsidian_vault_path` + `workspace_relative_path`、legacy `vault_root` 互換
- package resources から docs/prompts を seed（再実行は上書きしない、`--update-docs`）
- 中央 prompt template → 案件 snapshot（version/hash）
- `analyst-forecast start` wizard（6か月既定、媒体複数、cancel/redo）

### Fix 07 — 最終監査

- シナリオA/G unit、既存B～F相当テスト、品質ゲート実行
- `docs/06_実装/FINAL_REVIEW.md` 作成

## 検査結果（03～07後）

- pytest unit：63 passed
- ruff format / check：pass
- mypy src（strict）：pass
- Alembic 空DB upgrade head：pass
- Alembic 0001→head：pass
- ネットワーク integration：通常suite除外（external_blocked）
- wheel clean venv：未実施（external_blocked）

## 未完了・限定

- 実原文 5～10件の精度実証
- yfinance/FRED 実ネットワークの安定確認
- wheel のクリーンvenv導入確認
- 総合点・能力ランキング等は意図的に未実装

## 実行履歴（追記）

### 修正指示 03～07

- 関連：R-01, R-02, R-05～R-10、FIX-001～003, 009～021, 023
- 結果：完了（限定付き受入）
- 証拠：pytest 63、FINAL_REVIEW.md
