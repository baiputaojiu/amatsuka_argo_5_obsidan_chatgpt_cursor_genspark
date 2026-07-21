# 実装状況

## 最終更新

- 日時：2026-07-21
- 状態：Round3 Fix 01–09 を実装。pytest **94 passed**、ruff、mypy、docs sync、wheel clean venv、CHAT_HISTORY保持を確認
- MVP判定：`READY_FOR_REAL_SAMPLE_DIRECTIONAL_SLICE`（詳細は `FINAL_REVIEW_ROUND3.md`）
- 重大ブロッカー：なし
- 推奨する次の一作業：実Vaultで原文1件の縦断目視確認
- commit / push：未実施

## Round3 完了状況

### Fix 02 — 話者帰属（完了）

- `domain/attribution.py`：NFKC exact、canonical/alias照合
- `_is_formal_forecast` は verified `target_confirmed` のみ
- host/legacy_unknown の正式化防止、evidence→segment FK
- `analyst add-alias` CLI

### Fix 03 — knowledge cutoff（完了）

- `domain/knowledge_boundary.py` 共有境界
- P05/P07/P06/P09 で source境界超過をreject
- P11–P13 の made_at / corrected candidate 検証を維持・拡張

### Fix 04 — P09 correct 再検証（完了）

- corrected_payload を通常 `_validate_p05/p07/p08` と共有
- offset不正はreject、formal=0

### Fix 05 — 複数source workflow（完了）

- `pending_p08` があれば issuances があっても EXTRACT_FORECASTS
- `processed_no_formal_forecast` terminal

### Fix 06 — basket cache / CSV（完了）

- `series_kind` / `BASKET:{mapping_hash}` 分離
- AAA単独が basket 0% cache を読まない回帰test
- long-form CSV multi-symbol

### Fix 07 — Vault docs / prompts（完了）

- `scripts/sync_packaged_docs.py`（`--check`）
- FUTURE_ROADMAP 詳細版を resources へ同期
- P05/P07/P08/P09 prompt に Round3 規則を反映

### Fix 08 — migration / 品質ゲート（完了・一部限定）

- Alembic `0007`
- ruff / mypy / pytest / docs sync pass
- wheel clean venv：未実施

### Fix 09 — 最終監査（記録済）

- `FINAL_REVIEW_ROUND3.md` / `ROUND3_QUALITY_GATE.md`

## Round2 維持

- P11→P12→P13 状態機械、P07経路、mixed currency unevaluable、秘密非混入方針

## 検査結果（Round3後）

- pytest unit：94 passed
- ruff format / check：pass
- mypy src：pass
- packaged docs sync --check：pass
- Alembic 0007：pass
- ネットワーク integration：external_blocked
- wheel clean venv：未実施

## 未完了・限定

- wheelクリーン導入の縦断確認
- CHAT_HISTORY.pdf の blob hash 固定確認
- 受入マトリクス全52項目の完全証拠付け（一部は実装済・専用test未整備）
- 実原文の精度実証
- 総合点・能力ランキング・PNG・1/3/6/12観測など仕様上の将来項目
