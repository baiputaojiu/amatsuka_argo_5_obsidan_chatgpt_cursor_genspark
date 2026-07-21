# 実装状況

## 最終更新

- 日時：2026-07-21
- 状態：Round4 受入マトリクス R4-001〜R4-048 を証拠付きで PASS
- MVP判定：`READY_FOR_REAL_SAMPLE_DIRECTIONAL_SLICE`（詳細は `FINAL_REVIEW_ROUND4.md`）
- Round3時点：`READY_FOR_REAL_SAMPLE_DIRECTIONAL_SLICE`（`FINAL_REVIEW_ROUND3.md`）
- 重大ブロッカー：なし
- 推奨する次の一作業：実Vaultで原文1件の縦断目視確認
- commit / push：未実施

## Round4 完了状況

### Bugs A/B — lifecycle / P09（完了）

- supersede / generation / lineage
- results・workflow・evaluation は active + made_at 確定のみ

### Bug C — artifact reuse（完了）

- `artifact_reuse.py` / raw_sources lookup（prompt_version・model一致必須）
- workflow は `is_artifact_applicable_for_source` と共有判定

### Bug D — state machine（完了）

- reject retryable → EXTRACT_FORECASTS
- terminal / unresolved 無限 REVIEW 防止

### Bug E — time / cutoff（完了）

- P08 2.1.0 knowledge_cutoff、unknown made_at null
- legacy null made_at は active 集計から除外

### Bug F — basket（完了）

- 共通日1日 unevaluable、coverage_audit、単一symbol cache非汚染

### Package / migration / docs（完了）

- Alembic 0008/0009、FK/件数監査、docs sync、clean wheel `--help`

## Round3 完了状況（維持）

- Fix 02〜09（話者帰属、cutoff、P09、複数source、basket cache、Vault docs、migration）

## 検査結果（Round4後）

- pytest unit：**138 passed**, 3 deselected
- ruff check：pass
- mypy src：pass
- packaged docs sync --check：pass
- Alembic check：pass
- wheel clean venv：`test_r4_046_wheel_help` pass
- CHAT_HISTORY.pdf：保持確認（SHA256 `F1D95670…ADDD4A`）

## 未完了・限定（FULL MVP外）

- 総合点・能力ランキング・PNG・1/3/6/12観測など仕様上の将来項目
- 実原文の精度実証（directional slice の次ステップ）
- ネットワーク integration：external_blocked
