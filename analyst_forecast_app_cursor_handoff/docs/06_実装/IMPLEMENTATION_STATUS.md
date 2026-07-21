# 実装状況

## 最終更新

- 日時：2026-07-21
- 状態：Round5 コア修正（A–G）実装済み。受入マトリクス R5-001〜052 は **52 PASS / 0 GAP**（`FINAL_REVIEW_ROUND5.md`）
- MVP判定：`READY_FOR_REAL_SAMPLE_DIRECTIONAL_SLICE`
- Round4時点：`READY_FOR_REAL_SAMPLE_DIRECTIONAL_SLICE`（`FINAL_REVIEW_ROUND4.md`）
- Round3時点：`READY_FOR_REAL_SAMPLE_DIRECTIONAL_SLICE`（`FINAL_REVIEW_ROUND3.md`）
- 重大ブロッカー：なし
- 推奨する次の一作業：実Vault原文1件の方向評価目視（directional slice）
- commit / push：未実施

## Round5 状況

- Fix 07 migration FK-safe + 0010 + data-bearing / legacy_conflict tests
- Fix 02 forecast_operations lineage（update/add/remove + ALREADY_IMPORTED）
- Fix 03 require_active_component_context（P11/P12/P13/eval）
- Fix 04 P09 cutoff vs corrected made_at / source boundary（超過拒否・同値許可・+1µs拒否）
- Fix 05 single-symbol / basket coverage audit（common_date_count=1）+ same-day method unsupported
- Fix 06 reject_disposition 2.1.0 + P09 prompt
- Fix 08 品質ゲート実ログ（pytest 181 / ruff / mypy / docs sync / alembic / wheel xfail除去）
- 追加受入：`tests/unit/test_round5_acceptance_extra.py` / `tests/unit/test_round5_gaps.py`

## Round5 残GAP

- なし（R5-001〜052 全 PASS）

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

## 検査結果（Round5 本セッション）

- pytest unit：**181 passed**, 3 deselected
- round5 modules：**43 passed**
- ruff check：pass
- ruff format --check：pass（`test_round5_gaps.py` 再フォーマット後）
- mypy src：pass
- packaged docs sync --check：pass
- Alembic upgrade head / check：pass
- wheel：`test_r4_046_wheel_help` pass（xfail除去済み）
- git diff --check：pass（CRLF警告のみ）

## 未完了・限定（FULL MVP外）

- 総合点・能力ランキング・PNG・1/3/6/12観測など仕様上の将来項目
- 実原文の精度実証（directional slice の次ステップ）
- ネットワーク integration：external_blocked
