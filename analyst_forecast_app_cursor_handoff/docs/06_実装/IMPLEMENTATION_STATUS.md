# 実装状況

## 最終更新

- 日時：2026-07-22
- 状態：Round6 完了。受入マトリクス R6-001〜050 は **50 PASS / 0 GAP**（`FINAL_REVIEW_ROUND6.md`）
- MVP判定：`READY_FOR_REAL_SAMPLE_DIRECTIONAL_SLICE`
- Round5時点：独立レビューで追加指摘 → Round6で解消
- 重大ブロッカー：なし
- 推奨する次の一作業：実Vault原文1件の方向評価目視（directional slice）
- commit / push：未実施

## Round6 完了状況

- Fix 02: forecast_operations 完全被覆・一対一（多対一/未申告拒否）
- Fix 03: coverage_audit schema_version + instruments[]
- Fix 04: 固定JSON Schema allOf + legacy 2.0 + prompt例
- Fix 05: PK/legacy hash + DDL後失敗復元
- Fix 06: build dev依存、wheel正式縦断、品質ゲート全通（pytest 212）

## Round6 残GAP

- なし（R6-001〜050 全 PASS）

## Round5 完了状況（維持）

- lifecycle / reuse / P09 state / cutoff / basket / migration 0007→head 等

## 検査結果（Round6 本セッション）

- pytest unit：**212 passed**, 3 deselected
- ruff check / format --check：pass
- mypy：pass
- packaged docs sync --check：pass
- Alembic check：pass
- python -m build：pass
- git diff --check from 88864c2：pass（CRLF warningのみ）


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
