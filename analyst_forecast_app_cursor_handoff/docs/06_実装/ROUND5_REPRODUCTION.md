# Round5 バグ再現記録

実施日: 2026-07-21  
基準: Round4完了後の実装（独立レビュー判定 `NEEDS_CHANGES_BEFORE_REAL_SAMPLE`）

## A. データ入り0007 → head migration（Fix 07）

**入力:** `upgrade_database(revision="0007")` 後、issuance / component / evidence / evaluation / snapshot を含む参照グラフをSQL seed。`PRAGMA foreign_keys=ON`。

**期待:** headへ到達し FK・件数が保持される。

**修正前の実結果（コード根拠）:**
- `0009` の `batch_alter_table("forecast_issuances")` が SQLite で table rebuild（DROP+CREATE）を行う。
- `env.py` / `upgrade_database` は migration を transaction 内で実行し、接続レベルの `PRAGMA foreign_keys=OFF` を保証しない。
- 子行がある状態で DROP すると `FOREIGN KEY constraint failed`。
- `upgrade_database` は失敗時に backup 復元せず、部分適用DBを残し得る。

**関連コード:** `migrations/versions/0009_*.py`, `infrastructure/db/migration.py`, `migrations/env.py`

## B. 複数forecastのP09 correct lineage（Fix 02）

**入力:** 同一accepted P08に forecast_ref の異なる正式予想2件 → P09 correct。

**期待:** 各旧forecastが対応する新forecastへ一対一で supersede。lineage_root は別々。

**修正前の実結果（コード根拠）:**
- `ai_pipeline.py` `_apply_review_decision` correct 分岐で `old_issuances[0]` を全新issuanceの親にしている。
- `superseded_by_issuance_id` も `issuance_ids[0]` に偏る。

```1866:1875:src/analyst_forecast/application/ai_pipeline.py
                        if old_issuances:
                            old_first = old_issuances[0]
                            root = old_first.lineage_root_id or old_first.forecast_issuance_id
                            new_iss.lineage_root_id = root
                            ...
                for old_iss in old_issuances:
                    ...
                    old_iss.superseded_by_issuance_id = issuance_ids[0] if issuance_ids else None
```

## C. superseded component の P11/P12/P13（Fix 03）

**入力:** P09 correct 後の旧component IDでP11をingest。

**期待:** `inactive_forecast_component` で拒否。candidate行を増やさない。

**修正前の実結果:** `_validate_p11` に lifecycle_status / active 検査がない（存在・run/source一致のみ）。P12/P13も同様。evaluation のみ Round4 で active 検査あり。

## D. unknown→known 訂正時の P09 cutoff（Fix 04）

**入力:** 元 made_at=null、corrected made_at=08:00、P09 cutoff=08:30、source boundary=09:00。

**期待:** P09 を拒否（`p09_cutoff_exceeds_corrected_made_at`）。

**修正前の実結果:** `_validate_review_artifact` は reviewed P08 の non-null made_at のみと比較。元が null だと P09 cutoff と訂正後 made_at の比較をスキップ。

## E. 単一symbol 1取引日（Fix 05）

**入力:** normalized_start < normalized_end、provider が期間内1日だけ返す。

**期待:** unevaluable + coverage_audit 保存。hit/miss/return 非確定。

**修正前の実結果:** basket のみ `common_dates < 2`。単一銘柄は `bars` が1件でも open→close を計算。unevaluable 時 `_store_without_values` で coverage_audit が null になり得る。

## F. P09 reject 契約（Fix 06）

**入力:** `decision=reject` で `reject_terminal` / `reject_reason` 省略。

**期待:** Schema/Pydantic で必須 disposition + reason。

**修正前の実結果:** `reject_terminal: bool = False` が省略可能で暗黙 retryable。retryable 時 reason 不要。prompt に区分規則なし。

## G. 品質ゲート（Fix 08）

**修正前:** 独立レビューで `ruff format --check` が8ファイル fail、`git diff --check` 未合格でも最終監査で pass 扱い。

**対応方針:** format 適用、diff check、ゲート結果を ROUND5_QUALITY_GATE / FINAL_REVIEW に実ログで記載。失敗を pass と書かない。

## Remediation 対応

| 項目 | 指示 |
|------|------|
| A | 07 migration 原子性 + FK-safe rebuild |
| B | 02 forecast_operations + partial unique index |
| C | 03 require_active_component_context |
| D | 04 P09 corrected time boundary |
| E | 05 coverage audit 単一/basket統一 |
| F | 06 reject_disposition Schema/prompt |
| G | 08 format + 正直な品質ゲート文書 |
