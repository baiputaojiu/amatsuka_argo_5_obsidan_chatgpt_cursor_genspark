# Round6 Reproduction（修正前baseline）

- 日時: 2026-07-22
- HEAD: `88864c289750f8323c27b6e3f2c09fd70a79923d`（`round5対応完了後`）
- branch: `fetch2_1_アナリスト調査の計画を追加`
- 判定対象: 独立レビュー `NEEDS_CHANGES_BEFORE_REAL_SAMPLE`
- commit/push: 未実施

## Preflight 状態

- project root: `analyst_forecast_app_cursor_handoff/`（二重階層なし）
- ユーザー未commit（本作業開始時）: Obsidian vault 変更 + Round6 Fix_instructions 追加のみ。Round5 実装は HEAD に含まれる。
- Round5 非回帰対象: data-bearing 0007→head、active lineage unique、inactive P11/P12/P13/eval gate、P09 cutoff、単一銘柄1日 unevaluable、ruff/mypy/docs sync

## A. forecast_operations 多対一（R6-001/005）

### 入力（正式 ingest 想定）

- reviewed P08: forecast_ref `A`, `B`
- corrected P08: forecast_ref `X` のみ
- operations: `update A→X`, `update B→X`

### 修正前コード観測

- `_validate_review_artifact`（`ai_pipeline.py`）: ops が非空なら集合契約を検査せず通過
- `_apply_review_decision`: 各 update を `paired` に追加し、同一 `new_iss` へ複数 old を supersede 可能（`corrected_forecast_ref` 重複チェックなし）

### 修正前期待される実結果

- P09 **accepted**
- 旧 A/B は superseded、新 X が active（多対一 lineage）
- 正式 negative test へ固定（修正後は拒否）

## B. 旧・新 forecast 未申告（R6-001/008/009）

### 入力

- reviewed: `A`, `B` / corrected: `A2`, `B2`, `C`
- operations: `update A→A2`, `add C` のみ（旧 B・新 B2 未申告）

### 修正前観測

- set equality `O = Uo ⊎ R` / `N = Un ⊎ A` **未実装**
- ops に無い old は `lifecycle_status=active` のまま残る
- `_insert_p08` 後、ops 外の new も active のまま残る → active 増殖

### 追加 parameter case（正式 test へ）

| case | 期待（修正後） |
|------|----------------|
| 未申告旧のみ | reject + 件数不変 |
| 未申告新のみ | reject + 件数不変 |
| 存在しない ref | reject |
| 同一 ref の op 間重複 | reject |

## C. coverage_audit 欠落（R6-018〜023）

### 形状

1. 単一 symbol・1取引日 → unevaluable
2. 単一 symbol・2取引日 → success
3. basket・共通1日 → unevaluable
4. basket・共通2日 → success

### 修正前 DB `EvaluationRecord.coverage_audit` に存在するキー（例）

`requested_*`, `effective_*`, `evaluation_as_of`, `method_version`, `series_kind`, `coverage_status`, `reason_code`, `common_date_rule`, `basket_weights`, `mapping_hash`, `input_series_hashes`, success 時の `selected_*` / `common_date_count`

### 修正前に欠落するキー（独立レビュー指摘）

- `schema_version`
- `instruments[]`（symbol/currency/weight、input_first/last、in_range、series_hash、duplicate/invalid/dropped）
- instrument 別統計の統一構造

## D. P09 契約不整合（R6-028〜033）

### 修正前 probe（Draft202012Validator vs Pydantic、valid ID 使用）

| case | 固定 Schema | Pydantic |
|------|-------------|----------|
| 2.1 reject + disposition/reason なし | error **0件**（受理） | **拒否**（disposition 必須） |
| 2.1 accept + reject_disposition/reason | error **0件**（受理） | **拒否**（reject 以外に disposition 不可） |
| 2.0 reject で terminal 省略 | Schema 条件なし | disposition 単独等で緩い経路あり |

固定 `p09_forecast_review.schema.json` に `allOf` / `if`/`then` **なし**。  
正式 matrix test で Schema ≡ Pydantic を強制する。

## E. migration 証拠（R6-038〜042）

### 現状（`test_round5_migration.py`）

- 0007→head: **機能的には成功**（件数比較中心）
- PK 集合・legacy column projection SHA-256 比較: **未実施**
- 強制失敗: `alembic_command.upgrade` を **開始前** に `side_effect` → DDL 未適用のまま restore 試験

### 強化計画

- child graph + sentinel 付き fixture
- per-table PK + legacy projection hash
- backup 後に実 `ALTER` + `UPDATE` + `COMMIT` してから例外 → restore 検証

## F. wheel・品質ゲート（R6-044〜049）

### 修正前

- `pyproject.toml` `dev` extra: `mypy`, `pytest`, `ruff` のみ → **`build` なし**
- `test_round4_wheel.py`: `pytest.importorskip("build")` で skip 可能
- wheel test は help/docs 中心（正式縦断不足）

### `git diff --check`（Round5 base 区間）

```text
git diff --check 2f826edc7a5bfd5559e8c9a32cc8d9e58d598106..88864c289750f8323c27b6e3f2c09fd70a79923d -- analyst_forecast_app_cursor_handoff
```

→ Round5 差分に空白 error（独立レビュー: 14件）。Round6 で修正し、`88864c2` からの差分でも `--check` pass を要件とする。

## R6 ID 対応付け

| 再現 | 主 ID |
|------|-------|
| A 多対一 | R6-001, R6-005 |
| B 未申告 | R6-001, R6-003, R6-004, R6-008, R6-009 |
| C coverage | R6-018〜R6-027 |
| D Schema | R6-028〜R6-037 |
| E migration | R6-038〜R6-043 |
| F wheel/gate | R6-044〜R6-050 |

## 非回帰（壊してはならない）

- Round5: active lineage unique、inactive component gate、cutoff vs made_at、1日 unevaluable、migration 0007→head 成功、FK/integrity

## 次セッション

02〜06 の正式 test / 実装へ A〜F を移管。推測 PASS 禁止。
