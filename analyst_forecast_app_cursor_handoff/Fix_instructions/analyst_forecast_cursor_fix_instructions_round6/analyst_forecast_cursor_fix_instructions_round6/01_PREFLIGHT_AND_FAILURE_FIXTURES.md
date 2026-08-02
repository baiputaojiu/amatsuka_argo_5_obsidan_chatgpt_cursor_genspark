# 01 — 事前確認と失敗fixtureの固定

## 目的

Round5独立レビューで実際に受理された不正入力と、証拠不足だった試験を修正前baselineとして固定します。happy pathへ置き換えず、同じ入力を正式な回帰testへ移してください。

## Cursorへ渡す依頼文

```text
00_START_HERE.mdの拘束条件に従い、Round6 preflightを実施してください。commit・pushはしないでください。

project rootとbranchを確認し、HEADが88864c289750f8323c27b6e3f2c09fd70a79923dであることを記録してください。ユーザーの未commit変更がある場合は上書きせず、対象を報告して停止してください。

次のA～Fを、修正前probeまたは修正によってPASSへ変わる正式testとして再現してください。

A. forecast_operationsの多対一
- public AI ingest経路で、accepted P08にforecast_ref A/Bの2件を作る。
- corrected P08にはforecast_ref Xを1件だけ置く。
- operationsをupdate A→X、update B→Xとする。
- 現状P09がacceptedになること、旧A/B、新X、lineage root、lifecycle、結果件数を記録する。
- 単にSchema classを直接生成するだけでなく、正式P09 ingestを通す。

B. 旧・新forecastの未申告
- reviewed側をA/B、corrected側をA2/B2/Cとする。
- operationsにはupdate A→A2とadd Cだけを申告し、旧Bと新B2を説明しない。
- 現状acceptedとなり、activeがA2、旧B、B2、Cへ増殖することを確認する。
- 次に、未申告旧だけ、未申告新だけ、存在しないref、同一refの重複を別parameterで用意する。
- 各caseでbefore/afterのissuance/component/evidence/operation/review resolution件数を保存する。

C. coverage_auditの欠落
- 単一symbol・1取引日でunevaluableを作る。
- 単一symbol・2取引日で成功評価を作る。
- basket・共通1日と共通2日を作る。
- 各EvaluationRecord.coverage_auditを取得し、symbol、currency、input_first_date、input_last_date、in_range_row_count、series_hash、duplicate/invalid/dropped countsの欠落を記録する。
- ログやprovider requestではなく、DBへ保存されたJSONを確認する。

D. P09契約の不整合
- jsonschema.Draft202012Validatorで固定p09_forecast_review.schema.jsonを直接検証する。
- schema_version=2.1.0、decision=reject、reject_disposition/reject_reasonなしがerror 0件になる現状を記録する。
- decision=acceptへreject_disposition/reject_reasonを付けたcaseを固定SchemaとPydanticの両方へ通す。
- schema_version=2.0.0について、reject field省略、reject_terminalのみ、reject_dispositionのみ、両方一致、両方矛盾、accept/correct/unresolvedへのreject field混入を行列化する。
- 固定Schema、Pydantic runtime、model_json_schema、promptの結果差を表にする。

E. migration証拠
- 既存test_round5_migration.pyのfixtureを確認する。
- 0007→head成功を再実行し、現状は機能的に成功することを確認する。
- before/after比較がrow count中心で、全ID・全legacy列projection hashを比較していないことを記録する。
- 強制失敗testがAlembic/DDL変更前に例外化していることを、実際のhook位置とSchema before/afterで確認する。

F. wheel・品質ゲート
- cleanな`.[dev]`環境または依存定義の静的確認で、`build`がdev extraにないことを記録する。
- tests/unit/test_round4_wheel.pyがpytest.importorskip("build")でskipされることを再現する。
- wheel testがhelpとdocs probeだけであることを記録する。
- `git diff --check 2f826edc7a5bfd5559e8c9a32cc8d9e58d598106..88864c289750f8323c27b6e3f2c09fd70a79923d`を実行し、空白errorを保存する。
- ruff format --check、ruff check、mypy、pytest -q、docs sync、Alembic check、python -m buildを個別return code付きで記録する。

結果をdocs/06_実装/ROUND6_REPRODUCTION.mdへ、入力、期待、修正前実結果、DB query、error数、command、関連コードとして記載してください。

一時probeだけで終了せず、A～Fを02～06の正式testへ移してください。再現できないcaseを推測でPASSにせず、入力と観測差を報告して停止してください。
```

## 必須成果物

- `docs/06_実装/ROUND6_REPRODUCTION.md`
- A/Bの不正payloadをそのまま使う正式negative test
- Cの4形状をDB保存JSONまで確認するtest
- Dの固定Schema/Pydantic契約matrix test
- Eのdata-bearing migration test強化計画
- Fのnon-skipping wheel test強化計画

## 完了条件

- A～Fのすべてに具体的入力と観測結果がある。
- 修正対象をR6-001～R6-050へ対応付けている。
- 既に改善済みのactive guard、cutoff、migration成功等を壊さないnon-regression対象も列挙している。

