# 第6次修正指示 — START HERE

## 対象と今回の位置づけ

- Repository: `baiputaojiu/amatsuka_argo_5_obsidan_chatgpt_cursor_genspark`
- Review branch: `fetch2_1_アナリスト調査の計画を追加`
- Review base commit: `88864c289750f8323c27b6e3f2c09fd70a79923d`
- Python project root: repository直下の `analyst_forecast_app_cursor_handoff/`
- 独立レビュー判定: `NEEDS_CHANGES_BEFORE_REAL_SAMPLE`
- Round6の役割: 新機能開発ではなく、`READY_FOR_REAL_SAMPLE_DIRECTIONAL_SLICE`へ進めるかを判断する最終修正・最終評価

Round5で、実データ入り0007 DBのmigration、DBによるactive lineage一意制約、superseded componentのP11/P12/P13/評価拒否、P09訂正時のcutoff、単一銘柄1取引日の誤評価防止、Ruff/mypy/docs syncは改善されました。これらを壊さず維持してください。

一方、コミット`88864c2`の独立レビューでは次が未達でした。

1. `forecast_operations`が存在するだけで受理され、旧2件→新1件、未申告の旧・新forecastを含む訂正がacceptedになる。
2. `coverage_audit`に銘柄別symbol/currency、input first/last date、in-range件数、series hash、duplicate/invalid/dropped件数がない。
3. 固定P09 JSON Schemaがreject時の区分・理由を必須化しておらず、legacy契約ではreject以外へのreject fieldや新旧フィールドの矛盾を受理できる。
4. `build`がdev依存に含まれず、wheel testが`pytest.importorskip("build")`でskipされる。wheel testもhelp/docsだけで、指定した正式縦断を行っていない。
5. migration本体の主要不具合は改善したが、before/after証拠が件数中心で、強制失敗もDDL適用前に発生している。
6. `git diff --check 2f826ed..88864c2`が空白エラー14件で失敗したまま、Round5最終文書が「52 PASS / 0 GAP」としている。

## 配置場所

このフォルダを次の位置へ配置してください。

```text
analyst_forecast_app_cursor_handoff/
├─ pyproject.toml
├─ src/
├─ tests/
├─ docs/
└─ Fix_instructions/
   └─ analyst_forecast_cursor_fix_instructions_round6/
```

project rootを二重階層にしないでください。

## 実行順序

原則として1ファイルを1セッションで実行してください。

1. `01_PREFLIGHT_AND_FAILURE_FIXTURES.md`
2. `02_FORECAST_OPERATIONS_TOTAL_MAPPING_FIX.md`
3. `03_COVERAGE_AUDIT_CONTRACT_FIX.md`
4. `04_P09_SCHEMA_MODEL_PROMPT_CONTRACT_FIX.md`
5. `05_MIGRATION_EVIDENCE_AND_FAILURE_RECOVERY.md`
6. `06_WHEEL_VERTICAL_AND_QUALITY_GATE.md`
7. `07_DOCUMENTATION_AND_COMPLETION_EVIDENCE.md`
8. `08_FINAL_AUDIT.md`
9. `09_ACCEPTANCE_MATRIX.md`で最終照合

## Round6の拘束条件

- 新機能を追加しない。総合点、複数アナリスト統合、PNG、1/3/6/12か月観測等へ着手しない。
- commit、push、branch作成、PR作成を行わない。
- `reference/CHAT_HISTORY.pdf`を削除・改変しない。
- raw原文、訂正前artifact、旧issuance/component/evidence/mapping/evaluationを削除・上書きしない。
- SQLiteを機械処理の正本とし、Markdown、CSV、NEXT_ACTIONSは再生成可能な表示物とする。
- 市場結果をP05/P07/P08/P09/P11/P12/P13の意味判断へ渡さない。
- migration fixture以外では、目的状態を作るためにDB lifecycle列を直接書き換えない。public application API、CLIまたは正式AI ingest経路を使う。
- 既存`0001`～`0009` migrationを安易に編集しない。既適用DBとの両立が必要なら新revisionを優先する。
- 既存testを削除、弱体化、assert削減して通さない。変更が必要な場合は、誤っていた仕様・旧assert・新しい根拠を文書化する。
- 合計test件数、コードの存在、mock呼出し回数だけで個別受入項目をPASSにしない。
- Round6必須testとinternal wheel/migration/Schema testでは、`skip`、`skipif`、`importorskip`、`xfail`、例外握り潰し、成功扱いの早期returnを禁止する。
- live network専用integration testの既存deselectは許容するが、今回の受入項目をintegration markerへ移して通常試験から外してはならない。
- build toolやローカル依存がない場合は品質ゲートをFAILとする。自動skipしない。
- invalid payloadの拒否では、review resolution、issuance、component、evidence、operation、結果件数を変えない。保存が許される監査行がある場合は、仕様と件数を明示する。
- API key、token、実Vault絶対path、SQLite、backup、raw、market cache、AI実出力、venv、distをGitへ追加しない。

## Round6で変更してよい範囲

- P09 correction operationの集合検証とtransaction境界
- P09固定JSON Schema、Pydantic model/runtime validator、legacy adapter、prompt、packaged resource
- evaluation coverage auditの生成・保存・表示
- migration試験fixture、backup/restore試験、必要最小限のmigration wrapper修正
- wheel build/clean install/縦断test、dev依存、package data
- Round6 reproduction、quality gate、final review、implementation status等の関連文書
- Round5で発生した空白・format不整合

## セッションごとの報告形式

```text
対象指示:
修正前に再現した入力と実結果:
設計した不変条件:
変更ファイル:
追加したnegative test:
追加したpositive test:
DB before/after:
Schema / prompt / package同期:
実行した品質ゲートとreturn code:
skip / xfail / deselected:
未完了・限定:
次へ進めるか:
commit / push: 未実施
```

## 完了判定

- `NOT_READY`: R6-001～R6-050に1件でも内部fail、未実装、未検証、skip、xfail、証拠不足がある。
- `READY_FOR_REAL_SAMPLE_DIRECTIONAL_SLICE`: R6-001～R6-050がすべて具体的証拠付きでPASSし、実Vault原文1件のdirectional sliceへ進める。
- `FULL_MVP_READY`: Round6では使用禁止。正式MVP全要件を満たす場合だけ別途使用する。

Round6完了報告に「READY」「全件PASS」と書く前に、`09_ACCEPTANCE_MATRIX.md`を1行ずつ照合してください。1項目でも説明不能なら`NOT_READY`と報告してください。

