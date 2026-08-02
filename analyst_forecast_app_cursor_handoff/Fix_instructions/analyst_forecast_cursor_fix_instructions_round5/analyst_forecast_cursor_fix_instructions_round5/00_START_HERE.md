# 第5次修正指示 — START HERE

## 対象

- Repository: `baiputaojiu/amatsuka_argo_5_obsidan_chatgpt_cursor_genspark`
- Review branch: `fetch2_1_アナリスト調査の計画を追加`
- Review base commit: `2f826edc7a5bfd5559e8c9a32cc8d9e58d598106`
- Python project root: repository直下の `analyst_forecast_app_cursor_handoff/`
- 独立レビュー判定: `NEEDS_CHANGES_BEFORE_REAL_SAMPLE`
- 独立baseline:
  - pytest: `137 passed, 1 skipped, 3 deselected`
  - `ruff check`、mypy、packaged docs sync: pass
  - `ruff format --check`: 8 files fail
  - clean wheel代表縦断: pass
  - issuance/component/evidence入り0007 DB → head: foreign key errorでfail

Round4で、accepted P08の単一予想に対する再レビュー、別URLへの前処理再利用、P09の基本的な有限状態遷移、unknown日時の保持、basket共通日1日の評価抑止は改善されました。

一方、独立レビューでは次の7項目が未達でした。

1. issuance、component、evidence、evaluationを含む0007 DBを0009へ上げると、`DROP TABLE forecast_issuances`でforeign key errorになり、失敗後は0007表記のまま0008の一部列だけが残る。
2. 1つのP08に複数予想がある状態でP09 `correct`を行うと、すべての新予想が最初の旧予想のlineageへ束ねられ、同一lineageに複数activeができる。
3. superseded済みcomponentをP11へ渡してもacceptedになる。P12/P13にも同じactive世代検査がない。
4. unknown日時のP08をP09で時刻確定する際、P09自身の`knowledge_cutoff`が訂正後`made_at`より後でも受理される。
5. 単一銘柄の複数日予想は市場データが1取引日しかなくてもhit/missを計算し、coverage不足時の監査情報も保存されない。
6. P09のreject区分が省略可能なboolで、retryable理由も不要であり、実promptに区分・理由の説明がない。
7. `ruff format --check`と`git diff --check`が合格していないのに、最終監査でpass扱いになっている。

Round5の目的は新機能追加ではありません。既存DBの安全な更新、複数予想の一対一lineage、active componentの入口制御、訂正レビューの時点境界、評価coverage監査、P09 reject契約、品質ゲートの証拠性を閉じ、実原文1件のdirectional sliceへ進める状態にすることです。

## 配置場所

このフォルダを次の位置へ配置してください。

```text
analyst_forecast_app_cursor_handoff/
├─ pyproject.toml
├─ src/
├─ tests/
├─ docs/
└─ Fix_instructions/
   └─ analyst_forecast_cursor_fix_instructions_round5/
```

project rootを二重階層にしないでください。

## Cursorでの実行順序

原則として1ファイルを1セッションで実行してください。

1. `01_PREFLIGHT_AND_REPRODUCTION.md`
2. `02_MULTI_FORECAST_LINEAGE_FIX.md`
3. `03_ACTIVE_COMPONENT_GATE_FIX.md`
4. `04_P09_CORRECTED_TIME_BOUNDARY_FIX.md`
5. `05_MARKET_COVERAGE_AUDIT_FIX.md`
6. `06_P09_REJECT_DISPOSITION_SCHEMA_PROMPT_FIX.md`
7. `07_DATA_BEARING_MIGRATION_FIX.md`
8. `08_PACKAGE_DOCS_AND_QUALITY_GATE.md`
9. `09_FINAL_AUDIT.md`
10. `10_ACCEPTANCE_MATRIX.md`で最終照合

## 全工程の拘束条件

- 意味判断を伴うAI処理は、ユーザーが設定した高性能モデルだけを使う。廉価モデルへ自動切替しない。
- raw原文は変更禁止。正規化、segment、予想、レビュー、訂正、再利用物は別artifactとして保存する。
- 市場結果をP05/P07/P08/P09/P11/P12/P13の意味判断へ渡さない。後知恵を禁止する。
- SQLiteを機械処理の正本とし、Markdown、CSV、NEXT_ACTIONSは再生成可能な表示物とする。
- 訂正前のartifact、issuance、component、evidence、mapping、evaluationを削除・上書きしない。
- active/latestの判定は共通serviceまたはrepository queryを正本にし、各処理へ別実装しない。
- 既存`0001`～`0007` migrationは編集しない。Round4で追加された`0008`、`0009`の修正が不可避な場合は、理由、既適用DBへの対応、新headへの経路を文書化する。
- legacy行を自動的にactive、本人確認済み、時点確認済みへ昇格させない。
- migration試験でだけ、過去Schemaの実データfixtureをSQLで作成してよい。通常application試験はpublic application APIまたはCLIを使用する。
- migration失敗後に元DBを半端なSchemaで残さない。backup復元または原子的なupgradeを必須とする。
- `reference/CHAT_HISTORY.pdf`は削除・改変しない。
- API key、token、実Vault絶対path、raw、DB、市場cache、AI実データをGitへ追加しない。
- ループエンジニアリングは導入しない。
- commit、push、branch作成、PR作成を行わない。
- 指示外の大規模リファクタリングをしない。必要なら着手前に理由、代替案、影響範囲を報告する。
- pytest合計件数や「コードが存在する」ことだけでpassにしない。各negative caseを実行し、DB件数、状態、理由codeまでassertする。
- internalなmigration、Schema、SQLite、CSV、wheel、format失敗をskip、xfail、external_blockedで隠さない。

## Round5で変更してよいもの

- P09 Schema、Pydantic model、固定JSON Schema、prompt、legacy adapter
- P08 forecast_refとP09 correction operationの対応契約
- ForecastIssuance lifecycle、lineage、active一意制約、transaction処理
- P11/P12/P13/評価のactive component guard
- P09/corrected P08の時点検証共通service
- Evaluation coverage audit、unevaluable保存経路
- `0008`以降のAlembic migration、upgrade wrapper、migration tests
- result、workflow、NEXT_ACTIONS、docs、package、wheel test、品質ゲート文書

## 今回の範囲外

- 総合点・能力ランキング
- 複数アナリストの現在予想統合
- 上昇候補ランキング
- 時期・程度・早期実現の完全採点
- 期間不明予想の1/3/6/12か月自動観測
- PNG・ヒートマップ
- 情報源の完全自動収集
- exe化

これらを便乗実装せず、既存`FUTURE_ROADMAP`の詳細を維持してください。

## 各セッションの報告形式

```text
対象指示:
修正前に再現した問題:
設計判断と不変条件:
変更ファイル:
DB migration:
Schema変更:
追加したnegative test:
追加したpositive test:
実行した品質ゲート:
未完了・限定:
次の指示へ進めるか:
commit / push: 未実施
```

## 完了判定

- `NOT_READY`: Round5受入項目に1件でも内部fail、未実装、未検証、skip、xfailがある。
- `READY_FOR_REAL_SAMPLE_DIRECTIONAL_SLICE`: `10_ACCEPTANCE_MATRIX.md`の全項目がpassし、実原文1件の手動投入へ進める。
- `FULL_MVP_READY`: 今回は使用しない。正式仕様の全MVP要件を満たす場合だけ使用する。

