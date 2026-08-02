# 第4次修正指示 — START HERE

## 対象

- Repository: `baiputaojiu/amatsuka_argo_5_obsidan_chatgpt_cursor_genspark`
- Review branch: `fetch2_1_アナリスト調査の計画を追加`
- Review base commit: `bb7d167a849421b3441dcf06c054e497263ccfa0`
- Python project root: repository直下の `analyst_forecast_app_cursor_handoff/`
- 独立レビュー判定: `NEEDS_CHANGES_BEFORE_REAL_SAMPLE`
- 独立品質baseline: `pytest 94 passed, 3 deselected`、Ruff、mypy、Alembic、wheelはpass

Round3では、話者のPython検証、未来cutoff拒否、P09修正版の意味検証、複数sourceのP08優先、basket cache分離、完全版Vault docsが改善されました。

一方、独立レビューでは次の5件を再現しました。

1. accepted済みP08へP09 `accept`を行うとDB一意制約違反になり、`correct`では正式予想が二重化する。
2. 同一rawの別URL/source occurrenceへ再利用されたP05/P07を、P08がsource不一致として拒否する。
3. P09 `reject`後にP08ではなくP05/P07を案内し、`unresolved`後は同じレビューを無期限に案内する。
4. `made_at_source=unknown`でも架空の`made_at`を必須入力でき、正式予想・P11へ進む。P08 Schemaにtop-level `knowledge_cutoff`がない。P09 cutoffも予想時点を越え得る。
5. 複数日basketで共通取引日が1日しかなくても、的中・外れを計算する。

Round4の目的は、新機能を広げることではありません。正式予想の「有効な最新版」、source occurrenceごとのartifact適用関係、AIレビュー後の終端・再試行状態、時点境界、市場データ最低条件をDB正本として閉じ、実原文1件の縦断試験へ進める状態にすることです。

## 配置場所

このフォルダを次の位置へ配置してください。

```text
analyst_forecast_app_cursor_handoff/
├─ pyproject.toml
├─ src/
├─ tests/
├─ docs/
└─ Fix_instructions/
   └─ analyst_forecast_cursor_fix_instructions_round4/
```

project rootを二重階層にしないでください。

## Cursorでの実行順序

原則として1ファイルを1セッションで実行してください。

1. `01_PREFLIGHT_AND_REPRODUCTION.md`
2. `02_P08_REREVIEW_AND_FORECAST_LIFECYCLE_FIX.md`
3. `03_SOURCE_OCCURRENCE_REUSE_AND_LINEAGE_FIX.md`
4. `04_P09_STATE_MACHINE_AND_NEXT_ACTIONS_FIX.md`
5. `05_UNKNOWN_TIME_AND_CUTOFF_FIX.md`
6. `06_BASKET_COMMON_DATE_COVERAGE_FIX.md`
7. `07_MIGRATION_RESULTS_PACKAGE_AND_DOCS.md`
8. `08_FINAL_AUDIT.md`
9. `09_ACCEPTANCE_MATRIX.md`で最終照合

## 全工程の拘束条件

- 意味判断を伴うAI処理は、ユーザーが設定した高性能モデルだけを使う。廉価モデルへ自動切替しない。
- 人間承認を必須にしない。曖昧な判断は別AIレビューで解決し、解決不能なら理由付きterminal/excluded状態にする。
- raw原文は変更禁止。正規化、segment、予想、レビュー、修正版、再利用物は別artifactとして保存する。
- 話者推定、予想抽出、対象解決へ市場結果を渡さない。後知恵を禁止する。
- SQLiteを機械処理の正本とし、Markdown、CSV、NEXT_ACTIONSは再生成可能な表示物とする。
- accepted済みデータを訂正する場合も履歴を削除・上書きしない。append-onlyなlineageとactive/latestの明示を使う。
- 既存Schema、既存DB、既存案件を削除しない。DB変更は`0008`以降の新しいAlembic revisionで行う。`0001`～`0007`を編集しない。
- legacyデータを自動的にactive、本人確認済み、時点確認済みへ昇格させない。
- `reference/CHAT_HISTORY.pdf`は削除・改変しない。
- API key、token、実Vault絶対パス、raw、DB、市場cache、AI実データをGitへ追加しない。
- ループエンジニアリングは導入しない。
- commit、push、branch作成、PR作成を行わない。
- 指示外の大規模リファクタリングをしない。必要なら着手前に理由、代替案、影響範囲を報告する。
- pytest合計件数だけでREADYと判定しない。各negative caseをpublic APIまたはCLI経路で実際に再現・拒否する。

## Round4で変更してよいもの

- P08/P09 Schema、Pydantic model、固定JSON Schema、prompt
- AI artifact、forecast issuance、component、evidenceのrevision・active lifecycle
- source occurrenceと再利用artifactのassociation/derived artifact
- RunSourceのP08 review状態とworkflow/NEXT_ACTIONS
- made_at、made_at_source、knowledge_cutoff、time evidence
- basket共通取引日、evaluation audit metadata、unevaluable reason
- 結果・対象解決・市場評価のactive query
- Alembic、unit/integration test、packaged docs、最終監査文書

## 今回の範囲外

- 総合点・能力ランキング
- 複数アナリストの現在予想統合
- 上昇候補ランキング
- 時期・程度・早期実現の完全採点
- 期間不明予想の1/3/6/12か月自動観測
- PNG・ヒートマップ
- 情報源の完全自動収集
- exe化

これらを便乗実装せず、既存`FUTURE_ROADMAP`の記載を維持してください。

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

- `NOT_READY`: Round4受入項目に1件でも内部fail、未実装、未検証がある。
- `READY_FOR_REAL_SAMPLE_DIRECTIONAL_SLICE`: `09_ACCEPTANCE_MATRIX.md`の全項目がpassし、実原文1件の手動投入へ進める。
- `FULL_MVP_READY`: 今回は使用しない。正式仕様の全MVP要件を満たす場合だけ使用する。

