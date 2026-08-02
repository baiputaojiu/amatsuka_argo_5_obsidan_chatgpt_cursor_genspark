# 07 — Migration、active結果、package、説明書、品質ゲート

## 目的

02～06の変更を既存DBへ安全に反映し、active最新版だけを成果物へ出し、wheelから同じ仕様で動作することを確認します。metadata drift、packaged resourceの更新漏れ、legacyの誤昇格を防ぎます。

## Cursorへ渡す依頼文

```text
00～06の変更を統合し、migration、結果生成、package、docs、品質ゲートを実行してください。commit・pushはしないでください。

Migration:

- 既存0001～0007を編集しない。必要なDB変更は0008以降の新revisionにする。
- SQLiteでは必要に応じてrender_as_batchを使い、upgrade中のdata lossを防ぐ。
- upgrade前backupを維持する。
- active lineage、artifact applicability、source review state、unknown time、evaluation coverageに必要なFK、unique constraint、check/indexをmetadataと一致させる。
- active 1件制約をSQLiteでpartial unique index等にできない場合、transaction guardと整合性auditを併用し、その理由を記録する。
- legacy行を本人確認済み、時点確認済み、active最新版へ無条件に昇格させない。
- legacyに複数issuanceがありactiveを決められない場合はlegacy_conflict/excludedとして保持し、警告を出す。
- downgradeで新lifecycleを失う場合は、不可逆情報とbackup前提をmigration docstringへ明記する。

Migration試験:

- empty database → head
- 0001 → head
- 0005 → head
- 0007 → head
- Round3 fixture data入りDB → head
- accepted P08とissuance/evaluation入りfixture → head
- headでalembic check差分0
- PRAGMA foreign_key_check pass
- upgrade前後のraw/source/artifact/issuance/component/evaluation件数保持
- legacy conflictがactive集計へ入らないこと

Active結果:

- results、workflow、target resolution、evaluationが同じactive queryを使用する。
- 04_resultsのforecast CSV/Markdown、evaluation CSV/Markdown、summaryはactive generationだけを通常表示する。
- 監査用に履歴を表示する場合はactive結果と別section/fileにし、superseded/rejected/unresolvedをstatus付きで明確に分離する。
- summaryのforecast count、hit/miss/unevaluableはsuperseded componentの過去評価を二重計上しない。
- NEXT_ACTIONSはsuperseded/excluded componentを案内しない。
- result再生成は冪等で、DB正本と一致する。

Package/Schema/docs:

- P08/P09固定JSON Schema、Pydantic、Jinja prompt、生成例、prompt catalogを同期する。
- repo docsを編集正本とし、packaged docs sync --checkをpassさせる。
- USER_MANUAL、SYSTEM_SPECIFICATION、DATA_MODEL、WORKFLOW_DIAGRAMS、TROUBLESHOOTINGへ次を追記する。
  - accepted P08再レビューのaccept/correct/reject/unresolved
  - active/superseded lineageと結果への反映
  - source occurrenceへのpreprocess reuse条件とlineage
  - P08再抽出とterminal unresolvedの違い
  - made_at unknown/nullとknowledge cutoff
  - basket共通日不足のunevaluable理由
- FUTURE_ROADMAPの既存詳細を短縮しない。
- runtimeでrepo相対pathに依存せず、wheel内resourceだけでinitできるようにする。

Wheel縦断:

- sdistとwheelをbuildする。
- repositoryのsrcをPYTHONPATHへ入れずclean venvへwheelをinstallする。
- analyst-forecast --help、init --vault-root、model設定、run create、YouTube/blog source import、NEXT_ACTIONSを実行する。
- wheel環境でP05/P07→P08→P09の代表caseを取り込み、DB、NEXT_ACTIONS、04_resultsを確認する。
- packaged migration、Schema、prompt、完全版docsが含まれることを確認する。

品質ゲート:

- ruff format --check .
- ruff check .
- mypy src
- pytest。deselected integrationは理由を記録する。
- git diff --check
- packaged docs sync --check
- 全固定JSON Schema parse
- Pydantic generated schemaと固定Schemaのrequired/enum主要項目照合
- alembic upgrade/check/foreign_key_check
- buildしたwheelのcontents検査

Git衛生:

- reference/CHAT_HISTORY.pdfをbb7d167から変更しない。
- .env.example以外の.env、secret、token、API keyを追跡しない。
- SQLite、raw、market cache、backup、AI出力、実Vault絶対pathを追跡しない。
- build、dist、egg-info、venv、pytest cacheを追跡しない。

外部network試験が環境制約で実行できない場合だけexternal_blockedを使い、command、error分類、代替mock/CSV試験を記録してください。内部Schema、SQLite、CSV、state machine、wheelのfailをexternal_blockedで隠さないでください。
```

## 必須成果物

- `0008`以降の新Alembic revision
- migration compatibility tests
- active/superseded query regression tests
- wheel clean venv縦断testまたは再現可能script
- 更新済みrepo docsとpackaged docs
- `docs/06_実装/ROUND4_QUALITY_GATE.md`

## 完了条件

- 既存DBの履歴・件数を失わずheadへupgradeできる。
- active最新版だけが通常結果と次行動へ出る。
- clean wheelでもsource import、P08/P09、Vault docsまで同じ仕様で動く。
- 全内部品質ゲートがpassする。

