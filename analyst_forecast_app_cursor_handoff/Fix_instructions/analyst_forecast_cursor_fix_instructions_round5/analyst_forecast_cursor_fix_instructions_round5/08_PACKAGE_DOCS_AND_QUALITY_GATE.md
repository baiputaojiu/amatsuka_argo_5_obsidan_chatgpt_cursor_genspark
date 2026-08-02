# 08 — Package、説明書、wheel、品質ゲート

## 目的

02～07の変更をrepo、packaged resources、wheel、説明書へ一貫して反映し、formatやdiffの失敗、簡略化されたwheel test、最終監査文書の過大判定を防ぎます。

## Cursorへ渡す依頼文

```text
00～07を統合し、package、docs、wheel、品質ゲートを実行してください。commit・pushはしないでください。

Schema/prompt同期:

- P09新SchemaのPydantic model、generated schema、固定JSON Schema、Jinja prompt、prompt catalog、生成例を一致させる。
- P09 reject_disposition/reasonのconditional requiredとlegacy adapterを説明する。
- 02でforecast_operations等を追加した場合、そのrequired/enum/ref整合も全resourceへ反映する。
- repo resourcesとwheel内resourcesをscripts/sync_packaged_docs.py --check等で一致させる。
- 全固定JSON SchemaをDraft 2020-12としてparseし、Pydantic generated schemaとの主要required/enum/if-thenを比較する。

説明書:

- USER_MANUAL、SYSTEM_SPECIFICATION、DATA_MODEL、WORKFLOW_DIAGRAMS、METHODOLOGY、TROUBLESHOOTING、PROMPT_CATALOG、IMPLEMENTATION_STATUSを必要範囲で更新する。
- 次を後から別AIが復元できる詳しさで記載する。
  - 複数forecast correctionのupdate/add/removeとlineage
  - active一意制約と旧componentの扱い
  - P11/P12/P13のactive guard
  - P09 cutoffとcorrected made_atの関係
  - 単一銘柄・basketの2取引日条件とcoverage audit
  - P09 reject dispositionの有限状態遷移
  - 0007/0009からのmigrationと失敗時backup復元
- FUTURE_ROADMAPの既存詳細を短縮しない。
- FULL_MVP未実装項目をREADY対象と混同しない。

結果/NEXT_ACTIONS:

- 04_resultsの通常forecast/evaluation/summaryはactive generationだけを含む。
- 履歴表示を作る場合は通常結果と別section/fileにし、statusを明記する。
- inactive component、削除lineage、legacy conflictをNEXT_ACTIONSへ出さない。
- coverage不足evaluationはunevaluable reasonとaudit概要を表示する。
- 結果再生成を冪等にする。

Wheel縦断:

- sdistとwheelをbuildする。
- repository srcをPYTHONPATHへ入れないclean venvへwheelだけをinstallする。
- wheelからhelp、init --vault-root、model設定、run create、YouTube/blog source import、NEXT_ACTIONSを実行する。
- P05/P07→複数forecast P08→P09 correctを取り込み、lineageをDBで確認する。
- superseded旧componentのP11を拒否する。
- CSV providerで単一銘柄1取引日をunevaluableにし、coverage auditを確認する。
- packaged migrationでデータ入り0007 fixtureをheadへ上げる。少なくともwheelに全revisionが含まれることを確認する。
- 完全版docs、P09 Schema、P09 promptがwheelに含まれることを確認する。
- build失敗やinternal縦断失敗をskip/xfailにしない。環境上build toolがない場合も依存を明示し、品質ゲートをfailとして報告する。

品質ゲート:

1. python -m ruff format .
2. python -m ruff format --check .
3. python -m ruff check .
4. python -m mypy src/analyst_forecast --ignore-missing-imports
5. python -m pytest -q
6. git diff --check
7. python scripts/sync_packaged_docs.py --check
8. alembic upgrade head
9. alembic check
10. PRAGMA foreign_key_check / integrity_check
11. python -m build
12. wheel contents inspection

formatter実行後にtestを再実行してください。pytest合計だけでなく、R5-001～052をtest名・DB assertion・CLI outputへ対応付けてください。

Git衛生:

- reference/CHAT_HISTORY.pdfをbase commitから変更しない。
- .env.example以外の.env、secret、token、API keyを追跡しない。
- SQLite、backup、raw、market cache、AI実出力、実Vault絶対pathを追跡しない。
- dist、build、egg-info、venv、pytest cache、mypy cacheを追跡しない。
- git status --shortの全行を確認し、意図したsource/test/docs/migration以外を残さない。
```

## 必須成果物

- `docs/06_実装/ROUND5_REPRODUCTION.md`
- `docs/06_実装/ROUND5_QUALITY_GATE.md`
- 更新済み`docs/06_実装/IMPLEMENTATION_STATUS.md`
- migration compatibility/rollback tests
- wheel clean-venv vertical testまたは再現可能な非skip script
- Schema/prompt/docs sync tests

## 完了条件

- format、check、mypy、pytest、diff check、docs sync、Alembic、buildがすべてpassする。
- clean wheelでも代表negative/positive経路がrepo実行時と同じになる。
- packaged docs/Schema/promptがrepo正本と一致する。
- 文書の判定が実行結果を超えない。

