# 07 — Alembic・format・package品質ゲート

## 目的

最終監査でpassと記載されたものの、独立確認で失敗した`ruff format --check`と`alembic check`を修正し、修正後のwheelをクリーン環境で検証します。

## 確認済みの結果

```text
pytest                 64 passed, 3 deselected
ruff check             pass
mypy src               pass
ruff format --check    fail: 3 files
Alembic upgrade head   pass: 0005
Alembic check          fail
wheel build            pass
clean venv install     pass
wheel版 init           pass
```

Alembic checkの差分：

- `sources.raw_artifact_id`へmodel上は存在する外部キーを追加要求
- `ix_evaluations_component_as_of_method`をmigrationは作るがmodel metadataは削除要求

## コピペ用依頼文

```text
fix_instructions_round2/00_START_HERE.mdの共通拘束条件に従い、01～06の全変更後にmigration、format、package品質ゲートを修正してください。commit・pushはしないでください。

既存0001～0005を履歴書換えせず、必要な変更は0006以降の連続Alembic revisionで実装してください。既存DBを削除して作り直す方法は禁止です。migration前backupを維持してください。

少なくとも次のmetadata差分を解消してください。

- sources.raw_artifact_idのForeignKeyがmodelと実DBで一致する。
- ix_evaluations_component_as_of_methodを維持するならmodelへ正式定義し、不要なら新migrationで削除する。目的を文書化する。
- 01～06で追加したSchema/DB列/index/constraintがAlembic headとSQLAlchemy metadataで一致する。

次のmigration経路を実DBファイルのコピーまたはfixtureで試験してください。

1. 空DB→head。
2. 0001→head。
3. 0005→head。
4. raw artifact、source occurrence、legacy AI import、forecast、evaluationを含む既存DB→head。
5. migration失敗時にtransaction rollbackまたは復旧可能なbackupが残る。

`alembic check`を実行し、内部差分が0であることを確認してください。環境依存として未実行扱いにせず、内部SQLite fixtureで検証してください。

ruff formatを実行してから、ruff format --check、ruff check、mypy src、pytestをすべて実行してください。テストを通すために検証を弱めたり、integration markerへ内部テストを移したりしないでください。

wheelを作成し、source treeをimportできないクリーンvenvへwheelをインストールしてください。その環境で次を確認してください。

- analyst-forecast --help
- package内Schema、prompt、詳細docsを読める
- temp Obsidian Vaultへinitできる
- startまたは非対話run createで案件を作れる
- Alembic headまで作成される
- P05/P07/P08/P11/P12/P13/P06/P09 snapshotが生成される

Git追跡対象を確認し、database.sqlite、raw、案件AI出力、API key、実Vault path、ローカルconfigが含まれていないことを確認してください。reference/CHAT_HISTORY.pdfはユーザー許可済みなので削除しないでください。

品質ゲートの実行コマンド、終了コード、件数をdocs/06_実装へ正確に記録してください。実行していない検査をpassと書かないでください。ネットワークintegrationだけは、ネットワークまたはFRED keyがない場合にexternal_blockedとできます。
```

## 完了条件

- `ruff format --check .`がpass。
- `alembic check`が差分0。
- 空DBと既存DBの両方をheadへ移行できる。
- wheelのクリーン導入とVault初期化がpass。
- 内部失敗をexternal_blockedとして扱っていない。
