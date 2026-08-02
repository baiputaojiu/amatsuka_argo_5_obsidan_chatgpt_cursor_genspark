# 05 — migrationのデータ保持証拠とDDL後失敗復元

## 目的

Round5で改善したdata-bearing 0007→head migrationを維持しつつ、「件数が同じ」だけではなくID・legacy値・参照関係が保持されたことを証明します。また、migration開始前の失敗ではなく、実際にDDLとデータ変更が一部commitされた後のbackup復元を試します。

## Cursorへ渡す依頼文

```text
00～04に従い、migration testと必要最小限のbackup/restore処理を強化してください。commit・pushはしないでください。

現状の0007→head成功、FK check、integrity check、active unique indexは改善済みです。新しい失敗を再現しない限りmigration本体を不要に書き換えないでください。

1. data-bearing 0007 fixture
- upgrade_database(database, revision="0007")でSchemaを作る。
- migration fixtureに限り0007 tableへ直接SQLでseedしてよい。
- analyst、run、source、run_source、AI import/artifact、forecast_group、forecast_issuanceを最低2件、componentを最低2件、evidence、target、mapping、evaluation、evaluation_snapshotを参照整合付きで入れる。
- nullable、Unicode、日本語、quote、長文、decimal、datetime、JSON、nullを含むsentinel値を用意する。
- foreign_keys=ONでseed後のforeign_key_checkが0件であることを確認する。

2. before/after snapshot
- 各対象tableについてrow countだけでなくprimary key集合を比較する。
- 0007時点に存在した全columnをlegacy projectionとし、全rowをprimary key順、column名順、型を失わないcanonical representationへ変換してSHA-256化する。
- migrationで追加されたcolumnはlegacy projection hashから除外し、別途期待default/backfillをassertする。
- BLOBはhex、datetime/decimal/text/nullは区別し、JSON textを勝手に意味正規化して差を隠さない。
- impacted parent tableだけでなくcomponent/evidence/evaluation/snapshot等の子tableにもsnapshotを持つ。
- selected sentinel値、old→child FK ID、lineage/lifecycle backfillを個別assertする。
- SQLiteファイル全体のhashは成功migrationの前後比較には使わない。Schema変更で変わるためである。

3. DDL後の強制失敗
- 既存testのようにAlembic処理開始前に例外を投げて完了としない。
- testでは同じupgrade wrapperがbackupを作った後、Alembic呼出しを差し替え、対象DBへ実際に`ALTER TABLE ... ADD COLUMN forced_partial_marker TEXT`を行い、sentinel rowもUPDATEし、commitした後に例外を投げる。
- production DBへtest markerを残すmigrationを追加しない。monkeypatch/fake commandはtest内だけにする。
- wrapperが例外を呼出元へ返しつつbackupを原位置へ復元することを確認する。
- 復元後、alembic_version、table/column/index/FK一覧、row count、PK集合、legacy projection hash、sentinel値がbeforeと一致する。
- `forced_partial_marker`が存在せず、更新したsentinel値も元に戻ることを確認する。
- backup copyからの復元であるため、可能なら元DB file SHA-256も一致させる。WAL/SHMを使用する場合はcheckpoint/close手順を正本化する。

4. migration matrix
- empty→head、0001→head、0005→head、0007→head、既適用0009→head、head→headを実行する。
- PRAGMA integrity_check=ok、foreign_key_check=0。
- Alembic check差分0。
- active lineage partial unique index/constraintが実DBに存在し、重複active insertを拒否する。
- migration/restore testをskip/xfailにしない。
```

## 必須テスト

1. child graph付き0007→head成功。
2. 全tableのPK集合とlegacy projection hash一致。
3. sentinel値と全FK関係保持。
4. 新column/default/backfillが期待どおり。
5. FK/integrity/Alembic metadataが正常。
6. 0009→headとhead→headが冪等。
7. DDL add column + data UPDATE + commit後の強制例外。
8. 復元後にmarker columnなし、旧値・Schema・revision・logical hash一致。
9. failureをsuccessに変換せず呼出元がMigrationError等を受け取る。
10. backup、WAL/SHM、fixture DBがGit追跡されない。

## 完了条件

- R5-002の証拠不足を、全PK・全legacy column projection hashで解消する。
- R5-006の証拠不足を、実DDL・data commit後の復元で解消する。
- Round5で成功したmigration機能を非回帰にする。

