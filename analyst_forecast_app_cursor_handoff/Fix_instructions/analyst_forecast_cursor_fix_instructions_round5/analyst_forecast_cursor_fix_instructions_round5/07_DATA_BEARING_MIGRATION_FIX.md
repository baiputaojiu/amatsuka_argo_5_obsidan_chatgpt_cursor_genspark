# 07 — 子行を含む既存DBのmigrationと失敗時原子性

## 目的

ForecastIssuanceを参照するcomponent/evidence/evaluation等が存在する0007 DBを0009へ上げるとforeign key errorになり、失敗後に一部だけSchema変更されたDBが残る問題を修正します。

## Cursorへ渡す依頼文

```text
00～06の最終DB modelを、データ入りhistorical DBへ安全に反映してください。commit・pushはしないでください。

既存0001～0007は編集禁止です。Round4の0008/0009はまだ受入未完了ですが、すでに0009まで適用済みの開発DBも存在し得ます。次の両経路を成立させてください。

1. データ入り0007 DB → fixed migration chain → new head
2. 既に0009のDB → new head

0009を修正しなければ0007からheadへ到達できない場合、0009を変更してよいですが、次を必須にします。

- なぜ後続revisionだけでは救えないかをmigration docstringとROUND5_QUALITY_GATEへ記載する。
- 既適用0009 DBには新revision（例: 0010）で不足constraint/backfill/auditを適用する。
- 0009のrevision IDを変えない。
- empty DBだけでなく子行付き0007 DBで検証する。

SQLite table rebuild:

- PRAGMA foreign_keys=ONを試験開始時に明示する。
- forecast_issuancesをbatch rebuildする際、forecast_components、forecast_evidenceその他のFK参照を破壊しない。
- foreign_keysを一時無効化する設計なら、transaction境界、再有効化、foreign_key_check、例外時復旧をコードとtestで保証する。接続任せの黙示動作にしない。
- raw/source/artifact/issuance/evidence/component/target/mapping/evaluation/snapshotのID、row count、主要値を保持する。
- FKが別table名へ書換わっていないことをPRAGMA foreign_key_listで確認する。

active/lineage migration:

- 02のactive partial unique indexまたは等価制約をmetadataとmigrationへ入れる。
- 既存単一active lineageは決定的にbackfillしてよい。
- 同一lineage候補が複数、または対応不能なlegacy rowsを勝手に最新activeへしない。legacy_conflict/excludedと監査理由を残す。
- 0008の「全legacy行をactiveにする」処理がこの方針と矛盾する場合は修正する。

失敗時原子性:

- upgrade_database開始前backupを作る。
- Alembic/SQLite DDLが完全rollbackできない場合、失敗時にbackupを原位置へ復元し、元alembic_version、Schema、row count、hashを維持する。
- 失敗を呼出元へ明示的に返し、部分変更DBでアプリを継続しない。
- backup復元testでは意図的にmigration途中で例外を発生させ、0008列だけ残らないことを確認する。

historical fixture:

- test内でupgrade_database(revision="0007")後、0007 Schemaへ直接SQLを使って参照グラフをseedしてよい。
- analyst/runだけの簡易fixtureは不可。
- 少なくともforecast_issuance、2 component、evidence、target mapping、evaluation、snapshotを含める。
- 可能ならRound3のaccepted P08→P09 correct相当として2 issuanceを含むfixtureも追加する。
- binary SQLite、実DB、raw実データをGitへ追加しない。fixtureはtest実行時に生成する。

Alembic metadata:

- empty/0001/0005/0007/0009/headの各経路を確認する。
- alembic check差分0。
- PRAGMA integrity_check=ok。
- PRAGMA foreign_key_checkが0件。
- Base.metadataとtable/index/FK/nullability/defaultが一致する。
```

## 必須テスト

1. 子行付き0007 fixtureをnew headへupgradeできる。
2. issuance/component/evidence/evaluation/snapshotの件数と主要値が保持される。
3. `PRAGMA foreign_key_check`が空、`integrity_check`がokである。
4. component/evidence FKが再作成後の`forecast_issuances`を参照する。
5. alembic_versionがnew headになる。
6. empty/0001/0005/0007の各経路がpassする。
7. 既適用0009 DBをnew headへupgradeできる。
8. head→head再実行が冪等である。
9. alembic check差分0である。
10. migration途中の強制例外後、元revision、Schema、row count、content hashが復元される。
11. 失敗後に0008/0009の一部列だけ残らない。
12. legacy conflictをactiveへ自動昇格させない。
13. active一意index/constraintが実DBに存在する。
14. 同一lineage active重複insertがDBで失敗する。
15. backup pathや実DBをGitへ追跡しない。

## 完了条件

- 実データ参照グラフを保ったまま0007/0009からnew headへ上げられる。
- 成功後のFK・metadataが整合する。
- 失敗時に利用不能な半端DBを残さない。

