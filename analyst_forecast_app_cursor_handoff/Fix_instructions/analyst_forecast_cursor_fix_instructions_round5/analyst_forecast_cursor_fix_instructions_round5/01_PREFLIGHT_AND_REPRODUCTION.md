# 01 — 事前確認とRound5不具合再現

## 目的

独立レビューで確認した7項目を修正前の実行結果として固定します。単一予想やempty DBだけの試験で完了扱いにせず、今回失敗したデータ形状と操作順をそのまま回帰試験へ移します。

## Cursorへ渡す依頼文

```text
00_START_HERE.mdの拘束条件に従い、Round5 preflightを実施してください。commit・pushはしないでください。

project rootはrepository直下のanalyst_forecast_app_cursor_handoff/です。base commit 2f826edc7a5bfd5559e8c9a32cc8d9e58d598106を確認してください。

次のA～Gを最小probeまたは修正前にfailする回帰testで再現してください。

A. データ入り0007 DB migration
- upgrade_database(..., revision="0007")で過去Schemaを作る。
- migration fixtureに限りSQLで、analyst、run、source、run_source、AI import/artifact、forecast_group、forecast_issuance、forecast_evidence、forecast_component、target/mapping、evaluation、evaluation_snapshotの参照グラフを投入する。
- PRAGMA foreign_keys=ONの状態でheadへupgradeする。
- 0009のforecast_issuances再作成時にFOREIGN KEY constraint failedとなること、alembic_version、追加済み列、row countを記録する。
- 子行のないanalyst/runだけのfixtureではこの不具合を再現したことにしない。

B. 複数forecastのP09 correct
- public AI ingest経路で、同じaccepted P08内にforecast_refの異なる正式予想を2件作る。
- forecastごとに異なるcomponent/evidenceを持たせる。
- P09 correctで2件をそれぞれ訂正する。
- active 2件のlineage_root_idが同一になること、old_issuances[0]だけが全新世代の親になること、superseded_byが先頭新issuanceへ偏ることを記録する。
- 配列順を入れ替えたcorrected_payloadの挙動も記録する。

C. superseded componentのP11/P12/P13
- accepted P08 → P09 correctで旧componentをsupersededにする。
- 旧component IDを指定した正当な形式のP11を取り込み、acceptedになることを確認する。
- P11を旧componentがactiveな間に作成し、その後supersedeしてからP12を取り込むcaseを作る。
- 同様にP13へ進めるcaseを確認する。
- artifact classification、candidate/review/mapping行の増加、NEXT_ACTIONSを記録する。

D. unknown時刻の訂正とP09 cutoff
- 元P08: made_at=null、made_at_source=unknown。
- corrected P08: made_at=2026-01-10T08:00:00+00:00、必要なtime evidenceとP08 cutoff=08:00。
- P09自身: knowledge_cutoff=08:30。
- source boundary=09:00。
- P09 correctがacceptedになり、正式予想ができる現状を記録する。
- P09 cutoffが訂正後made_atと同値の場合もpositive controlとして用意する。

E. 単一symbolの1取引日
- normalized_start < normalized_endの複数日予想を作る。
- 単一symbol providerが期間内の1取引日だけ返す。
- direction-v2.0.0がexpired_hitまたはexpired_missとreturn/MFE/MAEを計算することを記録する。
- basket共通日1日のunevaluable caseではcommon_date_countとcoverage_auditがnullになることも記録する。

F. P09 reject契約
- decision=rejectでreject_terminalとreject_reasonを両方省略してSchema validationが通ることを確認する。
- retryable rejectでreasonなしが通ることを確認する。
- P09.md.j2、固定Schema、Pydantic generated schemaにretryable/terminal選択規則が揃っていないことを記録する。

G. 品質ゲート
- ruff format --check .
- ruff check .
- mypy src/analyst_forecast --ignore-missing-imports
- pytest -q
- git diff --check
- scripts/sync_packaged_docs.py --check
- alembic check
を実行し、format対象8ファイルとdiff checkの実結果を記録する。

各caseについて、入力、期待結果、実結果、DB query、例外、生成file、関連コードをdocs/06_実装/ROUND5_REPRODUCTION.mdへ記録してください。

一時probeを残す場合は02以降で正式testへ移し、最終状態で一時scriptへ依存しないでください。再現できない項目をYESやpassと書かず、入力・command・観測値を報告して停止してください。
```

## 必須成果物

- `docs/06_実装/ROUND5_REPRODUCTION.md`
- 修正前にfailする、または修正によってpassへ変わる正式test
- 次の不変条件を含むremediation plan
  - historical DBを子行込みで安全にheadへ上げる
  - forecast correctionは旧refと新refを一対一で対応させる
  - inactive componentをP11/P12/P13/評価へ通さない
  - P09の知識上限は訂正後made_atも越えない
  - current direction methodは1取引日から複数日returnを作らない
  - reject区分をAI出力で明示必須にする

## 完了条件

- A～Gを実行証拠で確認している。
- test名だけでなく、修正前DB値またはerror codeを記録している。
- 02～08の修正対象とR5受入IDを対応付けている。

