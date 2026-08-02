# 09 — Round5最終監査

## 目的

個別unit testの合計ではなく、historical DB、AI ingest、SQLite、workflow、結果、CSV評価、wheelをつないだ縦断シナリオでRound5修正とRound4合格項目の非回帰を確認します。

## Cursorへ渡す依頼文

```text
00_START_HERE.mdから08までを、自己申告ではなく実行証拠で最終監査してください。commit・pushはしないでください。

次のシナリオA～Rを実行してください。migration historical fixtureを除き、目的状態を作るためにDB lifecycle列を直接書き換えず、public application APIまたはCLI/AI ingest経路を使ってください。

A. データ入り0007 migration
0007 Schemaへissuance、2 components、evidence、mapping、evaluation、snapshotをseedし、foreign_keys=ONでheadへupgrade。件数・主要値保持、FK check 0、integrity ok。

B. 既適用0009 migration
0009 DBをnew headへupgradeし、active constraint、coverage列、Schema metadataを確認。head再実行もpass。

C. migration失敗復元
upgrade途中へ強制例外を入れ、alembic_version、Schema、row count、content hashが元DBと同じで、0008/0009列の部分残留がない。

D. 複数forecast update
accepted P08にA/Bの2正式予想を作り、P09 correctでA/Bを更新。異なるlineage root、各old→各newの一対一、active各1件。

E. reorder/add/remove
corrected payloadを並べ替え、A更新、B削除、C追加。配列順に依存せずA/Cだけactive、B履歴保持。

F. ambiguous correction
1旧→2新、未申告追加、存在しないrefを拒否。review/active世代に部分変更なし。

G. active unique/rollback
同一lineage active 2件をDB constraintで拒否。P09適用途中の強制失敗でも旧activeとreview状態を保持。

H. superseded P11
P09 correct後の旧componentへP11。inactive_forecast_componentで拒否し、candidate/mappingを増やさない。

I. stale P12/P13
P11またはP12後、次工程前にP09 correct。古いchainのP12/P13を拒否し、review/adjudication/lockを増やさない。新componentのchainはpass。

J. corrected time boundary
old unknown/null、corrected made_at=08:00、corrected cutoff=08:00、P09 cutoff=08:30、source boundary=09:00を拒否。同じP09 cutoff=08:00は他条件が正当なら一度だけ正式化。

K. multi-forecast time boundary
訂正A=08:00、B=09:00でP09 cutoff=08:30を拒否。配列順変更、add/removeも同じ規則。

L. single-symbol coverage
複数日予想へ1取引日だけを返し、unevaluable、全評価値null、coverage audit保存。2取引日では正しい数値。

M. basket coverage
共通日1日をunevaluableにしcommon_date_count=1と銘柄別auditを保存。共通2日では正しいweighted return。後続単一AAA cacheは汚染されない。

N. P09 reject contract
新Schemaでdisposition/reason省略を拒否。retryable、terminal、unresolvedを別fixtureで有限の期待状態へ進める。legacy省略rejectを黙ってretryable化しない。

O. active結果
旧世代、削除lineage、reject、unresolved、legacy conflictを含む状態から04_resultsとNEXT_ACTIONSを再生成し、active世代だけを通常count・次行動へ出す。

P. Round3/Round4非回帰
司会者/legacy_unknown非正式化、不正P09 offset拒否、別URL reuse、2 source P08優先、mixed currency unevaluable、basket cache分離、unknown日時非正式化、Vault完全版docsを再確認。

Q. Wheel
clean venvのwheelからhelp、init、model設定、run/source、複数P08/P09、旧P11拒否、CSV 1日coverage、docs/Schema/promptを確認。internal testはskip/xfailなし。

R. 品質・Git
ruff format/check、mypy、pytest、diff check、docs sync、Alembic check、buildがpass。CHAT_HISTORY維持、secret/DB/raw/cache/build生成物を追跡しない。

各シナリオについて、入力、command/test名、期待結果、実結果、DB query値、error code、生成file、判定をdocs/06_実装/FINAL_REVIEW_ROUND5.mdへ記載してください。

10_ACCEPTANCE_MATRIXの全項目をtest名、DB constraint/列、生成成果物、CLI出力へ対応付けてください。「実装した」「コードがある」「pytest全体がpass」だけで個別項目をpassにしないでください。
```

## 判定規則

- A～RまたはR5-001～R5-052に内部fail、未実装、未検証、skip、xfailがあれば`NOT_READY`。
- networkだけが外部制約で、同じ内部処理のmock/CSVがpassしている場合だけ`external_blocked`を許容する。
- 全Round5項目pass時だけ`READY_FOR_REAL_SAMPLE_DIRECTIONAL_SLICE`。
- 総合点、PNG、1/3/6/12か月観測等が未実装なら`FULL_MVP_READY`を使用しない。

## 最終報告形式

```text
判定:
対象base commit:
変更ファイル数:
pytest:
ruff format/check:
mypy:
docs sync:
Alembic historical/head/rollback:
wheel clean venv:
scenarios A-R:
acceptance R5-001～R5-052:
skip / xfail:
external_blocked:
意図的未実装:
次の一作業:
commit / push: 未実施
```

