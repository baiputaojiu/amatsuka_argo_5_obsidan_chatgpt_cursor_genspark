# 07 — 最終横断監査

## 目的

各修正が個別テストだけで成功し、実ワークフロー全体では接続されていない状態を防ぐ。

## コピペ用依頼文

```text
01～06の修正結果を、コード変更を最小限に抑えながら最終監査してください。まず監査し、失敗が見つかった場合だけ修正してください。

正式仕様、REMEDIATION_PLAN.md、IMPLEMENTATION_STATUS.md、Git差分を確認し、次の縦断シナリオを一時ディレクトリと匿名fixtureで実行してください。

シナリオA：予想あり
1. 新しいVault／workspaceを初期化。
2. 必須docsと中央promptsを確認。
3. 対話wizardまたはflagでYouTube案件を作成。
4. rawを取込み。
5. P05結果を取込み。
6. P08結果を取込み。
7. P11提案を取込み。
8. 別prompt executionのP12レビューを取込み。
9. mappingをlock。
10. AI取込み表示とNEXT_ACTIONSからcomponent ID入りコマンドを取得。
11. CSV市場評価を実行。
12. 04_resultsのMarkdown／CSVを確認。
13. status再生成後に次行動が正しいことを確認。

シナリオB：予想なし
- raw処理後、forecasts=[]を受理する。
- processed_no_forecastとなり、同じP08を無限に要求しない。
- 調査網羅性には処理済み情報源として数える。

シナリオC：テーマ対象解決不能
- dummy symbolなしでunresolvableを保存する。
- 不的中ではなくunevaluableとして表示する。
- 不要な市場取得を実行しない。

シナリオD：AI再レビュー
- 最初の低確信度出力をneeds_reviewにする。
- 別AI出力で解決する。
- 元ファイルを監査用に残したままworkflowが先へ進む。

シナリオE：重複原文
- 同一bytesを同じアナリストの別案件へ取込む。
- 同一bytesを別アナリストへ取込む。
- source occurrence、URL、analyst、raw artifact、案件入力が混線しない。
- 同一処理versionの不要な再処理をしない。

シナリオF：複数評価
- component Aを異なる2つのas_ofで評価。
- component Bを未評価のままにする。
- workflowがBを未評価として案内する。
- Aの古いunevaluable後に最新成功を作り、現在状態が最新成功になることを確認する。

シナリオG：方向
- 上昇、下落、flatについてdirection、MFE、MAEを確認。
- 下落予想の有利・不利方向が逆転していないことを確認。

また次を実行してください。
- ruff format --check .
- ruff check .
- mypy src
- pytest
- 空DBからalembic upgrade head
- 既存1.0.0 DB fixtureからupgrade head
- alembic check
- buildしたwheelをクリーンvenvへ入れて、initとCLI helpを確認

yfinance/FREDの実ネットワーク試験は外部rate limitやキー未設定をコード不良と混同せず、成功・skip・provider unavailableを分けて報告してください。市場値の取得不能を推測で成功扱いしないでください。

次の点を特に監査してください。
- reference/CHAT_HISTORY.pdfはユーザー許容済みのため、削除・履歴除去されていない。
- 新たな秘密情報、個人パス、DB、rawデータがGit追跡されていない。
- 人間承認が必須工程に追加されていない。
- P12がP11と別実行として記録される。
- 市場結果が対象解決入力へ入っていない。
- rawが上書きされていない。
- 総合点一つを新設していない。
- ループエンジニアリングを導入していない。

監査結果をdocs/06_実装/FINAL_REVIEW.mdへ、pass / fail / not_applicable / external_blockedで記録してください。各failには再現コマンド、期待、実際、原因、修正を記載してください。

最後に、MVPを次のいずれかで判定してください。
- READY_FOR_REAL_SAMPLE：匿名fixtureの全縦断が成功し、実原文5～10件の検証へ進める。
- READY_WITH_LIMITATIONS：限定条件付きで進める。条件を列挙。
- NOT_READY：実データ投入を止める重大問題がある。

commit・pushは行わず、変更ファイル、テスト結果、判定、残課題、次の一作業をユーザーへ報告してください。
```

