# 01 — 事前確認とRound4不具合再現

## 目的

今回の5件を修正前に実行結果で固定し、happy pathだけを増やして完了扱いにすることを防ぎます。この工程では原因と不変条件を整理し、無関係な実装変更はしません。

## Cursorへ渡す依頼文

```text
00_START_HERE.mdの拘束条件に従い、Round4のpreflightを実施してください。commit・pushはしないでください。

project rootはrepository直下のanalyst_forecast_app_cursor_handoff/です。base commit bb7d167a849421b3441dcf06c054e497263ccfa0を確認してください。

次の不具合を、DB直接書換えではなく既存のimport/public application APIを使う最小probeまたはfailする回帰testで再現してください。

A. accepted済みP08 + P09 accept
- P05/P07 → accepted P08 → ForecastIssuance 1件を作る。
- そのaccepted P08をreviewed_artifact_idに指定してP09 decision=acceptを取り込む。
- 現状のDB一意制約違反、RuntimeError、artifact/fileの残留有無を記録する。

B. accepted済みP08 + P09 correct
- ForecastIssuance 1件の状態から、正当なcorrected_payloadを持つP09を取り込む。
- issuance/component/evidence件数、どちらが結果・P11へ進むかを記録する。
- 訂正前後が両方activeとして数えられることを確認する。

C. 別source occurrenceへの前処理再利用
- 同じraw bytes、同じanalyst、同じmedium、別URLのsource occurrenceを2件作る。
- source 1でP05またはP07をacceptedにし、source 2へ再利用させる。
- workflowがsource 2をpreprocess済みと扱う一方、そのartifactを参照するsource 2のP08がinvalid_upstream_referenceになることを記録する。

D. P09 reject/unresolved
- needs_review P08へP09 rejectを取り込み、next actionがRUN_P08ではなくRUN_PREPROCESSになることを確認する。
- needs_review P08へP09 unresolvedを取り込み、同じreviewed artifactへのREVIEW_AI_OUTPUTが繰り返されることを確認する。
- accepted済みP08に対するreject/unresolvedも現状挙動を確認する。

E. 日時不明とcutoff
- made_at_source=unknownと任意のmade_atを持つP08がaccepted、formal issuance作成、RUN_P11となることを確認する。
- P08固定JSON Schema/Pydantic/top-level payloadにknowledge_cutoffがないことを確認する。
- P09 knowledge_cutoffがsource boundary以内だが、review対象ForecastIssuance.made_atより後でも受理されるcaseを再現する。

F. basket共通日1日
- 2銘柄以上、複数日予想、各symbolの共通日が1日だけのmarket dataを与える。
- 現状がunevaluableではなくdirection_resultを計算することを確認する。

各caseで、入力、期待結果、実結果、DB件数、DB lifecycle、次行動、生成成果物、例外、関連コードをdocs/06_実装/ROUND4_REPRODUCTION.mdへ記録してください。

baselineとしてpytest、ruff format --check、ruff check、mypy、packaged docs sync、alembic checkを実行してください。既存0001～0007 migrationは編集しないでください。

一時probeを残す場合は、02以降で正式testへ移し、最終状態で一時scriptへ依存しないでください。再現できない項目をYESやpassにせず、入力と観測値を報告して停止してください。
```

## 必須成果物

- `docs/06_実装/ROUND4_REPRODUCTION.md`
- 修正前にfailする、または理由を限定したxfail test
- 次の不変条件を含むremediation plan
  - 同一論理予想についてactive issuanceは最大1世代
  - artifact再利用はsource occurrenceへ明示的に適用可能
  - P09の各decisionは有限状態遷移を持つ
  - 時刻不明を架空datetimeへ変換しない
  - 複数日returnを1観測点から計算しない

## 完了条件

- A～Fを実行証拠で確認している。
- baseline合格項目を壊してはならないことを記録している。
- 02～07の修正対象と受入IDを対応付けている。

