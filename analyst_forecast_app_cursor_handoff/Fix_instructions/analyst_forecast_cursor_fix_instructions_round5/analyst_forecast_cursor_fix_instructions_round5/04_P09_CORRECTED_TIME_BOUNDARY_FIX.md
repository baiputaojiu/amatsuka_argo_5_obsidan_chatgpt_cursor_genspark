# 04 — P09訂正後時刻とknowledge cutoff境界

## 目的

元P08の日時がunknownの場合、P09 `correct`自身の`knowledge_cutoff`が訂正後`made_at`より後でも正式化できる問題を修正します。P09レビューとcorrected P08の両方が、訂正後の各予想時点以前の知識だけを使うようにします。

## Cursorへ渡す依頼文

```text
00～03に従い、P09 decision=correctの時点検証を修正してください。commit・pushはしないでください。

現在は、P09 cutoffをreviewed P08内のnon-null made_atと比較し、corrected P08の自分自身のcutoffだけを訂正後made_atと比較しています。元made_atがnullの場合、P09自身のcutoff > corrected made_atを見逃します。

次の順序で共通time-boundary serviceを実行してください。

1. P09.knowledge_cutoffはtimezone付きである。
2. P09.knowledge_cutoff <= source allowed boundary。
3. decision=accept/reject/unresolvedでreview対象に既知made_atがある場合、P09 cutoff <= 対象となる各既知made_at。
4. decision=correctではcorrected_payloadをP08 Schemaと通常意味検証へ通す。
5. corrected P08.knowledge_cutoff <= source allowed boundary。
6. corrected P08.knowledge_cutoff <= corrected payload内の各formal forecast made_at。
7. P09.knowledge_cutoff <= corrected payload内の各update/add対象formal forecast made_at。
8. remove対象についてはP09.knowledge_cutoff <= 対応する旧forecast made_at。
9. unknown/nullのまま残るforecastは正式化しない。nullをtoday、executed_at、retrieved_at、published_atへ自動補完しない。

複数forecastをcorrectする場合、P09 cutoffは影響を受ける全forecastのうち最も早い境界を越えてはいけません。最初の配列要素だけを確認しないでください。02のforecast_operationsを使ってupdate/add/removeの対象を決めてください。

境界と同値は許可し、1 microsecondでも後なら拒否してください。すべてUTCへ正規化して比較し、naive datetimeを許可しないでください。

P09 prompt_execution.executed_atは後日でよく、knowledge_cutoffの代用にしません。市場データ、evaluation、04_results、現在の株価をP09入力・time evidenceへ渡してはいけません。

通常P08 ingest、P09 correct、legacy migration auditで時点規則を共有してください。promptだけの禁止ではなくPython側のValidationIssueとして拒否し、少なくとも次のcodeを区別してください。

- p09_cutoff_exceeds_source_boundary
- p09_cutoff_exceeds_corrected_made_at
- corrected_p08_cutoff_exceeds_made_at
- missing_or_unverified_corrected_time

既存code名へ合わせる場合も、どの境界に違反したか機械判別できるようにしてください。
```

## 必須テスト

1. old made_at=null、corrected made_at=08:00、P09 cutoff=08:30、source boundary=09:00を拒否する。
2. 1のP09 cutoff=08:00は、他の証拠・Schemaが正当なら受理する。
3. corrected P08自身のcutoff=08:30、corrected made_at=08:00を拒否する。
4. P09 cutoffとcorrected P08 cutoffの片方だけ正当でも、もう片方が未来なら拒否する。
5. corrected forecast A made_at=08:00、B=09:00、P09 cutoff=08:30をA境界違反で拒否する。
6. corrected payloadの配列順を変えても判定が同じである。
7. add forecastのmade_atもP09 cutoff比較対象になる。
8. remove forecastの旧made_atも比較対象になる。
9. source boundaryがmade_atより早い場合は早い方を採用する。
10. 同値を許可し、+1 microsecondを拒否する。
11. timezoneなしdatetimeをSchemaで拒否する。
12. 正当なunknown→known訂正を一度だけ正式化する。
13. unknownのままの訂正をactive/P11へ進めない。
14. P09 prompt inputにmarket/evaluation/resultsが含まれない。

## 完了条件

- P09の意味判断cutoffとcorrected P08の意味判断cutoffが、訂正後予想時点以前に固定される。
- 単一・複数・追加・削除forecastで同じ規則が働く。
- 正当なunknown時刻解決だけが正式化される。

