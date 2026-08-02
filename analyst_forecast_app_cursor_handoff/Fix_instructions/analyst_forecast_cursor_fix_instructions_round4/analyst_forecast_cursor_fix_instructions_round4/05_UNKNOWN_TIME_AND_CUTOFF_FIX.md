# 05 — 日時不明の表現とP08/P09 knowledge cutoff境界

## 目的

`made_at_source=unknown`なのに架空の`made_at`を必須入力して正式化できる問題、P08 promptとSchemaの`knowledge_cutoff`不一致、P09が予想時点より後の知識を使える問題を修正します。

## Cursorへ渡す依頼文

```text
00～04に従い、P08/P09の時刻model、固定JSON Schema、Pydantic、DB、prompt、検証を一元化してください。commit・pushはしないでください。

時刻の意味を維持してください。

- made_at: 予想が実際に表明された時刻。分からなければnull。
- made_at_source: explicit / source_metadata / context_inferred / unknown。
- publicly_available_at: 一般利用可能になった時刻。実発言時刻の代用としてmade_atへ捏造しない。
- knowledge_cutoff: AIが意味判断に使用してよい知識の上限。
- executed_at/retrieved_at: 後日でもよいがmade_atには使わない。

P08 Schemaを次の不変条件へ変更してください。

- made_at_source=unknownならmade_atはnullでなければならない。
- made_atがnullなら正式ForecastIssuanceを作らず、P11へ進めない。
- made_at_sourceがexplicit/source_metadata/context_inferredならmade_atはtimezone付きで必須。
- context_inferredはtime evidence、raw offset/segment、precision、inference basisを必須にし、P09 review対象にする。
- source_metadataはsource.recorded_at/published_atとの決定的整合を検証する。
- publicly_available_atはtimezone付きで必須、made_atが存在する場合はmade_at <= publicly_available_at。
- P08 top-levelにtimezone付きknowledge_cutoffを追加し、固定JSON Schema、Pydantic、prompt、生成例、packaged resourceを一致させる。
- P08 knowledge_cutoff <= source allowed boundary。
- made_atが確定している予想ではknowledge_cutoff <= verified made_at。複数forecastでmade_atが違うなら、各forecast別cutoffを持つ設計でもよいが、すべての意味判断が該当予想時点以前であることを機械検証する。

後方互換:

- legacy P08にknowledge_cutoffがないからといってtoday/executed_atを補わない。
- legacy made_at_source=unknown + non-null made_atをverifiedへ昇格させない。legacy_time_unverified等としてactive集計・P11から除外する。
- Schema versionを上げる場合は旧Schema読込みと新規作成を区別し、新規P08では新fieldを必須にする。

P09:

- review knowledge_cutoff <= source allowed boundaryだけでなく、review対象となる各forecastのverified made_at以前であることを検証する。
- reviewed P08のmade_atがunknown/nullなら、review cutoffは少なくともsourceの保守的boundary以内とし、P09 acceptだけで日時を確定しない。
- unknown日時を解決するにはdecision=correctのcorrected_payload内で、made_at、made_at_source、time evidence、knowledge_cutoffを完全に示し、通常P08と同じ検証を通す。
- P09 correct payload内部のP08 knowledge_cutoffも検証する。
- P09 unresolvedなら04のterminal/excludedへ進める。
- P09実行日や公開後の市場結果をcutoffとして使わない。

時点検証は共通serviceにまとめ、通常P08、P09 correct、migration auditで同じ規則を使ってください。promptで禁止するだけでなくPython側で拒否してください。
```

## 必須テスト

1. made_at_source=unknown + made_at=nullをSchema上で表現できる。
2. unknown + 任意datetimeを新規P08で拒否する。
3. unknown/null P08からformal issuanceを作らない。
4. unknown/null sourceをP11へ進めない。
5. retrieved_at、executed_at、todayをmade_atへ自動補完しない。
6. explicit + made_at nullを拒否する。
7. source_metadataとsource時刻不一致を拒否する。
8. context_inferredでtime evidenceなしをneeds_reviewまたはrejectにする。
9. P08 fixed JSON Schemaがknowledge_cutoffを必須とする。
10. P08 cutoff > source boundaryを拒否する。
11. P08 cutoff > verified made_atを拒否する。
12. P09 cutoffがsource boundary以内でもverified made_atより後なら拒否する。
13. P09 correct内cutoff > corrected made_atを拒否する。
14. 正当なP09 correctでunknown日時を証拠付き確定し、一度だけ正式化できる。
15. P09 acceptだけでunknown日時を正式化しない。
16. legacy unknown timeをactive集計へ入れない。
17. prompt、Schema、Pydantic generated schemaのrequired/enumが一致する。
18. 市場結果、evaluation file、04_resultsがP08/P09 prompt inputへ入らない。

## 完了条件

- 不明日時を不明のまま保存できる。
- 日時未確認の予想が対象解決・市場評価へ進まない。
- P08/P09のすべての意味判断cutoffが予想時点以前に固定される。

