# 05 — 単一銘柄・basketの最低coverageと監査情報

## 目的

単一銘柄の複数日予想を1取引日だけでhit/miss判定する問題と、coverage不足でunevaluableにした際に`common_date_count`や`coverage_audit`がnullになる問題を修正します。

## Cursorへ渡す依頼文

```text
00～04に従い、evaluationの最低データ条件とcoverage監査を単一銘柄・basketで統一してください。commit・pushはしないでください。

現在のdirection-v2.0.0は、最初のadjusted_openと最後のadjusted_closeからreturnを作ります。normalized_start < normalized_endの複数日予想では、1つの取引日しかないseriesからhit/missを作らないでください。

最低条件:

- current direction-v2.0.0の複数日評価は、単一銘柄でも2つ以上の異なる有効取引日を必須にする。
- basketは全instrument共通の異なる取引日を2日以上必須にする。
- selected_start_date < selected_end_dateを必須にする。
- 1日だけならevaluation_status=unevaluable、理由code=insufficient_trading_datesまたはinsufficient_common_dates。
- direction_result、start/end/current price、actual_return、MFE、MAEを確定値として保存しない。
- 欠損symbolを落とした部分basketを作らない。
- invalid/duplicate/NaN/非正価格を黙って落として最低件数を満たしたことにしない。

同日予想:

- normalized_start == normalized_endを、複数日methodへ暗黙に流さない。
- 明示的なsingle-day open→close methodとmethod version、必要OHLC、仕様、testを実装する場合だけ評価してよい。
- 今回そのmethodを実装しない場合はsingle_day_method_not_supportedでunevaluableにする。この選択は新機能不足ではなく、誤評価防止として許容する。

coverage auditを評価成功・coverage不足の両方で保存してください。構造化JSONには少なくとも次を含めます。

- requested_start_date / requested_end_date
- effective_start_date / effective_end_date
- evaluation_as_of
- method_version
- series_kind
- instrumentごとのsymbol、currency、requested range、input_first_date、input_last_date、in_range_row_count、unique_valid_date_count、series_hash
- basketではweights、mapping_hash、intersection rule、common_date_count
- selected_start_date / selected_end_date
- duplicate_date_count、invalid_row_count、dropped_date_countまたは拒否理由
- coverage_statusとreason_code

insufficient coverageを例外catchして_store_without_valuesへ落とす場合も、検査済みaudit contextを引き渡してください。_store_without_valuesがcoverage_audit、common_date_count、selected datesを受け取れるようにし、nullへ捨てないでください。

単一銘柄成功時にもcoverage_auditを保存してください。basketだけ監査可能という分岐を残さないでください。requested periodと実際に使った期間の違いを後から説明できる必要があります。

市場cache identity、mixed currency、CSV long形式、basket合成seriesを先頭symbolで保存しないRound3/Round4の合格項目を維持してください。
```

## 必須テスト

1. 単一symbol・複数日予想・市場データ1取引日をunevaluableにする。
2. 1でdirection_result、price、return、MFE、MAEがnullである。
3. 1でcoverage_auditにrequested/effective period、row count、first/last、hash、reason codeが残る。
4. basket共通日1日をunevaluableにする既存挙動を維持する。
5. 4でcommon_date_count=1とcoverage_auditを保存する。
6. 各symbolに2日あってもintersection 1日ならcommon count=1を保存する。
7. 単一symbol 2取引日で正しいreturn/MFE/MAEを計算する。
8. basket共通2取引日で正しいweighted return/MFE/MAEを計算する。
9. 成功した単一symbolにも全必須coverage audit keyがある。
10. 成功したbasketにも銘柄別row count/first/last/hash/weightsがある。
11. selected_start_date < selected_end_dateを保証する。
12. 同日予想は明示methodがなければsingle_day_method_not_supportedになる。
13. duplicate/invalid/NaN/非正価格を黙って除外しない。
14. 欠損instrumentを落として部分basketを評価しない。
15. CSV providerとmock providerでcoverage判定と数値が一致する。
16. basket評価後の単一symbol cache値が汚染されない。
17. unevaluable再実行が冪等でauditを失わない。

## 完了条件

- 1取引日から複数日hit/missを作らない。
- 成功・失敗とも入力coverageをDBから監査できる。
- 単一銘柄とbasketが同じ最低条件・audit contractを使う。

