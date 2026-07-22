# 03 — coverage_auditの完全な保存契約

## 目的

評価を`unevaluable`にするだけでなく、「何を要求し、何行を受け取り、何を有効と判断し、なぜ評価できた／できなかったか」をDBから再現できる監査情報へ統一します。単一symbolとbasket、成功と不足で同じ構造を使用します。

## coverage_audit必須構造

`EvaluationRecord.coverage_audit`へ、少なくとも次を保存してください。key名は既存互換性のため合理的に調整してよいですが、意味と情報量を減らしてはいけません。

```json
{
  "schema_version": "1.0.0",
  "coverage_status": "sufficient | insufficient | invalid",
  "reason_code": "insufficient_trading_dates | insufficient_common_dates | ...",
  "requested_start_date": "YYYY-MM-DD",
  "requested_end_date": "YYYY-MM-DD",
  "effective_start_date": "YYYY-MM-DD",
  "effective_end_date": "YYYY-MM-DD",
  "evaluation_as_of": "YYYY-MM-DD",
  "method_version": "direction-v2.0.0",
  "series_kind": "single | basket",
  "selected_start_date": "YYYY-MM-DD or null",
  "selected_end_date": "YYYY-MM-DD or null",
  "common_date_count": "integer or null",
  "intersection_rule": "string or null",
  "mapping_hash": "sha256 or null",
  "instruments": [
    {
      "symbol": "AAA",
      "currency": "USD",
      "weight": "decimal or null",
      "requested_start_date": "YYYY-MM-DD",
      "requested_end_date": "YYYY-MM-DD",
      "input_row_count": 2,
      "input_first_date": "YYYY-MM-DD or null",
      "input_last_date": "YYYY-MM-DD or null",
      "in_range_row_count": 2,
      "unique_valid_date_count": 2,
      "duplicate_date_count": 0,
      "invalid_row_count": 0,
      "dropped_out_of_range_count": 0,
      "dropped_row_count": 0,
      "series_hash": "sha256 or null"
    }
  ]
}
```

## fieldの意味

- `input_row_count`: provider/cache/CSVから受け取った当該instrumentの全bar数。
- `input_first_date` / `input_last_date`: parse可能なinput date全体の最小・最大。0件ならnull。
- `in_range_row_count`: requested/effective range内にある行数。duplicateを除く前の件数。
- `unique_valid_date_count`: 日付、OHLC、価格正数等のvalidationを通り、重複方針適用後に使える異なる日付数。
- `duplicate_date_count`: 同一instrument・同一dateの2行目以降の件数。
- `invalid_row_count`: date/price/NaN/必要field等が不正な件数。
- `dropped_out_of_range_count`: requested/effective range外のため使用しない件数。
- `dropped_row_count`: 使用しなかった総数。内訳との関係を文書化し、二重計上しない。
- `series_hash`: evaluation候補となるvalid in-range rowsを、symbol/currency/date/OHLC/adjustmentを含むcanonical JSONまたは既存canonical CSVへ正規化し、date順でSHA-256化した値。同じ意味の入力で順序により変わらないこと。valid rowが0件ならnullを許可するがkeyは必須。
- `selected_start_date` / `selected_end_date`: 実際にreturnへ使用した2日。1日不足時は同日を両方へ入れるかendをnullにするかを一貫して定義し、`coverage_status=insufficient`と組み合わせる。成功時は必ずstart < end。
- basketの`common_date_count`: 全instrumentのvalid date intersection件数。不足時もnullにしない。

## Cursorへ渡す依頼文

```text
00～02に従い、evaluation coverage auditの生成、保存、表示、testを修正してください。commit・pushはしないでください。

- audit生成を、単一symbol/basketおよび成功/例外catchで別々に手書きしない。共通builderまたは同一contractへ集約する。
- providerからseriesを得た直後にinstrument別input統計を収集し、最低日数判定で例外になる前にaudit contextを完成させる。
- `_store_without_values`等へcoverage auditを必ず渡し、DBのcoverage_auditをnullにしない。
- 単一symbol複数日予想の1取引日は、status=unevaluable、reason_code=insufficient_trading_datesとし、direction_result、start/end/current price、actual_return、MFE、MAEをすべてnullにする。
- basket共通1日はstatus=unevaluable、reason_code=insufficient_common_dates、common_date_count=1とし、各instrumentのauditを残す。
- 単一symbol 2日、basket共通2日以上の成功時にも同じaudit構造を保存する。
- invalid/duplicate/NaN/非正価格を黙って落とし、残り2日で成功扱いにしない。既存validation方針に従ってinvalid/unevaluableとし、件数と理由をauditへ残す。
- out-of-range rowは使用しなくてよいが、件数とinput first/last dateに反映する。
- 欠損instrumentを除外した部分basketを作らない。
- auditの生成に現在時刻やdict挿入順等の非決定要素を入れず、同じ入力の再評価でcanonical hashを再現可能にする。
- cache hit/miss、CSV/mock providerでcoverage意味を変えない。basket評価後の単一symbol cache非汚染を維持する。
- 04_results等へunevaluable reasonとrequested/effective/selected date、instrument別valid件数の要約を出す。完全JSONの表示は不要だがDBに全項目を保持する。
```

## 必須テスト

1. 単一symbol・1取引日: 全評価値null、reason、完全audit、series hashあり。
2. 単一symbol・2取引日: 正しいreturn/MFE/MAE、完全audit、selected start < end。
3. basket・各seriesは2日以上だが共通1日: common_date_count=1、全値null、全instrument audit。
4. basket・共通2日: 正しいweighted return/MFE/MAE、weights/mapping hash/intersection rule/audit。
5. inputにrange外の前後barを含み、input first/last、in-range、dropped countが正しい。
6. duplicate dateを含み、duplicate countとreasonが正しい。
7. NaN、非正価格、不正date等を含み、invalid countを残して誤評価しない。
8. input順を入れ替えてもseries hashが同じ。
9. 値を1つ変えるとseries hashが変わる。
10. cache hitとCSV/mockでauditのcoverage部分が一致する。
11. 同日予想を暗黙の複数日methodで評価しない。
12. insufficient後の再実行が重複評価行を増やさず、保存auditを失わない。

## 完了条件

- 成功・不足、単一・basketの4区分すべてで完全なDB auditがある。
- Round5独立レビューで欠落したsymbol/currency、first/last、in-range、hash、duplicate/invalid/droppedがassertされる。
- 1日やinvalid inputからhit/missを作らない。

