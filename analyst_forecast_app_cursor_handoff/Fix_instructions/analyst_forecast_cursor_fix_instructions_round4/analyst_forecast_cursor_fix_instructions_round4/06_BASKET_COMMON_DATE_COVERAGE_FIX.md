# 06 — basket共通取引日と最低データ条件

## 目的

複数日予想のbasketで全銘柄の共通取引日が1日しかないのに、return、MFE、MAE、hit/missを計算する問題を修正します。合成前にデータ区間の成立性を検証し、理由付きunevaluableにします。

## Cursorへ渡す依頼文

```text
00～05に従い、basket evaluationの共通日付・期間coverage検証を修正してください。commit・pushはしないでください。

現状のlen(common_dates) < 1相当の判定では、1観測点から複数日returnを計算できてしまいます。次をPython側の明示規則にしてください。

複数日評価:

- 全instrumentに存在する有効なcommon trading dateを昇順で作る。
- 2つ以上の異なる共通日を必須にする。
- selected start date < selected end dateを必須にする。
- start/end price、actual return、MFE、MAEは同じcommon-date basket seriesから計算する。
- 1日しかなければhit/missを付けず、evaluation_status=unevaluable、reason=insufficient_common_dates等にする。
- 欠損symbolを落として残りだけで部分basketを作らない。
- NaN、非正価格、重複symbol/date、currency不一致を除外して黙って続行しない。既存規則に従って明示的に拒否/unevaluableにする。

明示的な単一取引日予想:

- 1つのclose観測だけから方向returnを作らない。
- 同日open→close等、仕様化済みの独立した1日methodがあり、全instrumentで必要なopen/closeが揃う場合だけ、そのmethod versionと価格選択根拠を記録して評価してよい。
- 単一日methodが未実装ならunevaluable_single_day_method_not_supported等にする。

監査情報として少なくとも次をEvaluationRecordまたは関連auditへ保存してください。

- requested/effective period
- common date count
- selected start/end date
- instrumentごとのinput first/last date、row count、series hash
- dropped date countまたはintersection rule
- basket mapping hash、weights、currency
- evaluation method version
- unevaluable reason

単一銘柄評価の既存挙動を不必要に変えないでください。ただし単一銘柄でもreturnに2点が必要なmethodでは1点からhit/missを計算しないでください。市場cache identity分離、mixed currency、CSV long形式等のRound3合格項目を維持してください。
```

## 必須テスト

1. 複数日・2銘柄・共通日1日をinsufficient_common_datesでunevaluableにする。
2. そのcaseでdirection_result、actual_return、MFE、MAEを確定値として保存しない。
3. 共通日2日で正しいweighted returnを計算できる。
4. 共通日2日で上昇予想のMFE/MAEが正しい。
5. 共通日2日で下落予想のMFE/MAEがdirection-v2規約に従う。
6. 各symbolに2日以上あってもintersectionが1日ならunevaluableにする。
7. 1銘柄欠損時に残りだけでbasket評価しない。
8. duplicate symbol/dateを黙って除外しない。
9. NaN/非正価格を含む必要日を黙って除外しない。
10. 明示単一日予想を1 close値だけで評価しない。
11. 対応済み単一日open→close methodがある場合はmethod version付きで正しく評価する。未実装なら理由付きunevaluableにする。
12. auditにcommon date count、start/end、各input hash、weightが残る。
13. basket評価後も単一symbol cacheが汚染されない。
14. CSV providerとmock providerで同じcoverage判定・数値になる。
15. legacy単一銘柄の正常な2点以上評価が回帰しない。

## 完了条件

- 1観測点から複数日hit/missを生成しない。
- basketの全instrumentが同じ日付集合で評価される。
- unevaluableの理由と入力coverageを後から監査できる。

