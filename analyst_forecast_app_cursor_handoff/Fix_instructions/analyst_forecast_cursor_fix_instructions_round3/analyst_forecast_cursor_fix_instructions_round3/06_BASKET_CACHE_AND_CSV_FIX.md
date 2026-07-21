# 06 — バスケット市場系列、cache分離、CSV入力

## 目的

合成バスケットを先頭銘柄の市場seriesとしてcacheし、後続の単一銘柄評価を汚染する問題を直します。複数instrument評価を再現可能にし、無料・簡単なCSV fallbackを実際にバスケットでも使えるようにします。

## Cursorへ渡す依頼文

    00_START_HERE.md、01～05に従い、市場series cacheとbasket評価を修正してください。commit・pushはしないでください。

    cache原則:

    - providerから取得したraw seriesは、実symbol、currency、provider、adjustment type、期間、data hashで個別に保存する。
    - 合成basket seriesを先頭instrumentのsymbolとして保存しない。
    - 合成seriesにはseries_kind=basket等の区別と、BASKET:<mapping_hash>等の衝突しないidentityを持たせる。
    - raw symbol lookupはbasket seriesを絶対に返さない。
    - basket evaluationは入力した全raw series ID/data hash、symbol、currency、weight、共通日、計算法version、mapping hashを監査可能にする。
    - mappingが変われば古いbasket cacheを再利用しない。

    provider検証:

    - providerが返したseries.symbolとrequest.symbolを確認する。
    - providerが返したcurrencyとmapping instrument currencyを確認する。
    - mapping上はJPYなのにproviderがUSDを返す等の不一致をunevaluableまたは明示errorにする。
    - 同じsymbolの重複instrumentをrejectする。
    - weight合計1、weight正、全instrument取得済みを確認する。

    basket計算:

    - 同一通貨のみMVPで評価可能。
    - 全instrumentに共通する取引日だけを使う。
    - 無断forward fill、back fill、欠損銘柄の除外をしない。
    - 複数日評価には原則2つ以上の共通日を要求する。明示的一日評価だけ例外を仕様化する。
    - weight付きadjusted returnで基準値1のbasketを作る。
    - MFE/MAEは共通日付上のbasket seriesとdirection-v2規約で計算する。
    - mixed currencyはFX換算未実装ならunevaluable_mixed_currency。
    - 先頭銘柄のみへ縮退しない。

    CSV fallbackは、初学者が一つのCSVを指定できる形式を正本として実装してください。

    推奨形式:

    - 単一銘柄の既存CSVは後方互換を維持。
    - basket CSVはsymbol、currency、date、open、high、low、close、adjusted_open、adjusted_close列を持つlong形式。
    - CsvMarketDataProviderはrequest.symbolでfilterし、該当symbolがない場合はunevaluableとする。
    - 同一symbol/date重複、currency不一致、欠損列、不正価格を拒否する。
    - CLIの--csv-path一つでbasket全instrumentを取得できる。

    market cacheとevaluation DBにはraw seriesとsynthetic basketを区別するmigrationを追加してください。単一銘柄の既存cacheを失わず、legacy recordをrawとみなす条件を明記してください。

## 必須テスト

1. AAA +20%、BBB -20%の50/50 basketは0%になる。
2. そのbasket評価後にAAAを評価すると+20%になり、basket 0% cacheを読まない。
3. raw cache lookupがseries_kind=basketを返さない。
4. mapping hash違いのbasket cacheを再利用しない。
5. provider currency不一致を検出する。
6. provider symbol不一致を検出する。
7. duplicate symbolを拒否する。
8. weight合計不一致と負weightを拒否する。
9. 1銘柄欠損、共通日不足をunevaluableにする。
10. mixed currencyを先頭銘柄で代用しない。
11. 下落予想のMFE/MAEがdirection-v2に従う。
12. legacy単一銘柄CSV評価が以前と同じ結果になる。
13. long形式basket CSVとmock providerで同じreturn、MFE、MAEになる。
14. CSVに同一symbol/date重複があれば拒否する。
15. evaluationから全input series hash、weight、common-date ruleを追跡できる。

## 完了条件

- basket cacheが単一symbol cacheを汚染しない。
- basketの全構成instrumentと入力データを監査できる。
- 一つのCSVで無料・簡単にbasket評価できる。

