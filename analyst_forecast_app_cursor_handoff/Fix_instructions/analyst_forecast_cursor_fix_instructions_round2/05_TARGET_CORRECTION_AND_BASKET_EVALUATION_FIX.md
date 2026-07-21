# 05 — 対象修正候補・参照整合性・バスケット評価

## 目的

P12が提示した修正候補を採用できない問題、別componentのP11/P12を組み合わせられる参照不整合、複数銘柄mappingを先頭1銘柄だけで評価する問題を直します。

## コピペ用依頼文

```text
fix_instructions_round2/00_START_HERE.mdの共通拘束条件に従い、P11/P12/P13の候補lineageと、複数instrument mappingの市場評価を修正してください。commit・pushはしないでください。

まず参照整合性を強化してください。

- P12が参照するP11 proposalは、同じrun_id、source_id、forecast_component_idでなければならない。
- P13が参照するP11 proposalとP12 reviewは、互いに同じrun/source/componentを対象にし、P12.reviewed proposal IDがP13.proposal IDと一致しなければならない。
- input_hashは直前artifactのoutput hashと一致させる。
- 別componentのproposal/reviewを同一source内で取り違えても拒否する。

次にP12のcorrected_candidateを正式採用できるようにしてください。現在はcorrected_candidateを保存できても、P12 agreedとP13 selected_candidate_refがP11元候補しか選べません。

原候補と修正候補の由来を失わないstable referenceを設計してください。例えばcandidate_originをp11_proposal / p12_correctionとして区別し、P12で合意済み修正候補をlockするか、不一致時にP13がP12修正候補を選択できるようにします。最終mappingには、proposal artifact、review artifact、必要時adjudication artifact、採用候補origin、元候補、修正差分を保存してください。

複数instrument mappingを実際の評価へ反映してください。mappingに2件以上のsymbolとweightがある場合、target.tickerの先頭1件だけを評価して成功扱いしてはいけません。

MVPのバスケット評価ルール：

- 各instrumentのsymbol、exchange、currency、weightを完全保持する。
- weight合計は1。負のweightは初期版では禁止。
- 単一instrumentは既存挙動を維持する。
- 同一通貨の複数instrumentは、それぞれの調整済みリターンをweightで合成する。
- 評価開始・終了は共通して取得できる取引日を使用する。
- MFE/MAEは共通日付上のweight付きバスケット指数を基準値1として計算する。
- 欠損日を無断でforward fillしない。共通日が不足する場合は理由付きunevaluableとする。
- 異なる通貨を含むbasketは、FX換算を未実装ならunevaluable_mixed_currencyとし、先頭銘柄だけへ縮退しない。
- provider、symbol、currency、取得期間、data hash、weight、共通日付ルールを監査可能にする。

CSV fallbackをバスケットでも利用できるようにしてください。既存単一銘柄CSV互換を維持しつつ、symbol列を含む一つのCSV、または明示的なsymbol→CSV指定方式を定義し、説明書とCLI helpへ記載してください。AIが市場値を作らないようにしてください。

必須テスト：

1. P12が別componentのP11を参照すると拒否。
2. P13が不一致なP11/P12組を参照すると拒否。
3. P12 corrected_candidateへの合意で修正候補をlockできる。
4. P13がP12修正候補を採用できる。
5. 採用候補のoriginとlineageをDBから追跡できる。
6. 2銘柄50/50 basketの上昇リターン、MFE、MAEが期待値と一致する。
7. 下落予想basketのMFE/MAEがdirection-v2規約に従う。
8. 3銘柄weight合計不一致を拒否する。
9. 1銘柄欠損または共通日不足をunevaluableとする。
10. mixed currencyを先頭銘柄で代用せずunevaluableとする。
11. 単一銘柄の既存評価結果が変わらない。
12. CSV basketとmock provider basketの結果が一致する。

market resultをP11/P12/P13へ渡さない境界を維持してください。Schema、DB migration、結果Markdown/CSV、NEXT_ACTIONS、方法論、CSV仕様を更新し、ruff format、ruff check、mypy、pytestを実行してください。
```

## 完了条件

- 修正候補を正式採用できる。
- P11/P12/P13のcomponent取り違えを拒否する。
- 複数代理指標が先頭1銘柄へ縮退しない。
- 同一通貨バスケットを再現可能に評価できる。
- 未対応の通貨換算や欠損を成功扱いしない。
