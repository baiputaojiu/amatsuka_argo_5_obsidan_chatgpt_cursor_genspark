# 08 — 最終横断監査

## 目的

個別unit testだけでなく、ターミナルとNEXT_ACTIONSに従った縦断シナリオで、今回の修正が実際につながっているかを確認します。

## コピペ用依頼文

```text
fix_instructions_round2/00_START_HERE.mdと01～07の完了結果を読み、最終横断監査を行ってください。この指示では新機能を追加せず、失敗を隠すためのテスト変更もしないでください。内部不具合が見つかった場合だけ最小修正し、影響するテストを再実行してください。commit・pushはしないでください。

次のシナリオを、可能な限りCLI、生成prompt相当fixture、ai ingest、market evaluate、statusを通して実行してください。DBをテストから直接完成状態へ書き換えないでください。

Scenario A: YouTube高確信度、単一銘柄、P05→P08→P11→P12 agreed→CSV評価→04_results。
Scenario B: YouTube話者不明、P05→P06 review→P08。人間承認なし。
Scenario C: 司会者の予想を対象者の正式成績へ入れない。
Scenario D: blog本人執筆、P07→P08→対象解決→CSV評価。
Scenario E: X本人投稿、予想0件をprocessed_no_forecastとして終了。
Scenario F: Web記事の記者要約と本人直接引用を区別する。
Scenario G: P08低確信度または高重要度→P09→解決→次工程。
Scenario H: P11/P12 agreedと、P11/P12 disagreed→P13の両経路。
Scenario I: P12 corrected_candidateまたはP13による修正候補採用。
Scenario J: 2銘柄同一通貨basketの上昇・下落方向評価。
Scenario K: mixed currencyまたは欠損basketを先頭銘柄で代用せずunevaluable。
Scenario L: 同一rawの同条件別run再利用と、別analyst非再利用。
Scenario M: 複数component・複数as_ofで未評価componentを隠さない。
Scenario N: init後のVault docs、詳細FUTURE_ROADMAP、prompt snapshot、モデル設定。
Scenario O: wheelクリーン導入後のtemp Vault縦断。

各段階でrecommended_actionをassertしてください。最低限、P08後P11、P11後P12、P12 disagreed後P13、mapping lock後評価、評価後結果確認であることを確認してください。

最終品質ゲート：

- ruff format --check .
- ruff check .
- mypy src
- pytest（unit件数とdeselected件数を記録）
- Alembic empty→head
- Alembic 0001→head
- Alembic 0005→head
- alembic check
- wheel build
- clean venv install
- wheel版CLI help/init/run create
- git diff --check
- Git追跡対象の秘密値、DB、raw検査

yfinance/FRED実ネットワークは失敗理由をrate limit、network、invalid symbol、no data、missing keyに分類してください。外部環境で実行できない場合はexternal_blockedとできますが、CSVによる内部縦断は必ず実行してください。

docs/06_実装/FINAL_REVIEW.mdとIMPLEMENTATION_STATUS.mdを実測値へ更新してください。判定は次から選んでください。

- NOT_READY: 内部シナリオまたは品質ゲートにfailがある。
- READY_FOR_REAL_SAMPLE_DIRECTIONAL_SLICE: 今回定義した媒体・AI・対象解決・単一/同一通貨basket方向評価が通り、未実装機能を明示している。
- FULL_MVP_READY: 仕様書のMVP完成条件をすべて満たす場合だけ。時期・程度・早期実現、1/3/6/12観測、PNG等が未実装なら使用しない。

FINAL_REVIEWには、pass/fail/external_blocked、証拠テスト名、実行結果、既知制約、次の一作業を記載してください。最終報告にcommit / push未実施を明記してください。
```

## 完了条件

- unitの寄せ集めではなく、NEXT_ACTIONSを含む縦断が通る。
- 内部failが0。
- 実装済みと未実装を正確に区別している。
- `FULL_MVP_READY`を過大に宣言していない。
