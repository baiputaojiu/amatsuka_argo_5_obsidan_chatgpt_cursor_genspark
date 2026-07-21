# 05 — 市場評価・下落方向・取得失敗の修正

## 目的

最小方向評価の計算を方向対応させ、無料データ取得失敗を正しく分類・案内する。

## コピペ用依頼文

```text
市場評価部分を修正してください。今回の範囲は、既存の最小方向評価を正確かつ運用可能にすることです。未決定の総合点、時期・程度の0～100式、能力ランキングを先回り実装しないでください。

1. MFE／MAEを予想方向に対応させる

上昇予想：
- favorable = (period_high - start_price) / start_price
- adverse = (period_low - start_price) / start_price

下落予想：
- favorableは価格下落の大きさが正の値になる表現とする。
- adverseは価格上昇の大きさが正の値になる表現とする。

DB列名と既存データの意味を確認し、符号規約を文書化してください。既存direction-v1.0.0の意味を黙って変更せず、必要ならevaluation method versionを上げて再評価可能にしてください。

2. 複合予想の親状態

一つのcomponentを評価するたびにForecastIssuanceRecord.current_statusを最後のcomponent状態で上書きしないでください。親予想の状態は全componentから明示的に集約するか、集約未実装ならcomponent状態とissuance状態を分離し、誤った完了表示を避けてください。

3. provider失敗分類

yfinanceはrate limit時に空DataFrameを返し、内部ログだけにYFRateLimitErrorが出る場合があります。「指定期間データなし」と「rate limited」「network error」「invalid symbol」を可能な範囲で区別してください。取得不能値を推測してはいけません。

次を実装してください。
- 回数と待機上限を小さく固定したretry／exponential backoff。
- rate limit時の具体的な再試行・CSV切替案内。
- provider error codeと原メッセージの監査記録。
- 同じsymbol、期間、調整方式の有効cacheがある場合は、再ネットワーク取得前に利用できる設計。
- CSVへ切り替えてもprovider、取得日時、通貨、調整方式が分かる記録。
- integration testのrate limitを市場データ不存在と誤判定しない。

4. データ品質

- adjusted_open/adjusted_closeとhigh/lowの調整基準を同一にする。
- 休場日、範囲外データ、重複日、欠損、0以下の不正価格を検査する。
- yfinanceの株式分割をまたぐfixtureまたは保存済みfixtureで回帰試験する。
- 実ネットワークtestは通常pytestから分離したままにする。

必須テスト：
- 上昇予想のhit/missとMFE/MAE。
- 下落予想のhit/missと方向対応MFE/MAE。
- flat実績。
- active、expired、not_started、unevaluable。
- 同一componentの複数as_of履歴。
- 複数componentで親状態を誤上書きしない。
- rate limit、network error、invalid symbol、true no dataの分類。
- retry上限超過後は推測せず終了する。
- cache hit時にネットワークを呼ばない。
- CSV fallbackで同じ計算結果を再現できる。

Ruff、mypy、通常pytestを実行してください。ネットワークintegrationは、環境が許す場合だけ実行し、外部rate limitによる失敗とコード不良を区別して報告してください。実装文書と評価method versionを更新し、commit・pushは行わないでください。
```

