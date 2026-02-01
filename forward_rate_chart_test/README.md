# シルバー / ゴールド 1ヶ月フォワードレート チャート（テスト）

XAG1M=TTKL風の1ヶ月フォワードレート時系列グラフのテスト用アプリです。

## 起動方法（仮想環境で実行）

**ワークスペースのルート**（`forward_rate_chart_test` の親フォルダ）にいる場合のみ、最初に `cd forward_rate_chart_test` を実行する。既に `forward_rate_chart_test` 内にいる場合は不要。

```powershell
# ルートにいる場合のみ
cd forward_rate_chart_test

python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

ブラウザで http://127.0.0.1:8052 を開いてください。

- 既に `venv` がある場合は作成を飛ばし、`.\venv\Scripts\Activate.ps1` から実行すればよい。
- venv 作成で「Unable to copy」などが出た場合は、`venv` フォルダを削除してから `python -m venv venv` をやり直す。

## データ

- `yield_curve_3d/data/silver_forward_curve.csv` および `gold_forward_curve.csv` を参照
- 列: Date, 0.0（現物）, 1（フロント月）, 2, 3, …。レート = (1 - 0.0) / 0.0 × 100 (%)

## 機能

- 金属切り替え: シルバー (XAG) / ゴールド (XAU)
- 期間切り替え: 1日 / 5日 / 10日 / 1ヶ月 / 3ヶ月 / 6ヶ月 / 2025年4月〜 / 年初来 / 1年 / 最大
- 限月データを更新: yfinance で新規日付のみ追加（既存データは変更しない）
- 現在価格が無い日: 日付を選んで現物価格を手入力して保存
- Y=0（Par）基準線、棒グラフ、ダークテーマ
