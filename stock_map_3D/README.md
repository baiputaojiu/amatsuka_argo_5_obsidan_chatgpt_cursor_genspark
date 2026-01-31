# 株価 3D マップ（日付 × 出来高 × VWAP）

銘柄の日付・出来高・VWAP を 3D で表示し、左側に VWAP 時系列と出来高時系列を表示する Dash アプリです。

- **右側**: 3D グラフ（日付インデックス × 出来高 × VWAP）、線でつなぐ
- **左上**: VWAP の時系列（3D ホバー位置を強調）
- **左下**: 出来高の時系列（3D ホバー位置を強調）
- **時間足**: 日足（1d）と 1 時間足（1h）をサポート。1 時間足は **最大約 60 日分**（yfinance の intraday 制約）
- **VWAP**: 各足の典型価格 `(High + Low + Close) / 3` を Z 軸に使用
- **データ**: `data/` に CSV があれば読み込み。無い場合は画面上の「データを取得」ボタンで対話的に yfinance 取得して保存（ハイブリッド）

---

## 目次

- [前提](#前提)
- [1. 仮想環境の作成](#1-仮想環境の作成)
- [2. 依存パッケージのインストール](#2-依存パッケージのインストール)
- [3. アプリの起動方法](#3-アプリの起動方法)
- [4. データの取得（fetch_stock_data.py）](#4-データの取得fetch_stock_datapy)
- [5. 1 時間足の制限について](#5-1-時間足の制限について)
- [6. VWAP について](#6-vwap-について)

---

## 前提

- OS: Windows 10 以降（PowerShell を想定）
- Python: 3.10 〜 3.12 程度を想定
- 作業ディレクトリ: `stock_map_3D` フォルダ

---

## 1. 仮想環境の作成

```powershell
cd D:\workspace\amatsuka_argo_5_obsidan_chatgpt_cursor_genspark\stock_map_3D
python -m venv venv
.\venv\Scripts\Activate.ps1
```

プロンプトに `(venv)` が付いていれば OK です。

---

## 2. 依存パッケージのインストール

```powershell
cd D:\workspace\amatsuka_argo_5_obsidan_chatgpt_cursor_genspark\stock_map_3D
python -m pip install --upgrade pip
pip install -r requirements.txt
```

含まれる主なパッケージ: Dash, Plotly, pandas, numpy, yfinance

---

## 3. アプリの起動方法

1. 仮想環境をアクティブにした状態で `app.py` を実行します。

   ```powershell
   cd D:\workspace\amatsuka_argo_5_obsidan_chatgpt_cursor_genspark\stock_map_3D
   python app.py
   ```

2. コンソールに表示される URL（例: `http://127.0.0.1:8050/`）をブラウザで開きます。
3. 銘柄・時間足・期間を選び、左の **「データを取得」** ボタンを押すと yfinance で取得し `data/` に保存されます。既に CSV がある場合はそれを自動で読み込みます。

---

## 4. データの取得（fetch_stock_data.py）

アプリ起動時に CSV が無ければ自動取得しますが、手動で事前に CSV を用意する場合は以下を実行します。

```powershell
# 日足・1年分
python fetch_stock_data.py AAPL 1d --period 1y

# 1時間足・60日分（最大約60日）
python fetch_stock_data.py AAPL 1h --period 60d

# 開始日・終了日指定（1時間足向き）
python fetch_stock_data.py 7203.T 1h --start 2024-11-01 --end 2025-01-31
```

保存先: `data/{ティッカー}_{interval}.csv`（例: `data/AAPL_1d.csv`, `data/7203.T_1h.csv`）

---

## 5. 1 時間足の制限について

yfinance の intraday データは **最大約 60 日分** までです。1 時間足を選んだ場合、期間は実質 60 日以内になります。UI で「60日（1h用）」を選ぶか、`fetch_stock_data.py` で `--period 60d` または `--start` / `--end` で指定してください。

---

## 6. VWAP について

本アプリでは、各足（1 本のローソク足）ごとに **典型価格** を VWAP として使用しています。

- **計算式**: `VWAP = (High + Low + Close) / 3`
- 1 本の足あたり 1 点のため、その足の出来高加重平均の近似として典型価格で代表しています。
- 本来の VWAP（セッション累積の出来高加重平均価格）とは異なりますが、時系列の 3D 可視化にはこの定義で統一しています。

---

## 銘柄リストの変更

`app.py` の `TICKER_OPTIONS` を編集すると、ドロップダウンに表示する銘柄を変更できます。Yahoo Finance のティッカーシンボル（例: AAPL, 7203.T）を指定します。
