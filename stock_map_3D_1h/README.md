# stock_map_3D_1h — 日付×株価×出来高の3D山脈（1時間足・730日版）

yfinance で取得した1時間足データを銘柄ごとにCSVに保存し、日付・価格帯・出来高を計算して「3D山脈」とローソク足を表示する Dash アプリです。

- **右側**: 3Dサーフェス（日付 × 株価 × 出来高）。山脈の下に1時間足ローソクを表示。透明度スライダーで透かし表示を調整可能。
- **左側**: 銘柄入力・取得/更新ボタン・お気に入りドロップダウン・価格刻み表示・軸範囲スライダー・平滑化選択・山脈の半透明度スライダー。
- **データ**: 1時間足・過去730日（約2年）。銘柄ごとに `data/{TICKER}_1h.csv` に保存。**更新ボタン**押下時のみ増分取得してCSVに追記。

---

## 前提

- OS: Windows 10 以降（PowerShell を想定）
- Python: 3.10 〜 3.12 程度
- 作業ディレクトリ: `stock_map_3D_1h` フォルダ

---

## 1. 仮想環境の作成

```powershell
cd D:\workspace\amatsuka_argo_5_obsidan_chatgpt_cursor_genspark\stock_map_3D_1h
python -m venv venv
.\venv\Scripts\Activate.ps1
```

---

## 2. 依存パッケージのインストール

```powershell
pip install -r requirements.txt
```

---

## 3. アプリの起動

```powershell
python app.py
```

ブラウザで `http://127.0.0.1:8052/` を開く。

- 銘柄コードを入力して「取得」で初回取得（730日分の1時間足）→CSV保存→描画。
- 「更新」で現在表示中の銘柄の1時間足を増分取得→CSV追記→再描画。
- お気に入りドロップダウンで銘柄を選ぶと、既存CSVを読んで描画。
- 価格刻みは直近終値の1％。X/Y/Z軸スライダーで3Dの表示範囲を変更。
- 平滑化（なし／現在の平滑化／ガウスぼかしのみ）と山脈の半透明度を選択可能。

---

## 4. データファイル

- `data/favorites.json`: お気に入り銘柄リスト（初期: `["7203.T"]`）
- `data/{TICKER}_1h.csv`: 銘柄ごとの1時間足（Open, High, Low, Close, Volume）
- `data/{TICKER}_graph_data.csv`: 描画用Z行列（自動保存）

730日分の1時間足取得には数秒〜十数秒かかることがあります。
