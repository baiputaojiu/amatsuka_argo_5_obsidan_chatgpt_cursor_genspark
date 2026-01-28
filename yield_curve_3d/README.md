# イールドカーブ 3D 可視化アプリ（日本 / 米国）

日本と米国のイールドカーブを 3D サーフェスで表示し、マウスホバーした日付の断面カーブと、10 年金利の推移を表示する Dash アプリです。

- 右側: 3D イールドカーブ（時間 × 残存期間 × 利回り）
- 左上: 選択日（ホバーした日）のイールドカーブ断面
- 左下: 10 年金利の時系列（ホバー位置に縦線）
- 上部ラジオボタン: 「日本 / 米国」の切替

---

## 前提

- OS: Windows 10 以降（PowerShell を想定）
- Python: 3.10 〜 3.12 程度を想定
- 作業ディレクトリ:
  - `D:\workspace\amatsuka_argo_5_obsidan_chatgpt_cursor_genspark\yield_curve_3d`

---

## 1. 仮想環境の作成

1. PowerShell でリポジトリのルート、または `yield_curve_3d` に移動します。

   ```powershell
   cd D:\workspace\amatsuka_argo_5_obsidan_chatgpt_cursor_genspark\yield_curve_3d
   ```

2. 仮想環境 `venv` を作成します。

   ```powershell
   python -m venv venv
   ```

3. 仮想環境をアクティベートします。

   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

   ※ プロンプトの先頭に `(venv)` のような表示が出れば OK です。

---

## 2. 依存パッケージのインストール

仮想環境をアクティブにした状態で、`requirements.txt` から必要なパッケージをインストールします。

```powershell
cd D:\workspace\amatsuka_argo_5_obsidan_chatgpt_cursor_genspark\yield_curve_3d
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` には、主に以下が含まれています。

- Dash
- Plotly
- pandas
- numpy

---

## 3. アプリの起動方法

1. 仮想環境が有効かどうか確認します（プロンプトに `(venv)` が付いているか）。
2. `yield_curve_3d` ディレクトリで `app.py` を実行します。

   ```powershell
   cd D:\workspace\amatsuka_argo_5_obsidan_chatgpt_cursor_genspark\yield_curve_3d
   python app.py
   ```

3. コンソールに以下のようなメッセージが出ます。

   ```text
   Dash is running on http://127.0.0.1:8050/
   ```

4. ブラウザで上記 URL（または表示されている URL）にアクセスするとアプリ画面が開きます。
   - 右側に 3D イールドカーブ
   - 左に 2 つの 2D グラフ（断面 / 10 年金利）
   - 上部ラジオボタンで「日本 / 米国」を選択

---

## 4. データファイルについて

`yield_curve_3d/data` ディレクトリに、以下の CSV が置かれている前提です。

- `japan_yield_curve.csv`
  - 1 行目: タイトル行（例: `国債金利情報`）
  - 2 行目: ヘッダー行（`基準日,1年,2年,3年,...,40年`）
  - 3 行目以降: 日付ごとの利回りデータ
- `usa_yield_curve.csv`
  - 1 行目: ヘッダー行（`Date,1 Mo,1.5 Month,2 Mo,...,30 Yr`）
  - 2 行目以降: 日付ごとの利回りデータ

これらの形式であれば、`app.py` が自動的に読み込んで 3D 表示を行います。

### 4.1 米国データを 2000 年〜現在にする

米国データを 2000 年〜現在まで拡張するには、`fetch_usa_data.py` を使います。

**方法 A: 自動ダウンロード（ネットワークが通る場合）**

```powershell
cd D:\workspace\amatsuka_argo_5_obsidan_chatgpt_cursor_genspark\yield_curve_3d
python fetch_usa_data.py
```

**方法 B: 手動ダウンロードしてから正規化**

1. ブラウザで以下を開き、Par Yield Curve の CSV をダウンロードします。
   - [U.S. Treasury Interest Rates Data CSV Archive](https://home.treasury.gov/interest-rates-data-csv-archive)
   - 例: **Daily Treasury Par Yield Curve Rates** の `par-yield-curve-rates-1990-2022.csv`
2. ダウンロードした CSV を `yield_curve_3d` フォルダなどに保存します。
3. 次のコマンドで 2000 年以降だけを正規化し、`data/usa_yield_curve.csv` に保存します。既存の `usa_yield_curve.csv` がある場合は、その中でアーカイブより新しい日付（例: 2023 年以降）はそのまま残してマージされます。

   ```powershell
   python fetch_usa_data.py ダウンロードしたファイル.csv
   ```

これで米国データが 2000 年〜現在（既存分とマージされた範囲）で表示されます。

---

## 5. よくあるトラブルと対処

- **仮想環境が有効にならない / 実行ポリシーのエラーが出る**
  - PowerShell で実行ポリシーが制限されている場合があります。
  - 管理者権限の PowerShell で一度だけ次を実行してください（会社 PC などの場合はルール要確認）。

    ```powershell
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    ```

- **`pip install --upgrade pip` で「To modify pip, please run...」と出る**
  - 仮想環境では `python -m pip install --upgrade pip` を使ってください（README の手順もこの形式にしています）。

- **`ModuleNotFoundError` が出る**
  - 仮想環境が有効な状態で、もう一度 `pip install -r requirements.txt` を実行してください。

---

## 6. カスタマイズのヒント

- 日付軸（日本の `S49.9.24` 形式）を西暦に変換して、より直感的な年表にする。
- 色のスケールやカメラ位置を変更して、日経サイトの見た目にさらに近づける。
- 断面グラフの残存期間軸を対数スケールに変更する。

これらの変更も一緒に進めたい場合は、どの部分を強化したいか教えてください。

