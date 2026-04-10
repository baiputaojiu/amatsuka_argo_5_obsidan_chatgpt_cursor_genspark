# 日経平均 月末・月初変動率ツール

過去 10 年間（既定）の各月について、**月末3日間の最後の取引日**から**翌月の月初3日間の最初の取引日**までの日経平均の変動率（%）を、yfinance で取得してリスト表示し、CSV と画像で保存するツールです。

## 前提条件

- **Python 3.x** がインストールされていること（3.10 以上を推奨）。
- ネットワーク接続があり、yfinance が Yahoo Finance からデータを取得できること。

## 作業手順（仮想環境の構築から実行まで）

### 1. フォルダへ移動

プロジェクトの `nikkei_month_turn` フォルダに移動します。

```bash
cd nikkei_month_turn
```

（リポジトリルートから実行する場合の例）

```bash
cd d:\workspace\amatsuka_argo_5_obsidan_chatgpt_cursor_genspark\nikkei_month_turn
```

### 2. 仮想環境の作成

同じフォルダ内に仮想環境を作成します。

```bash
python -m venv .venv
```

`.venv` フォルダが作成されます。

### 3. 仮想環境の有効化

**Windows（PowerShell / コマンドプロンプト）:**

```bash
.venv\Scripts\activate
```

**macOS / Linux:**

```bash
source .venv/bin/activate
```

有効化に成功すると、プロンプトの先頭に `(.venv)` と表示されます。

### 4. 依存パッケージのインストール

```bash
pip install -r requirements.txt
```

`yfinance`, `pandas`, `matplotlib` がインストールされます。

### 5. 実行

```bash
python app.py
```

- 標準出力に、各月の「月末日 → 月初日」と変動率（%）の一覧が表示されます。
- 同じタイミングで以下が保存されます。
  - **CSV**: `output/nikkei_month_turn.csv`
  - **画像**: `output/nikkei_month_turn.png`

`output` フォルダは存在しない場合に自動作成されます。

### 6. オプション: 取得年数の指定

過去 5 年分だけ取得したい場合などは、`--years` を指定します。

```bash
python app.py --years 5
```

### 7. 仮想環境の終了

作業が終わったら、仮想環境を無効化します。

```bash
deactivate
```

## 出力ファイルの説明

### CSV（`output/nikkei_month_turn.csv`）

| 列名 | 説明 |
|------|------|
| year_month | 対象月（YYYY-MM） |
| last_date_end_of_month | 月末3日間の最後の取引日 |
| first_date_start_of_month | 翌月月初3日間の最初の取引日 |
| close_end | 月末側の終値 |
| close_start | 月初側の終値 |
| change_pct | 変動率（%） |

- 文字コード: UTF-8（BOM 付き）のため、Excel でそのまま開けます。

### 画像（`output/nikkei_month_turn.png`）

- **横軸**: 年月（YYYY-MM）
- **縦軸**: 変動率（%）
- 棒グラフで表示。プラスは緑、マイナスは赤で色分けしています。

## ディレクトリ構成

```
nikkei_month_turn/
├── .venv/             # 仮想環境
├── app.py             # メインスクリプト
├── requirements.txt
├── output/            # 実行後に CSV / PNG が保存される
├── README.md          # 本ファイル
└── Specification.md   # 詳細仕様（フォルダ・システム・プログラム構成）
```

詳細な仕様・プログラム構成は [Specification.md](Specification.md) を参照してください。
