# アナリスト予想検証アプリ — 仕様・計画書セット

このフォルダは、経済アナリストの過去および現在の発言を収集し、予想の正確性・具体性・得意分野を検証する個人用アプリケーションの仕様・計画書です。

最終的には、過去実績を踏まえて複数アナリストの現在予想を比較し、今後上昇する可能性がある銘柄・指数・為替・コモディティ・業種・テーマの候補を見つけることを目指します。

## Cursorで実装を始める方へ

最初に [00_START_HERE.md](00_Cursor引継ぎ/00_START_HERE.md) を読み、次に [01_INITIAL_IMPLEMENTATION_REQUEST.md](00_Cursor引継ぎ/01_INITIAL_IMPLEMENTATION_REQUEST.md) の依頼文をCursorへ貼り付けてください。

ChatGPTとの会話全文を併用する場合は [CHAT_HISTORY_GUIDE.md](reference/CHAT_HISTORY_GUIDE.md) に従い、背景資料として `reference/chat_history.md` へ置いてください。正式仕様は `docs/` であり、会話全文より優先されます。

## 最初に読むもの

1. 初回準備：[STARTUP_GUIDE.md](docs/01_スタートアップガイド/STARTUP_GUIDE.md)
2. 普段の操作：[USER_MANUAL.md](docs/02_取扱説明書/USER_MANUAL.md)
3. 全体設計：[SYSTEM_SPECIFICATION.md](docs/03_仕様書/SYSTEM_SPECIFICATION.md)
4. 開発順序：[MVP_PLAN.md](docs/05_計画/MVP_PLAN.md)

## このアプリの基本的な使い方

1. Windowsで実行ファイルを起動する。
2. 分析対象者、調査期間、調査媒体等を入力する。
3. ターミナルに表示された「次にすること」に従う。
4. AI処理が必要な場合は、生成されたCursor用またはChatGPT用プロンプトを実行する。
5. AI出力を指定フォルダへ保存し、実行ファイルから取り込む。
6. Pythonが市場データの取得、採点、集計、グラフ作成を行う。
7. Obsidianでレポートを確認する。

通常利用で、処理順序や全プロンプト名を暗記する必要はありません。ターミナルと各案件の `NEXT_ACTIONS.md` が次の作業を案内します。

## AIとPythonの役割

| 処理 | 担当 |
|---|---|
| ブログ・X・Web記事等の探索 | 高性能AI |
| YouTube文字起こしの整理・話者推定 | 高性能AI |
| 原文から予想を抽出 | 高性能AI |
| テーマ・業種の評価対象候補を決定 | 高性能AI＋AI再検証 |
| JSON形式、引用存在、ID等の検査 | Python |
| 市場データ取得 | Python |
| 方向・時期・程度・早期実現の計算 | Python |
| 集計・表・グラフ作成 | Python |
| 曖昧な条件や分析品質の判断 | 高性能AI |
| 次に行う処理の機械判定 | Python |

## 主要文書

- [製品目的](docs/05_計画/PRODUCT_VISION.md)
- [評価方法](docs/03_仕様書/METHODOLOGY.md)
- [データモデル](docs/03_仕様書/DATA_MODEL.md)
- [フォルダ構造](docs/03_仕様書/FOLDER_STRUCTURE.md)
- [フローチャート](docs/03_仕様書/WORKFLOW_DIAGRAMS.md)
- [プロンプト一覧](docs/04_参考資料/PROMPT_CATALOG.md)
- [Python処理一覧](docs/04_参考資料/PYTHON_TASK_CATALOG.md)
- [将来課題](docs/05_計画/FUTURE_ROADMAP.md)
- [決定履歴](docs/05_計画/DECISION_LOG.md)
- [未決定事項](docs/05_計画/OPEN_QUESTIONS.md)

## 初期版の前提

- 個人利用専用であり、外部公開機能は持たない。
- Windows 11を初期対応環境とする。
- ターミナルとObsidianを主な操作・閲覧環境とする。
- AI処理はCursorまたはChatGPTで手動実行できる。
- AIを使う意味判断ではすべて高性能モデルを使用する。
- 市場データは無料・簡単さを優先し、`yfinance`、FRED、CSV取込みを使用する。
- ループエンジニアリングによる継続改善は初期版へ含めず、将来課題とする。

## 開発版のセットアップ

Windows 11とPython 3.12系を対象とする。初回実装はPython 3.12.10で検査している。

```powershell
py install 3.12
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.lock
python -m pip install -e . --no-deps
```

`requirements.lock` は開発、テスト、市場providerを含む全依存を固定している。APIキーや実Vaultのパスはコミットせず、`--config` または `ANALYST_FORECAST_CONFIG`、必要時は `FRED_API_KEY` をローカル環境へ設定する。`.env.example` に変数名だけを記載している。

## 代表コマンド

```powershell
analyst-forecast --help
analyst-forecast init --vault-root "D:\任意のVault内\30_Permanent\★アナリスト調査" --config ".\config.local.yaml"
analyst-forecast run create --name "対象者名" --period-start 2026-01-01 --period-end 2026-06-30 --evaluation-as-of 2026-07-20 --media youtube --config ".\config.local.yaml"
analyst-forecast source import RUN-YYYYMMDD-NNN ".\source.txt" --medium youtube --config ".\config.local.yaml"
analyst-forecast ai ingest ".\forecast_extraction.json" --config ".\config.local.yaml"
analyst-forecast market evaluate RUN-YYYYMMDD-NNN FCC-000001 --as-of 2026-07-20 --provider csv --csv-path ".\market.csv" --config ".\config.local.yaml"
analyst-forecast status RUN-YYYYMMDD-NNN --config ".\config.local.yaml"
```

raw取込みでは元ファイルをUTF-8のまま別名保存し、SHA-256で同一内容を検出する。AI出力は固定JSON Schema、Pydantic、ID参照、原文引用と文字位置を検証し、成功するまで正式予想テーブルへ登録しない。

## 開発時の検査

```powershell
ruff format --check .
ruff check .
mypy src
pytest
```

通常の `pytest` はネットワークを使用しない。明示的な実ネットワーク確認だけを次で実行する。FRED試験にはローカルの `FRED_API_KEY` が必要である。

```powershell
$env:RUN_NETWORK_TESTS = "1"
pytest -m integration
```

実装は `src/analyst_forecast/` の `domain`、`application`、`infrastructure`、`cli`、`schemas` に分離している。実装計画、暫定判断、検査結果は `docs/06_実装/` を参照する。
