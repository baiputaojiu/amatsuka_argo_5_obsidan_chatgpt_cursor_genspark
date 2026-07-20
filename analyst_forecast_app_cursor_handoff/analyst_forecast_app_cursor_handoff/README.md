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
