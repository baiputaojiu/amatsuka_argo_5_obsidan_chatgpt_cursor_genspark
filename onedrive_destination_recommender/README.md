# OneDrive保存先レコメンダー

OneDrive業務フォルダの保存先候補を表示する、個人用Windowsデスクトップアプリです。

現在はMVP 0のStep 2まで実装済みです。独立したPython環境、Tkinter画面、Outlook COM疎通、設定検証、カタログ更新に加え、検索語の正規化と保存先候補の判定コアを実装しています。画面からのカタログ更新と保存先候補の表示はまだ利用できません。

## 開発環境

- Windows
- Python 3.12
- Outlookデスクトップアプリ（MSG確認時だけ必要）

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

## 起動

```powershell
.\run_app.bat
```

現時点ではStep 2確認用の最小画面だけが開く。

MVP 0では診断ログを作らないため、起動経路は`run_app.bat`に限定し、エラー確認用のコンソールを意図的に表示する。

## テスト

単体テストと静的検査：

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check src tests
```

Outlook COM結合テストは、実MSGをリポジトリへ置かず、リポジトリ外のローカルパスを環境変数で指定して明示実行する。

```powershell
$env:ODR_TEST_MSG_PATH = "C:\path\to\local-test.msg"
.\.venv\Scripts\python.exe -m pytest -m integration
Remove-Item Env:ODR_TEST_MSG_PATH
```

テストは件名・本文・添付ファイル名を出力・保存せず、項目へアクセスできたことだけを確認する。環境変数が未指定の場合、結合テストはスキップする。

## 設定

実装後の設定例は`settings.example.json`を参照する。実際の`settings.json`、`catalog.json`、`audit.jsonl`は次のリポジトリ外へ置く。

```text
%LOCALAPPDATA%\OneDriveDestinationRecommender\
```

`settings.json`はアプリから作成・変更しない。`catalog.json`は、設定された今年度・昨年度フォルダを読み取り専用で完全走査できた場合だけ、同じランタイム領域で原子的に置き換える。

今年度・昨年度フォルダには、Windowsの大文字小文字差を含めて同一ではなく、互いに入れ子でないパスを指定する。

## データの扱い

- 実業務のMSG、メール本文、添付内容をGitへ保存しない
- OneDrive上のファイル・フォルダを変更しない
- Outlook COMはローカルMSGの読み取りだけに使う
- MSGを読めない場合も、手動検索と通常ファイルの処理は継続できる構成にする
