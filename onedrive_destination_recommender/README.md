# OneDrive保存先レコメンダー

OneDrive業務フォルダの保存先候補を表示する、個人用Windowsデスクトップアプリです。

現在はMVP 0のStep 4まで実装済みです。設定・カタログ、検索語と候補判定、Outlook MSGの読み取り、Audit記録、Codex相談用プロンプト生成を実装しています。画面からのカタログ更新、候補表示、確定、コピーはStep 5で接続するため、まだ利用できません。

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

現時点ではStep 4確認用の最小画面だけが開く。

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

`audit.jsonl`は、Step 5で候補選択・保存先未定・却下を確定したときだけ1行追記する。日常の候補判定はこのログを読み込まず、自動学習や自動ルール変更を行わない。

今年度・昨年度フォルダには、Windowsの大文字小文字差を含めて同一ではなく、互いに入れ子でないパスを指定する。

## データの扱い

- 実業務のMSG、メール本文、添付内容をGitへ保存しない
- OneDrive上のファイル・フォルダを変更しない
- Outlook COMはローカルMSGの読み取りだけに使う
- MSGを読めない場合も、手動検索と通常ファイルの処理は継続できる構成にする
- MSG本文、検索語全文、Codex相談用プロンプトをAuditへ記録しない
- Codex相談用の案内とプロンプトはメモリ上だけで生成し、ユーザー操作なしに送信・コピー・保存しない

## Auditを使った改善相談

判定の改善を望む場合だけ、次の手順で行う。アプリがAuditを自動的に学習へ使うことはない。

1. `%LOCALAPPDATA%\OneDriveDestinationRecommender\audit.jsonl`から、手動で修正した案件または却下した案件を確認する
2. 該当する元MSGまたは元ファイルを自分で選ぶ
3. Auditログと対象ファイルをCodexへ添付し、原因分析と改善案を依頼する
4. 提案内容を確認する
5. 採用する場合だけ、改めて明示的にプログラム変更を依頼する

Auditには業務ファイル名と業務フォルダの絶対パスが含まれるため、公開リポジトリへ置かない。
