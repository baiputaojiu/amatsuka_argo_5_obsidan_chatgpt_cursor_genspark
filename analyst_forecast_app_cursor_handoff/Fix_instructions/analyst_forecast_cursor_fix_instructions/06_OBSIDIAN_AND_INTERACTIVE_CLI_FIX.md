# 06 — Obsidian文書・中央プロンプト・対話CLI修正

## 目的

初回利用者が実パスや全コマンドを暗記せず、Obsidian Vault内の説明とターミナル案内だけで開始できる状態にする。

## コピペ用依頼文

```text
Obsidian Vault初期化、中央プロンプト、対話式案件作成を修正してください。

1. Vaultの意味を明確化する

現在の--vault-rootは、実際にはObsidian Vaultそのものではなく、Vault内の30_Permanent/★アナリスト調査を指します。名称と説明を整理してください。

望ましい設定例：
- obsidian_vault_path: Obsidian Vaultの絶対パス
- workspace_relative_path: 30_Permanent/★アナリスト調査
- workspace_root: 上記2つから安全に算出

既存configのvault_rootは後方互換として読み込めるようにしてください。workspace_relative_pathがVault外へ抜ける`..`や絶対パスを不正として扱ってください。個人の絶対パスをGitへ保存しないでください。

2. docsとpromptsを空にしない

init後、workspace_root内へ最低限次を配置してください。
- README.md
- AI_WORK_GUIDE.md
- docs/01_スタートアップガイド/STARTUP_GUIDE.md
- docs/02_取扱説明書/USER_MANUAL.md
- docs/03_仕様書/の主要仕様
- docs/04_参考資料/PROMPT_CATALOG.md、PYTHON_TASK_CATALOG.md、TROUBLESHOOTING.md
- docs/05_計画/FUTURE_ROADMAP.md、DECISION_LOG.md、OPEN_QUESTIONS.md
- prompts/以下の中央プロンプトテンプレート

Python package／将来のexeでも利用できるversioned resourceとして同梱し、開発リポジトリの偶然の相対位置へ依存しないでください。中央prompt templateを正本とし、案件の01_promptsは変数を埋めたsnapshotにしてください。runs.pyへ長いprompt本文をハードコードし続けないでください。

既存ユーザーが編集したVault内文書をinit再実行で無条件上書きしないでください。app version付きsnapshot、差分案内、バックアップ、または`--update-docs`等の明示操作を設計してください。

3. 対話式スタート

`analyst-forecast start`等の対話コマンドを追加してください。将来exeをダブルクリックしたときにも同じwizardを呼び出せる構造にします。

質問項目：
- 分析対象者
- 調査開始・終了。既定は今日から過去6か月。
- evaluation_as_of。既定は今日。
- 媒体：YouTube、ブログ、X、その他Webを複数選択。
- 重点対象。空欄可。
- Cursor／ChatGPTの使用方針と高性能モデル名。未設定なら実行前警告。
- 最終確認。

キャンセル可能にし、日本語で入力例と修正方法を表示してください。非対話環境やテスト、スクリプト利用のため、既存のflag方式も維持してください。

4. 初回設定案内

- 実Vaultを選ぶ方法。
- `30_Permanent/★アナリスト調査`を作ること。
- configの保存場所。
- APIキーを保存しない場所。
- 初回案件後のNEXT_ACTIONS。
- Cursorへ「次に何をすべき？」と尋ねる方法。

必須テスト：
- Windows形式パス、空白、日本語、★を含むパス。
- 既存vault_root configの移行。
- path traversal拒否。
- initで必須文書と中央promptが配置される。
- init再実行でユーザー編集を上書きしない。
- 案件snapshotが中央prompt versionとhashを記録する。
- 対話wizardの既定6か月、媒体複数選択、キャンセル、入力やり直し。
- flag方式の既存CLI回帰。
- 個人絶対パスや秘密値が生成prompt、監査ログ、Git管理ファイルへ漏れない。

Ruff、mypy、pytestを実行し、README、スタートアップガイド、取扱説明書、実装状況を更新してください。PyInstallerによるexe完成は今回必須ではありませんが、resource読込みが将来packaging可能な設計であることを確認してください。commit・pushは行わないでください。
```

