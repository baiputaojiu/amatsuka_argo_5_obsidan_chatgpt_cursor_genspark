# 取り扱い手順書（USER GUIDE）

本書は Outlook→Google カレンダー同期ツールを、実作業の順番どおりに迷わず進めるための手順書です。  
上から順に読めば、基本的に後戻りせずに作業できます。

---

## 0. 最初に確認（1分）

- このツールは **プロジェクト専用仮想環境 (`.venv`)** を前提に動作します。
- **`credentials.json` 配置や Google 認証より先に**、必ず仮想環境を作成してください。
- ターミナル実行は `outlook_google_sync/` をカレントディレクトリにして行います。

```powershell
cd D:\workspace\amatsuka_argo_5_obsidan_chatgpt_cursor_genspark\outlook_google_sync
```

> 注意: `cd scripts\setup_env.bat` のようにファイルへ `cd` はできません。  
> `.bat` は `cd` 後に `.\scripts\setup_env.bat` / `.\scripts\run_gui.bat` で実行します。

---

## 1. 初回セットアップ（初回だけ）

### 1-1. Python を確認

- `python --version` が 3.10 以上なら次へ進みます。
- 未導入または古い場合のみ Python 3.10 以上をインストールします。

### 1-2. リポジトリを用意

- `outlook_google_sync/` フォルダが未取得なら、クローンまたは展開して配置します。

### 1-3. 仮想環境を作成（最重要）

```powershell
cd D:\workspace\amatsuka_argo_5_obsidan_chatgpt_cursor_genspark\outlook_google_sync
.\scripts\setup_env.bat
```

- 完了後に `outlook_google_sync\.venv\` が作成されます。
- これは **初回のみ**（または `.venv` を再作成したいときのみ）実行します。

### 1-4. `credentials.json` を配置（Google連携する場合）

1. [Google Cloud Console](https://console.cloud.google.com/) でプロジェクトを用意
2. Google Calendar API を有効化
3. OAuth 同意画面を設定（テスト運用ならテストユーザー追加）
4. OAuth クライアント ID を **デスクトップアプリケーション** で作成
5. JSON をダウンロードして `credentials.json` にリネーム
6. `%USERPROFILE%\.outlook_google_sync\credentials.json` に配置

配置確認:

```powershell
Test-Path "$env:USERPROFILE\.outlook_google_sync\credentials.json"
```

- `True` なら配置完了です。
- `credentials.json` はリポジトリ配下に置かないでください。

---

## 2. 初回起動と認証（初回だけ）

### 2-1. GUI を起動

```powershell
cd D:\workspace\amatsuka_argo_5_obsidan_chatgpt_cursor_genspark\outlook_google_sync
.\scripts\run_gui.bat
```

### 2-2. Google 認証

1. 設定ウィンドウの「Google」タブを開く
2. 「Google認証」を押す
3. ブラウザで許可
4. `%USERPROFILE%\.outlook_google_sync\token.json` が作成されたら完了

`403: access_denied` / 「アクセスをブロック」の場合:

- OAuth 同意画面が Testing のときは、認証に使う Gmail を「テストユーザー」に追加します。

---

## 3. 日常運用（毎回ここから）

### 3-1. GUI を起動

```powershell
cd D:\workspace\amatsuka_argo_5_obsidan_chatgpt_cursor_genspark\outlook_google_sync
.\scripts\run_gui.bat
```

### 3-2. 入力方法を確認

| 入力方法 | 使う条件 | 注意 |
|----------|----------|------|
| COM | Classic Outlook + Windows + pywin32 | 企業GPOで制限される場合あり |
| 手動 ICS | Outlook から `.ics` を出力 | ファイルを都度選択 |
| マクロ連携 | VBA で `.ics` 出力 | `samples/macro/export_calendar.bas` を参照 |

**重要**: COM と ICS を切り替えると `sync_key` が変わり、重複が起きることがあります。

### 3-3. 同期の実行

1. 開始日・終了日を指定
2. 必要なモードを選んで実行

- **通常同期**: 日々の運用（差分反映）
- **通常プレビュー**: 本番前確認（推奨）
- **強制フル同期**: 大きな設定変更後、件数異常、重複疑い時
- **フルプレビュー**: 強制フル同期前の確認

> プレビューから5分以上経過すると再確認ダイアログが出ます。

### 3-4. 重複修復（必要時のみ）

1. 「重複修復」ボタンを押す
2. 検出モードを選ぶ（同一 `sync_key` / 同名同開始同終了）
3. グループごとにマージ方針を選択
4. 「選択内容で一括マージを実行」

- 場所が異なる予定が混在するグループはマージ不可です。
- 同名同時刻でも別会議の可能性があるため、プレビュー確認を推奨します。

---

## 4. 作業後・問題時の確認

### 4-1. 接続テスト（必要時）

- 実施タイミング: 環境構築直後、PC/入力方法変更後、エラー切り分け時
- 目的: 疎通確認（全同期の成功保証ではありません）

### 4-2. ログ確認

- 保存先: `%USERPROFILE%\.outlook_google_sync\logs\`
- ファイル名: `sync_YYYY-MM-DD.log`
- `standard` は要約、`detailed` は詳細解析向け

### 4-3. よくあるエラーと対処

| 症状 | 対処 |
|------|------|
| `ModuleNotFoundError` / GUI起動不可 | `.\scripts\setup_env.bat` で `.venv` を再構築 |
| `run_gui.bat` が認識されない | カレントを確認し `.\scripts\run_gui.bat` で実行 |
| 認証失敗 / トークン期限切れ | `credentials.json` 配置確認後、再認証 |
| `403: access_denied` | OAuth 同意画面のテストユーザーに該当 Gmail を追加 |
| 重複イベント発生 | 重複修復ツールを実行 |
| ICS パースエラー | 接続テストと ICS ファイル内容を確認 |

---

## 5. 実行コマンド早見表

| 操作 | コマンド / 実行場所 |
|------|---------------------|
| 初回セットアップ | ターミナルで `.\scripts\setup_env.bat` |
| GUI 起動 | ターミナルで `.\scripts\run_gui.bat` |
| Google 認証 | GUI（設定 → Google）+ ブラウザ |
| 同期 / プレビュー / 重複修復 | GUI 操作 |
| テスト実行 | ターミナルで `.\.venv\Scripts\python.exe -m pytest tests/ -v` |

---

## 6. 補足（必要なときだけ）

### 6-1. 手動で仮想環境を作る場合

`setup_env.bat` が使えない場合のみ実施します。

```powershell
cd D:\workspace\amatsuka_argo_5_obsidan_chatgpt_cursor_genspark\outlook_google_sync
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 6-2. 非公開予定の取り扱い

- 非公開予定を含める設定では、タイトル・本文・場所も Google に送信されます。
- 組織ポリシーに従って利用してください。

### 6-3. 処理中の画面操作

- 同期/プレビュー/接続テスト中は主要ボタンが一時無効になります。
- メインウィンドウの `×` は確認なしで即終了します（未保存入力は失われます）。
