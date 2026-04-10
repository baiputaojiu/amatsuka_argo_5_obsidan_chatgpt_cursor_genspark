# Outlook → Google Calendar 同期ツール

Outlook の予定を Google カレンダーへ **一方向同期** するデスクトップアプリケーションです。  
組織のポリシーにより Outlook Google Calendar Sync (OGCS) や Microsoft Graph API が利用できない環境を想定し、**Outlook COM / ICS ファイル** から予定を読み取り、**Google Calendar API** で書き込みます。

---

## 正本

- 仕様の正本: [`docs/SPEC.md`](docs/SPEC.md)
- 変更時は SPEC・USER_GUIDE・図・CHANGELOG を同一変更セットで更新すること（Ch44）

---

## 主な特徴

| 機能 | 概要 |
|------|------|
| **3 つの入力方法** | Outlook COM / 手動 ICS / VBA マクロ連携 |
| **2 つの同期モード** | 通常同期（差分） / 強制フル同期 |
| **プレビュー** | 本番前に想定アクション一覧を確認（dry-run） |
| **衝突ポリシー** | overwrite（既定）/ detach_new / merge |
| **除外フィルタ** | 終日・Free・Tentative・Private・カテゴリ・キーワード |
| **カテゴリ色同期** | OlCategoryColor → Google colorId 固定マッピング |
| **削除の安全性** | 本ツール作成イベントのみ対象、UI で個別/一括承認 |
| **重複修復** | sync_key 一致の重複を検出し修復 |
| **設定集約** | 全設定をタブ型ウィンドウに統合（ポップアップ多発なし） |

---

## 前提環境

| 項目 | 要件 |
|------|------|
| OS | Windows 64 ビット |
| Python | 3.10 以上 |
| Outlook | Classic Outlook（COM モード時） |
| ネットワーク | Google API への HTTPS が可能 |

---

## クイックスタート

**最初に仮想環境（`.venv`）を構築してから** 続きの手順に進んでください。`credentials.json` より先に実施します。

### 実行場所（先に確認）

- **推奨**: Cursor のターミナル（PowerShell）で実行
- 代替: Windows PowerShell（単体）
- エクスプローラから `.bat` ダブルクリックでも実行可能だが、エラー確認のしやすさのためターミナル実行を推奨

カレントディレクトリ変更が必要な場合は、先に次を実行:

```powershell
cd D:\workspace\amatsuka_argo_5_obsidan_chatgpt_cursor_genspark\outlook_google_sync
```

### 1. 仮想環境の構築（必須・最初のステップ）

```bat
cd D:\workspace\amatsuka_argo_5_obsidan_chatgpt_cursor_genspark\outlook_google_sync
.\scripts\setup_env.bat
```

上記を実行すると **`outlook_google_sync\.venv\`** が作成され、依存パッケージがその中にインストールされます。2回目以降はスキップしてよい。

### 2. Google 認証情報の配置

1. [Google Cloud Console](https://console.cloud.google.com/) で OAuth 2.0 クライアント ID を作成
2. `credentials.json` をダウンロード
3. 以下に配置：

```
%USERPROFILE%\.outlook_google_sync\credentials.json
```

### 3. アプリの起動

```bat
.\scripts\run_gui.bat
```

上記は Cursor ターミナル（PowerShell）で `outlook_google_sync/` 配下から実行してください。`.venv` が存在する場合は **その Python で GUI が起動** します。

### 4. 初回 Google 認証

1. 起動後、**設定 → Google タブ** から「Google 認証」をクリック
2. ブラウザで Google アカウントにログイン・許可
3. 認証成功後、「カレンダー一覧を取得」で対象カレンダーを選択

### 5. 同期の実行

1. メインウィンドウで **開始日・終了日** を選択
2. **通常プレビュー** で想定アクションを確認
3. 問題なければ **通常同期** を実行

---

## フォルダ構成

```
outlook_google_sync/           ← 開発・納品フォルダ（ソースコード等）
├── docs/
│   ├── SPEC.md                ← 仕様書（正本）
│   ├── USER_GUIDE.md          ← 取り扱い手順書
│   ├── SYSTEM_ARCHITECTURE.md ← システム構成図
│   ├── FLOWS.md               ← フロー図
│   └── diagrams/              ← 図の元データ（drawio / svg）
├── src/outlook_google_sync/
│   ├── gui/                   ← GUI 層（tkinter）
│   ├── sync/                  ← 同期ロジック（エンジン・衝突・差分・削除等）
│   ├── connectors/            ← Outlook COM / ICS / Google Calendar 接続
│   ├── models/                ← EventModel, Profile, SyncResult
│   ├── config/                ← 設定読込・保存・migration・export/import
│   ├── services/              ← 接続テスト・通知・監査
│   ├── logging/               ← ロガー・ファイルシンク・UIシンク
│   ├── utils/                 ← ハッシュ・テキスト・日時ユーティリティ
│   └── main.py                ← エントリーポイント
├── tests/
│   ├── unit/                  ← 単体テスト（44件）
│   └── integration/           ← 統合テスト
├── samples/
│   ├── ics/                   ← サンプル ICS（匿名・架空データ）
│   ├── macro/                 ← VBA マクロサンプル (.bas)
│   └── log_examples/          ← サンプルログ（機密マスク済み）
├── scripts/
│   ├── run_gui.bat            ← GUI 起動スクリプト
│   └── setup_env.bat          ← 環境セットアップスクリプト
├── requirements.txt
├── CHANGELOG.md
└── README.md                  ← 本ファイル
```

---

## 実行時データ保存先

実行時データは **開発フォルダとは別** の場所に保存されます。  
**開発フォルダに実行時データを混在させないでください。**

| ファイル | パス |
|----------|------|
| `credentials.json` | `%USERPROFILE%\.outlook_google_sync\credentials.json` |
| `token.json` | `%USERPROFILE%\.outlook_google_sync\token.json` |
| `config.json` | `%USERPROFILE%\.outlook_google_sync\config.json` |
| ログ | `%USERPROFILE%\.outlook_google_sync\logs\` |
| 監査記録 | `%USERPROFILE%\.outlook_google_sync\audit.json` |

---

## 入力方法と読み取りエンジン

| UI 表示 | `input_method` | 読み取りエンジン | 説明 |
|---------|----------------|-----------------|------|
| COM | `com` | `outlook_com` | Outlook COM で直接読取（Classic Outlook 必須） |
| 手動 ICS | `ics_manual` | `ics` | `.ics` ファイルをパス指定で読取 |
| マクロ連携 | `ics_macro` | `ics` | VBA マクロで出力した `.ics` を読取 |

- **マクロ連携**: `samples/macro/export_calendar.bas` を Outlook VBA にインポートして実行後、出力 ICS をアプリで選択
- **COM と ICS で同一予定の `sync_key` は一致しない場合がある**ため、入力方法を切り替えると重複が発生する可能性があります（切替時に確認ダイアログが表示されます）

---

## 同期モード

### 通常同期（差分）
前回の同期メタデータを参照し、ソース側で変更があった予定のみ Google に書き込みます。  
ただし、Google 側の手編集は差分スキップせず衝突検知を実行します。

### 強制フル同期
差分メタデータを参照せず、期間内の全予定を再評価します。  
設定変更後の整合再構築や、不整合の修復に使用します。

### プレビュー同期（dry-run）
書込みを一切行わず、想定アクション（新規作成・更新・削除候補・スキップ等）を一覧表示します。  
**通常用プレビュー** と **フル用プレビュー** を区別して実行できます。

---

## 衝突ポリシー

Google 上の本ツール作成イベントが、ツール以外の操作で編集された場合の挙動を制御します。

| ポリシー | 説明 |
|----------|------|
| `overwrite`（既定） | Outlook/ICS を正として全フィールドを上書き |
| `detach_new` | 手編集済みイベントから管理を外し、ソースから新規作成 |
| `merge` | 既存を保持しつつ項目優先設定でマージ |

衝突の検知は以下の **3 条件すべて** を満たした場合です：
1. `tool_marker` が本ツールの固定値と一致
2. Google `updated` が `last_tool_write_utc` より新しい
3. 同期対象フィールドに差分がある

---

## 除外フィルタ

設定ウィンドウの「フィルタ」タブで、以下の条件で予定を除外できます：

| 条件 | 説明 |
|------|------|
| 終日予定 | `AllDayEvent = True` の予定を除外 |
| Free（空き） | `BusyStatus = 0` の予定を除外 |
| Tentative（仮） | `BusyStatus = 1` の予定を除外 |
| Private | 非公開予定を除外（除外設定は `include_private` より優先） |
| カテゴリ | 指定カテゴリのみ含む / 指定カテゴリを除外 |
| 件名キーワード | 部分一致で除外（大文字小文字無視） |
| 場所キーワード | 部分一致で除外（大文字小文字無視） |

---

## 接続テスト

**疎通と最小読取の確認** であり、**本番同期の成功を保証するものではありません。**

| 対象 | 確認内容 |
|------|----------|
| **Outlook COM** | `Outlook.Application` 取得 → カレンダーオープン → 1件読取 |
| **ICS** | ファイル存在 → パース → VCALENDAR/VEVENT → UID → 期間内イベント確認（計11項目） |
| **Google** | トークン有効性 → `calendarList.list`（基本接続） → 対象カレンダー確認（選択時のみ） |

- Outlook/ICS が 0 件でも **警告であり失敗ではない**
- 対象カレンダー未選択時: 「対象カレンダー未選択のため、カレンダー存在確認は未実施です。」と表示（成功扱い）

---

## 削除仕様

削除候補は以下の **すべて** を満たすイベントのみ：

1. **DEL-01**: `tool_marker` が本ツール固定値と一致
2. **DEL-02**: イベントの開始・終了が同期対象期間 `R` と重なる
3. **DEL-03**: `sync_key` が今回の読取結果 `K_now` に含まれない

- `R` 外に移動しただけのイベントは候補にならない（DEL-05）
- Google 手作成イベントは候補に含まれない（Ch16）
- 削除前に必ず **確認ダイアログ** が表示され、個別/一括で承認できる

---

## 「タイトルのみ」モード

`detail_level = title_only` に設定した場合：
- 新規作成時: `summary` と `start`/`end` のみ送信（`description`/`location` は含めない）
- 更新時: `description`/`location` を PATCH に含めない（空文字で消さない）

> **注意**: 過去に「詳細」モードで同期した予定の `description`/`location` は Google 上に残り続けます。  
> 非公開予定を含めて同期した場合、情報が Google に残ります。

---

## 非公開予定

- **含める設定 ON**: `visibility=private` を付与し、タイトル・本文・場所を含む全フィールドを送信
- **含める設定 OFF**: Private の予定は読取から除外

> **注意**: 非公開予定を Google に送信すると情報持ち出しリスクがあります。  
> 組織ポリシーの制約により利用できない場合があります。

---

## カテゴリ色同期

Outlook カテゴリの **色（OlCategoryColor 列挙値）** を Google `colorId` に変換する固定マッピングです。  
**カテゴリ名は参照しません**（利用者がカテゴリ名を変更しても色が同じなら同じ `colorId` になります）。

複数カテゴリがある場合は **先頭の認識された色** を採用します。  
未マッピング・色なし・ICS にカテゴリ色情報がない場合は `colorId` を送信しません（Google 既定色）。

---

## ログ

| 種別 | 場所 | 説明 |
|------|------|------|
| 実行時ログ | `%USERPROFILE%\.outlook_google_sync\logs\` | 詳細なスタックトレースを含む |
| UI ログ | メインウィンドウ下部 | 短いメッセージのみ |
| サンプルログ | `samples/log_examples/` | 機密マスク済みの例示のみ |

- トークンや個人情報はログに出力しない方針です
- 詳細度は設定ウィンドウの「ログ/通知」タブで `standard` / `detailed` を切替可能

---

## セキュリティ

- `credentials.json` / `token.json` を **リポジトリに含めない**
- `samples/log_examples/` に **生ログを置かない**
- 非公開予定を含めると **本文・場所も Google へ送信される**（情報持ち出しリスク）
- 組織ポリシーにより COM アクセスが制限される場合がある

---

## トラブルシューティング

| 症状 | 対処 |
|------|------|
| COM で「Outlook.Application 取得失敗」 | Classic Outlook がインストールされているか確認。新しい Outlook では COM 非対応 |
| ICS で「VEVENT なし」 | `.ics` ファイルに有効な VEVENT ブロックがあるか確認 |
| Google 認証エラー | `credentials.json` の配置場所を確認。ブラウザのポップアップブロッカーを確認 |
| 「開始日が終了日より後です」 | 日付範囲を正しく設定してください |
| 重複が発生した | COM ↔ ICS 切替で発生しやすい。**重複修復** 機能で対応可能 |
| 同期後に予定が消えた | 本ツールが作成したイベントのみが削除対象。削除は事前確認あり |
| 「プレビュー結果が古い」と表示される | プレビュー取得から 5 分以上経過。再プレビューを推奨 |
| レート制限エラー (429) | 再試行ダイアログで回数を選択。指数バックオフで自動再試行 |
| 設定が消えた | `config.json` の `schema_version` 不整合。バックアップから復元可能 |

---

## テスト

```powershell
cd D:\workspace\amatsuka_argo_5_obsidan_chatgpt_cursor_genspark\outlook_google_sync
.\.venv\Scripts\python.exe -m pytest tests/ -v
```

テスト実行場所: Cursor ターミナル（PowerShell）。仮想環境構築後に実行すること。

| テスト種別 | 件数 | 対象 |
|-----------|------|------|
| Unit | 41 | sync_key, 削除候補, 衝突検知, migration, フィルタ, 色マッピング等 |
| Integration | 3 | ICS コネクタの読取・日付範囲 |

---

## 将来拡張（初版非対象）

| ID | 機能 |
|----|------|
| OOS-01 | 定期同期（タスクスケジューラ連携） |
| OOS-02 | システムトレイ常駐 |
| OOS-03 | リマインダー同期 |
| OOS-04 | 出席者（attendees）同期 |
| OOS-05 | 起動時の同期自動開始 |

---

## ライセンス

プロジェクト固有。詳細は組織のポリシーに従ってください。
