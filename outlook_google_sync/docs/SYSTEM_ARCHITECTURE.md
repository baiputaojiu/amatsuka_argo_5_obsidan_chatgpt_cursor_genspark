# システムアーキテクチャ（Outlook → Google Calendar 同期ツール）

本書は、Outlook 由来の予定を Google Calendar に同期するデスクトップアプリケーションの構成をまとめたものです。仕様書 Ch43.1 で求められる構成要素（GUI、コネクタ、正規化モデル、同期エンジン、設定・ログ・通知・監査、フィルタと色マッピングなど）を網羅します。

---

## 1. 目的と全体像

アプリケーションは **tkinter ベースの GUI** から操作し、**Outlook 側は COM または ICS** で読み取り、**EventModel** に正規化したうえで **同期エンジン** が差分またはフルモードで **Google Calendar API** に書き込みます。設定・認証トークン・ログ・監査はユーザーホーム配下の **ランタイムディレクトリ** に保持します。

---

## 1.1 実行場所（運用ルール）

- **初回**: `scripts/setup_env.bat` で **`outlook_google_sync/.venv/`** を作成し依存を入れる（**他の手順より先**）。
- `scripts/run_gui.bat` は **`.venv` があればその `python.exe` で起動** する。
- `scripts/setup_env.bat`、`scripts/run_gui.bat`、`python -m pytest` などのコマンドは、**`outlook_google_sync/` で Cursor ターミナル（PowerShell）から実行**する。
- GUI 起動後の操作（同期・プレビュー・接続テスト・重複修復・設定）は、**アプリ画面のボタン操作**で実行する。
- エクスプローラから `.bat` を直接実行する方法は補助的な手段とし、標準運用はターミナル実行とする。

---

## 2. 依存関係の方向（レイヤリング）

上位から下位へ一方向の利用を基本とします。**設定（config）** と **ログ（logging）** は各層から参照される横断的関心事です。

| 方向 | 意味 |
|------|------|
| **GUI** → **Sync / Services** | 画面は同期パイプラインと接続テスト・通知・監査保存などを呼び出す。 |
| **Sync / Services** → **Connectors** | エンジンとサービスが Outlook / Google の I/O をコネクタ経由で行う。 |
| **Connectors** → **Models / Utils** | 外部 API・ファイルの結果を EventModel 等に載せ、日時・ハッシュ・色などは utils を利用。 |
| **Config / Logging** | 全層が paths・設定読込・マイグレーション、およびロガー／UI シンクを利用可能。 |

### 2.1 テキストベースのアーキテクチャ図

```
                    ┌─────────────────────────────────────────┐
                    │  Cross-cutting: config/, logging/         │
                    │  (paths, settings_store, migration,       │
                    │   logger_factory, file/ui sinks)         │
                    └─────────────────┬───────────────────────┘
                                      │ 参照
    ┌─────────────────────────────────┼─────────────────────────────────┐
    │                                 ▼                                 │
    │  ┌──────────────┐    ┌──────────────────┐    ┌─────────────────┐ │
    │  │ gui/         │───▶│ sync/, services/ │───▶│ connectors/     │ │
    │  │ MainWindow   │    │ engine, conflict │    │ outlook_com     │ │
    │  │ Settings…    │    │ preview, diff/   │    │ outlook_ics     │ │
    │  │ Preview…     │    │ full, delete_…   │    │ google_calendar │ │
    │  │ Duplicate…   │    │ merge, filters,  │    └────────┬────────┘ │
    │  │ dialogs      │    │ key_utils, dup…  │             │          │
    │  └──────────────┘    └────────┬─────────┘             │          │
    │                               │                         ▼          │
    │                               └────────────────▶ models/, utils/  │
    │                                                    EventModel,       │
    │                                                    Profile, Filter…  │
    └────────────────────────────────────────────────────────────────────┘

    ランタイム: ~/.outlook_google_sync/  (config.json, token.json, logs/, audit.json 等)
```

---

## 3. ソースモジュール構成

```
src/outlook_google_sync/
├── gui/          # MainWindow, SettingsWindow, PreviewWindow, DuplicateRepairWindow, dialogs
├── models/       # EventModel, Profile, FilterConfig, SyncResult, config_schema (SCHEMA_VERSION)
├── sync/         # engine, conflict, preview, diff_sync, full_sync, delete_candidates,
│                 # merge, duplicate_repair, filters, key_utils
├── connectors/   # outlook_com, outlook_ics, google_calendar
├── config/       # settings_store, migration, paths, export_import
├── logging/      # logger_factory, ui_log_sink, file_log_sink
├── services/     # connection_test, notifications, audit_store
└── utils/        # hash_utils, datetime_utils, text_utils, color_mapping
```

---

## 4. Ch43.1 構成要素（機能別）

### 4.1 メイン GUI（tkinter MainWindow）

- **役割**: 同期実行、日付範囲、プレビュー・設定・重複修復への遷移、ログ表示領域との連携。
- **実装**: `gui/main_window.py`
- **依存**: `sync`（プレビュー・エンジン呼び出し）、`config`（設定の読み書き）、`logging`（UI ログ）、`services`（通知・接続テスト等）。

### 4.2 設定ウィンドウ（タブ構成）

- **役割**: 全設定を一画面に集約（SET-WIN 系要件）。
- **タブ**: **基本**、**同期**、**フィルタ**、**Google**、**ログ/通知**、**高度な設定**（`SettingsWindow._build_tabs`）。
- **実装**: `gui/settings_window.py`
- **状態**: プロファイル／フィルタ／衝突ポリシー等は `Profile`・設定 JSON と対応。

### 4.3 プレビューウィンドウ

- **役割**: 同期前の対象件数・スナップショット確認。
- **実装**: `gui/preview_window.py`、ロジックは `sync/preview.py`。

### 4.4 重複修復ウィンドウ

- **役割**: Google 側の重複やツール管理外イベントの整理 UI。
- **実装**: `gui/duplicate_repair_window.py`、`sync/duplicate_repair.py`。

### 4.5 Outlook COM コネクタ（pywin32）

- **役割**: 実行中の Outlook から予定アイテムを取得。
- **要点**: アイテム制限に **`IncludeRecurrences = True`** を設定し、繰り返しの個別インスタンスを列挙対象に含める（`connectors/outlook_com.py`）。
- **依存**: `pywin32`（Windows のみ）。

### 4.6 ICS コネクタ（icalendar + dateutil）

- **役割**: 手動／マクロ出力の `.ics` を読み込み `EventModel` へ変換。
- **繰り返し**: `icalendar` で VEVENT を解析し、**RRULE** は `dateutil.rrule.rrulestr` により指定期間内へ展開（未インストール時は展開スキップ、`connectors/outlook_ics.py`）。

### 4.7 正規化モデル（EventModel）

- **役割**: COM / ICS いずれからも同一の内部表現へ揃え、Google 用ボディ生成（`to_google_body`）、フィンガープリント、`extendedProperties.private` の同期キー等を担う。
- **実装**: `models/event.py`（dataclass）。

### 4.8 同期エンジン

- **衝突検知**: `sync/conflict.py`（Google 既存と書き込み内容の比較）。
- **モード**: **diff**（フィンガープリント差分、`diff_sync.filter_diff_targets`）と **full**（`full_sync.full_targets`）。
- **削除候補**: `delete_candidates.select_delete_candidates` とエンジン内の削除フロー。
- **マージ・detach**: 衝突ポリシーに応じ `merge.merge_fields`、`google_calendar.detach_event` 等。
- **実装の中心**: `sync/engine.py`。

### 4.9 Google Calendar API コネクタ

- **認証**: OAuth 2.0（`credentials.json` / `token.json` をランタイムディレクトリに配置、`get_service`）。
- **操作**: **upsert**（作成・更新）、**detach**（拡張プロパティの切り離し）、**delete**、一覧・カレンダー列挙（`connectors/google_calendar.py`）。

### 4.10 設定とマイグレーション

- **スキーマ版**: `models/config_schema.SCHEMA_VERSION`。
- **読込**: `config/migration.migrate_config` — 既知の古い版はフィールド補完で昇格。
- **非互換**: 保存版が **新しすぎる** 場合は `config.json` を `config.backup.v{old}.json` にコピーしたうえで例外（ロールバック用バックアップ）。
- **その他**: `settings_store`、`paths`、`export_import`。

### 4.11 ログ（ファイル + UI、標準 / 詳細）

- **ファイル**: `logging/logger_factory.build_logger` — 日付付きファイルを `logs/` に出力し、`StreamHandler` を併設（`gui/main_window` 等から直接利用）。`file_log_sink` は同一ファクトリに基づくモジュール用ロガーを提供。
- **UI**: `logging/ui_log_sink.UILogSink` — コールバックで MainWindow のログ欄へ出力。
- **詳細度**: 設定 `log_verbosity` は **`standard` / `detailed`**（設定 UI・`Profile`）。

### 4.12 通知

- **実装**: `services/notifications.py`（tkinter `messagebox` による情報表示）。設定の `notification_enabled` と連動。

### 4.13 監査ストア（Audit Store）

- **役割**: 同期結果などのスナップショットを JSON で保存（調査・トレーサビリティ）。
- **実装**: `services/audit_store.save_audit` → ランタイム直下 `audit.json`。

### 4.14 ランタイムデータ格納（`~/.outlook_google_sync/`）

- **代表パス**（`config/paths.py`）:
  - `config.json` — 設定本体
  - `token.json` / `credentials.json` — Google OAuth
  - `logs/` — ファイルログ
  - `audit.json` — 監査出力

### 4.15 フィルタシステム（Ch8）

- **モデル**: `models/profile.FilterConfig`（終日、空き、仮、非公開、カテゴリ include/exclude、件名・場所キーワード）。
- **適用**: `sync/filters.py`（設定に基づき `EventModel` リストを除外）。

### 4.16 カテゴリ色マッピング（OlCategoryColor → Google colorId）

- **役割**: Outlook のカテゴリ色列挙値を Google Calendar の `colorId` 文字列へ固定対応。
- **実装**: `utils/color_mapping.py`（`OL_COLOR_TO_GOOGLE`、`outlook_color_to_google_color_id`）。

---

## 5. 主要データフロー（概要）

1. **読取**: `outlook_com` または `outlook_ics` が期間内アイテムを取得し `EventModel` 化。
2. **フィルタ**: `sync/filters` が `FilterConfig` に従い除外。
3. **プレビュー**: `sync/preview` が件数・代表項目を算出し GUI に表示。
4. **同期**: `SyncEngine.run` が既存 Google イベントを取得 → diff/full で対象決定 → 衝突処理・upsert/detach/delete → `SyncResult`。
5. **永続化**: 設定・フィンガープリント・メタデータを `config.json` 側に更新、`audit_store` 任意保存、ログはファイルと UI の両方へ。

---

## 6. 外部ライブラリと実行環境

- **GUI**: 標準 `tkinter`。
- **Outlook COM**: `pywin32`（Windows）。
- **ICS**: `icalendar`、`python-dateutil`（RRULE 展開）。
- **Google**: Google API クライアント（OAuth、Calendar v3）。

---

## 7. 関連ドキュメント

- ユーザー向け手順・画面: `docs/USER_GUIDE.md`
- 同期フロー詳細: `docs/FLOWS.md`
- 要件の要約: `docs/SPEC.md`
