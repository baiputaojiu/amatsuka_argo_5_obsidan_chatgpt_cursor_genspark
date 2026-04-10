# Outlook→Google カレンダー同期ツール 仕様書（正本）

本書は本ツールの **唯一の正本** であり、単体で読み、実装判断・受け入れ確認ができる。
他文書（README、手順書、図、CHANGELOG）は本書と矛盾してはならない。

正本の原本は `C:/Users/a1050455/.cursor/plans/outlook-google同期gui_98449998.plan.md` に置かれた
改訂仕様書であり、本ファイルはその正式コピーとして `docs/SPEC.md` に配置する。

## 概要

- **入力方法**: COM（Outlook COM）、手動 ICS、マクロ連携（3種）
- **読み取りエンジン**: `outlook_com`、`ics`（2種）
- **同期方式**: 通常同期（差分）、強制フル同期、プレビュー同期（dry-run）
- **衝突検知**: 固定3条件アルゴリズム（tool_marker一致 + updated > last_tool_write_utc + フィールド差分）
- **衝突ポリシー**: overwrite（既定）、detach_new、merge
- **削除**: DEL-01〜05 ルール。本ツール作成イベントのみ。確認UI必須
- **フィルタ**: 終日・Free・Tentative・Private・カテゴリ・件名KW・場所KW
- **カテゴリ色**: OlCategoryColor色コード→Google colorId 固定マッピング
- **GUI**: tkinter + tkcalendar。設定は1つの設定ウィンドウに集約（タブ構成）
- **ワーカースレッド**: 重い処理はメインスレッド外。キャンセル対応あり

## 主要な仕様ポイント

### 実行場所の統一ルール（運用）

- **初回および環境再構築時**: `scripts/setup_env.bat` で **`outlook_google_sync/.venv/` に仮想環境を作成**し依存をインストールする。`credentials.json` の配置や GUI 利用より **先行** する。
- **GUI 起動**: `scripts/run_gui.bat` は **`.venv` が存在すればその Python で起動** する（グローバル Python への誤依存を防ぐ）。
- `scripts/setup_env.bat`、`scripts/run_gui.bat`、`python -m pytest` 等のコマンド実行は、**`outlook_google_sync/` をカレントディレクトリにした Cursor ターミナル（PowerShell）**を正規手順とする。
- 同等の実行環境として Windows PowerShell（単体）も許容する。
- エクスプローラから `.bat` を直接実行する方法は補助的手段であり、標準手順ではない。
- 同期実行・プレビュー・接続テスト・重複修復・設定変更は GUI で行う。
- ドキュメント上でカレントディレクトリ変更が必要な手順には、必ず `cd` コマンドを併記する。

### 一意キー戦略（Ch11）
- `sync_key` = primary_id（EntryID / UID+RECURRENCE-ID）または fallback_id（SHA-256ハッシュ）
- fallback ハッシュ入力: `reader_engine` + UTC開始 + UTC終了 + タイトル
- **禁止**: ファイルパス（FB-BAN-01）、ファイルmtime（FB-BAN-02）、input_method（FB-BAN-03）
- `extendedProperties.private` 必須キー: tool_marker, sync_key, sync_key_kind, reader_engine, input_method, last_tool_write_utc

### 繰り返し予定（Ch12）
- 各occurrence を個別Googleイベントとして同期（シリーズマスタ方式は不使用）
- COM: `IncludeRecurrences=True` + Restrict で展開。繰り返し発生は原則fallbackキー
- ICS: `dateutil.rrule` で RRULE 展開

### 差分同期での Google 側変更検知（Ch13.5 DIFF-GC-01〜03）
- ソース変更なしでも `updated > last_tool_write_utc` なら差分スキップ対象外
- Google 側手編集の検知を省略してはならない

### プレビュー（Ch15）
- スナップショット方式: プレビュー結果をそのまま本番に引き継ぐ（PRE-SNAP-01）
- 古化判定: 5分超で再プレビュー確認（PRE-SNAP-03、固定・変更不可）
- カテゴリ: 新規、更新、削除候補、スキップ、マージ候補、重複候補

### 削除（Ch26 DEL-01〜05）
- DEL-01: tool_marker 一致
- DEL-02: R（同期対象期間）と重なる（**必須**、推奨ではない）
- DEL-03: sync_key が K_now に含まれない
- DEL-04: R外の全体差分で候補決定してはならない
- DEL-05: R外へ移動したイベントは候補外

### 接続テスト（Ch24）
- 疎通と最小読取の確認。同期成功は保証しない（CT-01〜05）
- Google: 2段構成（基本接続 + 対象カレンダー確認）
- カレンダー未選択時: 成功扱い + 「未実施」メッセージ（CT-CAL-01〜03）
- ICS: 11項目の詳細チェック（Ch25）

### 設定保存先（Ch22.3）
- 実行時データ: `~/.outlook_google_sync/`（credentials.json, token.json, config.json, logs/）
- 開発フォルダ（`outlook_google_sync/`）とは明確に分離

### 必須成果物（Ch38 ART-01〜13）
- SPEC.md, USER_GUIDE.md, 構成図, フロー図, ソース, requirements.txt, CHANGELOG.md, テスト, .basサンプル, スクリプト, README

## 詳細

本仕様書の完全版は plan ファイルに記載されている。本ファイルは要約版であり、
判断に迷った場合は plan ファイル本文を参照すること。
変更時は Ch44（同時更新ルール）に従い、本ファイル・USER_GUIDE・図・CHANGELOG を同一変更セットで更新すること。
