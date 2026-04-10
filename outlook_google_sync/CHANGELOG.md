# Changelog

## 0.3.2 - 2026-03-31

- **ドキュメント**: `docs/USER_GUIDE.md` に **手順の要否一覧**（必要な場合／不要な場合）と **§1 と §2 の関係** を追記。初回セットアップ表に要否の短い注記を付与。
- **ドキュメント**: 各章・主要小見出しの先頭に `*` 形式の **要否（必要／不要）** を追記。

## 0.3.1 - 2026-03-31

- **ドキュメント**: 手順書・README・SPEC・FLOWS で **仮想環境（`.venv`）構築を先に行う** 旨を統一。USER_GUIDE の章番号を振り直し。
- **`scripts/run_gui.bat`**: `.venv\Scripts\python.exe` があればそれで起動し、グローバル Python への誤依存を防止。

## 0.3.0 - 2026-03-30

- **変更概要**: 仕様書再突合による追加修正 + README 充実
  - Ch13.2: sync_metadata 全フィールド保存（last_success_utc, last_run_mode, last_D_start, last_D_end, last_conditions_hash）
  - Ch6.2: 設定ウィンドウに「プロファイル」タブ追加（エクスポート/インポート導線）
  - Ch26 DEL-02: SyncEngine の削除候補抽出に range_start/range_end を正しく渡すよう修正
  - Ch31: 監査情報の履歴保存（直近50件）と閲覧機能（サマリーダイアログから「監査履歴」ボタン）
  - Ch20.4 TTL-PRIV-03: 設定ウィンドウ同期タブに title_only 切替注意文を追加
  - プレビュー → 本番実行時の range_start/range_end 引き渡し修正
  - samples/ics/ にサンプル ICS ファイル追加（匿名・架空データ）
  - README を大幅充実（フォルダ構成・入力方法・同期モード・衝突ポリシー・フィルタ・接続テスト・削除仕様・トラブルシューティング等）
- **互換性影響**: config.json schema_version=1 を維持。後方互換。

## 0.2.0 - 2026-03-30

- **変更概要**: 仕様書との突合に基づく大規模な抜け漏れ補完
  - 全サブパッケージに `__init__.py` 追加
  - Ch8: 除外フィルタ実装（終日/Free/Tentative/Private/カテゴリ/件名KW/場所KW）
  - Ch12: 繰り返し予定の展開（COM `IncludeRecurrences`、ICS RRULE via dateutil）
  - Ch15: プレビュー拡充（削除候補/マージ候補/重複候補カテゴリ、スナップショット引継ぎ、5分古化チェック）
  - Ch17: 衝突検知の固定3条件アルゴリズム + overwrite/detach_new/merge ポリシーをエンジンに統合
  - Ch19: OlCategoryColor → Google colorId 固定マッピング実装
  - Ch20: title_only モードのテスト追加（description/location 保持確認）
  - Ch24-25: 接続テスト強化（ICS 11項目チェック、COM テスト、Google 2段構成）
  - Ch26: 削除実行フロー + 確認UI（個別/一括許可 DeleteConfirmDialog）
  - Ch28: サマリーダイアログ + 再試行ダイアログ
  - Ch32: 設定エクスポート/インポート
  - EventModel に sync_key_kind, is_all_day, busy_status, categories, ol_category_color を追加
  - SyncResult に merged, delete_candidates_list を追加
  - Profile/FilterConfig モデルを仕様準拠に拡充
  - 設定ウィンドウを全タブ構成に改修（基本/同期/フィルタ/Google/ログ通知/高度な設定）
  - SyncEngine 全面改修（衝突検知統合、削除実行、キャンセル対応、エラーコールバック）
  - MainWindow 改修（フィルタ統合、COM↔ICS切替警告、削除確認UI連携、プレビュー→本番引継ぎ）
  - DuplicateRepairWindow 拡充（残す選択、古い方一括保持）
  - Google Calendar コネクタに detach_event, delete_event, last_tool_write_utc スタンプ追加
  - テスト 41件（unit: 39 + integration: 2）— detach_new, title_only, フィルタ, 色マッピング等を追加
  - docs/SPEC.md 正本化、SYSTEM_ARCHITECTURE.md / FLOWS.md / USER_GUIDE.md 充実
- **互換性影響**: config.json schema_version=1 を維持。後方互換。

## 0.1.0 - 2026-03-30

- **変更概要**: 初期実装（スキャフォールド + 基本機能）
- **互換性影響**: 初版
