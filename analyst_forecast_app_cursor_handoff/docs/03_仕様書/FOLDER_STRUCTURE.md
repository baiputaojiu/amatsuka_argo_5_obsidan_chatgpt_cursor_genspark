# フォルダ構造

## 1. ルート

```text
30_Permanent/
└─ ★アナリスト調査/
   ├─ README.md
   ├─ AI_WORK_GUIDE.md
   ├─ _system/
   ├─ docs/
   ├─ prompts/
   └─ analysts/
```

## 2. システム領域

```text
_system/
├─ config.yaml
├─ database.sqlite
├─ analysts.yaml
├─ forecast_targets.yaml
├─ target_mappings.yaml
├─ prompt_versions.yaml
├─ task_catalog.yaml
├─ market_cache/
└─ backups/
   ├─ database/
   ├─ configuration/
   └─ mappings/
```

- `database.sqlite`：機械処理上の正本。
- `task_catalog.yaml`：処理依存関係と担当。
- `market_cache`：取得済み市場データ。
- `backups`：DB、設定、マッピングの自動バックアップ。

## 3. 共通文書

```text
docs/
├─ 01_スタートアップガイド/
├─ 02_取扱説明書/
├─ 03_仕様書/
├─ 04_参考資料/
└─ 05_計画/
```

## 4. 共通プロンプト

```text
prompts/
├─ source_discovery/
├─ youtube_processing/
├─ forecast_extraction/
├─ target_resolution/
├─ validation/
├─ analysis/
└─ reporting/
```

中央フォルダは最新版の正本。案件作成時に、分析対象者、期間、入出力先等を埋めたスナップショットを案件へ生成する。

## 5. 対象者・案件

```text
analysts/
└─ analyst-name__A0001/
   ├─ analyst_profile.md
   └─ RUN-20260720-001__20260120_20260720/
      ├─ request.yaml
      ├─ status.yaml
      ├─ WORKFLOW_STATE.json
      ├─ NEXT_ACTIONS.md
      ├─ OPEN_ISSUES.md
      ├─ README.md
      ├─ 01_prompts/
      ├─ 02_sources/
      ├─ 03_ai_outputs/
      ├─ 04_results/
      └─ 05_audit/
```

## 6. 原文

```text
02_sources/
├─ youtube/
│  ├─ raw/
│  ├─ processed/
│  └─ metadata/
├─ blog/
│  ├─ raw/
│  ├─ processed/
│  └─ metadata/
├─ x/
│  ├─ raw/
│  ├─ processed/
│  └─ metadata/
└─ web/
   ├─ raw/
   ├─ processed/
   └─ metadata/
```

- `raw`：変更禁止。
- `processed`：話者、段落、句読点等を付与。
- `metadata`：URL、日時、媒体、取得情報、証拠レベル。

## 7. AI出力

```text
03_ai_outputs/
├─ inbox/
├─ accepted/
├─ needs_review/
└─ rejected/
```

- 最初は `inbox`。
- Python検証後に分類。
- `accepted`だけDBへ取込み。
- 却下・要確認も削除せず理由を保存。

## 8. 結果

```text
04_results/
├─ forecasts/
├─ evaluations/
├─ target_mappings/
├─ tables/
├─ charts/
└─ reports/
```

過去実績と現在予想を別の正本にしない。`all_forecasts`から状態で表示を分ける。

## 9. 監査

```text
05_audit/
├─ search_logs/
├─ processing_logs/
├─ evaluation_snapshots/
└─ errors/
```

## 10. ファイル名

長い日本語タイトルをファイル名にせず、短い固定IDと日付を使う。

```text
SRC-000001__2026-07-01__youtube.txt
forecast_extraction_BATCH-0001.json
target_resolution_BATCH-0001.json
```

Windowsの使用禁止文字、パス長、同姓同名を避ける。完全なタイトルはメタデータへ保存する。

