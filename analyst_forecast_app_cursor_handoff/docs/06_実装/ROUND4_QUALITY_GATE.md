# Round4 品質ゲート

実施日: 2026-07-21

## コマンド結果（R4-047）

実行環境: `.\.venv\Scripts\python.exe`

| コマンド | 結果 |
|---------|------|
| `python -m ruff check src tests` | pass（All checks passed） |
| `python -m mypy src/analyst_forecast --ignore-missing-imports` | pass（52 files） |
| `python -m pytest -q` | **138 passed**, 3 deselected |
| `python scripts/sync_packaged_docs.py --check` | pass（packaged docs sync check: OK） |
| `python -m alembic check` | pass（No new upgrade operations detected） |

## Migration

- `0008_round4_lifecycle_and_reuse.py` — lifecycle, applicability, run_source axes
- `0009_round4_metadata_alignment.py` — nullable made_at, NOT NULL lifecycle/generation
- Active集計は `lifecycle_status=active` かつ `made_at IS NOT NULL`（R4-036）

## 主要テスト（gap埋め）

| ID | テスト |
|----|--------|
| R4-009 | `test_r4_009_summary_forecast_count_active_only` |
| R4-016 | `test_r4_016_model_and_prompt_version_mismatch` |
| R4-019 | `test_r4_019_pending_p08_after_reuse_matches_applicability` |
| R4-036 | `test_r4_036_null_made_at_not_in_active_aggregation` |
| R4-042 | `test_basket_cache_does_not_pollute_single_symbol`（docstring） |
| R4-044 | `test_r4_044_upgrade_preserves_rows_fk_and_metadata` |
| R4-045 | `test_r4_045_sync_packaged_docs_check` |
| R4-046 | `test_r4_046_wheel_help` |

## 判定

受入マトリクス全48項目 PASS → `READY_FOR_REAL_SAMPLE_DIRECTIONAL_SLICE`（詳細は `FINAL_REVIEW_ROUND4.md`）
