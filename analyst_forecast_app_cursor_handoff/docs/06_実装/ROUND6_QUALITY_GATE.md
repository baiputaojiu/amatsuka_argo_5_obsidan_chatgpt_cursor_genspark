# Round6 Quality Gate

実行日: 2026-07-22  
Python: `.\.venv\Scripts\python.exe`  
cwd: `analyst_forecast_app_cursor_handoff/`  
base HEAD: `88864c289750f8323c27b6e3f2c09fd70a79923d`  
commit/push: **未実施**

## Commands and results

| Gate | Command | Result |
|------|---------|--------|
| ruff format --check | `python -m ruff format --check src tests` | **pass** (rc 0) |
| ruff check | `python -m ruff check src tests` | **pass** (rc 0) |
| mypy | `python -m mypy src/analyst_forecast --ignore-missing-imports` | **pass** (rc 0) |
| docs sync | `python scripts/sync_packaged_docs.py --check` | **pass** — packaged docs sync check: OK |
| alembic check | `python -m alembic check` | **pass** — No new upgrade operations detected |
| build wheel | `python -m build --wheel`（本セッションで成功） | **pass** |
| pytest full | `python -m pytest -q --tb=line` | **pass** — `212 passed, 3 deselected in 1232.47s` |
| pytest round6 | operations/coverage/migration/schema/wheel | **pass**（wheel単独も pass） |
| git diff --check | `git diff --check 88864c2 -- analyst_forecast_app_cursor_handoff` | **pass**（CRLF warningのみ） |

## skip / xfail / deselected

- Round6 required tests: **skip/xfail/importorskip なし**
- live network integration: 既存 `@pytest.mark.skipif` + deselected 3件（許容）

## Honesty notes

- full pytest は Schema `false`→`null` 許容修正後に **212 passed** を再確認。
- wheel 縦断は `test_r6_045_048_wheel_formal_vertical`（site-packages、P09 ops、inactive gate、coverage builders、0007→head）。
- P09 fixed Schema の allOf は `model_json_schema()` と完全一致しないため、dual-validator matrix で契約を担保。
