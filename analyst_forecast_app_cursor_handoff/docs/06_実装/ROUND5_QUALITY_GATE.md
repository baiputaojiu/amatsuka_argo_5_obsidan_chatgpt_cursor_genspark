# Round5 Quality Gate

実行日: 2026-07-21（GAP閉鎖セッションで再実行）  
Python: `.\.venv\Scripts\python.exe`

## Commands and results

| Gate | Command | Result |
|------|---------|--------|
| pytest full | `python -m pytest -q --tb=line` | **pass** — `181 passed, 3 deselected in 668.35s` |
| pytest round5 | `pytest tests/unit/test_round5_*.py` | **pass** — `43 passed` |
| ruff check | `python -m ruff check src tests` | **pass** |
| ruff format | `python -m ruff format --check src tests` | **pass**（`test_round5_gaps.py` を再フォーマット後） |
| mypy | `python -m mypy src/analyst_forecast --ignore-missing-imports` | **pass** |
| docs sync | `python scripts/sync_packaged_docs.py --check` | **pass** |
| alembic upgrade | `python -m alembic upgrade head` | **pass**（前回ゲート維持） |
| alembic check | `python -m alembic check` | **pass** |
| wheel | `pytest tests/unit/test_round4_wheel.py -q` | **pass** — xfail 除去済み、失敗時は fail |
| git diff --check | `git diff --check` | **pass** (LF/CRLF warnings only) |

## Round5 pytest subset (this session)

```text
tests/unit/test_round5_gaps.py: 10 passed
round5 modules (critical/functional/migration/acceptance_extra/gaps): 43 passed
full suite: 181 passed, 3 deselected
```

## Honesty notes (Fix 08)

- Full pytest was re-run after closing R5-012〜052 GAP.
- R5-049 evidence remains Round4 wheel vertical (`test_r4_046_wheel_help`) on the current tree.
- R5-050: `pytest.xfail` on wheel build failure **removed**; round5 modules audited for skip/xfail markers.
- R5-052: `CHAT_HISTORY.pdf` / `.gitignore` / `git ls-files` hygiene covered by `test_r5_052_*`.
- Matrix all PASS → `READY_FOR_REAL_SAMPLE_DIRECTIONAL_SLICE`（`FINAL_REVIEW_ROUND5.md`）。

## Migration paths verified by test

- data-bearing 0007 → head (`0010`) with FK check / integrity / active lineage index
- 0009 → head + head→head idempotent
- 0009 multi-active lineage seed → `legacy_conflict` on upgrade (R5-007)
- forced Alembic failure → backup restore (hash + revision + no partial columns)
- active lineage partial unique index rejects duplicate insert
- empty / 0001 / 0005 → head (Round3 / compatibility tests, still in full suite)
- P09 correct mid-materialize forced failure → atomic rollback (R5-012)
