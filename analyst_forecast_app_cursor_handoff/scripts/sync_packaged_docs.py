#!/usr/bin/env python3
"""Sync repository docs/ into packaged resources/docs/.

Usage:
  python scripts/sync_packaged_docs.py          # copy
  python scripts/sync_packaged_docs.py --check  # exit 1 on drift
"""

from __future__ import annotations

import argparse
import filecmp
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PACKAGED = ROOT / "src" / "analyst_forecast" / "resources" / "docs"

# repo docs path (relative to docs/) -> packaged relative path
SYNC_MAP: list[tuple[str, str]] = [
    ("README.md", "README.md"),
    ("01_スタートアップガイド/STARTUP_GUIDE.md", "01_スタートアップガイド/STARTUP_GUIDE.md"),
    ("02_取扱説明書/USER_MANUAL.md", "02_取扱説明書/USER_MANUAL.md"),
    ("03_仕様書/SYSTEM_SPECIFICATION.md", "03_仕様書/SYSTEM_SPECIFICATION.md"),
    ("03_仕様書/FOLDER_STRUCTURE.md", "03_仕様書/FOLDER_STRUCTURE.md"),
    ("03_仕様書/WORKFLOW_DIAGRAMS.md", "03_仕様書/WORKFLOW_DIAGRAMS.md"),
    ("03_仕様書/DATA_MODEL.md", "03_仕様書/DATA_MODEL.md"),
    ("03_仕様書/METHODOLOGY.md", "03_仕様書/METHODOLOGY.md"),
    ("04_参考資料/PROMPT_CATALOG.md", "04_参考資料/PROMPT_CATALOG.md"),
    ("04_参考資料/PYTHON_TASK_CATALOG.md", "04_参考資料/PYTHON_TASK_CATALOG.md"),
    ("04_参考資料/TROUBLESHOOTING.md", "04_参考資料/TROUBLESHOOTING.md"),
    ("05_計画/DECISION_LOG.md", "05_計画/DECISION_LOG.md"),
    ("05_計画/OPEN_QUESTIONS.md", "05_計画/OPEN_QUESTIONS.md"),
    ("05_計画/FUTURE_ROADMAP.md", "05_計画/FUTURE_ROADMAP.md"),
]


def _pairs() -> list[tuple[Path, Path]]:
    pairs: list[tuple[Path, Path]] = []
    for src_rel, dst_rel in SYNC_MAP:
        src = DOCS / src_rel
        dst = PACKAGED / dst_rel
        if not src.is_file():
            # README may live only in resources; allow packaged-only for AI_WORK_GUIDE
            continue
        pairs.append((src, dst))
    # AI_WORK_GUIDE is packaged-only canonical if missing from docs
    ai_guide_docs = DOCS / "AI_WORK_GUIDE.md"
    ai_guide_pkg = PACKAGED / "AI_WORK_GUIDE.md"
    if ai_guide_docs.is_file():
        pairs.append((ai_guide_docs, ai_guide_pkg))
    return pairs


def sync() -> None:
    for src, dst in _pairs():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        print(f"synced {src.relative_to(ROOT)} -> {dst.relative_to(ROOT)}")


def check() -> int:
    missing: list[str] = []
    drifted: list[str] = []
    for src, dst in _pairs():
        if not dst.is_file():
            missing.append(str(dst.relative_to(ROOT)))
            continue
        if not filecmp.cmp(src, dst, shallow=False):
            drifted.append(str(dst.relative_to(ROOT)))
    required_names = [
        "README.md",
        "STARTUP_GUIDE.md",
        "USER_MANUAL.md",
        "SYSTEM_SPECIFICATION.md",
        "FOLDER_STRUCTURE.md",
        "WORKFLOW_DIAGRAMS.md",
        "DATA_MODEL.md",
        "METHODOLOGY.md",
        "PROMPT_CATALOG.md",
        "PYTHON_TASK_CATALOG.md",
        "TROUBLESHOOTING.md",
        "DECISION_LOG.md",
        "OPEN_QUESTIONS.md",
        "FUTURE_ROADMAP.md",
        "AI_WORK_GUIDE.md",
    ]
    present = {path.name for path in PACKAGED.rglob("*.md")}
    for name in required_names:
        if name not in present:
            missing.append(f"(missing packaged) {name}")
    roadmap = PACKAGED / "05_計画" / "FUTURE_ROADMAP.md"
    if roadmap.is_file() and roadmap.stat().st_size < 2000:
        drifted.append("FUTURE_ROADMAP.md too short (expected detailed roadmap)")
    if missing or drifted:
        for item in missing:
            print(f"MISSING: {item}", file=sys.stderr)
        for item in drifted:
            print(f"DRIFT: {item}", file=sys.stderr)
        return 1
    print("packaged docs sync check: OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        return check()
    sync()
    return check()


if __name__ == "__main__":
    raise SystemExit(main())
