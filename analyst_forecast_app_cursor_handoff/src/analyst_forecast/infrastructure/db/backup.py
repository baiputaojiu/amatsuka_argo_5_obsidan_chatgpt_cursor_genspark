from __future__ import annotations

import re
import sqlite3
from datetime import UTC, datetime
from pathlib import Path


def backup_database(
    database_path: Path,
    *,
    backup_dir: Path,
    reason: str,
) -> Path | None:
    if not database_path.is_file() or database_path.stat().st_size == 0:
        return None
    backup_dir.mkdir(parents=True, exist_ok=True)
    safe_reason = re.sub(r"[^A-Za-z0-9_-]", "_", reason)[:40] or "update"
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    destination = backup_dir / f"database__before_{safe_reason}__{stamp}.sqlite"
    with (
        sqlite3.connect(database_path) as source,
        sqlite3.connect(destination) as target,
    ):
        source.backup(target)
    return destination
