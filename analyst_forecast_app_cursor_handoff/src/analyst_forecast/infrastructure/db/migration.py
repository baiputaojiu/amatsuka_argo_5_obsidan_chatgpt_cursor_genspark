"""Round5: atomic upgrade with backup restore; FK-safe migration connection.

Revision note: application-level wrapper. Schema changes live in 0009 fix / 0010+.
"""

from __future__ import annotations

import hashlib
import shutil
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from alembic import command
from alembic.config import Config
from sqlalchemy import text

from analyst_forecast.infrastructure.db.session import create_sqlite_engine


class MigrationError(RuntimeError):
    """Raised when upgrade fails after attempted restore."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def upgrade_database(
    database_path: Path,
    backup_dir: Path | None = None,
    *,
    revision: str = "head",
) -> Path | None:
    """Upgrade Alembic revisions with pre-upgrade backup and restore-on-failure.

    SQLite table rebuilds (batch_alter) require foreign_keys=OFF outside the
    migration transaction. This wrapper sets that pragma on the connection
    before Alembic runs, then re-enables and runs foreign_key_check.
    """
    backup_path: Path | None = None
    pre_hash: str | None = None
    pre_version: str | None = None
    if database_path.is_file() and database_path.stat().st_size > 0:
        pre_hash = _file_sha256(database_path)
        with sqlite3.connect(database_path) as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='alembic_version'"
            ).fetchone()
            if row is not None:
                ver = conn.execute("SELECT version_num FROM alembic_version").fetchone()
                pre_version = ver[0] if ver else None
        if backup_dir is not None:
            backup_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
            backup_path = backup_dir / f"database__before_migration__{stamp}.sqlite"
            shutil.copy2(database_path, backup_path)

    migrations = Path(__file__).with_name("migrations")
    config = Config()
    config.set_main_option("script_location", str(migrations))
    engine = create_sqlite_engine(database_path)
    try:
        with engine.connect() as connection:
            # PRAGMA foreign_keys cannot be changed inside a multi-statement
            # transaction; commit so OFF takes effect for subsequent work.
            connection.execute(text("PRAGMA foreign_keys=OFF"))
            connection.commit()
            with connection.begin():
                config.attributes["connection"] = connection
                command.upgrade(config, revision)
            connection.execute(text("PRAGMA foreign_keys=ON"))
            connection.commit()
            violations = connection.execute(text("PRAGMA foreign_key_check")).fetchall()
            if violations:
                raise MigrationError(f"foreign_key_check failed after upgrade: {violations[:5]}")
    except Exception as exc:
        engine.dispose()
        if backup_path is not None and backup_path.is_file():
            shutil.copy2(backup_path, database_path)
            if pre_hash is not None and _file_sha256(database_path) != pre_hash:
                raise MigrationError("migration failed and backup restore hash mismatch") from exc
            if pre_version is not None:
                with sqlite3.connect(database_path) as conn:
                    ver = conn.execute("SELECT version_num FROM alembic_version").fetchone()
                    restored = ver[0] if ver else None
                    if restored != pre_version:
                        raise MigrationError(
                            f"migration failed; restored version {restored!r} "
                            f"!= expected {pre_version!r}"
                        ) from exc
        raise MigrationError(f"migration failed: {exc}") from exc
    finally:
        engine.dispose()
    return backup_path
