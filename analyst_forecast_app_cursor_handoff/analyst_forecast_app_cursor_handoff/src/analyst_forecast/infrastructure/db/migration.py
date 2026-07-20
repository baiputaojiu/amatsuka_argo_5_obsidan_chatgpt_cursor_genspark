from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

from alembic import command
from alembic.config import Config

from analyst_forecast.infrastructure.db.session import create_sqlite_engine


def upgrade_database(database_path: Path, backup_dir: Path | None = None) -> Path | None:
    backup_path: Path | None = None
    if database_path.is_file() and database_path.stat().st_size > 0 and backup_dir is not None:
        backup_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        backup_path = backup_dir / f"database__before_migration__{stamp}.sqlite"
        shutil.copy2(database_path, backup_path)

    migrations = Path(__file__).with_name("migrations")
    config = Config()
    config.set_main_option("script_location", str(migrations))
    engine = create_sqlite_engine(database_path)
    with engine.begin() as connection:
        config.attributes["connection"] = connection
        command.upgrade(config, "head")
    engine.dispose()
    return backup_path
