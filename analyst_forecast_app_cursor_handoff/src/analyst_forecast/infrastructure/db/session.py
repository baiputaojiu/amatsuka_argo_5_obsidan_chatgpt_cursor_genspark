from __future__ import annotations

from pathlib import Path

from sqlalchemy import Engine, event
from sqlalchemy.engine import create_engine
from sqlalchemy.orm import Session, sessionmaker

type SessionFactory = sessionmaker[Session]


def create_sqlite_engine(database_path: Path) -> Engine:
    database_path.parent.mkdir(parents=True, exist_ok=True)
    url = f"sqlite+pysqlite:///{database_path.resolve().as_posix()}"
    engine = create_engine(url, future=True)

    @event.listens_for(engine, "connect")
    def configure_sqlite(dbapi_connection: object, connection_record: object) -> None:
        del connection_record
        cursor = dbapi_connection.cursor()  # type: ignore[attr-defined]
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()

    return engine


def create_session_factory(database_path: Path) -> SessionFactory:
    return sessionmaker(
        bind=create_sqlite_engine(database_path),
        class_=Session,
        expire_on_commit=False,
    )
