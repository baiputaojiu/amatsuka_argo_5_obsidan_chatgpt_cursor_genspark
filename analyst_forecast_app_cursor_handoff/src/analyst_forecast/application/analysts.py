"""分析対象者の alias 管理。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from sqlalchemy import select

from analyst_forecast.application.settings import AppSettings
from analyst_forecast.domain.attribution import normalize_person_name
from analyst_forecast.infrastructure.db.models import AnalystRecord
from analyst_forecast.infrastructure.db.session import create_session_factory


@dataclass(frozen=True, slots=True)
class AliasUpdateResult:
    analyst_id: str
    canonical_name: str
    aliases: tuple[str, ...]


def add_analyst_alias(
    settings: AppSettings,
    *,
    analyst_id: str | None = None,
    canonical_name: str | None = None,
    alias: str,
) -> AliasUpdateResult:
    normalized_alias = normalize_person_name(alias)
    if not normalized_alias:
        raise ValueError("aliasは空にできません")

    session_factory = create_session_factory(settings.database_file)
    with session_factory.begin() as session:
        analyst: AnalystRecord | None = None
        if analyst_id:
            analyst = session.get(AnalystRecord, analyst_id)
        elif canonical_name:
            analyst = session.scalar(
                select(AnalystRecord).where(
                    AnalystRecord.normalized_name == normalize_person_name(canonical_name)
                )
            )
        if analyst is None:
            raise ValueError("分析対象者が見つかりません")

        current = list(analyst.aliases or [])
        existing_normalized = {normalize_person_name(item) for item in current}
        existing_normalized.add(normalize_person_name(analyst.canonical_name))
        if normalized_alias not in existing_normalized:
            current.append(alias.strip())
            analyst.aliases = current
            analyst.aliases_updated_at = datetime.now(UTC)
        return AliasUpdateResult(
            analyst_id=analyst.analyst_id,
            canonical_name=analyst.canonical_name,
            aliases=tuple(analyst.aliases or []),
        )


def list_analyst_aliases(
    settings: AppSettings,
    *,
    analyst_id: str,
) -> AliasUpdateResult:
    session_factory = create_session_factory(settings.database_file)
    with session_factory() as session:
        analyst = session.get(AnalystRecord, analyst_id)
        if analyst is None:
            raise ValueError("分析対象者が見つかりません")
        return AliasUpdateResult(
            analyst_id=analyst.analyst_id,
            canonical_name=analyst.canonical_name,
            aliases=tuple(analyst.aliases or []),
        )
