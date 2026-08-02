"""発言時点境界と knowledge_cutoff の共通検証。"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

from analyst_forecast.application.ai_ingestion import ValidationIssue

TimeBasis = Literal[
    "recorded_at",
    "published_at",
    "made_at",
    "publicly_available_at",
    "unknown",
]


@dataclass(frozen=True)
class KnowledgeBoundary:
    boundary: datetime | None
    basis: TimeBasis
    note: str


def as_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("日時にはタイムゾーンが必要です")
    return value.astimezone(UTC)


def source_knowledge_boundary(source: Any) -> KnowledgeBoundary:
    """SOURCEごとの許容 knowledge 上限。"""
    medium = getattr(source, "medium", None)
    recorded_at = getattr(source, "recorded_at", None)
    published_at = getattr(source, "published_at", None)

    if medium == "youtube":
        if recorded_at is not None:
            return KnowledgeBoundary(
                boundary=as_utc(recorded_at),
                basis="recorded_at",
                note="YouTubeは収録時刻を境界とする",
            )
        if published_at is not None:
            return KnowledgeBoundary(
                boundary=as_utc(published_at),
                basis="published_at",
                note="recorded_at不明のためpublished_atを代替境界とする",
            )
        return KnowledgeBoundary(
            boundary=None,
            basis="unknown",
            note="YouTubeのrecorded_at/published_atが未設定",
        )

    if published_at is not None:
        return KnowledgeBoundary(
            boundary=as_utc(published_at),
            basis="published_at",
            note="本人執筆または公開記事の公開時刻を境界とする",
        )
    if recorded_at is not None:
        return KnowledgeBoundary(
            boundary=as_utc(recorded_at),
            basis="recorded_at",
            note="published_at不明のためrecorded_atを境界とする",
        )
    return KnowledgeBoundary(
        boundary=None,
        basis="unknown",
        note="公開・収録時刻が未設定",
    )


def validate_knowledge_cutoff(
    knowledge_cutoff: datetime | None,
    boundary: KnowledgeBoundary,
    *,
    path: str = "$['knowledge_cutoff']",
    code: str = "future_knowledge_cutoff",
) -> list[ValidationIssue]:
    if knowledge_cutoff is None:
        return [
            ValidationIssue(
                "missing_knowledge_cutoff",
                "knowledge_cutoffが必要です。",
                path,
            )
        ]
    if boundary.boundary is None:
        return [
            ValidationIssue(
                "unknown_source_time_boundary",
                "SOURCEの発言・公開時刻が未解決のためknowledge_cutoffを検証できません。",
                path,
            )
        ]
    if as_utc(knowledge_cutoff) > boundary.boundary:
        return [
            ValidationIssue(
                code,
                (f"knowledge_cutoffが許容境界({boundary.basis})より後です。 {boundary.note}"),
                path,
            )
        ]
    return []


def validate_cutoff_against_made_at(
    knowledge_cutoff: datetime,
    made_at: datetime,
    *,
    path: str = "$['knowledge_cutoff']",
    code: str = "future_knowledge_cutoff",
) -> list[ValidationIssue]:
    if as_utc(knowledge_cutoff) > as_utc(made_at):
        return [
            ValidationIssue(
                code,
                "knowledge_cutoffが発言日時より後です。",
                path,
            )
        ]
    return []
