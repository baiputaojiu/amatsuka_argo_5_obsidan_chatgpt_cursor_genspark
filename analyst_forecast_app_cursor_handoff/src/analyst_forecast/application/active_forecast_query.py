"""Shared query helpers for active-only forecasts (Round4/Round5).

All results, workflow, evaluation, and NEXT_ACTIONS must use these helpers
to exclude superseded/rejected/unresolved issuances from active aggregation.
Legacy rows with made_at IS NULL are never treated as active formal forecasts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session

from analyst_forecast.infrastructure.db.models import (
    AiArtifactRecord,
    ForecastComponentRecord,
    ForecastIssuanceRecord,
)

ACTIVE_LIFECYCLE_STATUSES = {"active"}
EXCLUDED_LIFECYCLE_STATUSES = {
    "superseded",
    "rejected_or_withdrawn",
    "withdrawn_by_correction",
    "review_unresolved_excluded",
    "legacy_conflict",
    "legacy_time_unverified",
}


@dataclass(frozen=True)
class ActiveComponentContext:
    component: ForecastComponentRecord
    issuance: ForecastIssuanceRecord
    artifact: AiArtifactRecord | None


@dataclass(frozen=True)
class InactiveComponentError:
    code: str
    message: str
    lifecycle_status: str | None = None


def active_issuances_query(*, run_filter: Select[Any] | None = None) -> Select[Any]:
    """Base query returning only active lifecycle issuances with verified made_at."""
    q = select(ForecastIssuanceRecord).where(
        ForecastIssuanceRecord.lifecycle_status.in_(ACTIVE_LIFECYCLE_STATUSES),
        ForecastIssuanceRecord.made_at.is_not(None),
    )
    if run_filter is not None:
        q = q.where(ForecastIssuanceRecord.forecast_issuance_id.in_(run_filter))
    return q


def active_issuance_ids(session: Session, issuance_ids: list[str]) -> list[str]:
    """Filter a list of issuance IDs to only those with active lifecycle and made_at."""
    if not issuance_ids:
        return []
    return list(
        session.scalars(
            select(ForecastIssuanceRecord.forecast_issuance_id).where(
                ForecastIssuanceRecord.forecast_issuance_id.in_(issuance_ids),
                ForecastIssuanceRecord.lifecycle_status == "active",
                ForecastIssuanceRecord.made_at.is_not(None),
            )
        )
    )


def active_components(session: Session, issuance_ids: list[str]) -> list[ForecastComponentRecord]:
    """Return components belonging to active issuances only."""
    active_ids = active_issuance_ids(session, issuance_ids)
    if not active_ids:
        return []
    return list(
        session.scalars(
            select(ForecastComponentRecord).where(
                ForecastComponentRecord.forecast_issuance_id.in_(active_ids)
            )
        )
    )


def is_active_issuance(session: Session, issuance_id: str) -> bool:
    """Check if a specific issuance is active and has a verified made_at."""
    iss = session.get(ForecastIssuanceRecord, issuance_id)
    if iss is None:
        return False
    return iss.lifecycle_status == "active" and iss.made_at is not None


def is_active_component(session: Session, component_id: str) -> bool:
    """Check if a component belongs to an active issuance."""
    comp = session.get(ForecastComponentRecord, component_id)
    if comp is None:
        return False
    return is_active_issuance(session, comp.forecast_issuance_id)


def require_active_component_context(
    session: Session,
    component_id: str,
    *,
    run_id: str | None = None,
    source_id: str | None = None,
    analyst_id: str | None = None,
) -> ActiveComponentContext | InactiveComponentError:
    """Gate for P11/P12/P13/evaluation: only active formal components pass."""
    component = session.get(ForecastComponentRecord, component_id)
    if component is None:
        return InactiveComponentError(
            code="inactive_forecast_component",
            message=f"構成予想が存在しません: {component_id}",
        )
    issuance = session.get(ForecastIssuanceRecord, component.forecast_issuance_id)
    if issuance is None:
        return InactiveComponentError(
            code="inactive_forecast_component",
            message=f"親issuanceが存在しません: {component.forecast_issuance_id}",
        )
    status = issuance.lifecycle_status or "unknown"
    if status != "active":
        return InactiveComponentError(
            code="inactive_forecast_component",
            message=(f"issuance lifecycle_status={status} のため対象解決/評価できません"),
            lifecycle_status=status,
        )
    if issuance.made_at is None:
        return InactiveComponentError(
            code="inactive_forecast_component",
            message="made_at未確定のissuanceは対象解決/評価できません",
            lifecycle_status=status,
        )
    root = issuance.lineage_root_id or issuance.forecast_issuance_id
    active_count = session.scalar(
        select(func.count())
        .select_from(ForecastIssuanceRecord)
        .where(
            (
                (ForecastIssuanceRecord.lineage_root_id == root)
                | (
                    (ForecastIssuanceRecord.lineage_root_id.is_(None))
                    & (ForecastIssuanceRecord.forecast_issuance_id == root)
                )
            ),
            ForecastIssuanceRecord.lifecycle_status == "active",
            ForecastIssuanceRecord.made_at.is_not(None),
        )
    )
    if active_count is not None and int(active_count) > 1:
        return InactiveComponentError(
            code="inactive_forecast_component",
            message="同一lineageにactive issuanceが複数ありlegacy_conflictです",
            lifecycle_status="legacy_conflict",
        )
    artifact = None
    if issuance.ai_artifact_id:
        artifact = session.get(AiArtifactRecord, issuance.ai_artifact_id)
    if run_id is not None and artifact is not None and artifact.run_id != run_id:
        return InactiveComponentError(
            code="inactive_forecast_component",
            message="componentのrun_idが要求と一致しません",
            lifecycle_status=status,
        )
    if source_id is not None and issuance.source_id != source_id:
        return InactiveComponentError(
            code="inactive_forecast_component",
            message="componentのsource_idが要求と一致しません",
            lifecycle_status=status,
        )
    if analyst_id is not None and issuance.analyst_id != analyst_id:
        return InactiveComponentError(
            code="inactive_forecast_component",
            message="componentのanalyst_idが要求と一致しません",
            lifecycle_status=status,
        )
    return ActiveComponentContext(component=component, issuance=issuance, artifact=artifact)
