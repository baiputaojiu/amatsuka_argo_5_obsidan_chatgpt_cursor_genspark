"""Shared query helpers for active-only forecasts (Round4).

All results, workflow, evaluation, and NEXT_ACTIONS must use these helpers
to exclude superseded/rejected/unresolved issuances from active aggregation.
Legacy rows with made_at IS NULL are never treated as active formal forecasts.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import Select, select
from sqlalchemy.orm import Session

from analyst_forecast.infrastructure.db.models import (
    ForecastComponentRecord,
    ForecastIssuanceRecord,
)

ACTIVE_LIFECYCLE_STATUSES = {"active"}
EXCLUDED_LIFECYCLE_STATUSES = {
    "superseded",
    "rejected_or_withdrawn",
    "review_unresolved_excluded",
    "legacy_conflict",
    "legacy_time_unverified",
}


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
    rows = list(
        session.scalars(
            select(ForecastIssuanceRecord.forecast_issuance_id).where(
                ForecastIssuanceRecord.forecast_issuance_id.in_(issuance_ids),
                ForecastIssuanceRecord.lifecycle_status == "active",
                ForecastIssuanceRecord.made_at.is_not(None),
            )
        )
    )
    return rows


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
