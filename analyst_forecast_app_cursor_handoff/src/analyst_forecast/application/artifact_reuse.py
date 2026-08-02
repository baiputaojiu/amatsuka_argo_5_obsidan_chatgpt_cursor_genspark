"""Artifact reuse: safely apply preprocessing results to different source occurrences.

When the same raw content (by hash) appears at a different URL/source occurrence,
the preprocessing (P05/P07) can be reused without re-running AI, provided conditions
are met. This module implements the reuse logic and records applicability.
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from analyst_forecast.domain.knowledge_boundary import as_utc, source_knowledge_boundary
from analyst_forecast.infrastructure.db.ids import next_id
from analyst_forecast.infrastructure.db.models import (
    AiArtifactRecord,
    ArtifactApplicabilityRecord,
    RunSourceRecord,
    SourceRecord,
)


class ReuseError(Exception):
    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(message)


def can_reuse_processed_artifact(
    session: Session,
    *,
    original_artifact: AiArtifactRecord,
    target_source: SourceRecord,
    target_run_id: str,
) -> tuple[bool, str]:
    """Check if an artifact can be reused for a different source occurrence.

    Returns (can_reuse, reason).
    """
    if original_artifact.classification != "accepted":
        return False, "original_not_accepted"
    if original_artifact.resolution_status == "superseded":
        return False, "original_superseded"

    original_source = session.get(SourceRecord, original_artifact.source_id)
    if original_source is None:
        return False, "original_source_missing"

    if original_source.raw_hash != target_source.raw_hash:
        return False, "raw_hash_mismatch"

    # analyst must match (via run)
    from analyst_forecast.infrastructure.db.models import RunRecord

    original_run = session.get(RunRecord, original_artifact.run_id)
    target_run = session.get(RunRecord, target_run_id)
    if original_run is None or target_run is None:
        return False, "run_missing"
    if original_run.analyst_id != target_run.analyst_id:
        return False, "analyst_mismatch"

    if original_source.medium != target_source.medium:
        return False, "medium_mismatch"

    # prompt family check
    expected_prompt = "P05" if target_source.medium == "youtube" else "P07"
    if original_artifact.prompt_id != expected_prompt:
        return False, "prompt_mismatch"

    # cutoff boundary check
    if original_artifact.knowledge_cutoff is not None:
        boundary = source_knowledge_boundary(target_source)
        cutoff = original_artifact.knowledge_cutoff
        cutoff = cutoff.replace(tzinfo=UTC) if cutoff.tzinfo is None else as_utc(cutoff)
        if boundary.boundary is not None and cutoff > boundary.boundary:
            return False, "cutoff_exceeds_target_boundary"

    # target already has its own accepted preprocessing
    existing_accepted = session.scalar(
        select(AiArtifactRecord).where(
            AiArtifactRecord.source_id == target_source.source_id,
            AiArtifactRecord.prompt_id == expected_prompt,
            AiArtifactRecord.classification == "accepted",
            AiArtifactRecord.run_id == target_run_id,
        )
    )
    if existing_accepted is not None:
        return False, "target_has_own_accepted"

    return True, "ok"


def reuse_artifact_for_source(
    session: Session,
    *,
    original_artifact_id: str,
    target_run_id: str,
    target_source_id: str,
) -> ArtifactApplicabilityRecord:
    """Create or return existing applicability record for reuse.

    Idempotent: returns existing record if already applied.
    """
    original = session.get(AiArtifactRecord, original_artifact_id)
    if original is None:
        raise ReuseError("artifact_not_found", "Original artifact not found")

    target_source = session.get(SourceRecord, target_source_id)
    if target_source is None:
        raise ReuseError("source_not_found", "Target source not found")

    # Idempotent check
    existing = session.scalar(
        select(ArtifactApplicabilityRecord).where(
            ArtifactApplicabilityRecord.ai_artifact_id == original_artifact_id,
            ArtifactApplicabilityRecord.target_source_id == target_source_id,
        )
    )
    if existing is not None:
        return existing

    can_reuse, reason = can_reuse_processed_artifact(
        session,
        original_artifact=original,
        target_source=target_source,
        target_run_id=target_run_id,
    )
    if not can_reuse:
        raise ReuseError(reason, f"Cannot reuse artifact: {reason}")

    record = ArtifactApplicabilityRecord(
        applicability_id=next_id(session, "APL-", width=6, sequence_key="APPLICABILITY"),
        ai_artifact_id=original_artifact_id,
        target_run_id=target_run_id,
        target_source_id=target_source_id,
        reused_from_artifact_id=original_artifact_id,
        raw_artifact_id=target_source.raw_artifact_id,
        raw_hash=target_source.raw_hash,
        applicability_status="active",
        created_at=datetime.now(UTC),
    )
    session.add(record)

    # Update the run_source link to mark preprocessing as reused/accepted
    link = session.get(RunSourceRecord, {"run_id": target_run_id, "source_id": target_source_id})
    if link is not None:
        link.processing_status = "accepted"
        link.preprocess_status = "preprocess_accepted"
        link.latest_ai_artifact_id = original_artifact_id

    session.flush()
    return record


def is_artifact_applicable_for_source(
    session: Session,
    *,
    artifact_id: str,
    source_id: str,
) -> bool:
    """Check if an artifact is applicable for a given source (original or via reuse)."""
    artifact = session.get(AiArtifactRecord, artifact_id)
    if artifact is None:
        return False
    if artifact.source_id == source_id:
        return True
    exists = session.scalar(
        select(ArtifactApplicabilityRecord).where(
            ArtifactApplicabilityRecord.ai_artifact_id == artifact_id,
            ArtifactApplicabilityRecord.target_source_id == source_id,
            ArtifactApplicabilityRecord.applicability_status == "active",
        )
    )
    return exists is not None
