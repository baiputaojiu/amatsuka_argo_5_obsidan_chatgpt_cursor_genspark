from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    JSON,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utc_now() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class AuditMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now
    )
    version: Mapped[int] = mapped_column(Integer, default=1)


class IdSequenceRecord(Base):
    __tablename__ = "id_sequences"

    sequence_key: Mapped[str] = mapped_column(String(32), primary_key=True)
    current_value: Mapped[int] = mapped_column(Integer, nullable=False)


class AnalystRecord(AuditMixin, Base):
    __tablename__ = "analysts"

    analyst_id: Mapped[str] = mapped_column(String(16), primary_key=True)
    canonical_name: Mapped[str] = mapped_column(String(200), nullable=False)
    normalized_name: Mapped[str] = mapped_column(String(200), nullable=False, unique=True)
    aliases: Mapped[list[str]] = mapped_column(JSON, default=list)
    aliases_updated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    affiliation: Mapped[str | None] = mapped_column(String(200))
    specialties: Mapped[list[str]] = mapped_column(JSON, default=list)
    official_youtube: Mapped[str | None] = mapped_column(Text)
    official_blog: Mapped[str | None] = mapped_column(Text)
    official_x: Mapped[str | None] = mapped_column(Text)
    profile_notes: Mapped[str | None] = mapped_column(Text)


class RunRecord(AuditMixin, Base):
    __tablename__ = "runs"

    run_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    analyst_id: Mapped[str] = mapped_column(ForeignKey("analysts.analyst_id"), index=True)
    period_start: Mapped[date] = mapped_column(Date, nullable=False)
    period_end: Mapped[date] = mapped_column(Date, nullable=False)
    evaluation_as_of: Mapped[date] = mapped_column(Date, nullable=False)
    selected_media: Mapped[list[str]] = mapped_column(JSON, default=list)
    focus_targets: Mapped[list[str]] = mapped_column(JSON, default=list)
    ai_environment: Mapped[list[str]] = mapped_column(JSON, default=list)
    model_configuration: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(40), default="not_started", index=True)
    run_path: Mapped[str] = mapped_column(Text, nullable=False, unique=True)


class RawArtifactRecord(AuditMixin, Base):
    __tablename__ = "raw_artifacts"

    raw_artifact_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    canonical_path: Mapped[str] = mapped_column(Text, nullable=False)
    byte_size: Mapped[int] = mapped_column(Integer, nullable=False)
    encoding: Mapped[str] = mapped_column(String(40), default="utf-8")
    first_seen_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class SourceRecord(AuditMixin, Base):
    __tablename__ = "sources"
    __table_args__ = (
        Index(
            "ix_sources_artifact_analyst_medium",
            "raw_artifact_id",
            "analyst_id",
            "medium",
        ),
    )

    source_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    analyst_id: Mapped[str] = mapped_column(ForeignKey("analysts.analyst_id"), index=True)
    medium: Mapped[str] = mapped_column(String(20), index=True)
    url: Mapped[str | None] = mapped_column(Text)
    external_source_id: Mapped[str | None] = mapped_column(String(300))
    title: Mapped[str | None] = mapped_column(Text)
    publisher_or_channel: Mapped[str | None] = mapped_column(String(300))
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    recorded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    evidence_level: Mapped[str | None] = mapped_column(String(8))
    raw_artifact_id: Mapped[str | None] = mapped_column(String(20), index=True)
    raw_file_path: Mapped[str] = mapped_column(Text, nullable=False)
    raw_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    acquisition_status: Mapped[str] = mapped_column(String(40), default="acquired")
    source_relation: Mapped[str] = mapped_column(String(40), default="original")
    original_source_id: Mapped[str | None] = mapped_column(
        ForeignKey("sources.source_id"), nullable=True
    )


class RunSourceRecord(Base):
    __tablename__ = "run_sources"

    run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id"), primary_key=True)
    source_id: Mapped[str] = mapped_column(ForeignKey("sources.source_id"), primary_key=True)
    observed_url: Mapped[str | None] = mapped_column(Text)
    observed_medium: Mapped[str] = mapped_column(String(20), nullable=False)
    observed_published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    linked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    processing_status: Mapped[str] = mapped_column(
        String(40), default="raw_imported", nullable=False, index=True
    )
    latest_ai_artifact_id: Mapped[str | None] = mapped_column(
        ForeignKey("ai_artifacts.ai_artifact_id")
    )
    local_input_path: Mapped[str | None] = mapped_column(Text)
    input_kind: Mapped[str] = mapped_column(String(20), default="copy")
    artifact_manifest_path: Mapped[str | None] = mapped_column(Text)
    # Round4 separate state axes
    preprocess_status: Mapped[str | None] = mapped_column(String(40), nullable=True)
    p08_review_status: Mapped[str | None] = mapped_column(String(40), nullable=True)
    p09_attempt_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    terminal_reason: Mapped[str | None] = mapped_column(Text, nullable=True)


class AiImportRecord(AuditMixin, Base):
    __tablename__ = "ai_imports"

    ai_import_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id"), index=True)
    source_id: Mapped[str] = mapped_column(ForeignKey("sources.source_id"), index=True)
    output_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    schema_version: Mapped[str] = mapped_column(String(20))
    classified_file_path: Mapped[str] = mapped_column(Text)
    classification: Mapped[str] = mapped_column(String(30))
    validation_status: Mapped[str] = mapped_column(String(30))


class PromptExecutionRecord(AuditMixin, Base):
    __tablename__ = "prompt_executions"

    prompt_execution_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    ai_import_id: Mapped[str | None] = mapped_column(
        ForeignKey("ai_imports.ai_import_id"), unique=True
    )
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id"), index=True)
    prompt_id: Mapped[str] = mapped_column(String(20))
    prompt_version: Mapped[str] = mapped_column(String(20))
    environment: Mapped[str] = mapped_column(String(20))
    model: Mapped[str] = mapped_column(String(200))
    input_files: Mapped[list[str]] = mapped_column(JSON, default=list)
    output_file: Mapped[str] = mapped_column(Text)
    executed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    validation_status: Mapped[str] = mapped_column(String(30))


class AiArtifactRecord(AuditMixin, Base):
    __tablename__ = "ai_artifacts"

    ai_artifact_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id"), index=True)
    source_id: Mapped[str | None] = mapped_column(ForeignKey("sources.source_id"), index=True)
    prompt_execution_id: Mapped[str] = mapped_column(
        ForeignKey("prompt_executions.prompt_execution_id"), unique=True
    )
    prompt_id: Mapped[str] = mapped_column(String(20), index=True)
    schema_version: Mapped[str] = mapped_column(String(20))
    input_hash: Mapped[str] = mapped_column(String(64))
    output_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    classified_file_path: Mapped[str] = mapped_column(Text)
    classification: Mapped[str] = mapped_column(String(30), index=True)
    resolution_status: Mapped[str] = mapped_column(String(40), index=True)
    confidence: Mapped[float | None] = mapped_column(Float)
    importance: Mapped[str] = mapped_column(String(20), default="normal")
    high_importance_reason: Mapped[str | None] = mapped_column(Text)
    knowledge_cutoff: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    supersedes_artifact_id: Mapped[str | None] = mapped_column(
        ForeignKey("ai_artifacts.ai_artifact_id")
    )
    resolved_by_artifact_id: Mapped[str | None] = mapped_column(
        ForeignKey("ai_artifacts.ai_artifact_id")
    )
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class SegmentRecord(AuditMixin, Base):
    __tablename__ = "segments"
    __table_args__ = (
        UniqueConstraint("ai_artifact_id", "local_ref", name="uq_segment_artifact_ref"),
    )

    segment_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    ai_artifact_id: Mapped[str] = mapped_column(
        ForeignKey("ai_artifacts.ai_artifact_id"), index=True
    )
    source_id: Mapped[str] = mapped_column(ForeignKey("sources.source_id"), index=True)
    local_ref: Mapped[str] = mapped_column(String(100))
    sequence_number: Mapped[int] = mapped_column(Integer)
    raw_start_offset: Mapped[int] = mapped_column(Integer)
    raw_end_offset: Mapped[int] = mapped_column(Integer)
    raw_text: Mapped[str] = mapped_column(Text)
    normalized_text: Mapped[str] = mapped_column(Text)
    speaker_status: Mapped[str] = mapped_column(String(30))
    speaker_candidate: Mapped[str | None] = mapped_column(String(200))
    speaker_confidence: Mapped[float] = mapped_column(Float)
    attribution_basis: Mapped[str] = mapped_column(Text)
    review_status: Mapped[str] = mapped_column(String(30))
    importance: Mapped[str] = mapped_column(String(20), default="normal")
    high_importance_reason: Mapped[str | None] = mapped_column(Text)


class ForecastGroupRecord(AuditMixin, Base):
    __tablename__ = "forecast_groups"

    forecast_group_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    analyst_id: Mapped[str] = mapped_column(ForeignKey("analysts.analyst_id"), index=True)
    central_thesis: Mapped[str] = mapped_column(Text)
    first_issued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    latest_issued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    current_stance: Mapped[str | None] = mapped_column(String(20))
    reaffirmation_count: Mapped[int] = mapped_column(Integer, default=0)
    revision_count: Mapped[int] = mapped_column(Integer, default=0)
    withdrawal_status: Mapped[str] = mapped_column(String(30), default="active")


class ForecastIssuanceRecord(AuditMixin, Base):
    __tablename__ = "forecast_issuances"
    __table_args__ = (
        UniqueConstraint("ai_import_id", "local_ref", name="uq_issuance_import_ref"),
        UniqueConstraint("ai_artifact_id", "local_ref", name="uq_issuance_artifact_ref"),
        Index("ix_forecast_issuance_analyst_status", "analyst_id", "current_status"),
    )

    forecast_issuance_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    analyst_id: Mapped[str] = mapped_column(ForeignKey("analysts.analyst_id"), index=True)
    forecast_group_id: Mapped[str] = mapped_column(
        ForeignKey("forecast_groups.forecast_group_id"), index=True
    )
    ai_import_id: Mapped[str | None] = mapped_column(ForeignKey("ai_imports.ai_import_id"))
    ai_artifact_id: Mapped[str | None] = mapped_column(
        ForeignKey("ai_artifacts.ai_artifact_id"), index=True
    )
    source_id: Mapped[str] = mapped_column(ForeignKey("sources.source_id"), index=True)
    local_ref: Mapped[str] = mapped_column(String(100))
    made_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    publicly_available_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    made_at_source: Mapped[str | None] = mapped_column(String(40))
    forecast_type: Mapped[str] = mapped_column(String(40))
    commitment_strength: Mapped[str] = mapped_column(String(40))
    evidence_level: Mapped[str] = mapped_column(String(8))
    extraction_confidence: Mapped[float] = mapped_column(Float)
    human_readable_summary: Mapped[str] = mapped_column(Text)
    relation_to_previous: Mapped[str] = mapped_column(String(40))
    current_status: Mapped[str] = mapped_column(String(40), index=True)
    speaker_candidate: Mapped[str | None] = mapped_column(String(200))
    speaker_attribution_status: Mapped[str | None] = mapped_column(String(40))
    verified_attribution_status: Mapped[str | None] = mapped_column(String(40), index=True)
    attribution_confidence: Mapped[float | None] = mapped_column(Float)
    attribution_basis: Mapped[str | None] = mapped_column(Text)
    statement_kind: Mapped[str | None] = mapped_column(String(40))
    attribution_verification_reason: Mapped[str | None] = mapped_column(Text)
    # Round4 lifecycle
    lifecycle_status: Mapped[str] = mapped_column(String(40), default="active", index=True)
    supersedes_forecast_issuance_id: Mapped[str | None] = mapped_column(String(20), nullable=True)
    superseded_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    superseded_by_issuance_id: Mapped[str | None] = mapped_column(String(20), nullable=True)
    lifecycle_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    review_artifact_id: Mapped[str | None] = mapped_column(String(20), nullable=True)
    generation: Mapped[int] = mapped_column(Integer, default=1)
    lineage_root_id: Mapped[str | None] = mapped_column(String(20), nullable=True, index=True)


class ForecastEvidenceRecord(AuditMixin, Base):
    __tablename__ = "forecast_evidence"

    forecast_evidence_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    forecast_issuance_id: Mapped[str] = mapped_column(
        ForeignKey("forecast_issuances.forecast_issuance_id"), index=True
    )
    source_id: Mapped[str] = mapped_column(ForeignKey("sources.source_id"), index=True)
    segment_id: Mapped[str | None] = mapped_column(ForeignKey("segments.segment_id"), index=True)
    quote: Mapped[str] = mapped_column(Text)
    start_offset: Mapped[int] = mapped_column(Integer)
    end_offset: Mapped[int] = mapped_column(Integer)
    role: Mapped[str] = mapped_column(String(40))


class TargetRecord(AuditMixin, Base):
    __tablename__ = "targets"
    __table_args__ = (
        UniqueConstraint("canonical_name", "ticker", "currency", name="uq_target_identity"),
    )

    target_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    raw_label: Mapped[str] = mapped_column(Text)
    canonical_name: Mapped[str] = mapped_column(Text)
    target_type: Mapped[str] = mapped_column(String(40), index=True)
    ticker: Mapped[str | None] = mapped_column(String(100), index=True)
    security_code: Mapped[str | None] = mapped_column(String(100))
    exchange: Mapped[str | None] = mapped_column(String(100))
    currency: Mapped[str | None] = mapped_column(String(20))
    aliases: Mapped[list[str]] = mapped_column(JSON, default=list)
    valid_from: Mapped[date | None] = mapped_column(Date)
    valid_to: Mapped[date | None] = mapped_column(Date)


class TargetMappingRecord(AuditMixin, Base):
    __tablename__ = "target_mappings"

    target_mapping_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    target_id: Mapped[str] = mapped_column(ForeignKey("targets.target_id"), index=True)
    mapping_method: Mapped[str] = mapped_column(String(40))
    evaluation_instruments: Mapped[list[Any]] = mapped_column(JSON, default=list)
    weights: Mapped[list[float] | None] = mapped_column(JSON)
    benchmark: Mapped[str | None] = mapped_column(String(100))
    knowledge_cutoff: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    source_evidence: Mapped[str] = mapped_column(Text)
    proposal_model: Mapped[str | None] = mapped_column(String(200))
    review_result: Mapped[str | None] = mapped_column(Text)
    adjudication_result: Mapped[str | None] = mapped_column(Text)
    proposal_artifact_id: Mapped[str | None] = mapped_column(
        ForeignKey("ai_artifacts.ai_artifact_id")
    )
    review_artifact_id: Mapped[str | None] = mapped_column(
        ForeignKey("ai_artifacts.ai_artifact_id")
    )
    adjudication_artifact_id: Mapped[str | None] = mapped_column(
        ForeignKey("ai_artifacts.ai_artifact_id")
    )
    mapping_status: Mapped[str] = mapped_column(String(40), index=True)
    mapping_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    locked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    unevaluable_reason: Mapped[str | None] = mapped_column(Text)


class ForecastComponentRecord(AuditMixin, Base):
    __tablename__ = "forecast_components"
    __table_args__ = (
        UniqueConstraint("forecast_issuance_id", "local_ref", name="uq_component_ref"),
    )

    forecast_component_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    forecast_issuance_id: Mapped[str] = mapped_column(
        ForeignKey("forecast_issuances.forecast_issuance_id"), index=True
    )
    parent_component_id: Mapped[str | None] = mapped_column(
        ForeignKey("forecast_components.forecast_component_id")
    )
    local_ref: Mapped[str] = mapped_column(String(100))
    sequence_number: Mapped[int] = mapped_column(Integer)
    prediction_form: Mapped[str] = mapped_column(String(40))
    direction: Mapped[str] = mapped_column(String(20))
    time_expression_raw: Mapped[str | None] = mapped_column(Text)
    time_source: Mapped[str] = mapped_column(String(30))
    normalized_start: Mapped[date | None] = mapped_column(Date)
    normalized_end: Mapped[date | None] = mapped_column(Date)
    time_precision: Mapped[str | None] = mapped_column(String(30))
    magnitude_value: Mapped[Decimal | None] = mapped_column(Numeric(24, 10))
    magnitude_unit: Mapped[str | None] = mapped_column(String(40))
    magnitude_operator: Mapped[str | None] = mapped_column(String(40))
    scenario_probability: Mapped[float | None] = mapped_column(Float)
    raw_target_label: Mapped[str | None] = mapped_column(Text)
    target_resolution_status: Mapped[str] = mapped_column(String(40), default="pending", index=True)
    importance: Mapped[str] = mapped_column(String(20), default="normal")
    high_importance_reason: Mapped[str | None] = mapped_column(Text)
    target_id: Mapped[str | None] = mapped_column(ForeignKey("targets.target_id"), index=True)
    target_mapping_id: Mapped[str | None] = mapped_column(
        ForeignKey("target_mappings.target_mapping_id"), index=True
    )


class TargetResolutionCandidateRecord(AuditMixin, Base):
    __tablename__ = "target_resolution_candidates"
    __table_args__ = (
        UniqueConstraint(
            "proposal_artifact_id",
            "candidate_ref",
            name="uq_target_candidate_artifact_ref",
        ),
    )

    target_resolution_candidate_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    proposal_artifact_id: Mapped[str] = mapped_column(
        ForeignKey("ai_artifacts.ai_artifact_id"), index=True
    )
    forecast_component_id: Mapped[str] = mapped_column(
        ForeignKey("forecast_components.forecast_component_id"), index=True
    )
    candidate_ref: Mapped[str] = mapped_column(String(100))
    rank: Mapped[int] = mapped_column(Integer)
    canonical_name: Mapped[str] = mapped_column(Text)
    target_type: Mapped[str] = mapped_column(String(40))
    mapping_method: Mapped[str] = mapped_column(String(40))
    instruments: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    existed_at: Mapped[date] = mapped_column(Date)
    knowledge_cutoff: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    source_evidence: Mapped[str] = mapped_column(Text)
    confidence: Mapped[float] = mapped_column(Float)
    candidate_status: Mapped[str] = mapped_column(String(30), default="proposed")


class TargetResolutionReviewRecord(AuditMixin, Base):
    __tablename__ = "target_resolution_reviews"

    target_resolution_review_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    review_artifact_id: Mapped[str] = mapped_column(
        ForeignKey("ai_artifacts.ai_artifact_id"), index=True
    )
    proposal_artifact_id: Mapped[str] = mapped_column(
        ForeignKey("ai_artifacts.ai_artifact_id"), index=True
    )
    forecast_component_id: Mapped[str] = mapped_column(
        ForeignKey("forecast_components.forecast_component_id"), index=True
    )
    candidate_ref: Mapped[str | None] = mapped_column(String(100))
    decision: Mapped[str] = mapped_column(String(30))
    confidence: Mapped[float] = mapped_column(Float)
    rationale: Mapped[str] = mapped_column(Text)
    corrected_candidate: Mapped[dict[str, Any] | None] = mapped_column(JSON)


class TargetResolutionAdjudicationRecord(AuditMixin, Base):
    __tablename__ = "target_resolution_adjudications"

    target_resolution_adjudication_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    adjudication_artifact_id: Mapped[str] = mapped_column(
        ForeignKey("ai_artifacts.ai_artifact_id"), index=True
    )
    proposal_artifact_id: Mapped[str] = mapped_column(
        ForeignKey("ai_artifacts.ai_artifact_id"), index=True
    )
    review_artifact_id: Mapped[str] = mapped_column(
        ForeignKey("ai_artifacts.ai_artifact_id"), index=True
    )
    forecast_component_id: Mapped[str] = mapped_column(
        ForeignKey("forecast_components.forecast_component_id"), index=True
    )
    final_status: Mapped[str] = mapped_column(String(30))
    selected_candidate_ref: Mapped[str | None] = mapped_column(String(100))
    rationale: Mapped[str] = mapped_column(Text)


class MarketSeriesRecord(AuditMixin, Base):
    __tablename__ = "market_series"
    __table_args__ = (
        Index(
            "ix_market_series_lookup",
            "provider",
            "symbol",
            "currency",
            "adjustment_type",
            "start_date",
            "end_date",
        ),
        Index(
            "ix_market_series_kind_lookup",
            "series_kind",
            "provider",
            "symbol",
            "currency",
            "start_date",
            "end_date",
        ),
    )

    market_series_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    provider: Mapped[str] = mapped_column(String(40), index=True)
    symbol: Mapped[str] = mapped_column(String(100), index=True)
    series_kind: Mapped[str] = mapped_column(String(20), default="raw", index=True)
    series_identity: Mapped[str | None] = mapped_column(String(200), index=True)
    mapping_hash: Mapped[str | None] = mapped_column(String(64), index=True)
    input_series_hashes: Mapped[list[str] | None] = mapped_column(JSON)
    basket_weights: Mapped[list[float] | None] = mapped_column(JSON)
    common_date_rule: Mapped[str | None] = mapped_column(String(80))
    currency: Mapped[str] = mapped_column(String(20))
    adjustment_type: Mapped[str] = mapped_column(String(60))
    frequency: Mapped[str] = mapped_column(String(20))
    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[date] = mapped_column(Date)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    raw_cache_path: Mapped[str] = mapped_column(Text)
    data_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    quality_status: Mapped[str] = mapped_column(String(30), default="valid")
    provider_error_code: Mapped[str | None] = mapped_column(String(60))
    provider_error_message: Mapped[str | None] = mapped_column(Text)
    retryable: Mapped[str | None] = mapped_column(String(10))
    attempt_count: Mapped[int | None] = mapped_column(Integer)
    cache_hit: Mapped[str | None] = mapped_column(String(10))


class ArtifactApplicabilityRecord(Base):
    __tablename__ = "artifact_applicability"
    __table_args__ = (
        UniqueConstraint(
            "ai_artifact_id", "target_source_id", name="uq_applicability_artifact_source"
        ),
    )

    applicability_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    ai_artifact_id: Mapped[str] = mapped_column(
        ForeignKey("ai_artifacts.ai_artifact_id"), nullable=False
    )
    target_run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id"), nullable=False)
    target_source_id: Mapped[str] = mapped_column(ForeignKey("sources.source_id"), nullable=False)
    reused_from_artifact_id: Mapped[str | None] = mapped_column(
        ForeignKey("ai_artifacts.ai_artifact_id"), nullable=True
    )
    raw_artifact_id: Mapped[str | None] = mapped_column(String(20), nullable=True)
    raw_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    applicability_status: Mapped[str] = mapped_column(String(40), default="active")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)


class EvaluationRecord(AuditMixin, Base):
    __tablename__ = "evaluations"
    __table_args__ = (
        UniqueConstraint(
            "forecast_component_id",
            "target_mapping_id",
            "evaluation_method_version",
            "evaluation_as_of",
            name="uq_evaluation_identity",
        ),
        Index(
            "ix_evaluations_component_as_of_method",
            "forecast_component_id",
            "evaluation_as_of",
            "evaluation_method_version",
        ),
    )

    evaluation_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    forecast_component_id: Mapped[str] = mapped_column(
        ForeignKey("forecast_components.forecast_component_id"), index=True
    )
    target_mapping_id: Mapped[str] = mapped_column(
        ForeignKey("target_mappings.target_mapping_id"), index=True
    )
    market_series_id: Mapped[str | None] = mapped_column(
        ForeignKey("market_series.market_series_id")
    )
    evaluation_method_version: Mapped[str] = mapped_column(String(20))
    evaluation_as_of: Mapped[date] = mapped_column(Date, index=True)
    start_price: Mapped[Decimal | None] = mapped_column(Numeric(24, 10))
    end_price: Mapped[Decimal | None] = mapped_column(Numeric(24, 10))
    current_price: Mapped[Decimal | None] = mapped_column(Numeric(24, 10))
    period_high: Mapped[Decimal | None] = mapped_column(Numeric(24, 10))
    period_low: Mapped[Decimal | None] = mapped_column(Numeric(24, 10))
    actual_return: Mapped[Decimal | None] = mapped_column(Numeric(24, 10))
    total_return: Mapped[Decimal | None] = mapped_column(Numeric(24, 10))
    base_currency_return: Mapped[Decimal | None] = mapped_column(Numeric(24, 10))
    benchmark_return: Mapped[Decimal | None] = mapped_column(Numeric(24, 10))
    excess_return: Mapped[Decimal | None] = mapped_column(Numeric(24, 10))
    direction_result: Mapped[str | None] = mapped_column(String(20))
    timing_result: Mapped[str | None] = mapped_column(String(40))
    magnitude_result: Mapped[str | None] = mapped_column(String(40))
    early_realization_result: Mapped[str | None] = mapped_column(String(40))
    evaluation_status: Mapped[str] = mapped_column(String(40), index=True)
    unevaluable_reason: Mapped[str | None] = mapped_column(Text)
    max_favorable_excursion: Mapped[Decimal | None] = mapped_column(Numeric(24, 10))
    max_adverse_excursion: Mapped[Decimal | None] = mapped_column(Numeric(24, 10))
    provider_error_code: Mapped[str | None] = mapped_column(String(60))
    provider_error_message: Mapped[str | None] = mapped_column(Text)
    retryable: Mapped[str | None] = mapped_column(String(10))
    attempt_count: Mapped[int | None] = mapped_column(Integer)
    cache_hit: Mapped[str | None] = mapped_column(String(10))
    # Round4 coverage audit
    common_date_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    selected_start_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    selected_end_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    coverage_audit: Mapped[dict[str, Any] | None] = mapped_column(JSON, nullable=True)


class WorkflowTaskRecord(AuditMixin, Base):
    __tablename__ = "workflow_tasks"
    __table_args__ = (
        UniqueConstraint("run_id", "task_key", name="uq_workflow_task_run_key"),
        Index("ix_workflow_tasks_run_status", "run_id", "status"),
    )

    workflow_task_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id"), index=True)
    task_key: Mapped[str] = mapped_column(String(80), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    executor: Mapped[str] = mapped_column(String(40), nullable=False)
    depends_on: Mapped[list[str]] = mapped_column(JSON, default=list)
    related_artifact_id: Mapped[str | None] = mapped_column(
        ForeignKey("ai_artifacts.ai_artifact_id")
    )
    related_source_id: Mapped[str | None] = mapped_column(ForeignKey("sources.source_id"))
    related_component_id: Mapped[str | None] = mapped_column(
        ForeignKey("forecast_components.forecast_component_id")
    )
    supersedes_task_id: Mapped[str | None] = mapped_column(
        ForeignKey("workflow_tasks.workflow_task_id")
    )
    resolved_by_task_id: Mapped[str | None] = mapped_column(
        ForeignKey("workflow_tasks.workflow_task_id")
    )
    retryable: Mapped[str] = mapped_column(String(10), default="yes")
    last_error: Mapped[str | None] = mapped_column(Text)
    recommended_rank: Mapped[int | None] = mapped_column(Integer)
    command_or_prompt: Mapped[str | None] = mapped_column(Text)
    inputs: Mapped[list[str]] = mapped_column(JSON, default=list)
    outputs: Mapped[list[str]] = mapped_column(JSON, default=list)
    details: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class EvaluationSnapshotRecord(AuditMixin, Base):
    __tablename__ = "evaluation_snapshots"
    __table_args__ = (
        UniqueConstraint("evaluation_id", "snapshot_at", name="uq_evaluation_snapshot"),
    )

    evaluation_snapshot_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    evaluation_id: Mapped[str] = mapped_column(ForeignKey("evaluations.evaluation_id"), index=True)
    snapshot_at: Mapped[date] = mapped_column(Date)
    status: Mapped[str] = mapped_column(String(40))
    interim_return: Mapped[Decimal | None] = mapped_column(Numeric(24, 10))
    max_favorable_excursion: Mapped[Decimal | None] = mapped_column(Numeric(24, 10))
    max_adverse_excursion: Mapped[Decimal | None] = mapped_column(Numeric(24, 10))
    first_realization_at: Mapped[date | None] = mapped_column(Date)
    days_early_or_late: Mapped[int | None] = mapped_column(Integer)
    notes: Mapped[str | None] = mapped_column(Text)
