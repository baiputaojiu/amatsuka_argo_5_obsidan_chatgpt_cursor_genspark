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


class SourceRecord(AuditMixin, Base):
    __tablename__ = "sources"

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
    raw_file_path: Mapped[str] = mapped_column(Text, nullable=False)
    raw_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
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
    ai_import_id: Mapped[str] = mapped_column(ForeignKey("ai_imports.ai_import_id"), unique=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("runs.run_id"), index=True)
    prompt_id: Mapped[str] = mapped_column(String(20))
    prompt_version: Mapped[str] = mapped_column(String(20))
    environment: Mapped[str] = mapped_column(String(20))
    model: Mapped[str] = mapped_column(String(200))
    input_files: Mapped[list[str]] = mapped_column(JSON, default=list)
    output_file: Mapped[str] = mapped_column(Text)
    executed_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utc_now)
    validation_status: Mapped[str] = mapped_column(String(30))


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
        Index("ix_forecast_issuance_analyst_status", "analyst_id", "current_status"),
    )

    forecast_issuance_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    analyst_id: Mapped[str] = mapped_column(ForeignKey("analysts.analyst_id"), index=True)
    forecast_group_id: Mapped[str] = mapped_column(
        ForeignKey("forecast_groups.forecast_group_id"), index=True
    )
    ai_import_id: Mapped[str] = mapped_column(ForeignKey("ai_imports.ai_import_id"))
    source_id: Mapped[str] = mapped_column(ForeignKey("sources.source_id"), index=True)
    local_ref: Mapped[str] = mapped_column(String(100))
    made_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    publicly_available_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    forecast_type: Mapped[str] = mapped_column(String(40))
    commitment_strength: Mapped[str] = mapped_column(String(40))
    evidence_level: Mapped[str] = mapped_column(String(8))
    extraction_confidence: Mapped[float] = mapped_column(Float)
    human_readable_summary: Mapped[str] = mapped_column(Text)
    relation_to_previous: Mapped[str] = mapped_column(String(40))
    current_status: Mapped[str] = mapped_column(String(40), index=True)


class ForecastEvidenceRecord(AuditMixin, Base):
    __tablename__ = "forecast_evidence"

    forecast_evidence_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    forecast_issuance_id: Mapped[str] = mapped_column(
        ForeignKey("forecast_issuances.forecast_issuance_id"), index=True
    )
    source_id: Mapped[str] = mapped_column(ForeignKey("sources.source_id"), index=True)
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
    ticker: Mapped[str] = mapped_column(String(100), index=True)
    security_code: Mapped[str | None] = mapped_column(String(100))
    exchange: Mapped[str | None] = mapped_column(String(100))
    currency: Mapped[str] = mapped_column(String(20))
    aliases: Mapped[list[str]] = mapped_column(JSON, default=list)
    valid_from: Mapped[date | None] = mapped_column(Date)
    valid_to: Mapped[date | None] = mapped_column(Date)


class TargetMappingRecord(AuditMixin, Base):
    __tablename__ = "target_mappings"

    target_mapping_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    target_id: Mapped[str] = mapped_column(ForeignKey("targets.target_id"), index=True)
    mapping_method: Mapped[str] = mapped_column(String(40))
    evaluation_instruments: Mapped[list[str]] = mapped_column(JSON, default=list)
    weights: Mapped[list[float] | None] = mapped_column(JSON)
    benchmark: Mapped[str | None] = mapped_column(String(100))
    knowledge_cutoff: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    source_evidence: Mapped[str] = mapped_column(Text)
    proposal_model: Mapped[str | None] = mapped_column(String(200))
    review_result: Mapped[str | None] = mapped_column(Text)
    adjudication_result: Mapped[str | None] = mapped_column(Text)
    mapping_status: Mapped[str] = mapped_column(String(40), index=True)
    mapping_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    locked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


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
    target_id: Mapped[str] = mapped_column(ForeignKey("targets.target_id"), index=True)
    target_mapping_id: Mapped[str] = mapped_column(
        ForeignKey("target_mappings.target_mapping_id"), index=True
    )


class MarketSeriesRecord(AuditMixin, Base):
    __tablename__ = "market_series"

    market_series_id: Mapped[str] = mapped_column(String(20), primary_key=True)
    provider: Mapped[str] = mapped_column(String(40), index=True)
    symbol: Mapped[str] = mapped_column(String(100), index=True)
    currency: Mapped[str] = mapped_column(String(20))
    adjustment_type: Mapped[str] = mapped_column(String(60))
    frequency: Mapped[str] = mapped_column(String(20))
    start_date: Mapped[date] = mapped_column(Date)
    end_date: Mapped[date] = mapped_column(Date)
    retrieved_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    raw_cache_path: Mapped[str] = mapped_column(Text)
    data_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    quality_status: Mapped[str] = mapped_column(String(30), default="valid")


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
