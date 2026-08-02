from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from analyst_forecast.domain.models import Direction, MappingStatus, TimeSource


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PromptExecution(StrictModel):
    prompt_id: Literal["P08"]
    prompt_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    environment: Literal["cursor", "chatgpt"]
    model: str = Field(min_length=1, max_length=200)


class EvidenceQuote(StrictModel):
    source_id: str = Field(pattern=r"^SRC-\d{6}$")
    quote: str = Field(min_length=1)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(gt=0)
    role: Literal["prediction", "condition", "target", "timing", "magnitude", "context"]

    @model_validator(mode="after")
    def validate_offsets(self) -> EvidenceQuote:
        if self.end_offset <= self.start_offset:
            raise ValueError("end_offsetはstart_offsetより後である必要があります")
        return self


class TargetMappingOutput(StrictModel):
    raw_label: str = Field(min_length=1)
    canonical_name: str = Field(min_length=1)
    target_type: Literal[
        "stock",
        "index",
        "etf",
        "fx",
        "commodity",
        "industry",
        "theme",
        "economic_indicator",
    ]
    symbol: str = Field(min_length=1, max_length=100)
    exchange: str | None = Field(default=None, max_length=100)
    currency: str = Field(min_length=1, max_length=20)
    mapping_method: Literal[
        "explicit",
        "constituent_example",
        "official_index",
        "etf_proxy",
        "fixed_basket",
    ]
    knowledge_cutoff: datetime
    source_evidence: str = Field(min_length=1)
    proposal_model: str = Field(min_length=1, max_length=200)
    mapping_status: MappingStatus
    review_result: str | None = None

    @field_validator("knowledge_cutoff")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("knowledge_cutoffにはタイムゾーンが必要です")
        return value

    @model_validator(mode="after")
    def require_independent_review(self) -> TargetMappingOutput:
        if self.mapping_status in {"verified", "corrected"} and not self.review_result:
            raise ValueError("検証済みマッピングには別AIのreview_resultが必要です")
        return self


class ForecastComponentOutput(StrictModel):
    component_ref: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,99}$")
    parent_component_ref: str | None = None
    sequence_number: int = Field(ge=1)
    prediction_form: Literal[
        "deadline_target",
        "point_in_time",
        "period_direction",
        "period_state",
        "turning_point",
        "relative",
        "range",
        "event",
    ]
    direction: Direction
    time_expression_raw: str | None = None
    time_source: TimeSource
    normalized_start: date | None = None
    normalized_end: date | None = None
    time_precision: str | None = None
    magnitude_value: float | None = None
    magnitude_unit: str | None = None
    magnitude_operator: (
        Literal[
            "approximate",
            "minimum",
            "maximum",
            "range",
            "exact_target",
            "threshold",
        ]
        | None
    ) = None
    scenario_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    target: TargetMappingOutput

    @model_validator(mode="after")
    def validate_period(self) -> ForecastComponentOutput:
        if (
            self.normalized_start is not None
            and self.normalized_end is not None
            and self.normalized_end < self.normalized_start
        ):
            raise ValueError("normalized_endはnormalized_start以後である必要があります")
        return self


class ForecastIssuanceOutput(StrictModel):
    forecast_ref: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,99}$")
    forecast_group_ref: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,99}$")
    existing_forecast_group_id: str | None = Field(default=None, pattern=r"^FCG-\d{6}$")
    made_at: datetime
    publicly_available_at: datetime
    forecast_type: Literal[
        "directional",
        "numeric",
        "conditional",
        "recommendation",
        "scenario",
    ]
    commitment_strength: Literal["explicit", "directional", "weak", "recommendation"]
    evidence_level: Literal["A", "B", "C", "D", "E", "F"]
    extraction_confidence: float = Field(ge=0.0, le=1.0)
    human_readable_summary: str = Field(min_length=1)
    relation_to_previous: Literal[
        "initial",
        "reaffirmation",
        "strengthened",
        "weakened",
        "numeric_revision",
        "timing_revision",
        "condition_added",
        "reversal",
        "withdrawal",
    ]
    evidence: list[EvidenceQuote] = Field(min_length=1)
    components: list[ForecastComponentOutput] = Field(min_length=1)

    @field_validator("made_at", "publicly_available_at")
    @classmethod
    def require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("日時にはタイムゾーンが必要です")
        return value

    @model_validator(mode="after")
    def validate_references_and_cutoff(self) -> ForecastIssuanceOutput:
        refs = [component.component_ref for component in self.components]
        if len(refs) != len(set(refs)):
            raise ValueError("component_refが重複しています")
        ref_set = set(refs)
        for component in self.components:
            if (
                component.parent_component_ref is not None
                and component.parent_component_ref not in ref_set
            ):
                raise ValueError("parent_component_refが同じ予想表明内に存在しません")
            if component.target.knowledge_cutoff > self.made_at:
                raise ValueError("knowledge_cutoffが発言日時より後です")
        return self


class ForecastExtractionOutput(StrictModel):
    model_config = ConfigDict(
        extra="forbid",
        title="ForecastExtractionOutput",
        json_schema_extra={
            "$id": "https://local.invalid/schemas/forecast-extraction-1.0.0.json",
            "$schema": "https://json-schema.org/draft/2020-12/schema",
        },
    )

    schema_version: Literal["1.0.0"]
    run_id: str = Field(pattern=r"^RUN-\d{8}-\d{3}$")
    source_id: str = Field(pattern=r"^SRC-\d{6}$")
    prompt_execution: PromptExecution
    forecasts: list[ForecastIssuanceOutput] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_local_ids(self) -> ForecastExtractionOutput:
        forecast_refs = [forecast.forecast_ref for forecast in self.forecasts]
        if len(forecast_refs) != len(set(forecast_refs)):
            raise ValueError("forecast_refが重複しています")
        component_refs = [
            component.component_ref
            for forecast in self.forecasts
            for component in forecast.components
        ]
        if len(component_refs) != len(set(component_refs)):
            raise ValueError("出力全体でcomponent_refが重複しています")
        groups: dict[str, set[str | None]] = {}
        for forecast in self.forecasts:
            groups.setdefault(forecast.forecast_group_ref, set()).add(
                forecast.existing_forecast_group_id
            )
        if any(len(group_ids) > 1 for group_ids in groups.values()):
            raise ValueError("同じforecast_group_refに異なるexisting_forecast_group_idがあります")
        return self


def schema_path() -> Path:
    return Path(__file__).with_name("forecast_extraction.schema.json")
