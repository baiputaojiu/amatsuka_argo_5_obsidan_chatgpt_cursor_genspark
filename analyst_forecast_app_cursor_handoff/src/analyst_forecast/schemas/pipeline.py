from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from analyst_forecast.domain.models import Direction, TimeSource
from analyst_forecast.schemas.ai_output import EvidenceQuote

HashValue = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]
ArtifactId = Annotated[str, Field(pattern=r"^AIF-\d{6}$")]
ComponentId = Annotated[str, Field(pattern=r"^FCC-\d{6}$")]
SourceId = Annotated[str, Field(pattern=r"^SRC-\d{6}$")]
RunId = Annotated[str, Field(pattern=r"^RUN-\d{8}-\d{3}$")]
LocalRef = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,99}$")]


class PipelineModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _require_timezone(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("日時にはタイムゾーンが必要です")
    return value


class PromptExecutionBase(PipelineModel):
    prompt_id: str
    prompt_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    environment: Literal["cursor", "chatgpt"]
    model: str = Field(min_length=1, max_length=200)
    executed_at: datetime

    @field_validator("executed_at")
    @classmethod
    def validate_executed_at(cls, value: datetime) -> datetime:
        return _require_timezone(value)


class P05PromptExecution(PromptExecutionBase):
    prompt_id: Literal["P05"]


class P06PromptExecution(PromptExecutionBase):
    prompt_id: Literal["P06"]


class P07PromptExecution(PromptExecutionBase):
    prompt_id: Literal["P07"]


class P08PromptExecution(PromptExecutionBase):
    prompt_id: Literal["P08"]


class P09PromptExecution(PromptExecutionBase):
    prompt_id: Literal["P09"]


class P11PromptExecution(PromptExecutionBase):
    prompt_id: Literal["P11"]


class P12PromptExecution(PromptExecutionBase):
    prompt_id: Literal["P12"]


class P13PromptExecution(PromptExecutionBase):
    prompt_id: Literal["P13"]


class SegmentOutput(PipelineModel):
    segment_ref: LocalRef
    sequence_number: int = Field(ge=1)
    raw_start_offset: int = Field(ge=0)
    raw_end_offset: int = Field(gt=0)
    raw_text: str = Field(min_length=1)
    normalized_text: str = Field(min_length=1)
    speaker_status: Literal["identified", "unknown"]
    speaker_candidate: str | None = Field(default=None, max_length=200)
    speaker_confidence: float = Field(ge=0.0, le=1.0)
    attribution_basis: str = Field(min_length=1)
    review_status: Literal["accepted", "needs_review", "reviewed"]
    importance: Literal["normal", "high"] = "normal"
    high_importance_reason: str | None = None

    @model_validator(mode="after")
    def validate_segment(self) -> SegmentOutput:
        if self.raw_end_offset <= self.raw_start_offset:
            raise ValueError("raw_end_offsetはraw_start_offsetより後にしてください")
        if self.speaker_status == "unknown" and self.speaker_candidate is not None:
            raise ValueError("unknown speakerにspeaker_candidateを設定できません")
        if self.speaker_status == "identified" and not self.speaker_candidate:
            raise ValueError("identified speakerにはspeaker_candidateが必要です")
        if self.importance == "high" and not self.high_importance_reason:
            raise ValueError("高重要度には理由が必要です")
        return self


class P05Output(PipelineModel):
    model_config = ConfigDict(
        extra="forbid",
        title="P05SpeakerProcessingOutput",
        json_schema_extra={
            "$id": "https://local.invalid/schemas/p05-speaker-processing-2.0.0.json",
            "$schema": "https://json-schema.org/draft/2020-12/schema",
        },
    )

    schema_version: Literal["2.0.0"]
    run_id: RunId
    source_id: SourceId
    prompt_execution: P05PromptExecution
    input_hash: HashValue
    knowledge_cutoff: datetime
    segments: list[SegmentOutput] = Field(min_length=1)

    @field_validator("knowledge_cutoff")
    @classmethod
    def validate_cutoff(cls, value: datetime) -> datetime:
        return _require_timezone(value)


class TextSegmentOutput(PipelineModel):
    """ブログ・X・Web向けの原文整理segment。"""

    segment_ref: LocalRef
    sequence_number: int = Field(ge=1)
    raw_start_offset: int = Field(ge=0)
    raw_end_offset: int = Field(gt=0)
    raw_text: str = Field(min_length=1)
    normalized_text: str = Field(min_length=1)
    author_status: Literal["identified", "unknown"]
    author_candidate: str | None = Field(default=None, max_length=200)
    author_confidence: float = Field(ge=0.0, le=1.0)
    # 記事著者と引用発言者を分離（direct_quote時）
    content_author: str | None = Field(default=None, max_length=200)
    statement_speaker: str | None = Field(default=None, max_length=200)
    statement_kind: Literal[
        "author_own",
        "direct_quote",
        "third_party_summary",
        "repost",
        "reply",
    ]
    attribution_basis: str = Field(min_length=1)
    review_status: Literal["accepted", "needs_review", "reviewed"]
    importance: Literal["normal", "high"] = "normal"
    high_importance_reason: str | None = None

    @model_validator(mode="after")
    def validate_segment(self) -> TextSegmentOutput:
        if self.raw_end_offset <= self.raw_start_offset:
            raise ValueError("raw_end_offsetはraw_start_offsetより後にしてください")
        if self.author_status == "unknown" and self.author_candidate is not None:
            raise ValueError("unknown authorにauthor_candidateを設定できません")
        if self.author_status == "identified" and not self.author_candidate:
            raise ValueError("identified authorにはauthor_candidateが必要です")
        if self.statement_kind == "direct_quote" and not self.statement_speaker:
            raise ValueError("direct_quoteにはstatement_speakerが必要です")
        if self.importance == "high" and not self.high_importance_reason:
            raise ValueError("高重要度には理由が必要です")
        return self


class P07Output(PipelineModel):
    model_config = ConfigDict(
        extra="forbid",
        title="P07TextSourceProcessingOutput",
        json_schema_extra={
            "$id": "https://local.invalid/schemas/p07-text-source-processing-2.0.0.json",
            "$schema": "https://json-schema.org/draft/2020-12/schema",
        },
    )

    schema_version: Literal["2.0.0"]
    run_id: RunId
    source_id: SourceId
    prompt_execution: P07PromptExecution
    input_hash: HashValue
    knowledge_cutoff: datetime
    segments: list[TextSegmentOutput] = Field(min_length=1)

    @field_validator("knowledge_cutoff")
    @classmethod
    def validate_cutoff(cls, value: datetime) -> datetime:
        return _require_timezone(value)


class ReviewFinding(PipelineModel):
    finding_ref: LocalRef
    severity: Literal["info", "warning", "error"]
    message: str = Field(min_length=1)
    evidence: str = Field(min_length=1)


class P06Output(PipelineModel):
    """話者・著者帰属レビュー（P05/P07共通）。"""

    model_config = ConfigDict(
        extra="forbid",
        title="P06SpeakerAttributionReviewOutput",
        json_schema_extra={
            "$id": "https://local.invalid/schemas/p06-speaker-review-2.0.0.json",
            "$schema": "https://json-schema.org/draft/2020-12/schema",
        },
    )

    schema_version: Literal["2.0.0"]
    run_id: RunId
    source_id: SourceId
    reviewed_artifact_id: ArtifactId
    prompt_execution: P06PromptExecution
    input_hash: HashValue
    decision: Literal["accept", "correct", "reject", "unresolved"]
    findings: list[ReviewFinding] = Field(default_factory=list)
    corrected_payload: dict[str, object] | None = None
    knowledge_cutoff: datetime

    @field_validator("knowledge_cutoff")
    @classmethod
    def validate_cutoff(cls, value: datetime) -> datetime:
        return _require_timezone(value)

    @model_validator(mode="after")
    def validate_decision(self) -> P06Output:
        if self.decision == "correct" and self.corrected_payload is None:
            raise ValueError("correctにはcorrected_payloadが必要です")
        if self.decision != "correct" and self.corrected_payload is not None:
            raise ValueError("correct以外にcorrected_payloadは設定できません")
        return self


class P09Output(PipelineModel):
    """予想抽出レビュー。"""

    model_config = ConfigDict(
        extra="forbid",
        title="P09ForecastExtractionReviewOutput",
        json_schema_extra={
            "$id": "https://local.invalid/schemas/p09-forecast-review-2.0.0.json",
            "$schema": "https://json-schema.org/draft/2020-12/schema",
        },
    )

    schema_version: Literal["2.0.0"]
    run_id: RunId
    source_id: SourceId
    reviewed_artifact_id: ArtifactId
    prompt_execution: P09PromptExecution
    input_hash: HashValue
    decision: Literal["accept", "correct", "reject", "unresolved"]
    findings: list[ReviewFinding] = Field(default_factory=list)
    corrected_payload: dict[str, object] | None = None
    knowledge_cutoff: datetime

    @field_validator("knowledge_cutoff")
    @classmethod
    def validate_cutoff(cls, value: datetime) -> datetime:
        return _require_timezone(value)

    @model_validator(mode="after")
    def validate_decision(self) -> P09Output:
        if self.decision == "correct" and self.corrected_payload is None:
            raise ValueError("correctにはcorrected_payloadが必要です")
        if self.decision != "correct" and self.corrected_payload is not None:
            raise ValueError("correct以外にcorrected_payloadは設定できません")
        return self


class ForecastComponentV2(PipelineModel):
    component_ref: LocalRef
    parent_component_ref: LocalRef | None = None
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
    raw_target_label: str = Field(min_length=1)
    target_resolution_status: Literal["pending"]

    @model_validator(mode="after")
    def validate_period(self) -> ForecastComponentV2:
        if (
            self.normalized_start is not None
            and self.normalized_end is not None
            and self.normalized_end < self.normalized_start
        ):
            raise ValueError("normalized_endはnormalized_start以後にしてください")
        return self


class ForecastIssuanceV2(PipelineModel):
    forecast_ref: LocalRef
    forecast_group_ref: LocalRef
    existing_forecast_group_id: str | None = Field(default=None, pattern=r"^FCG-\d{6}$")
    made_at: datetime
    publicly_available_at: datetime
    made_at_source: Literal[
        "explicit",
        "source_metadata",
        "context_inferred",
        "unknown",
    ] = "unknown"
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
    importance: Literal["normal", "high"] = "normal"
    high_importance_reason: str | None = None
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
    components: list[ForecastComponentV2] = Field(min_length=1)
    upstream_segment_refs: list[LocalRef] = Field(default_factory=list)
    speaker_candidate: str | None = Field(default=None, max_length=200)
    speaker_attribution_status: Literal[
        "target_confirmed",
        "uncertain",
        "not_target",
        "legacy_unknown",
    ] = "legacy_unknown"
    attribution_confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    attribution_basis: str | None = None
    statement_kind: Literal[
        "direct_statement",
        "direct_quote",
        "third_party_summary",
        "legacy_unknown",
    ] = "legacy_unknown"

    @field_validator("made_at", "publicly_available_at")
    @classmethod
    def validate_datetime(cls, value: datetime) -> datetime:
        return _require_timezone(value)

    @model_validator(mode="after")
    def validate_forecast(self) -> ForecastIssuanceV2:
        refs = {component.component_ref for component in self.components}
        if len(refs) != len(self.components):
            raise ValueError("component_refが重複しています")
        if any(
            component.parent_component_ref is not None
            and component.parent_component_ref not in refs
            for component in self.components
        ):
            raise ValueError("parent_component_refが同じ表明内に存在しません")
        if self.importance == "high" and not self.high_importance_reason:
            raise ValueError("高重要度には理由が必要です")
        if self.made_at > self.publicly_available_at:
            raise ValueError("made_atはpublicly_available_at以前にしてください")
        return self


class P08Output(PipelineModel):
    model_config = ConfigDict(
        extra="forbid",
        title="P08ForecastExtractionOutput",
        json_schema_extra={
            "$id": "https://local.invalid/schemas/p08-forecast-extraction-2.1.0.json",
            "$schema": "https://json-schema.org/draft/2020-12/schema",
        },
    )

    schema_version: Literal["2.0.0", "2.1.0"] = "2.1.0"
    run_id: RunId
    source_id: SourceId
    p05_artifact_id: ArtifactId | None = None
    upstream_artifact_id: ArtifactId | None = None
    upstream_prompt_id: Literal["P05", "P07"] | None = None
    prompt_execution: P08PromptExecution
    input_hash: HashValue
    processing_status: Literal["processed_with_forecasts", "processed_no_forecast"]
    forecasts: list[ForecastIssuanceV2]

    @model_validator(mode="after")
    def validate_processing_status(self) -> P08Output:
        upstream_id = self.upstream_artifact_id or self.p05_artifact_id
        upstream_prompt = self.upstream_prompt_id
        if upstream_id is None:
            raise ValueError("upstream_artifact_idまたはp05_artifact_idが必要です")
        if upstream_prompt is None:
            if self.p05_artifact_id is not None and self.upstream_artifact_id is None:
                upstream_prompt = "P05"
            else:
                raise ValueError("upstream_prompt_idが必要です")
        object.__setattr__(self, "upstream_artifact_id", upstream_id)
        object.__setattr__(self, "upstream_prompt_id", upstream_prompt)
        if self.p05_artifact_id is None and upstream_prompt == "P05":
            object.__setattr__(self, "p05_artifact_id", upstream_id)
        if self.processing_status == "processed_no_forecast" and self.forecasts:
            raise ValueError("processed_no_forecastではforecastsを空にしてください")
        if self.processing_status == "processed_with_forecasts" and not self.forecasts:
            raise ValueError("processed_with_forecastsにはforecastが必要です")
        return self


class ResolutionInstrument(PipelineModel):
    symbol: str = Field(min_length=1, max_length=100)
    exchange: str | None = Field(default=None, max_length=100)
    currency: str = Field(min_length=1, max_length=20)
    weight: float = Field(gt=0.0, le=1.0)


class TargetResolutionCandidate(PipelineModel):
    candidate_ref: LocalRef
    rank: int = Field(ge=1, le=3)
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
    mapping_method: Literal[
        "explicit",
        "constituent_example",
        "official_index",
        "etf_proxy",
        "fixed_basket",
    ]
    instruments: list[ResolutionInstrument] = Field(min_length=1)
    existed_at: date
    knowledge_cutoff: datetime
    source_evidence: str = Field(min_length=1)
    confidence: float = Field(ge=0.0, le=1.0)

    @field_validator("knowledge_cutoff")
    @classmethod
    def validate_cutoff(cls, value: datetime) -> datetime:
        return _require_timezone(value)

    @model_validator(mode="after")
    def validate_weights(self) -> TargetResolutionCandidate:
        total = sum(instrument.weight for instrument in self.instruments)
        if abs(total - 1.0) > 0.000001:
            raise ValueError("候補内instrumentのweight合計は1にしてください")
        return self


class P11Output(PipelineModel):
    model_config = ConfigDict(
        extra="forbid",
        title="P11TargetResolutionProposalOutput",
        json_schema_extra={
            "$id": "https://local.invalid/schemas/p11-target-resolution-2.0.0.json",
            "$schema": "https://json-schema.org/draft/2020-12/schema",
        },
    )

    schema_version: Literal["2.0.0"]
    run_id: RunId
    source_id: SourceId
    forecast_component_id: ComponentId
    prompt_execution: P11PromptExecution
    input_hash: HashValue
    knowledge_cutoff: datetime
    resolution_status: Literal["proposed", "unresolvable"]
    candidates: list[TargetResolutionCandidate] = Field(max_length=3)
    unevaluable_reason: str | None = None

    @field_validator("knowledge_cutoff")
    @classmethod
    def validate_cutoff(cls, value: datetime) -> datetime:
        return _require_timezone(value)

    @model_validator(mode="after")
    def validate_resolution(self) -> P11Output:
        if self.resolution_status == "proposed" and not self.candidates:
            raise ValueError("proposedには1件以上のcandidateが必要です")
        if self.resolution_status == "unresolvable":
            if self.candidates:
                raise ValueError("unresolvableではcandidatesを空にしてください")
            if not self.unevaluable_reason:
                raise ValueError("unresolvableにはunevaluable_reasonが必要です")
        refs = {candidate.candidate_ref for candidate in self.candidates}
        ranks = {candidate.rank for candidate in self.candidates}
        if len(refs) != len(self.candidates) or len(ranks) != len(self.candidates):
            raise ValueError("candidate_refまたはrankが重複しています")
        return self


class CandidateReview(PipelineModel):
    candidate_ref: LocalRef
    decision: Literal["accept", "correct", "reject", "unresolved"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)
    corrected_candidate: TargetResolutionCandidate | None = None

    @model_validator(mode="after")
    def validate_correction(self) -> CandidateReview:
        if self.decision == "correct" and self.corrected_candidate is None:
            raise ValueError("correctにはcorrected_candidateが必要です")
        if self.decision != "correct" and self.corrected_candidate is not None:
            raise ValueError("correct以外にcorrected_candidateは設定できません")
        return self


class P12Output(PipelineModel):
    model_config = ConfigDict(
        extra="forbid",
        title="P12TargetResolutionReviewOutput",
        json_schema_extra={
            "$id": "https://local.invalid/schemas/p12-target-review-2.0.0.json",
            "$schema": "https://json-schema.org/draft/2020-12/schema",
        },
    )

    schema_version: Literal["2.0.0"]
    run_id: RunId
    source_id: SourceId
    forecast_component_id: ComponentId
    proposal_artifact_id: ArtifactId
    prompt_execution: P12PromptExecution
    input_hash: HashValue
    knowledge_cutoff: datetime
    resolution_status: Literal["agreed", "disagreed", "unresolved"]
    reviews: list[CandidateReview]
    recommended_candidate_ref: LocalRef | None = None
    recommended_candidate_origin: Literal["p11_proposal", "p12_correction"] | None = None
    unevaluable_reason: str | None = None

    @field_validator("knowledge_cutoff")
    @classmethod
    def validate_cutoff(cls, value: datetime) -> datetime:
        return _require_timezone(value)

    @model_validator(mode="after")
    def validate_resolution(self) -> P12Output:
        if self.resolution_status == "agreed":
            if not self.recommended_candidate_ref:
                raise ValueError("agreedにはrecommended_candidate_refが必要です")
            origin = self.recommended_candidate_origin or "p11_proposal"
            object.__setattr__(self, "recommended_candidate_origin", origin)
            if origin == "p11_proposal":
                if not any(
                    review.candidate_ref == self.recommended_candidate_ref
                    and review.decision == "accept"
                    for review in self.reviews
                ):
                    raise ValueError("推奨candidateにaccept reviewが必要です")
            elif not any(
                review.candidate_ref == self.recommended_candidate_ref
                and review.decision == "correct"
                and review.corrected_candidate is not None
                for review in self.reviews
            ):
                raise ValueError("p12_correctionにはcorrect reviewが必要です")
        elif self.resolution_status == "disagreed":
            if not self.reviews:
                raise ValueError("disagreedにはreviewが必要です")
            if self.recommended_candidate_ref is not None:
                raise ValueError("disagreedでは推奨candidateを確定できません")
        else:
            if self.recommended_candidate_ref is not None:
                raise ValueError("unresolvedでは推奨candidateを設定できません")
            if not self.unevaluable_reason:
                raise ValueError("unresolvedにはunevaluable_reasonが必要です")
        return self


class P13Output(PipelineModel):
    model_config = ConfigDict(
        extra="forbid",
        title="P13TargetResolutionAdjudicationOutput",
        json_schema_extra={
            "$id": "https://local.invalid/schemas/p13-target-adjudication-2.0.0.json",
            "$schema": "https://json-schema.org/draft/2020-12/schema",
        },
    )

    schema_version: Literal["2.0.0"]
    run_id: RunId
    source_id: SourceId
    forecast_component_id: ComponentId
    proposal_artifact_id: ArtifactId
    review_artifact_id: ArtifactId
    prompt_execution: P13PromptExecution
    input_hash: HashValue
    knowledge_cutoff: datetime
    final_status: Literal["verified", "unresolvable"]
    selected_candidate_ref: LocalRef | None = None
    selected_candidate_origin: Literal["p11_proposal", "p12_correction"] | None = None
    rationale: str = Field(min_length=1)
    unevaluable_reason: str | None = None

    @field_validator("knowledge_cutoff")
    @classmethod
    def validate_cutoff(cls, value: datetime) -> datetime:
        return _require_timezone(value)

    @model_validator(mode="after")
    def validate_final_status(self) -> P13Output:
        if self.final_status == "verified" and not self.selected_candidate_ref:
            raise ValueError("verifiedにはselected_candidate_refが必要です")
        if self.final_status == "verified" and self.selected_candidate_origin is None:
            object.__setattr__(self, "selected_candidate_origin", "p11_proposal")
        if self.final_status == "unresolvable":
            if self.selected_candidate_ref is not None:
                raise ValueError("unresolvableではcandidateを選択できません")
            if not self.unevaluable_reason:
                raise ValueError("unresolvableにはunevaluable_reasonが必要です")
        return self


PipelineOutput = (
    P05Output | P06Output | P07Output | P08Output | P09Output | P11Output | P12Output | P13Output
)

PIPELINE_MODELS: dict[str, type[PipelineModel]] = {
    "P05": P05Output,
    "P06": P06Output,
    "P07": P07Output,
    "P08": P08Output,
    "P09": P09Output,
    "P11": P11Output,
    "P12": P12Output,
    "P13": P13Output,
}

PIPELINE_SCHEMA_FILENAMES = {
    "P05": "p05_speaker_processing.schema.json",
    "P06": "p06_speaker_review.schema.json",
    "P07": "p07_text_source_processing.schema.json",
    "P08": "p08_forecast_extraction_v2.schema.json",
    "P09": "p09_forecast_review.schema.json",
    "P11": "p11_target_resolution.schema.json",
    "P12": "p12_target_review.schema.json",
    "P13": "p13_target_adjudication.schema.json",
}


def pipeline_schema_path(prompt_id: str) -> Path:
    try:
        filename = PIPELINE_SCHEMA_FILENAMES[prompt_id]
    except KeyError as error:
        raise ValueError(f"未対応prompt_idです: {prompt_id}") from error
    return Path(__file__).with_name(filename)
