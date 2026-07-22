"""Round7 P09 Schema/Pydantic alignment and prompt example validation."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from analyst_forecast.schemas.pipeline import P09Output

SCHEMA_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "analyst_forecast"
    / "schemas"
    / "p09_forecast_review.schema.json"
)
PROMPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "analyst_forecast"
    / "resources"
    / "prompts"
    / "P09.md.j2"
)


def _base(**kwargs: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": "2.1.0",
        "run_id": "RUN-20260101-001",
        "source_id": "SRC-000001",
        "reviewed_artifact_id": "AIF-000001",
        "prompt_execution": {
            "prompt_id": "P09",
            "prompt_version": "2.1.0",
            "environment": "cursor",
            "model": "fixture",
            "executed_at": "2026-07-20T12:00:00+00:00",
        },
        "input_hash": "a" * 64,
        "decision": "accept",
        "findings": [],
        "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
    }
    payload.update(kwargs)
    return payload


def _schema_ok(payload: dict[str, object]) -> bool:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    return not list(Draft202012Validator(schema).iter_errors(payload))


def _pydantic_ok(payload: dict[str, object]) -> bool:
    try:
        P09Output.model_validate(payload)
        return True
    except ValidationError:
        return False


@pytest.mark.parametrize(
    ("extra", "expect_valid"),
    [
        pytest.param({}, True, id="accept-without-operations"),
        pytest.param({"forecast_operations": None}, True, id="accept-null-operations"),
        pytest.param({"forecast_operations": []}, False, id="accept-empty-operations"),
    ],
)
def test_r7_accept_forecast_operations_unified(
    extra: dict[str, object], expect_valid: bool
) -> None:
    payload = _base(**extra)
    assert _schema_ok(payload) is expect_valid
    assert _pydantic_ok(payload) is expect_valid
    assert _schema_ok(payload) == _pydantic_ok(payload)


def _extract_prompt_json_blocks() -> list[dict[str, object]]:
    text = PROMPT_PATH.read_text(encoding="utf-8")
    blocks: list[dict[str, object]] = []
    for match in re.finditer(r"```json\s*\n(.*?)\n```", text, re.DOTALL):
        blocks.append(json.loads(match.group(1)))
    return blocks


def test_r7_p09_prompt_examples_validate_against_fixed_schema() -> None:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    validator = Draft202012Validator(schema)
    blocks = _extract_prompt_json_blocks()
    assert len(blocks) >= 4
    for index, payload in enumerate(blocks):
        errors = list(validator.iter_errors(payload))
        assert not errors, f"example {index}: {[error.message for error in errors]}"


def test_r7_correct_example_has_full_p08_fields() -> None:
    blocks = _extract_prompt_json_blocks()
    correct = next(item for item in blocks if item.get("decision") == "correct")
    corrected = correct["corrected_payload"]
    assert isinstance(corrected, dict)
    for key in (
        "schema_version",
        "run_id",
        "source_id",
        "upstream_artifact_id",
        "upstream_prompt_id",
        "prompt_execution",
        "input_hash",
        "knowledge_cutoff",
        "processing_status",
        "forecasts",
    ):
        assert key in corrected
    forecast = corrected["forecasts"][0]
    for key in (
        "forecast_ref",
        "made_at",
        "evidence",
        "components",
        "speaker_candidate",
        "upstream_segment_refs",
    ):
        assert key in forecast
