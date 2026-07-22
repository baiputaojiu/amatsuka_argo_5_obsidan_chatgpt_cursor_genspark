"""Round6 P09 fixed Schema vs Pydantic contract matrix."""

from __future__ import annotations

import json
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
        "decision": "reject",
        "findings": [],
        "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
    }
    payload.update(kwargs)
    return payload


def _schema_ok(payload: dict[str, object]) -> bool:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    Draft202012Validator.check_schema(schema)
    return not list(Draft202012Validator(schema).iter_errors(payload))


def _pydantic_ok(payload: dict[str, object]) -> bool:
    try:
        P09Output.model_validate(payload)
        return True
    except ValidationError:
        return False


@pytest.mark.parametrize(
    ("payload", "expect_valid"),
    [
        pytest.param(
            _base(reject_disposition="retryable", reject_reason="needs reextract"),
            True,
            id="r6-028-valid-retryable",
        ),
        pytest.param(
            _base(reject_disposition="terminal", reject_reason="unsalvageable"),
            True,
            id="r6-028-valid-terminal",
        ),
        pytest.param(_base(), False, id="r6-028-missing-both"),
        pytest.param(
            _base(reject_reason="only reason"),
            False,
            id="r6-028-missing-disposition",
        ),
        pytest.param(
            _base(reject_disposition="retryable"),
            False,
            id="r6-028-missing-reason",
        ),
        pytest.param(
            _base(reject_disposition="retryable", reject_reason="   "),
            False,
            id="r6-035-blank-reason",
        ),
        pytest.param(
            _base(
                reject_disposition="terminal",
                reject_reason="ok",
                reject_terminal=True,
            ),
            False,
            id="r6-029-reject-terminal-on-2-1",
        ),
        pytest.param(
            _base(
                decision="accept",
                reject_disposition="retryable",
                reject_reason="nope",
            ),
            False,
            id="r6-029-accept-with-reject-fields",
        ),
        pytest.param(
            {
                **_base(
                    schema_version="2.0.0",
                    reject_terminal=True,
                    reject_reason="legacy terminal",
                ),
                "reject_disposition": None,
            },
            True,
            id="r6-031-legacy-terminal-valid",
        ),
        pytest.param(
            {
                **_base(
                    schema_version="2.0.0",
                    reject_terminal=False,
                    reject_reason="legacy retryable",
                )
            },
            True,
            id="r6-032-legacy-retryable-valid",
        ),
        pytest.param(
            {**_base(schema_version="2.0.0"), "reject_reason": "no terminal"},
            False,
            id="r6-031-legacy-missing-terminal",
        ),
        pytest.param(
            {
                **_base(schema_version="2.0.0"),
                "reject_disposition": "terminal",
                "reject_reason": "disposition only",
            },
            False,
            id="r6-033-legacy-disposition-only",
        ),
        pytest.param(
            {
                **_base(schema_version="2.0.0"),
                "reject_terminal": True,
                "reject_disposition": "terminal",
                "reject_reason": "mixed consistent",
            },
            False,
            id="r6-033-legacy-mixed-consistent",
        ),
    ],
)
def test_r6_schema_pydantic_agree(payload: dict[str, object], expect_valid: bool) -> None:
    # Drop explicit null reject_disposition keys that confuse fixed Schema presence checks
    cleaned = {k: v for k, v in payload.items() if v is not None}
    schema_valid = _schema_ok(cleaned)
    pydantic_valid = _pydantic_ok(cleaned)
    assert schema_valid is expect_valid
    assert pydantic_valid is expect_valid
    assert schema_valid == pydantic_valid


def test_r6_032_legacy_adapter_converts_disposition() -> None:
    model = P09Output.model_validate(
        {
            "schema_version": "2.0.0",
            "run_id": "RUN-20260101-001",
            "source_id": "SRC-000001",
            "reviewed_artifact_id": "AIF-000001",
            "prompt_execution": {
                "prompt_id": "P09",
                "prompt_version": "2.0.0",
                "environment": "cursor",
                "model": "fixture",
                "executed_at": "2026-07-20T12:00:00+00:00",
            },
            "input_hash": "a" * 64,
            "decision": "reject",
            "findings": [],
            "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
            "reject_terminal": False,
            "reject_reason": "retry please",
        }
    )
    assert model.reject_disposition == "retryable"
    assert model.is_reject_terminal is False


def test_r6_037_repo_schema_hash_stable() -> None:
    assert SCHEMA_PATH.is_file()
    text = SCHEMA_PATH.read_text(encoding="utf-8")
    assert "allOf" in text
    assert "reject_disposition" in text
