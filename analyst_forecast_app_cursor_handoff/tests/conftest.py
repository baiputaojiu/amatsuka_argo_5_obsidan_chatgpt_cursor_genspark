from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from analyst_forecast.application.bootstrap import initialize_workspace
from analyst_forecast.application.raw_sources import RawSourceRequest, import_raw_source
from analyst_forecast.application.runs import CreateRunRequest, create_run
from analyst_forecast.application.settings import AppSettings
from analyst_forecast.domain.models import Medium

RAW_TEXT = "日経平均は今後上昇する。これは現状分析ではなく予想です。"


@pytest.fixture
def settings(tmp_path: Path) -> AppSettings:
    value = AppSettings(
        vault_root=tmp_path / "vault",
        database_path=tmp_path / "vault" / "_system" / "database.sqlite",
        cursor_model="high-performance-fixture",
        chatgpt_model="high-performance-fixture",
    )
    initialize_workspace(value, config_path=tmp_path / "config.local.yaml")
    return value


@pytest.fixture
def run_result(settings: AppSettings) -> Any:
    request = CreateRunRequest(
        canonical_name="匿名アナリストA",
        period_start=date(2026, 1, 1),
        period_end=date(2026, 6, 30),
        evaluation_as_of=date(2026, 7, 20),
        selected_media=[Medium.YOUTUBE],
        focus_targets=["日経平均"],
    )
    return create_run(
        settings,
        request,
        now=datetime(2026, 7, 20, 12, 0, tzinfo=UTC),
    )


@pytest.fixture
def source_result(settings: AppSettings, run_result: Any, tmp_path: Path) -> Any:
    source_path = tmp_path / "source.txt"
    source_path.write_text(RAW_TEXT, encoding="utf-8")
    request = RawSourceRequest(
        run_id=run_result.run_id,
        input_path=source_path,
        medium=Medium.YOUTUBE,
        url="https://example.invalid/video/fixture",
        title="匿名fixture",
        recorded_at=datetime(2026, 1, 10, 9, 0, tzinfo=UTC),
        published_at=datetime(2026, 1, 10, 10, 0, tzinfo=UTC),
        retrieved_at=datetime(2026, 7, 20, 12, 30, tzinfo=UTC),
    )
    return import_raw_source(settings, request)


def make_ai_payload(
    *,
    run_id: str,
    source_id: str,
    raw_text: str = RAW_TEXT,
    confidence: float = 0.95,
    direction: str = "up",
    forecast_ref: str = "forecast-1",
    group_ref: str = "group-1",
) -> dict[str, Any]:
    quote = "日経平均は今後上昇する"
    start = raw_text.index(quote)
    end = start + len(quote)
    return {
        "schema_version": "1.0.0",
        "run_id": run_id,
        "source_id": source_id,
        "prompt_execution": {
            "prompt_id": "P08",
            "prompt_version": "1.0.0",
            "environment": "cursor",
            "model": "high-performance-fixture",
        },
        "forecasts": [
            {
                "forecast_ref": forecast_ref,
                "forecast_group_ref": group_ref,
                "made_at": "2026-01-10T09:00:00+00:00",
                "publicly_available_at": "2026-01-10T10:00:00+00:00",
                "forecast_type": "directional",
                "commitment_strength": "explicit",
                "evidence_level": "A",
                "extraction_confidence": confidence,
                "human_readable_summary": "日経平均は今後上昇する",
                "relation_to_previous": "initial",
                "evidence": [
                    {
                        "source_id": source_id,
                        "quote": quote,
                        "start_offset": start,
                        "end_offset": end,
                        "role": "prediction",
                    }
                ],
                "components": [
                    {
                        "component_ref": "component-1",
                        "sequence_number": 1,
                        "prediction_form": "period_direction",
                        "direction": direction,
                        "time_expression_raw": "今後3か月",
                        "time_source": "explicit",
                        "normalized_start": "2026-01-13",
                        "normalized_end": "2026-04-13",
                        "target": {
                            "raw_label": "日経平均",
                            "canonical_name": "日経平均株価",
                            "target_type": "index",
                            "symbol": "^N225",
                            "exchange": "JPX",
                            "currency": "JPY",
                            "mapping_method": "explicit",
                            "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
                            "source_evidence": "原文で予測対象として明示",
                            "proposal_model": "high-performance-fixture",
                            "mapping_status": "verified",
                            "review_result": "別の高性能AIが原文・時点・symbolを確認済み",
                        },
                    }
                ],
            }
        ],
    }
