"""Round2-01: 対象解決状態機械の回帰。ingest経路のみで検証する。"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pytest
import test_ai_pipeline_v2 as pipe  # pytest adds tests/unit to path for same-dir imports

from analyst_forecast.application.ai_ingestion import AiIngestStatus, ingest_ai_output
from analyst_forecast.application.evaluation import evaluate_component
from analyst_forecast.application.settings import AppSettings
from analyst_forecast.application.workflow import refresh_workflow
from analyst_forecast.domain.market import MarketBar, MarketDataRequest, MarketSeries
from helpers_pipeline_v2 import import_locked_component


class _FixtureProvider:
    name = "fixture"

    def fetch(self, request: MarketDataRequest) -> MarketSeries:
        from datetime import UTC, datetime

        return MarketSeries(
            provider=self.name,
            symbol=request.symbol,
            currency=request.currency,
            adjustment_type="split_adjusted_ohlc",
            frequency="1d",
            retrieved_at=datetime(2026, 7, 20, tzinfo=UTC),
            bars=(
                MarketBar.from_prices(date(2026, 1, 13), "100", "102", high="103", low="99"),
                MarketBar.from_prices(date(2026, 4, 13), "108", "110", high="111", low="107"),
            ),
        )


def _ingest_through_p11(
    settings: AppSettings,
    run_result: Any,
    source_result: Any,
    tmp_path: Path,
    *,
    unresolvable: bool = False,
) -> tuple[Any, Any, Any]:
    _, p08 = pipe._ingest_p05_p08(settings, run_result, source_result, tmp_path)
    p11 = ingest_ai_output(
        settings,
        pipe._write(
            tmp_path,
            "p11.json",
            pipe._p11_payload(
                run_result.run_id,
                source_result.source_id,
                p08.component_ids[0],
                p08.output_hash,
                unresolvable=unresolvable,
            ),
        ),
    )
    assert p11.status is AiIngestStatus.ACCEPTED, p11.issues
    return p08, p11, refresh_workflow(settings, run_result.run_id)


def test_p08_then_recommends_run_p11(
    settings: AppSettings,
    run_result: Any,
    source_result: Any,
    tmp_path: Path,
) -> None:
    _, p08 = pipe._ingest_p05_p08(settings, run_result, source_result, tmp_path)
    assert p08.status is AiIngestStatus.ACCEPTED
    state = refresh_workflow(settings, run_result.run_id)
    assert state.recommended_action.action_id == "RUN_P11"
    assert p08.component_ids[0] in (state.recommended_action.command_or_prompt or "")
    assert "<component-id>" not in (state.recommended_action.command_or_prompt or "")


def test_p11_proposed_then_recommends_run_p12(
    settings: AppSettings,
    run_result: Any,
    source_result: Any,
    tmp_path: Path,
) -> None:
    p08, p11, state = _ingest_through_p11(settings, run_result, source_result, tmp_path)
    assert state.recommended_action.action_id == "RUN_P12"
    assert state.recommended_action.action_id != "RUN_P11"
    assert p08.component_ids[0] in (state.recommended_action.command_or_prompt or "")
    assert p11.artifact_ids[0] in (state.recommended_action.command_or_prompt or "")


def test_p11_unresolvable_still_recommends_run_p12(
    settings: AppSettings,
    run_result: Any,
    source_result: Any,
    tmp_path: Path,
) -> None:
    _, _, state = _ingest_through_p11(
        settings, run_result, source_result, tmp_path, unresolvable=True
    )
    assert state.recommended_action.action_id == "RUN_P12"


def test_p12_agreed_then_evaluate_market(
    settings: AppSettings,
    run_result: Any,
    source_result: Any,
    tmp_path: Path,
) -> None:
    component_id = import_locked_component(settings, run_result, source_result, tmp_path)
    state = refresh_workflow(settings, run_result.run_id)
    assert state.recommended_action.action_id == "EVALUATE_MARKET"
    assert component_id in (state.recommended_action.command_or_prompt or "")
    assert "<component-id>" not in (state.recommended_action.command_or_prompt or "")


def test_p12_disagreed_then_run_p13_not_evaluate(
    settings: AppSettings,
    run_result: Any,
    source_result: Any,
    tmp_path: Path,
) -> None:
    p08, p11, _ = _ingest_through_p11(settings, run_result, source_result, tmp_path)
    p12 = ingest_ai_output(
        settings,
        pipe._write(
            tmp_path,
            "p12-dis.json",
            pipe._p12_payload(
                run_result.run_id,
                source_result.source_id,
                p08.component_ids[0],
                p11.artifact_ids[0],
                p11.output_hash,
                resolution_status="disagreed",
            ),
        ),
    )
    assert p12.status is AiIngestStatus.ACCEPTED, p12.issues
    state = refresh_workflow(settings, run_result.run_id)
    assert state.recommended_action.action_id == "RUN_P13"
    assert state.recommended_action.action_id != "EVALUATE_MARKET"
    assert p08.component_ids[0] in (state.recommended_action.command_or_prompt or "")
    assert p11.artifact_ids[0] in (state.recommended_action.command_or_prompt or "")
    assert p12.artifact_ids[0] in (state.recommended_action.command_or_prompt or "")


def test_p13_verified_then_evaluate_market(
    settings: AppSettings,
    run_result: Any,
    source_result: Any,
    tmp_path: Path,
) -> None:
    p08, p11, _ = _ingest_through_p11(settings, run_result, source_result, tmp_path)
    p12 = ingest_ai_output(
        settings,
        pipe._write(
            tmp_path,
            "p12-dis.json",
            pipe._p12_payload(
                run_result.run_id,
                source_result.source_id,
                p08.component_ids[0],
                p11.artifact_ids[0],
                p11.output_hash,
                resolution_status="disagreed",
            ),
        ),
    )
    assert p12.status is AiIngestStatus.ACCEPTED
    p13 = ingest_ai_output(
        settings,
        pipe._write(
            tmp_path,
            "p13.json",
            pipe._p13_payload(
                run_result.run_id,
                source_result.source_id,
                p08.component_ids[0],
                p11.artifact_ids[0],
                p12.artifact_ids[0],
                p12.output_hash,
            ),
        ),
    )
    assert p13.status is AiIngestStatus.ACCEPTED, p13.issues
    state = refresh_workflow(settings, run_result.run_id)
    assert state.recommended_action.action_id == "EVALUATE_MARKET"


def test_p12_unresolved_does_not_loop_p11_or_market_fetch(
    settings: AppSettings,
    run_result: Any,
    source_result: Any,
    tmp_path: Path,
) -> None:
    p08, p11, _ = _ingest_through_p11(settings, run_result, source_result, tmp_path)
    p12 = ingest_ai_output(
        settings,
        pipe._write(
            tmp_path,
            "p12-un.json",
            pipe._p12_payload(
                run_result.run_id,
                source_result.source_id,
                p08.component_ids[0],
                p11.artifact_ids[0],
                p11.output_hash,
                resolution_status="unresolved",
                candidate_ref=None,
            ),
        ),
    )
    assert p12.status is AiIngestStatus.ACCEPTED, p12.issues
    state = refresh_workflow(settings, run_result.run_id)
    assert state.recommended_action.action_id not in {"RUN_P11", "EVALUATE_MARKET"}
    assert state.recommended_action.action_id in {
        "REVIEW_UNRESOLVABLE",
        "REVIEW_RESULTS",
        "SUPPLY_MARKET_CSV",
    }


def test_unlocked_component_market_evaluate_is_rejected(
    settings: AppSettings,
    run_result: Any,
    source_result: Any,
    tmp_path: Path,
) -> None:
    _, p08 = pipe._ingest_p05_p08(settings, run_result, source_result, tmp_path)
    with pytest.raises(ValueError, match="mapping固定"):
        evaluate_component(
            settings,
            component_id=p08.component_ids[0],
            provider=_FixtureProvider(),
            as_of=date(2026, 4, 13),
            run_id=run_result.run_id,
        )


def test_multi_component_shows_remaining_stage(
    settings: AppSettings,
    run_result: Any,
    source_result: Any,
    tmp_path: Path,
) -> None:
    from datetime import UTC, datetime

    from analyst_forecast.application.raw_sources import RawSourceRequest, import_raw_source
    from analyst_forecast.domain.models import Medium
    from conftest import RAW_TEXT

    locked = import_locked_component(settings, run_result, source_result, tmp_path, label="done")
    evaluate_component(
        settings,
        component_id=locked,
        provider=_FixtureProvider(),
        as_of=date(2026, 4, 13),
        run_id=run_result.run_id,
    )
    second_raw = tmp_path / "second.txt"
    second_raw.write_text(RAW_TEXT + "追加", encoding="utf-8")
    second = import_raw_source(
        settings,
        RawSourceRequest(
            run_id=run_result.run_id,
            input_path=second_raw,
            medium=Medium.YOUTUBE,
            url="https://example.invalid/second-sm",
            published_at=datetime(2026, 1, 15, tzinfo=UTC),
            retrieved_at=datetime(2026, 7, 20, tzinfo=UTC),
        ),
    )
    (tmp_path / "second_pipe").mkdir(exist_ok=True)
    _, p08 = pipe._ingest_p05_p08(settings, run_result, second, tmp_path / "second_pipe")
    state = refresh_workflow(settings, run_result.run_id)
    assert state.recommended_action.action_id == "RUN_P11"
    assert p08.component_ids[0] in (state.recommended_action.command_or_prompt or "")
    assert locked not in (state.recommended_action.command_or_prompt or "")
