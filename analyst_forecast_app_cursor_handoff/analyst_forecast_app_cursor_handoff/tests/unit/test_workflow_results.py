import json
from datetime import UTC, date, datetime
from pathlib import Path

from sqlalchemy import select

from analyst_forecast.application.ai_ingestion import ingest_ai_output
from analyst_forecast.application.evaluation import evaluate_component
from analyst_forecast.application.raw_sources import RawSourceRequest, import_raw_source
from analyst_forecast.application.runs import CreateRunRequest, create_run
from analyst_forecast.application.settings import AppSettings
from analyst_forecast.application.workflow import (
    assert_component_belongs_to_run,
    refresh_workflow,
)
from analyst_forecast.domain.market import MarketBar, MarketDataRequest, MarketSeries
from analyst_forecast.domain.models import Medium
from analyst_forecast.infrastructure.db.models import AiArtifactRecord
from analyst_forecast.infrastructure.db.session import create_session_factory
from conftest import RAW_TEXT, make_ai_payload
from helpers_pipeline_v2 import import_locked_component


class FixtureProvider:
    name = "fixture"

    def fetch(self, request: MarketDataRequest) -> MarketSeries:
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


def test_workflow_shows_real_component_ids_and_results(
    settings: AppSettings,
    run_result,
    source_result,
    tmp_path: Path,
) -> None:
    component_id = import_locked_component(settings, run_result, source_result, tmp_path)
    state = refresh_workflow(settings, run_result.run_id)

    assert component_id in (state.recommended_action.command_or_prompt or "")
    assert "<component-id>" not in (state.recommended_action.command_or_prompt or "")

    evaluate_component(
        settings,
        component_id=component_id,
        provider=FixtureProvider(),
        as_of=date(2026, 4, 13),
        run_id=run_result.run_id,
    )
    complete = refresh_workflow(settings, run_result.run_id)
    run_path = run_result.run_path
    assert (run_path / "04_results" / "forecasts" / "all_forecasts.md").is_file()
    assert (run_path / "04_results" / "tables" / "all_forecasts.csv").is_file()
    assert (run_path / "04_results" / "evaluations" / "evaluations.md").is_file()
    assert (run_path / "04_results" / "tables" / "evaluations.csv").is_file()
    assert (run_path / "04_results" / "reports" / "vertical_mvp_summary.md").is_file()
    assert complete.recommended_action.action_id == "REVIEW_RESULTS"


def test_foreign_component_is_rejected(
    settings: AppSettings,
    run_result,
    source_result,
    tmp_path: Path,
) -> None:
    component_id = import_locked_component(settings, run_result, source_result, tmp_path)
    other = create_run(
        settings,
        CreateRunRequest(
            canonical_name="別案件",
            period_start=date(2026, 1, 1),
            period_end=date(2026, 6, 30),
            evaluation_as_of=date(2026, 7, 20),
            selected_media=[Medium.YOUTUBE],
        ),
        now=datetime(2026, 7, 23, tzinfo=UTC),
    )
    try:
        assert_component_belongs_to_run(
            settings,
            run_id=other.run_id,
            component_id=component_id,
        )
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_needs_review_uses_db_not_leftover_files(
    settings: AppSettings,
    run_result,
    source_result,
    tmp_path: Path,
) -> None:
    low = make_ai_payload(
        run_id=run_result.run_id,
        source_id=source_result.source_id,
        confidence=0.4,
    )
    low_path = tmp_path / "low.json"
    low_path.write_text(json.dumps(low, ensure_ascii=False), encoding="utf-8")
    ingest_ai_output(settings, low_path)
    leftover = run_result.run_path / "03_ai_outputs" / "needs_review" / "stale.json"
    leftover.write_text("{}", encoding="utf-8")

    fixed = make_ai_payload(
        run_id=run_result.run_id,
        source_id=source_result.source_id,
        confidence=0.95,
        forecast_ref="forecast-fixed",
        group_ref="group-fixed",
    )
    fixed_path = tmp_path / "fixed.json"
    fixed_path.write_text(json.dumps(fixed, ensure_ascii=False), encoding="utf-8")
    session_factory = create_session_factory(settings.database_file)
    with session_factory.begin() as session:
        for artifact in session.scalars(
            select(AiArtifactRecord).where(AiArtifactRecord.run_id == run_result.run_id)
        ):
            if artifact.classification == "needs_review":
                artifact.resolution_status = "resolved"
    ingested = ingest_ai_output(settings, fixed_path)
    state = refresh_workflow(settings, run_result.run_id)
    assert leftover.is_file()
    assert state.counts["needs_review"] == 0
    assert ingested.component_ids or state.recommended_action.action_id != "REVIEW_AI_OUTPUT"


def test_unevaluated_component_not_hidden_by_multi_as_of(
    settings: AppSettings,
    run_result,
    source_result,
    tmp_path: Path,
) -> None:
    component_a = import_locked_component(
        settings, run_result, source_result, tmp_path, label="a"
    )

    second_raw = tmp_path / "second.txt"
    second_raw.write_text(RAW_TEXT + "追加", encoding="utf-8")
    second_source = import_raw_source(
        settings,
        RawSourceRequest(
            run_id=run_result.run_id,
            input_path=second_raw,
            medium=Medium.YOUTUBE,
            url="https://example.invalid/second",
            published_at=datetime(2026, 1, 15, tzinfo=UTC),
            retrieved_at=datetime(2026, 7, 20, tzinfo=UTC),
        ),
    )
    component_b = import_locked_component(
        settings,
        run_result,
        second_source,
        tmp_path,
        label="b",
    )

    evaluate_component(
        settings,
        component_id=component_a,
        provider=FixtureProvider(),
        as_of=date(2026, 3, 13),
        run_id=run_result.run_id,
    )
    evaluate_component(
        settings,
        component_id=component_a,
        provider=FixtureProvider(),
        as_of=date(2026, 4, 13),
        run_id=run_result.run_id,
    )
    state = refresh_workflow(settings, run_result.run_id)
    assert component_b in (state.recommended_action.command_or_prompt or "")


def test_latest_success_overrides_past_unevaluable(
    settings: AppSettings,
    run_result,
    source_result,
    tmp_path: Path,
) -> None:
    component_id = import_locked_component(settings, run_result, source_result, tmp_path)

    class Boom:
        name = "boom"

        def fetch(self, request: MarketDataRequest) -> MarketSeries:
            raise __import__(
                "analyst_forecast.domain.market", fromlist=["MarketDataUnavailable"]
            ).MarketDataUnavailable("一時取得不能")

    evaluate_component(
        settings,
        component_id=component_id,
        provider=Boom(),
        as_of=date(2026, 3, 13),
        run_id=run_result.run_id,
    )
    evaluate_component(
        settings,
        component_id=component_id,
        provider=FixtureProvider(),
        as_of=date(2026, 4, 13),
        run_id=run_result.run_id,
    )
    state = refresh_workflow(settings, run_result.run_id)
    assert state.counts["unevaluable"] == 0
    assert state.counts["unevaluated_components"] == 0
