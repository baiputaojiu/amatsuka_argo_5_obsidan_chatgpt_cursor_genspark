"""Fix 07 縦断シナリオの最小回帰。"""

from __future__ import annotations

from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

from analyst_forecast.application.bootstrap import initialize_workspace
from analyst_forecast.application.evaluation import evaluate_component
from analyst_forecast.application.raw_sources import RawSourceRequest, import_raw_source
from analyst_forecast.application.runs import CreateRunRequest, create_run
from analyst_forecast.application.settings import AppSettings
from analyst_forecast.application.workflow import refresh_workflow
from analyst_forecast.domain.market import MarketBar, MarketDataRequest, MarketSeries
from analyst_forecast.domain.models import Medium
from conftest import RAW_TEXT
from helpers_pipeline_v2 import import_locked_component


class ScenarioProvider:
    name = "scenario-fixture"

    def __init__(self, direction: str = "up") -> None:
        self.direction = direction

    def fetch(self, request: MarketDataRequest) -> MarketSeries:
        if self.direction == "down":
            bars = (
                MarketBar.from_prices(date(2026, 1, 13), "100", "102", high="104", low="98"),
                MarketBar.from_prices(date(2026, 4, 13), "94", "92", high="104", low="88"),
            )
        elif self.direction == "flat":
            bars = (
                MarketBar.from_prices(date(2026, 1, 13), "100", "100", high="101", low="99"),
                MarketBar.from_prices(date(2026, 4, 13), "100", "100", high="101", low="99"),
            )
        else:
            bars = (
                MarketBar.from_prices(date(2026, 1, 13), "100", "102", high="104", low="98"),
                MarketBar.from_prices(date(2026, 4, 13), "108", "110", high="112", low="107"),
            )
        return MarketSeries(
            provider=self.name,
            symbol=request.symbol,
            currency=request.currency,
            adjustment_type="split_adjusted_ohlc",
            frequency="1d",
            retrieved_at=datetime(2026, 7, 20, tzinfo=UTC),
            bars=bars,
        )


def test_scenario_a_vertical_with_forecast(tmp_path: Path) -> None:
    vault = tmp_path / "scenario-a"
    settings = AppSettings(
        vault_root=vault,
        database_path=vault / "_system" / "database.sqlite",
        cursor_model="fixture-model",
        chatgpt_model="fixture-model",
    )
    initialize_workspace(settings, config_path=tmp_path / "config.yaml")
    assert (vault / "prompts" / "catalog.json").is_file()
    run = create_run(
        settings,
        CreateRunRequest(
            canonical_name="匿名シナリオA",
            period_start=date(2026, 1, 1),
            period_end=date(2026, 6, 30),
            evaluation_as_of=date(2026, 7, 20),
            selected_media=[Medium.YOUTUBE],
        ),
        now=datetime(2026, 7, 20, tzinfo=UTC),
    )
    raw = tmp_path / "raw.txt"
    raw.write_text(RAW_TEXT, encoding="utf-8")
    source = import_raw_source(
        settings,
        RawSourceRequest(
            run_id=run.run_id,
            input_path=raw,
            medium=Medium.YOUTUBE,
            url="https://example.invalid/a",
            published_at=datetime(2026, 1, 10, tzinfo=UTC),
            retrieved_at=datetime(2026, 7, 20, tzinfo=UTC),
        ),
    )
    run_ns = SimpleNamespace(run_id=run.run_id, run_path=run.run_path)
    component_id = import_locked_component(settings, run_ns, source, tmp_path)
    state = refresh_workflow(settings, run.run_id)
    assert component_id in (state.recommended_action.command_or_prompt or "")
    evaluate_component(
        settings,
        component_id=component_id,
        provider=ScenarioProvider("up"),
        as_of=date(2026, 4, 13),
        run_id=run.run_id,
    )
    final = refresh_workflow(settings, run.run_id)
    assert (run.run_path / "04_results" / "reports" / "vertical_mvp_summary.md").is_file()
    assert final.recommended_action.action_id == "REVIEW_RESULTS"


def test_scenario_g_direction_mfe(tmp_path: Path) -> None:
    vault = tmp_path / "scenario-g"
    settings = AppSettings(
        vault_root=vault,
        database_path=vault / "_system" / "database.sqlite",
        cursor_model="fixture-model",
        chatgpt_model="fixture-model",
    )
    initialize_workspace(settings, config_path=tmp_path / "config.yaml")
    run = create_run(
        settings,
        CreateRunRequest(
            canonical_name="匿名シナリオG",
            period_start=date(2026, 1, 1),
            period_end=date(2026, 6, 30),
            evaluation_as_of=date(2026, 7, 20),
            selected_media=[Medium.YOUTUBE],
        ),
        now=datetime(2026, 7, 20, tzinfo=UTC),
    )
    raw = tmp_path / "raw.txt"
    raw.write_text(RAW_TEXT, encoding="utf-8")
    source = import_raw_source(
        settings,
        RawSourceRequest(
            run_id=run.run_id,
            input_path=raw,
            medium=Medium.YOUTUBE,
            url="https://example.invalid/g",
            published_at=datetime(2026, 1, 10, tzinfo=UTC),
            retrieved_at=datetime(2026, 7, 20, tzinfo=UTC),
        ),
    )
    run_ns = SimpleNamespace(run_id=run.run_id, run_path=run.run_path)
    component_id = import_locked_component(settings, run_ns, source, tmp_path, direction="down")
    result = evaluate_component(
        settings,
        component_id=component_id,
        provider=ScenarioProvider("down"),
        as_of=date(2026, 4, 13),
        run_id=run.run_id,
    )
    assert result.direction_result == "hit"
    assert result.max_favorable_excursion == Decimal("0.12")
    assert result.max_adverse_excursion == Decimal("0.04")
