import json
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import yaml

from analyst_forecast.application.evaluation import evaluate_component
from analyst_forecast.application.settings import AppSettings
from analyst_forecast.application.workflow import refresh_workflow
from analyst_forecast.domain.market import MarketBar, MarketDataRequest, MarketSeries
from helpers_pipeline_v2 import import_locked_component


class WorkflowFixtureProvider:
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


def test_workflow_guides_each_vertical_stage(
    settings: AppSettings,
    run_result,
    source_result,
    tmp_path: Path,
) -> None:
    initial = refresh_workflow(settings, run_result.run_id)
    assert initial.recommended_action.executor == "[AI Cursor]"
    assert "予想抽出" in initial.recommended_action.title or "P05" in (
        initial.recommended_action.command_or_prompt or ""
    )

    component_id = import_locked_component(settings, run_result, source_result, tmp_path)

    after_ai = refresh_workflow(settings, run_result.run_id)
    assert after_ai.recommended_action.executor == "[PYTHON]"
    assert "市場" in after_ai.recommended_action.title
    assert after_ai.recommended_action.inputs
    assert after_ai.recommended_action.outputs
    assert after_ai.recommended_action.reason
    assert component_id in (after_ai.recommended_action.command_or_prompt or "")

    evaluated = evaluate_component(
        settings,
        component_id=component_id,
        provider=WorkflowFixtureProvider(),
        as_of=date(2026, 4, 13),
        run_id=run_result.run_id,
    )
    assert evaluated.actual_return == Decimal("0.1")

    complete = refresh_workflow(settings, run_result.run_id)
    assert complete.recommended_action.executor == "[USER]"
    assert "確認" in complete.recommended_action.title
    assert len(complete.alternatives) <= 2

    run_path = run_result.run_path
    yaml_state = yaml.safe_load((run_path / "status.yaml").read_text(encoding="utf-8"))
    json_state = json.loads((run_path / "WORKFLOW_STATE.json").read_text(encoding="utf-8"))
    next_actions = (run_path / "NEXT_ACTIONS.md").read_text(encoding="utf-8")
    open_issues = (run_path / "OPEN_ISSUES.md").read_text(encoding="utf-8")

    action_id = complete.recommended_action.action_id
    assert yaml_state["recommended_action"]["action_id"] == action_id
    assert json_state["recommended_action"]["action_id"] == action_id
    assert action_id in next_actions
    assert "ブロッカー" in open_issues
