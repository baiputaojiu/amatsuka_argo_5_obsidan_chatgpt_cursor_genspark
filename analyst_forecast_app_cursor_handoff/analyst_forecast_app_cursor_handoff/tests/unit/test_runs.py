from datetime import UTC, date, datetime
from pathlib import Path

import yaml

from analyst_forecast.application.runs import CreateRunRequest, create_run
from analyst_forecast.application.settings import AppSettings
from analyst_forecast.domain.models import Medium


def test_create_run_issues_ids_and_required_tree(
    settings: AppSettings,
    run_result,
) -> None:
    assert run_result.analyst_id == "A0001"
    assert run_result.run_id == "RUN-20260720-001"
    assert run_result.run_path.name.endswith("__20260101_20260630")

    required_files = {
        "request.yaml",
        "status.yaml",
        "WORKFLOW_STATE.json",
        "NEXT_ACTIONS.md",
        "OPEN_ISSUES.md",
        "README.md",
    }
    assert required_files <= {path.name for path in run_result.run_path.iterdir()}

    for medium in ("youtube", "blog", "x", "web"):
        for category in ("raw", "processed", "metadata"):
            assert (run_result.run_path / "02_sources" / medium / category).is_dir()

    prompt_files = {path.name for path in (run_result.run_path / "01_prompts").glob("*.md")}
    assert len(prompt_files) == 10
    assert any(name.startswith("P05") and "cursor" in name for name in prompt_files)
    assert any(name.startswith("P12") and "chatgpt" in name for name in prompt_files)
    assert any(name.startswith("P13") and "cursor" in name for name in prompt_files)
    assert (
        run_result.run_path / "01_prompts" / "schemas" / "forecast_extraction.schema.json"
    ).is_file()
    for schema_name in (
        "p05_speaker_processing.schema.json",
        "p08_forecast_extraction_v2.schema.json",
        "p11_target_resolution.schema.json",
        "p12_target_review.schema.json",
        "p13_target_adjudication.schema.json",
    ):
        assert (run_result.run_path / "01_prompts" / "schemas" / schema_name).is_file()

    p08_prompt = next((run_result.run_path / "01_prompts").glob("P08*cursor.md"))
    p11_prompt = next((run_result.run_path / "01_prompts").glob("P11*cursor.md"))
    p12_prompt = next((run_result.run_path / "01_prompts").glob("P12*cursor.md"))
    assert "p08_forecast_extraction_v2.schema.json" in p08_prompt.read_text(encoding="utf-8")
    assert "03_ai_outputs/inbox/P08_" in p08_prompt.read_text(encoding="utf-8")
    assert "市場結果を入力に含めない" in p11_prompt.read_text(encoding="utf-8")
    assert "p12_target_review.schema.json" in p12_prompt.read_text(encoding="utf-8")

    request = yaml.safe_load((run_result.run_path / "request.yaml").read_text(encoding="utf-8"))
    assert request["analyst_id"] == "A0001"
    assert request["run_id"] == "RUN-20260720-001"
    assert request["selected_media"] == ["youtube"]


def test_same_analyst_can_have_overlapping_runs(settings: AppSettings, run_result) -> None:
    second = create_run(
        settings,
        CreateRunRequest(
            canonical_name="匿名アナリストA",
            period_start=date(2026, 3, 1),
            period_end=date(2026, 6, 30),
            evaluation_as_of=date(2026, 7, 20),
            selected_media=[Medium.BLOG],
        ),
        now=datetime(2026, 7, 20, 13, 0, tzinfo=UTC),
    )

    assert second.analyst_id == run_result.analyst_id
    assert second.run_id == "RUN-20260720-002"
    assert second.run_path != run_result.run_path


def test_user_name_is_made_windows_safe(settings: AppSettings) -> None:
    result = create_run(
        settings,
        CreateRunRequest(
            canonical_name='匿名:対象/者*?"<>|',
            period_start=date(2026, 1, 1),
            period_end=date(2026, 1, 31),
            evaluation_as_of=date(2026, 2, 1),
            selected_media=[Medium.WEB],
        ),
        now=datetime(2026, 7, 21, tzinfo=UTC),
    )

    forbidden = set('<>:"/\\|?*')
    assert not forbidden.intersection(result.run_path.parent.name)


def test_workspace_config_snapshot_has_no_fixture_absolute_path(
    settings: AppSettings,
) -> None:
    snapshot = settings.vault_root / "_system" / "config.yaml"
    assert snapshot.is_file()
    assert str(Path.home()) not in snapshot.read_text(encoding="utf-8")
