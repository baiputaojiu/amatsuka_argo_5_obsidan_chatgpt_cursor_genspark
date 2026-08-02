import re
from pathlib import Path

from typer.testing import CliRunner

from analyst_forecast.cli.app import app

FIXTURES = Path(__file__).parents[1] / "fixtures"


def test_cli_help_is_japanese() -> None:
    result = CliRunner().invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "アナリスト予想検証" in result.stdout
    assert "作業領域を初期化" in result.stdout
    assert "案件" in result.stdout


def test_init_command_creates_configured_workspace(tmp_path) -> None:
    config_path = tmp_path / "local-config.yaml"
    vault_root = tmp_path / "vault"

    result = CliRunner().invoke(
        app,
        [
            "init",
            "--vault-root",
            str(vault_root),
            "--config",
            str(config_path),
        ],
    )

    assert result.exit_code == 0, result.stdout
    assert "初期化しました" in result.stdout
    assert config_path.is_file()
    assert (vault_root / "_system" / "database.sqlite").is_file()


def test_cli_runs_anonymous_vertical_fixture_end_to_end(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from analyst_forecast.application.settings import load_settings
    from analyst_forecast.infrastructure.db.models import SourceRecord
    from analyst_forecast.infrastructure.db.session import create_session_factory
    from helpers_pipeline_v2 import import_locked_component

    runner = CliRunner()
    config_path = tmp_path / "config.yaml"
    vault_root = tmp_path / "vault"
    initialized = runner.invoke(
        app,
        [
            "init",
            "--vault-root",
            str(vault_root),
            "--config",
            str(config_path),
        ],
    )
    assert initialized.exit_code == 0, initialized.stdout

    created = runner.invoke(
        app,
        [
            "run",
            "create",
            "--name",
            "匿名アナリストA",
            "--period-start",
            "2026-01-01",
            "--period-end",
            "2026-06-30",
            "--evaluation-as-of",
            "2026-07-20",
            "--media",
            "youtube",
            "--config",
            str(config_path),
        ],
    )
    assert created.exit_code == 0, created.stdout
    run_match = re.search(r"RUN-\d{8}-001", created.stdout)
    assert run_match is not None
    run_id = run_match.group()

    imported_source = runner.invoke(
        app,
        [
            "source",
            "import",
            run_id,
            str(FIXTURES / "raw" / "anonymous_analyst_a.txt"),
            "--medium",
            "youtube",
            "--recorded-at",
            "2026-01-10T09:00:00+00:00",
            "--published-at",
            "2026-01-10T10:00:00+00:00",
            "--config",
            str(config_path),
        ],
    )
    assert imported_source.exit_code == 0, imported_source.stdout
    assert "SRC-000001" in imported_source.stdout

    settings = load_settings(config_path)
    session_factory = create_session_factory(settings.database_file)
    with session_factory() as session:
        source = session.get(SourceRecord, "SRC-000001")
        assert source is not None
        source_ns = SimpleNamespace(
            source_id=source.source_id,
            raw_hash=source.raw_hash,
        )
    run_ns = SimpleNamespace(run_id=run_id)
    component_id = import_locked_component(
        settings,
        run_ns,
        source_ns,
        tmp_path,
        label="cli",
    )

    evaluated = runner.invoke(
        app,
        [
            "market",
            "evaluate",
            run_id,
            component_id,
            "--as-of",
            "2026-04-13",
            "--provider",
            "csv",
            "--csv-path",
            str(FIXTURES / "market" / "n225_direction_up.csv"),
            "--config",
            str(config_path),
        ],
    )
    assert evaluated.exit_code == 0, evaluated.stdout
    assert "expired_hit" in evaluated.stdout
    assert "REVIEW_RESULTS" in evaluated.stdout
