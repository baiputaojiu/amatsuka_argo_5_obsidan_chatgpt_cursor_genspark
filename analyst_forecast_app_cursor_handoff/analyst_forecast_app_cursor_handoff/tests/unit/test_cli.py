import json
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
            "--config",
            str(config_path),
        ],
    )
    assert imported_source.exit_code == 0, imported_source.stdout
    assert "SRC-000001" in imported_source.stdout

    payload = json.loads(
        (FIXTURES / "ai" / "forecast_extraction_anonymous.json").read_text(encoding="utf-8")
    )
    payload["run_id"] = run_id
    ai_path = tmp_path / "ai-output.json"
    ai_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    imported_ai = runner.invoke(
        app,
        ["ai", "ingest", str(ai_path), "--config", str(config_path)],
    )
    assert imported_ai.exit_code == 0, imported_ai.stdout
    assert "accepted" in imported_ai.stdout

    evaluated = runner.invoke(
        app,
        [
            "market",
            "evaluate",
            run_id,
            "FCC-000001",
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
