from datetime import date
from io import StringIO
from pathlib import Path

import pytest

from analyst_forecast.application.bootstrap import initialize_workspace
from analyst_forecast.application.settings import AppSettings, load_settings
from analyst_forecast.application.wizard import WizardCancelled, interactive_start
from analyst_forecast.domain.market import MarketDataRequest, ProviderError
from analyst_forecast.infrastructure.market.yfinance_provider import (
    YFinanceMarketDataProvider,
)


def test_init_seeds_docs_and_prompts_without_overwrite(tmp_path: Path) -> None:
    vault = tmp_path / "vault with space" / "★アナリスト調査"
    settings = AppSettings(
        vault_root=vault,
        database_path=vault / "_system" / "database.sqlite",
        cursor_model="model-a",
        chatgpt_model="model-b",
    )
    config = tmp_path / "config.yaml"
    initialize_workspace(settings, config_path=config)
    assert (vault / "README.md").is_file()
    assert (vault / "AI_WORK_GUIDE.md").is_file()
    assert (vault / "docs" / "01_スタートアップガイド" / "STARTUP_GUIDE.md").is_file()
    assert (vault / "prompts" / "catalog.json").is_file()

    readme = vault / "README.md"
    readme.write_text("ユーザー編集", encoding="utf-8")
    initialize_workspace(settings, config_path=config)
    assert readme.read_text(encoding="utf-8") == "ユーザー編集"

    initialize_workspace(settings, config_path=config, update_docs=True)
    assert "アナリスト予想検証" in readme.read_text(encoding="utf-8")


def test_legacy_vault_root_config_loads(tmp_path: Path) -> None:
    vault = tmp_path / "legacy-vault"
    vault.mkdir()
    config = tmp_path / "legacy.yaml"
    config.write_text(
        "vault_root: " + str(vault).replace("\\", "/") + "\n",
        encoding="utf-8",
    )
    loaded = load_settings(config)
    assert loaded.workspace_root == vault.resolve() or loaded.vault_root == vault


def test_path_traversal_rejected() -> None:
    with pytest.raises(ValueError, match=r"\.\."):
        AppSettings(
            obsidian_vault_path=Path("C:/vault"),
            workspace_relative_path="../escape",
        )


def test_wizard_defaults_and_cancel(tmp_path: Path) -> None:
    vault = tmp_path / "wizard-vault"
    settings = AppSettings(
        vault_root=vault,
        database_path=vault / "_system" / "database.sqlite",
        cursor_model="m1",
        chatgpt_model="m2",
    )
    initialize_workspace(settings, config_path=tmp_path / "cfg.yaml")
    answers = iter(
        [
            "匿名W",
            "",
            "",
            "",
            "youtube,blog",
            "",
            "cancel",
        ]
    )
    output = StringIO()
    with pytest.raises(WizardCancelled):
        interactive_start(settings, input_func=lambda: next(answers), output=output)


def test_wizard_creates_run(tmp_path: Path) -> None:
    vault = tmp_path / "wizard-ok"
    settings = AppSettings(
        vault_root=vault,
        database_path=vault / "_system" / "database.sqlite",
        cursor_model="m1",
        chatgpt_model="m2",
    )
    initialize_workspace(settings, config_path=tmp_path / "cfg.yaml")
    answers = iter(
        [
            "匿名W2",
            "2026-01-01",
            "2026-06-30",
            "2026-07-20",
            "youtube,x",
            "日経平均",
            "yes",
        ]
    )
    result = interactive_start(
        settings,
        input_func=lambda: next(answers),
        output=StringIO(),
    )
    assert result.run_id.startswith("RUN-")
    manifests = list(result.run_path.glob("01_prompts/*.manifest.json"))
    assert manifests
    text = (result.run_path / "01_prompts" / manifests[0].name).read_text(encoding="utf-8")
    assert "template_hash" in text
    assert "C:/" not in text
    assert "api_key" not in text.lower()


def test_yfinance_classifies_rate_limit() -> None:
    def boom(*args, **kwargs):
        raise RuntimeError("YFRateLimitError: Too Many Requests")

    provider = YFinanceMarketDataProvider(
        sleeper=lambda _: None,
        downloader=boom,
        max_attempts=2,
    )
    with pytest.raises(ProviderError) as error:
        provider.fetch(
            MarketDataRequest(
                symbol="^N225",
                currency="JPY",
                start=date(2026, 1, 1),
                end=date(2026, 1, 10),
            )
        )
    assert error.value.code == "rate_limit"
    assert error.value.attempt_count == 2
