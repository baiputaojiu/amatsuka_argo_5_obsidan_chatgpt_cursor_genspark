from __future__ import annotations

import os
import sys
from datetime import date, datetime
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from analyst_forecast.application.ai_ingestion import ingest_ai_output
from analyst_forecast.application.analysts import add_analyst_alias, list_analyst_aliases
from analyst_forecast.application.bootstrap import initialize_workspace
from analyst_forecast.application.evaluation import evaluate_component
from analyst_forecast.application.raw_sources import RawSourceRequest, import_raw_source
from analyst_forecast.application.runs import CreateRunRequest, create_run
from analyst_forecast.application.settings import (
    AppSettings,
    default_config_path,
    load_settings,
    save_settings,
)
from analyst_forecast.application.wizard import WizardCancelled, interactive_start
from analyst_forecast.application.workflow import (
    WorkflowState,
    assert_component_belongs_to_run,
    refresh_workflow,
)
from analyst_forecast.domain.market import MarketDataProvider
from analyst_forecast.domain.models import Medium
from analyst_forecast.infrastructure.market.csv_provider import CsvMarketDataProvider
from analyst_forecast.infrastructure.market.fred_provider import FredMarketDataProvider
from analyst_forecast.infrastructure.market.yfinance_provider import (
    YFinanceMarketDataProvider,
)


def _configure_windows_utf8() -> None:
    if os.name != "nt":
        return
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(encoding="utf-8")


_configure_windows_utf8()

console = Console()
DEFAULT_CONFIG_PATH = default_config_path()
app = typer.Typer(
    name="analyst-forecast",
    help="アナリスト予想検証を原文から再現可能に進めるローカルCLI。",
    no_args_is_help=True,
    rich_markup_mode="markdown",
)
run_app = typer.Typer(help="分析案件を作成・管理します。")
source_app = typer.Typer(help="変更禁止のraw原文を取り込みます。")
ai_app = typer.Typer(help="AI出力JSONを検証して取り込みます。")
market_app = typer.Typer(help="市場データを取得し最小方向評価を行います。")
app.add_typer(run_app, name="run")
app.add_typer(source_app, name="source")
app.add_typer(ai_app, name="ai")
app.add_typer(market_app, name="market")

config_app = typer.Typer(help="ローカル設定")
app.add_typer(config_app, name="config")

analyst_app = typer.Typer(help="分析対象者の alias などを管理します。")
app.add_typer(analyst_app, name="analyst")


@config_app.command("set-model", help="高性能モデル名をローカル設定へ保存します。")
def set_model_command(
    cursor: Annotated[str | None, typer.Option("--cursor", help="Cursor用モデル名")] = None,
    chatgpt: Annotated[str | None, typer.Option("--chatgpt", help="ChatGPT用モデル名")] = None,
    config: Annotated[
        Path, typer.Option("--config", help="ローカル設定ファイル")
    ] = DEFAULT_CONFIG_PATH,
) -> None:
    if not cursor and not chatgpt:
        raise typer.BadParameter("--cursor または --chatgpt が必要です")
    try:
        settings = load_settings(config)
        if cursor:
            settings.cursor_model = cursor
        if chatgpt:
            settings.chatgpt_model = chatgpt
        save_settings(settings, config)
        typer.echo(
            "モデル設定を保存しました。\n"
            f"cursor={settings.cursor_model or '未設定'}\n"
            f"chatgpt={settings.chatgpt_model or '未設定'}\n"
            f"設定: {config}"
        )
    except Exception as error:
        _fail("モデル設定に失敗しました", error)


@analyst_app.command("add-alias", help="分析対象者へ alias を登録します（NFKC exact照合用）。")
def analyst_add_alias_command(
    alias: Annotated[str, typer.Argument(help="追加する別名")],
    analyst_id: Annotated[str | None, typer.Option("--analyst-id", help="分析対象者ID")] = None,
    canonical_name: Annotated[
        str | None, typer.Option("--name", help="canonical_nameで検索")
    ] = None,
    config: Annotated[
        Path, typer.Option("--config", help="ローカル設定ファイル")
    ] = DEFAULT_CONFIG_PATH,
) -> None:
    try:
        settings = load_settings(config)
        result = add_analyst_alias(
            settings,
            analyst_id=analyst_id,
            canonical_name=canonical_name,
            alias=alias,
        )
    except Exception as error:
        _fail("alias登録に失敗しました", error)
    typer.echo(
        f"aliasを更新しました。\n"
        f"analyst_id={result.analyst_id}\n"
        f"canonical_name={result.canonical_name}\n"
        f"aliases={list(result.aliases)}"
    )


@analyst_app.command("list-aliases", help="分析対象者の alias 一覧を表示します。")
def analyst_list_aliases_command(
    analyst_id: Annotated[str, typer.Argument(help="分析対象者ID")],
    config: Annotated[
        Path, typer.Option("--config", help="ローカル設定ファイル")
    ] = DEFAULT_CONFIG_PATH,
) -> None:
    try:
        settings = load_settings(config)
        result = list_analyst_aliases(settings, analyst_id=analyst_id)
    except Exception as error:
        _fail("alias一覧の取得に失敗しました", error)
    typer.echo(
        f"analyst_id={result.analyst_id}\n"
        f"canonical_name={result.canonical_name}\n"
        f"aliases={list(result.aliases)}"
    )


@app.command("init", help="作業領域を初期化し、設定とSQLiteを作成します。")
def init_command(
    vault_root: Annotated[
        Path | None,
        typer.Option(
            "--vault-root",
            help="作業領域（Obsidian内の 30_Permanent/★アナリスト調査）",
        ),
    ] = None,
    obsidian_vault: Annotated[
        Path | None,
        typer.Option("--obsidian-vault", help="Obsidian Vault本体の絶対パス"),
    ] = None,
    workspace_relative: Annotated[
        str,
        typer.Option(
            "--workspace-relative",
            help="Vault内の相対作業パス",
        ),
    ] = "30_Permanent/★アナリスト調査",
    config: Annotated[
        Path,
        typer.Option("--config", help="ローカル設定ファイル"),
    ] = DEFAULT_CONFIG_PATH,
    update_docs: Annotated[
        bool,
        typer.Option("--update-docs", help="同梱docs/promptsで上書き更新（backup付き）"),
    ] = False,
    cursor_model: Annotated[
        str | None,
        typer.Option("--cursor-model", help="Cursor用の高性能モデル名"),
    ] = None,
    chatgpt_model: Annotated[
        str | None,
        typer.Option("--chatgpt-model", help="ChatGPT用の高性能モデル名"),
    ] = None,
) -> None:
    try:
        if vault_root is None and obsidian_vault is None:
            raise ValueError("--vault-root または --obsidian-vault が必要です")
        if obsidian_vault is not None:
            settings = AppSettings(
                vault_root=obsidian_vault / workspace_relative,
                obsidian_vault_path=obsidian_vault,
                workspace_relative_path=workspace_relative,
                database_path=(obsidian_vault / workspace_relative) / "_system" / "database.sqlite",
                cursor_model=cursor_model,
                chatgpt_model=chatgpt_model,
            )
        else:
            assert vault_root is not None
            settings = AppSettings(
                vault_root=vault_root,
                database_path=vault_root / "_system" / "database.sqlite",
                cursor_model=cursor_model,
                chatgpt_model=chatgpt_model,
            )
        initialize_workspace(settings, config_path=config, update_docs=update_docs)
    except Exception as error:
        _fail("初期化できませんでした", error)
    console.print(
        Panel.fit(
            f"初期化しました。\n設定: {config}\n作業領域: {settings.workspace_root}\n"
            "docs/promptsは初回seed済み（再実行は既存編集を保持、--update-docsで更新）。",
            title="完了",
        )
    )


@app.command("start", help="対話wizardで案件を作成します。")
def start_command(
    config: Annotated[
        Path, typer.Option("--config", help="ローカル設定ファイル")
    ] = DEFAULT_CONFIG_PATH,
) -> None:
    try:
        settings = load_settings(config)
        result = interactive_start(settings)
        state = refresh_workflow(settings, result.run_id)
    except WizardCancelled as error:
        console.print(str(error))
        raise typer.Exit(code=0) from error
    except Exception as error:
        _fail("対話作成できませんでした", error)
    console.print(
        Panel.fit(
            f"対象者ID: {result.analyst_id}\n案件ID: {result.run_id}\n"
            f"案件フォルダ: {result.run_path}",
            title="案件作成完了",
        )
    )
    _render_workflow(state)


@run_app.command("create", help="対象者ID・案件IDを発行して案件を作成します。")
def create_run_command(
    canonical_name: Annotated[str, typer.Option("--name", help="分析対象者名")],
    period_start: Annotated[str, typer.Option("--period-start", help="開始日 YYYY-MM-DD")],
    period_end: Annotated[str, typer.Option("--period-end", help="終了日 YYYY-MM-DD")],
    evaluation_as_of: Annotated[
        str, typer.Option("--evaluation-as-of", help="評価基準日 YYYY-MM-DD")
    ],
    media: Annotated[
        list[str],
        typer.Option("--media", help="youtube / blog / x / web。複数指定可"),
    ],
    focus_target: Annotated[
        list[str] | None,
        typer.Option("--focus-target", help="重点対象。複数指定可"),
    ] = None,
    config: Annotated[
        Path, typer.Option("--config", help="ローカル設定ファイル")
    ] = DEFAULT_CONFIG_PATH,
) -> None:
    try:
        settings = load_settings(config)
        request = CreateRunRequest(
            canonical_name=canonical_name,
            period_start=date.fromisoformat(period_start),
            period_end=date.fromisoformat(period_end),
            evaluation_as_of=date.fromisoformat(evaluation_as_of),
            selected_media=[Medium(value.lower()) for value in media],
            focus_targets=focus_target or [],
        )
        result = create_run(settings, request)
        state = refresh_workflow(settings, result.run_id)
    except Exception as error:
        _fail("案件を作成できませんでした", error)
    console.print(
        Panel.fit(
            f"対象者ID: {result.analyst_id}\n案件ID: {result.run_id}\n"
            f"案件フォルダ: {result.run_path}",
            title="案件作成完了",
        )
    )
    _render_workflow(state)


@source_app.command("import", help="raw原文を追記専用で取り込み、SHA-256を登録します。")
def import_source_command(
    run_id: Annotated[str, typer.Argument(help="案件ID")],
    input_path: Annotated[Path, typer.Argument(help="UTF-8原文ファイル")],
    medium: Annotated[str, typer.Option("--medium", help="youtube / blog / x / web")],
    url: Annotated[str | None, typer.Option("--url", help="原文のURL")] = None,
    recorded_at: Annotated[
        str | None, typer.Option("--recorded-at", help="発言日時 ISO 8601")
    ] = None,
    published_at: Annotated[
        str | None, typer.Option("--published-at", help="公開日時 ISO 8601")
    ] = None,
    title: Annotated[str | None, typer.Option("--title", help="原文タイトル")] = None,
    config: Annotated[
        Path, typer.Option("--config", help="ローカル設定ファイル")
    ] = DEFAULT_CONFIG_PATH,
) -> None:
    try:
        settings = load_settings(config)
        request = RawSourceRequest(
            run_id=run_id,
            input_path=input_path,
            medium=Medium(medium.lower()),
            url=url,
            title=title,
            recorded_at=_parse_datetime(recorded_at),
            published_at=_parse_datetime(published_at),
        )
        result = import_raw_source(settings, request)
        state = refresh_workflow(settings, run_id)
    except Exception as error:
        _fail("raw原文を取り込めませんでした", error)
    duplicate = "（既存原文へ関連付け）" if result.duplicate else ""
    console.print(
        f"source_id: [bold]{result.source_id}[/bold] {duplicate}\n"
        f"SHA-256: {result.raw_hash}\nraw: {result.raw_file_path}"
    )
    _render_workflow(state)


@ai_app.command("ingest", help="AI出力のSchema・参照・引用を検証後に取り込みます。")
def ingest_ai_command(
    input_path: Annotated[Path, typer.Argument(help="AI出力JSON")],
    config: Annotated[
        Path, typer.Option("--config", help="ローカル設定ファイル")
    ] = DEFAULT_CONFIG_PATH,
) -> None:
    try:
        settings = load_settings(config)
        result = ingest_ai_output(settings, input_path)
        run_id = _peek_run_id(input_path)
        state = refresh_workflow(settings, run_id) if run_id else None
    except Exception as error:
        _fail("AI出力を検証できませんでした", error)
    console.print(f"分類: [bold]{result.status.value}[/bold]\nSHA-256: {result.output_hash}")
    if result.forecast_issuance_ids:
        console.print("issuance IDs:")
        for item in result.forecast_issuance_ids:
            console.print(f"- {item}")
    if result.component_ids:
        table = Table(title="構成予想")
        table.add_column("component_id")
        table.add_column("issuance_id")
        for index, component_id in enumerate(result.component_ids):
            issuance = (
                result.forecast_issuance_ids[index]
                if index < len(result.forecast_issuance_ids)
                else "-"
            )
            table.add_row(component_id, issuance)
        console.print(table)
    for issue in result.issues:
        console.print(f"- {issue.path}: {issue.message}")
    console.print(result.guidance)
    if state is not None:
        _render_workflow(state)
    if result.status.value == "rejected":
        raise typer.Exit(code=2)


@market_app.command("evaluate", help="固定済み対象を市場データで最小方向評価します。")
def evaluate_market_command(
    run_id: Annotated[str, typer.Argument(help="案件ID")],
    component_id: Annotated[str, typer.Argument(help="構成予想ID")],
    as_of: Annotated[str, typer.Option("--as-of", help="評価基準日 YYYY-MM-DD")],
    provider_name: Annotated[
        str, typer.Option("--provider", help="yfinance / fred / csv")
    ] = "yfinance",
    csv_path: Annotated[Path | None, typer.Option("--csv-path", help="CSV providerの入力")] = None,
    config: Annotated[
        Path, typer.Option("--config", help="ローカル設定ファイル")
    ] = DEFAULT_CONFIG_PATH,
) -> None:
    try:
        settings = load_settings(config)
        assert_component_belongs_to_run(settings, run_id=run_id, component_id=component_id)
        provider = _provider(provider_name, csv_path)
        result = evaluate_component(
            settings,
            component_id=component_id,
            provider=provider,
            as_of=date.fromisoformat(as_of),
            run_id=run_id,
        )
        state = refresh_workflow(settings, run_id)
    except Exception as error:
        _fail("市場評価を実行できませんでした", error)
    console.print(
        f"評価ID: {result.evaluation_id}\n状態: {result.evaluation_status}\n"
        f"方向結果: {result.direction_result or '未判定'}\n"
        f"変化率: {result.actual_return if result.actual_return is not None else '取得不能'}\n"
        f"method: {result.method_version}"
    )
    if result.unevaluable_reason:
        console.print(f"理由: {result.unevaluable_reason}")
    _render_workflow(state)


@app.command("status", help="案件状態と次の推奨行動を再生成して表示します。")
def status_command(
    run_id: Annotated[str, typer.Argument(help="案件ID")],
    config: Annotated[
        Path, typer.Option("--config", help="ローカル設定ファイル")
    ] = DEFAULT_CONFIG_PATH,
) -> None:
    try:
        state = refresh_workflow(load_settings(config), run_id)
    except Exception as error:
        _fail("状態を更新できませんでした", error)
    _render_workflow(state)


def _provider(name: str, csv_path: Path | None) -> MarketDataProvider:
    normalized = name.lower()
    if normalized == "yfinance":
        return YFinanceMarketDataProvider()
    if normalized == "fred":
        return FredMarketDataProvider()
    if normalized == "csv":
        if csv_path is None:
            raise ValueError("CSV providerには--csv-pathが必要です")
        return CsvMarketDataProvider(csv_path=csv_path)
    raise ValueError(f"未対応providerです: {name}")


def _render_workflow(state: WorkflowState) -> None:
    action = state.recommended_action
    table = Table(title="次に行うこと", show_header=False)
    table.add_column("項目", style="bold")
    table.add_column("内容")
    table.add_row("推奨", f"{action.action_id}: {action.title}")
    table.add_row("理由", action.reason)
    table.add_row("担当", action.executor)
    table.add_row("入力", "、".join(action.inputs) or "なし")
    table.add_row("出力", "、".join(action.outputs) or "なし")
    table.add_row("操作", action.command_or_prompt or "成果物を確認")
    console.print(table)
    if state.alternatives:
        console.print("代替案:")
        for alternative in state.alternatives:
            console.print(f"- {alternative.title}: {alternative.reason}")
    if state.blockers:
        console.print("ブロッカー:")
        for blocker in state.blockers:
            console.print(f"- {blocker}")


def _peek_run_id(path: Path) -> str | None:
    try:
        import json

        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception:
        return None
    if isinstance(data, dict):
        value = data.get("run_id")
        return str(value) if value else None
    return None


def _parse_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    return datetime.fromisoformat(value)


def _fail(title: str, error: Exception) -> None:
    if isinstance(error, ValidationError):
        details = "\n".join(
            f"- {'.'.join(str(part) for part in item['loc'])}: {item['msg']}"
            for item in error.errors(include_url=False)
        )
    else:
        details = str(error)
    console.print(f"[bold red]{title}[/bold red]\n原因: {details}")
    if "次の操作" not in details:
        console.print("次の操作: 入力、設定、状態ファイルを確認して再実行してください。")
    raise typer.Exit(code=1)


def main() -> None:
    app()
