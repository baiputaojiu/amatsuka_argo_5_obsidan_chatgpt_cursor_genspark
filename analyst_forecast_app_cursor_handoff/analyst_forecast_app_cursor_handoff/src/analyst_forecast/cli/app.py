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
from analyst_forecast.application.bootstrap import initialize_workspace
from analyst_forecast.application.evaluation import evaluate_component
from analyst_forecast.application.raw_sources import RawSourceRequest, import_raw_source
from analyst_forecast.application.runs import CreateRunRequest, create_run
from analyst_forecast.application.settings import (
    AppSettings,
    default_config_path,
    load_settings,
)
from analyst_forecast.application.workflow import WorkflowState, refresh_workflow
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


@app.command("init", help="作業領域を初期化し、設定とSQLiteを作成します。")
def init_command(
    vault_root: Annotated[
        Path,
        typer.Option("--vault-root", help="★アナリスト調査フォルダとして使う保存先"),
    ],
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            help="ローカル設定ファイル",
        ),
    ] = DEFAULT_CONFIG_PATH,
) -> None:
    try:
        settings = AppSettings(
            vault_root=vault_root,
            database_path=vault_root / "_system" / "database.sqlite",
        )
        initialize_workspace(settings, config_path=config)
    except Exception as error:
        _fail("初期化できませんでした", error)
    console.print(
        Panel.fit(
            f"初期化しました。\n設定: {config}\nデータ領域: {vault_root}",
            title="完了",
        )
    )


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
    except Exception as error:
        _fail("AI出力を検証できませんでした", error)
    console.print(f"分類: [bold]{result.status.value}[/bold]\nSHA-256: {result.output_hash}")
    for issue in result.issues:
        console.print(f"- {issue.path}: {issue.message}")
    console.print(result.guidance)
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
        provider = _provider(provider_name, csv_path)
        result = evaluate_component(
            settings,
            component_id=component_id,
            provider=provider,
            as_of=date.fromisoformat(as_of),
        )
        state = refresh_workflow(settings, run_id)
    except Exception as error:
        _fail("市場評価を実行できませんでした", error)
    console.print(
        f"評価ID: {result.evaluation_id}\n状態: {result.evaluation_status}\n"
        f"方向結果: {result.direction_result or '未判定'}\n"
        f"変化率: {result.actual_return if result.actual_return is not None else '取得不能'}"
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
