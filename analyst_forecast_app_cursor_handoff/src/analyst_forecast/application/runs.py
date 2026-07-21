from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select

from analyst_forecast import __version__
from analyst_forecast.application.settings import AppSettings
from analyst_forecast.domain.models import Medium
from analyst_forecast.infrastructure.db.ids import next_id
from analyst_forecast.infrastructure.db.models import AnalystRecord, RunRecord
from analyst_forecast.infrastructure.db.session import create_session_factory
from analyst_forecast.schemas.ai_output import schema_path
from analyst_forecast.schemas.pipeline import (
    PIPELINE_SCHEMA_FILENAMES,
    pipeline_schema_path,
)


class CreateRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    canonical_name: str = Field(min_length=1, max_length=200)
    period_start: date
    period_end: date
    evaluation_as_of: date
    selected_media: list[Medium] = Field(min_length=1)
    focus_targets: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_dates(self) -> CreateRunRequest:
        if self.period_end < self.period_start:
            raise ValueError("調査終了日は開始日以後にしてください")
        return self


@dataclass(frozen=True, slots=True)
class CreateRunResult:
    analyst_id: str
    run_id: str
    run_path: Path


def create_run(
    settings: AppSettings,
    request: CreateRunRequest,
    *,
    now: datetime | None = None,
) -> CreateRunResult:
    timestamp = now or datetime.now(UTC)
    normalized_name = " ".join(request.canonical_name.split()).casefold()
    if not normalized_name:
        raise ValueError("分析対象者名は空にできません")

    session_factory = create_session_factory(settings.database_file)
    created_run_path: Path | None = None
    with session_factory.begin() as session:
        analyst = session.scalar(
            select(AnalystRecord).where(AnalystRecord.normalized_name == normalized_name)
        )
        if analyst is None:
            analyst = AnalystRecord(
                analyst_id=next_id(session, "A", width=4, sequence_key="ANALYST"),
                canonical_name=request.canonical_name.strip(),
                normalized_name=normalized_name,
            )
            session.add(analyst)
            session.flush()

        day_token = timestamp.astimezone(UTC).strftime("%Y%m%d")
        run_id = next_id(
            session,
            f"RUN-{day_token}-",
            width=3,
            sequence_key=f"RUN-{day_token}",
        )
        analyst_folder = (
            settings.vault_root
            / "analysts"
            / f"{_safe_windows_name(analyst.canonical_name)}__{analyst.analyst_id}"
        )
        run_path = analyst_folder / (
            f"{run_id}__{request.period_start:%Y%m%d}_{request.period_end:%Y%m%d}"
        )
        created_run_path = run_path
        relative_run_path = run_path.relative_to(settings.vault_root).as_posix()
        session.add(
            RunRecord(
                run_id=run_id,
                analyst_id=analyst.analyst_id,
                period_start=request.period_start,
                period_end=request.period_end,
                evaluation_as_of=request.evaluation_as_of,
                selected_media=[medium.value for medium in request.selected_media],
                focus_targets=request.focus_targets,
                ai_environment=["cursor", "chatgpt"],
                model_configuration={
                    "cursor": settings.cursor_model,
                    "chatgpt": settings.chatgpt_model,
                },
                status="not_started",
                run_path=relative_run_path,
            )
        )
        try:
            _create_run_tree(
                settings=settings,
                request=request,
                analyst_id=analyst.analyst_id,
                run_id=run_id,
                analyst_folder=analyst_folder,
                run_path=run_path,
                created_at=timestamp,
            )
        except Exception:
            if created_run_path.exists():
                shutil.rmtree(created_run_path)
            raise

    from analyst_forecast.application.workflow import refresh_workflow

    refresh_workflow(settings, run_id)
    return CreateRunResult(
        analyst_id=analyst.analyst_id,
        run_id=run_id,
        run_path=run_path,
    )


def _create_run_tree(
    *,
    settings: AppSettings,
    request: CreateRunRequest,
    analyst_id: str,
    run_id: str,
    analyst_folder: Path,
    run_path: Path,
    created_at: datetime,
) -> None:
    if run_path.exists():
        raise FileExistsError(f"案件フォルダが既に存在します: {run_path}")

    for path in _required_directories(run_path):
        path.mkdir(parents=True, exist_ok=path != run_path)

    profile_path = analyst_folder / "analyst_profile.md"
    if not profile_path.exists():
        profile_path.write_text(
            f"# {request.canonical_name}\n\n- analyst_id: `{analyst_id}`\n",
            encoding="utf-8",
            newline="\n",
        )

    request_data = {
        "schema_version": "1.0.0",
        "app_version": __version__,
        "analyst_id": analyst_id,
        "canonical_name": request.canonical_name,
        "run_id": run_id,
        "period_start": request.period_start,
        "period_end": request.period_end,
        "evaluation_as_of": request.evaluation_as_of,
        "selected_media": [medium.value for medium in request.selected_media],
        "focus_targets": request.focus_targets,
        "ai_environment": ["cursor", "chatgpt"],
        "model_configuration": {
            "cursor": settings.cursor_model,
            "chatgpt": settings.chatgpt_model,
        },
        "created_at": created_at.isoformat(),
    }
    (run_path / "request.yaml").write_text(
        yaml.safe_dump(request_data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )
    (run_path / "README.md").write_text(
        f"# 案件 {run_id}\n\n"
        f"- 分析対象者：{request.canonical_name} (`{analyst_id}`)\n"
        "- 機械状態は `WORKFLOW_STATE.json`、次の操作は `NEXT_ACTIONS.md` を参照してください。\n"
        "- `02_sources/*/raw/` の原文は変更しないでください。\n",
        encoding="utf-8",
        newline="\n",
    )
    _write_prompt_snapshots(settings, request, run_id, run_path)


def _required_directories(run_path: Path) -> list[Path]:
    directories = [run_path, run_path / "01_prompts" / "schemas"]
    for medium in Medium:
        for category in ("raw", "processed", "metadata"):
            directories.append(run_path / "02_sources" / medium.value / category)
    for category in ("inbox", "accepted", "needs_review", "rejected"):
        directories.append(run_path / "03_ai_outputs" / category)
    for category in (
        "forecasts",
        "evaluations",
        "target_mappings",
        "tables",
        "charts",
        "reports",
    ):
        directories.append(run_path / "04_results" / category)
    for category in (
        "search_logs",
        "processing_logs",
        "evaluation_snapshots",
        "errors",
    ):
        directories.append(run_path / "05_audit" / category)
    return directories


def _write_prompt_snapshots(
    settings: AppSettings,
    request: CreateRunRequest,
    run_id: str,
    run_path: Path,
) -> None:
    import json

    from analyst_forecast.resources import read_text_resource, resource_sha256

    catalog = json.loads(read_text_resource("prompts", "catalog.json"))
    template_version = str(catalog.get("template_version", "1.0.0"))
    prompt_meta = catalog["prompts"]
    prompt_ids = ["P08", "P11", "P12", "P13"]
    if Medium.YOUTUBE in request.selected_media:
        prompt_ids.insert(0, "P05")

    output_names = {
        "P05": f"P05_{run_id}_SOURCE_ID.json",
        "P08": f"P08_{run_id}_SOURCE_ID.json",
        "P11": f"P11_{run_id}_COMPONENT_ID.json",
        "P12": f"P12_{run_id}_COMPONENT_ID.json",
        "P13": f"P13_{run_id}_COMPONENT_ID.json",
    }
    for prompt_id in prompt_ids:
        meta = prompt_meta[prompt_id]
        title = meta["title"]
        purpose = meta["purpose"]
        warning = meta["warning"]
        input_path = meta["input_path"]
        template = read_text_resource("prompts", f"{prompt_id}.md.j2")
        template_hash = resource_sha256("prompts", f"{prompt_id}.md.j2")
        for environment in ("cursor", "chatgpt"):
            model = settings.cursor_model if environment == "cursor" else settings.chatgpt_model
            model_text = model or "未設定（実行前に高性能モデルを設定）"
            input_instruction = (
                f"案件内の `{input_path}` を読み込む"
                if environment == "cursor"
                else f"案件内の `{input_path}` を添付する"
            )
            output_name = output_names[prompt_id]
            schema_name = PIPELINE_SCHEMA_FILENAMES[prompt_id]
            content = (
                template.replace("{{prompt_id}}", prompt_id)
                .replace("{{title}}", title)
                .replace("{{environment}}", environment)
                .replace("{{template_version}}", template_version)
                .replace("{{template_hash}}", template_hash)
                .replace("{{run_id}}", run_id)
                .replace("{{purpose}}", purpose)
                .replace("{{model_text}}", model_text)
                .replace("{{input_instruction}}", input_instruction)
                .replace("{{output_name}}", output_name)
                .replace("{{schema_name}}", schema_name)
                .replace("{{warning}}", warning)
            )
            filename = f"{prompt_id}__{_safe_windows_name(title)}__{environment}.md"
            (run_path / "01_prompts" / filename).write_text(
                content,
                encoding="utf-8",
                newline="\n",
            )
            manifest = {
                "prompt_id": prompt_id,
                "environment": environment,
                "template_version": template_version,
                "template_hash": template_hash,
            }
            (run_path / "01_prompts" / f"{prompt_id}__{environment}.manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
                newline="\n",
            )

    shutil.copy2(
        schema_path(),
        run_path / "01_prompts" / "schemas" / "forecast_extraction.schema.json",
    )
    for prompt_id in ("P05", "P08", "P11", "P12", "P13"):
        shutil.copy2(
            pipeline_schema_path(prompt_id),
            run_path / "01_prompts" / "schemas" / PIPELINE_SCHEMA_FILENAMES[prompt_id],
        )


def _safe_windows_name(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", value.strip())
    cleaned = re.sub(r"\s+", "_", cleaned)
    cleaned = cleaned.rstrip(" .")[:40] or "analyst"
    if cleaned.upper() in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"LPT{i}" for i in range(1, 10)),
    }:
        return f"_{cleaned}"
    return cleaned
