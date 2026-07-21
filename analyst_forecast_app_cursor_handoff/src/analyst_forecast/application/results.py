from __future__ import annotations

import csv
import io
from pathlib import Path

from sqlalchemy import select

from analyst_forecast.application.settings import AppSettings
from analyst_forecast.infrastructure.db.models import (
    AiArtifactRecord,
    AiImportRecord,
    EvaluationRecord,
    ForecastComponentRecord,
    ForecastEvidenceRecord,
    ForecastIssuanceRecord,
    RunRecord,
    RunSourceRecord,
    SourceRecord,
    TargetMappingRecord,
    TargetRecord,
)
from analyst_forecast.infrastructure.db.session import create_session_factory


def generate_run_results(settings: AppSettings, run_id: str) -> dict[str, Path]:
    session_factory = create_session_factory(settings.database_file)
    with session_factory() as session:
        run = session.get(RunRecord, run_id)
        if run is None:
            raise ValueError(f"案件IDが存在しません: {run_id}")
        run_path = settings.vault_root / Path(run.run_path)
        results_root = run_path / "04_results"
        forecasts_dir = results_root / "forecasts"
        evaluations_dir = results_root / "evaluations"
        tables_dir = results_root / "tables"
        reports_dir = results_root / "reports"
        for path in (forecasts_dir, evaluations_dir, tables_dir, reports_dir):
            path.mkdir(parents=True, exist_ok=True)

        issuances = list(
            session.scalars(
                select(ForecastIssuanceRecord)
                .outerjoin(
                    AiImportRecord,
                    AiImportRecord.ai_import_id == ForecastIssuanceRecord.ai_import_id,
                )
                .outerjoin(
                    AiArtifactRecord,
                    AiArtifactRecord.ai_artifact_id == ForecastIssuanceRecord.ai_artifact_id,
                )
                .where((AiImportRecord.run_id == run_id) | (AiArtifactRecord.run_id == run_id))
            )
        )
        issuance_ids = [item.forecast_issuance_id for item in issuances]
        components = (
            list(
                session.scalars(
                    select(ForecastComponentRecord).where(
                        ForecastComponentRecord.forecast_issuance_id.in_(issuance_ids)
                    )
                )
            )
            if issuance_ids
            else []
        )
        component_ids = [item.forecast_component_id for item in components]
        evaluations = (
            list(
                session.scalars(
                    select(EvaluationRecord).where(
                        EvaluationRecord.forecast_component_id.in_(component_ids)
                    )
                )
            )
            if component_ids
            else []
        )

        forecast_rows: list[dict[str, str]] = []
        for component in components:
            issuance = session.get(ForecastIssuanceRecord, component.forecast_issuance_id)
            if issuance is None:
                continue
            source = session.get(SourceRecord, issuance.source_id)
            mapping = (
                session.get(TargetMappingRecord, component.target_mapping_id)
                if component.target_mapping_id
                else None
            )
            target = session.get(TargetRecord, component.target_id) if component.target_id else None
            evidence = session.scalar(
                select(ForecastEvidenceRecord).where(
                    ForecastEvidenceRecord.forecast_issuance_id == issuance.forecast_issuance_id
                )
            )
            link = session.get(
                RunSourceRecord,
                {"run_id": run_id, "source_id": issuance.source_id},
            )
            raw_path = (
                link.local_input_path
                if link and link.local_input_path
                else (source.raw_file_path if source else "")
            )
            forecast_rows.append(
                {
                    "forecast_issuance_id": issuance.forecast_issuance_id,
                    "forecast_component_id": component.forecast_component_id,
                    "source_id": issuance.source_id,
                    "raw_path": raw_path or "",
                    "quote": evidence.quote if evidence else "",
                    "target": component.raw_target_label
                    or (target.canonical_name if target else ""),
                    "symbol": (target.ticker or "") if target else "",
                    "direction": component.direction,
                    "period_start": (
                        component.normalized_start.isoformat() if component.normalized_start else ""
                    ),
                    "period_end": (
                        component.normalized_end.isoformat() if component.normalized_end else ""
                    ),
                    "mapping_status": mapping.mapping_status if mapping else "",
                    "summary": issuance.human_readable_summary,
                }
            )

        evaluation_rows: list[dict[str, str]] = []
        for evaluation in evaluations:
            evaluated = session.get(ForecastComponentRecord, evaluation.forecast_component_id)
            evaluation_rows.append(
                {
                    "evaluation_id": evaluation.evaluation_id,
                    "forecast_component_id": evaluation.forecast_component_id,
                    "evaluation_as_of": evaluation.evaluation_as_of.isoformat(),
                    "method_version": evaluation.evaluation_method_version,
                    "status": evaluation.evaluation_status,
                    "direction_result": evaluation.direction_result or "",
                    "start_price": _num(evaluation.start_price),
                    "end_price": _num(evaluation.end_price),
                    "actual_return": _num(evaluation.actual_return),
                    "mfe": _num(evaluation.max_favorable_excursion),
                    "mae": _num(evaluation.max_adverse_excursion),
                    "unevaluable_reason": evaluation.unevaluable_reason or "",
                    "target": (evaluated.raw_target_label or "") if evaluated else "",
                    "direction": evaluated.direction if evaluated else "",
                }
            )

        forecasts_md = results_root / "forecasts" / "all_forecasts.md"
        forecasts_csv = results_root / "tables" / "all_forecasts.csv"
        evaluations_md = results_root / "evaluations" / "evaluations.md"
        evaluations_csv = results_root / "tables" / "evaluations.csv"
        summary_md = results_root / "reports" / "vertical_mvp_summary.md"

        _atomic_write(forecasts_md, _forecasts_markdown(run_id, forecast_rows))
        _atomic_write(forecasts_csv, _to_csv(forecast_rows))
        _atomic_write(evaluations_md, _evaluations_markdown(run_id, evaluation_rows))
        _atomic_write(evaluations_csv, _to_csv(evaluation_rows))
        _atomic_write(
            summary_md,
            _summary_markdown(run_id, forecast_rows, evaluation_rows),
        )
        return {
            "forecasts_md": forecasts_md,
            "forecasts_csv": forecasts_csv,
            "evaluations_md": evaluations_md,
            "evaluations_csv": evaluations_csv,
            "summary_md": summary_md,
        }


def _forecasts_markdown(run_id: str, rows: list[dict[str, str]]) -> str:
    lines = [f"# 予想一覧（{run_id}）", ""]
    if not rows:
        lines.append("登録済みの予想はありません。")
        return "\n".join(lines) + "\n"
    for row in rows:
        lines.extend(
            [
                f"## {row['forecast_component_id']}",
                "",
                f"- issuance: `{row['forecast_issuance_id']}`",
                f"- source_id: `{row['source_id']}`",
                f"- raw path: `{row['raw_path']}`",
                f"- quote: {row['quote']}",
                f"- target: {row['target']}",
                f"- symbol: {row['symbol'] or '-'}",
                f"- direction: {row['direction']}",
                f"- period: {row['period_start']} ～ {row['period_end']}",
                f"- mapping: {row['mapping_status'] or '-'}",
                f"- summary: {row['summary']}",
                "",
            ]
        )
    return "\n".join(lines)


def _evaluations_markdown(run_id: str, rows: list[dict[str, str]]) -> str:
    lines = [f"# 評価一覧（{run_id}）", ""]
    if not rows:
        lines.append("評価結果はまだありません。")
        return "\n".join(lines) + "\n"
    for row in rows:
        lines.extend(
            [
                f"## {row['evaluation_id']}",
                "",
                f"- component: `{row['forecast_component_id']}`",
                f"- as_of: {row['evaluation_as_of']}",
                f"- method: {row['method_version']}",
                f"- status: {row['status']}",
                f"- direction_result: {row['direction_result'] or '-'}",
                f"- start/end: {row['start_price'] or '-'} / {row['end_price'] or '-'}",
                f"- actual_return: {row['actual_return'] or '-'}",
                f"- MFE/MAE: {row['mfe'] or '-'} / {row['mae'] or '-'}",
                f"- unevaluable_reason: {row['unevaluable_reason'] or '-'}",
                "",
            ]
        )
    return "\n".join(lines)


def _summary_markdown(
    run_id: str,
    forecasts: list[dict[str, str]],
    evaluations: list[dict[str, str]],
) -> str:
    latest_by_component: dict[str, dict[str, str]] = {}
    for row in evaluations:
        current = latest_by_component.get(row["forecast_component_id"])
        if current is None or row["evaluation_as_of"] >= current["evaluation_as_of"]:
            latest_by_component[row["forecast_component_id"]] = row
    hit = sum(1 for row in latest_by_component.values() if row.get("direction_result") == "hit")
    miss = sum(1 for row in latest_by_component.values() if row.get("direction_result") == "miss")
    unevaluable = sum(
        1 for row in latest_by_component.values() if row.get("status") == "unevaluable"
    )
    return (
        f"# 縦断MVPサマリー（{run_id}）\n\n"
        f"- 予想構成数: {len(forecasts)}\n"
        f"- 評価履歴数: {len(evaluations)}\n"
        f"- 最新評価 hit/miss/unevaluable: {hit}/{miss}/{unevaluable}\n"
        "- 原文全文は含めず、source_id・raw path・引用で追跡できます。\n"
        "- 本ファイルはSQLiteから再生成されます。手編集を正本にしないでください。\n"
    )


def _to_csv(rows: list[dict[str, str]]) -> str:
    if not rows:
        return ""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0].keys()), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _num(value: object) -> str:
    return "" if value is None else str(value)


def _atomic_write(path: Path, content: str) -> None:
    from analyst_forecast.application.io_utils import atomic_write_text

    atomic_write_text(path, content)
