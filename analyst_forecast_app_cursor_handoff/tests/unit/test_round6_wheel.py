"""Round6 clean-wheel formal vertical (no skip/xfail/importorskip)."""

# Embedded installed-script strings intentionally exceed line length.
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import textwrap
import venv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_SRC = (PROJECT_ROOT / "src").resolve()


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _run(
    cmd: list[str],
    *,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    base = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    base.pop("PYTHONPATH", None)
    base.pop("PYTHONHOME", None)
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env=base,
    )


def _assert_ok(result: subprocess.CompletedProcess[str], *, label: str) -> None:
    assert result.returncode == 0, (
        f"{label} failed rc={result.returncode}\n"
        f"stdout={(result.stdout or '')[-1200:]}\n"
        f"stderr={(result.stderr or '')[-1200:]}"
    )


def test_r6_045_048_wheel_formal_vertical(tmp_path: Path) -> None:
    """Build+install wheel outside repo; run vertical via installed API."""
    import build as _build  # noqa: F401 — must be present (R6-044)

    work = tmp_path / "outside_repo"
    work.mkdir()
    dist_dir = work / "dist"
    dist_dir.mkdir()

    build = _run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist_dir)],
        cwd=PROJECT_ROOT,
    )
    _assert_ok(build, label="python -m build --wheel")
    wheels = list(dist_dir.glob("*.whl"))
    assert wheels, "wheel artifact missing"
    wheel = wheels[0]

    venv_dir = work / "venv"
    venv.create(venv_dir, with_pip=True, clear=True)
    py = _venv_python(venv_dir)
    install = _run([str(py), "-m", "pip", "install", "--quiet", str(wheel)], cwd=work)
    _assert_ok(install, label="pip install wheel")

    schema_path = PROJECT_ROOT / "src/analyst_forecast/schemas/p09_forecast_review.schema.json"
    prompt_path = PROJECT_ROOT / "src/analyst_forecast/resources/prompts/P09.md.j2"
    repo_schema_hash = hashlib.sha256(schema_path.read_bytes()).hexdigest()
    repo_prompt_hash = hashlib.sha256(prompt_path.read_bytes()).hexdigest()
    repo_src = str(REPO_SRC)
    vault = str(work / "vault_space")

    script = work / "vertical.py"
    script.write_text(
        textwrap.dedent(
            f"""\
            from __future__ import annotations
            import hashlib, json, sqlite3
            from datetime import UTC, date, datetime
            from pathlib import Path
            import analyst_forecast
            from analyst_forecast.application.ai_ingestion import AiIngestStatus, ingest_ai_output
            from analyst_forecast.application.bootstrap import initialize_workspace
            from analyst_forecast.application.raw_sources import RawSourceRequest, import_raw_source
            from analyst_forecast.application.runs import CreateRunRequest, create_run
            from analyst_forecast.application.settings import AppSettings
            from analyst_forecast.domain.models import Medium
            from analyst_forecast.infrastructure.db.migration import upgrade_database
            from analyst_forecast.infrastructure.db.models import (
                AiArtifactRecord, AnalystRecord, ForecastComponentRecord,
                ForecastIssuanceRecord, RunRecord,
            )
            from analyst_forecast.infrastructure.db.session import create_session_factory
            from analyst_forecast.application.active_forecast_query import (
                InactiveComponentError, require_active_component_context,
            )
            from jsonschema import Draft202012Validator
            from sqlalchemy import func, select

            pkg = Path(analyst_forecast.__file__).resolve()
            assert "site-packages" in str(pkg).replace("\\\\", "/"), pkg
            assert {repo_src!r} not in str(pkg), pkg

            schema_file = pkg.parent / "schemas" / "p09_forecast_review.schema.json"
            prompt_file = pkg.parent / "resources" / "prompts" / "P09.md.j2"
            assert schema_file.is_file() and prompt_file.is_file()
            assert hashlib.sha256(schema_file.read_bytes()).hexdigest() == {repo_schema_hash!r}
            assert hashlib.sha256(prompt_file.read_bytes()).hexdigest() == {repo_prompt_hash!r}
            schema = json.loads(schema_file.read_text(encoding="utf-8"))
            Draft202012Validator.check_schema(schema)
            bare = {{
                "schema_version": "2.1.0",
                "run_id": "RUN-20260101-001",
                "source_id": "SRC-000001",
                "reviewed_artifact_id": "AIF-000001",
                "prompt_execution": {{
                    "prompt_id": "P09", "prompt_version": "2.1.0",
                    "environment": "cursor", "model": "x",
                    "executed_at": "2026-07-20T12:00:00+00:00",
                }},
                "input_hash": "a" * 64,
                "decision": "reject",
                "findings": [],
                "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
            }}
            assert list(Draft202012Validator(schema).iter_errors(bare))
            mig = pkg.parent / "infrastructure" / "db" / "migrations" / "versions"
            assert (mig / "0001_initial_mvp_schema.py").is_file()
            assert any(mig.glob("0010*.py"))

            root = Path({vault!r}); root.mkdir(parents=True, exist_ok=True)
            settings = AppSettings(
                vault_root=root / "vault",
                database_path=root / "vault" / "_system" / "database.sqlite",
                cursor_model="high-performance-fixture",
                chatgpt_model="high-performance-fixture",
            )
            initialize_workspace(settings, config_path=root / "config.local.yaml")
            RAW = "日経平均は今後上昇する。これは現状分析ではなく予想です。"
            run = create_run(
                settings,
                CreateRunRequest(
                    canonical_name="匿名アナリストA",
                    period_start=date(2026, 1, 1),
                    period_end=date(2026, 6, 30),
                    evaluation_as_of=date(2026, 7, 20),
                    selected_media=[Medium.YOUTUBE],
                    focus_targets=["日経平均"],
                ),
                now=datetime(2026, 7, 20, 12, 0, tzinfo=UTC),
            )
            src_path = root / "source.txt"; src_path.write_text(RAW, encoding="utf-8")
            source = import_raw_source(
                settings,
                RawSourceRequest(
                    run_id=run.run_id, input_path=src_path, medium=Medium.YOUTUBE,
                    url="https://example.invalid/video/fixture", title="fixture",
                    recorded_at=datetime(2026, 1, 10, 9, 0, tzinfo=UTC),
                    published_at=datetime(2026, 1, 10, 10, 0, tzinfo=UTC),
                    retrieved_at=datetime(2026, 7, 20, 12, 30, tzinfo=UTC),
                ),
            )

            def write(name, payload):
                path = root / name
                path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                return path

            def prompt(pid):
                return {{
                    "prompt_id": pid, "prompt_version": "2.0.0", "environment": "cursor",
                    "model": "high-performance-fixture",
                    "executed_at": "2026-07-20T12:00:00+00:00",
                }}

            sf = create_session_factory(settings.database_file)
            with sf() as session:
                speaker = session.get(AnalystRecord, session.get(RunRecord, run.run_id).analyst_id).canonical_name

            p05 = ingest_ai_output(settings, write("p05.json", {{
                "schema_version": "2.0.0", "run_id": run.run_id, "source_id": source.source_id,
                "prompt_execution": prompt("P05"), "input_hash": source.raw_hash,
                "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
                "segments": [{{
                    "segment_ref": "segment-main", "sequence_number": 1,
                    "raw_start_offset": 0, "raw_end_offset": len(RAW),
                    "raw_text": RAW, "normalized_text": RAW,
                    "speaker_status": "identified", "speaker_candidate": speaker,
                    "speaker_confidence": 0.95, "attribution_basis": "fixture",
                    "review_status": "accepted",
                }}],
            }}))
            assert p05.status is AiIngestStatus.ACCEPTED, p05.issues

            blog_path = root / "blog.txt"; blog_path.write_text(RAW, encoding="utf-8")
            blog_source = import_raw_source(
                settings,
                RawSourceRequest(
                    run_id=run.run_id, input_path=blog_path, medium=Medium.BLOG,
                    url="https://example.invalid/blog/fixture", title="blog-fixture",
                    recorded_at=datetime(2026, 1, 10, 9, 0, tzinfo=UTC),
                    published_at=datetime(2026, 1, 10, 10, 0, tzinfo=UTC),
                    retrieved_at=datetime(2026, 7, 20, 12, 30, tzinfo=UTC),
                ),
            )
            p07 = ingest_ai_output(settings, write("p07-blog.json", {{
                "schema_version": "2.0.0", "run_id": run.run_id, "source_id": blog_source.source_id,
                "prompt_execution": prompt("P07"), "input_hash": blog_source.raw_hash,
                "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
                "segments": [{{
                    "segment_ref": "segment-blog", "sequence_number": 1,
                    "raw_start_offset": 0, "raw_end_offset": len(RAW),
                    "raw_text": RAW, "normalized_text": RAW,
                    "author_status": "identified", "author_candidate": speaker,
                    "author_confidence": 0.95, "statement_kind": "author_own",
                    "attribution_basis": "fixture", "review_status": "accepted",
                    "importance": "normal", "high_importance_reason": None,
                }}],
            }}))
            assert p07.status is AiIngestStatus.ACCEPTED, p07.issues

            def forecast(label):
                quote = "日経平均は今後上昇する"
                return {{
                    "forecast_ref": f"forecast-{{label}}",
                    "forecast_group_ref": f"group-{{label}}",
                    "made_at": "2026-01-10T09:00:00+00:00",
                    "publicly_available_at": "2026-01-10T10:00:00+00:00",
                    "made_at_source": "explicit", "forecast_type": "directional",
                    "commitment_strength": "explicit", "evidence_level": "A",
                    "extraction_confidence": 0.95, "human_readable_summary": quote,
                    "relation_to_previous": "initial",
                    "upstream_segment_refs": ["segment-main"],
                    "speaker_candidate": speaker,
                    "speaker_attribution_status": "target_confirmed",
                    "attribution_confidence": 0.95, "attribution_basis": "本人segment",
                    "statement_kind": "direct_statement",
                    "evidence": [{{
                        "source_id": source.source_id, "quote": quote,
                        "start_offset": 0, "end_offset": len(quote), "role": "prediction",
                    }}],
                    "components": [{{
                        "component_ref": f"component-{{label}}", "sequence_number": 1,
                        "prediction_form": "period_direction", "direction": "up",
                        "time_source": "explicit", "time_expression_raw": "今後3か月",
                        "normalized_start": "2026-01-13", "normalized_end": "2026-04-13",
                        "raw_target_label": "日経平均", "target_resolution_status": "pending",
                    }}],
                }}

            p08 = ingest_ai_output(settings, write("p08.json", {{
                "schema_version": "2.1.0", "run_id": run.run_id, "source_id": source.source_id,
                "upstream_artifact_id": p05.artifact_ids[0], "upstream_prompt_id": "P05",
                "prompt_execution": prompt("P08"), "input_hash": p05.output_hash,
                "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
                "processing_status": "processed_with_forecasts",
                "forecasts": [forecast("a"), forecast("b")],
            }}))
            assert p08.status is AiIngestStatus.ACCEPTED, [i.message for i in p08.issues]
            with sf.begin() as session:
                art = session.get(AiArtifactRecord, p08.artifact_ids[0])
                art.classification = "needs_review"
                art.resolution_status = "needs_review"

            def counts():
                with sf() as session:
                    return {{
                        "iss": session.scalar(select(func.count()).select_from(ForecastIssuanceRecord)),
                        "active": session.scalar(
                            select(func.count()).select_from(ForecastIssuanceRecord).where(
                                ForecastIssuanceRecord.lifecycle_status == "active"
                            )
                        ),
                    }}

            before = counts()
            corrected_one = {{
                "schema_version": "2.1.0", "run_id": run.run_id, "source_id": source.source_id,
                "upstream_artifact_id": p05.artifact_ids[0], "upstream_prompt_id": "P05",
                "prompt_execution": prompt("P08"), "input_hash": p05.output_hash,
                "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
                "processing_status": "processed_with_forecasts",
                "forecasts": [forecast("x")],
            }}
            bad = ingest_ai_output(settings, write("p09-many.json", {{
                "schema_version": "2.1.0", "run_id": run.run_id, "source_id": source.source_id,
                "reviewed_artifact_id": p08.artifact_ids[0], "prompt_execution": prompt("P09"),
                "input_hash": p08.output_hash, "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
                "decision": "correct", "findings": [], "corrected_payload": corrected_one,
                "forecast_operations": [
                    {{"action": "update", "reviewed_forecast_ref": "forecast-a",
                      "corrected_forecast_ref": "forecast-x", "reason": "a"}},
                    {{"action": "update", "reviewed_forecast_ref": "forecast-b",
                      "corrected_forecast_ref": "forecast-x", "reason": "b"}},
                ],
            }}))
            assert bad.status is not AiIngestStatus.ACCEPTED
            assert counts() == before

            bad_undeclared = ingest_ai_output(settings, write("p09-undeclared.json", {{
                "schema_version": "2.1.0", "run_id": run.run_id, "source_id": source.source_id,
                "reviewed_artifact_id": p08.artifact_ids[0], "prompt_execution": prompt("P09"),
                "input_hash": p08.output_hash, "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
                "decision": "correct", "findings": [], "corrected_payload": corrected_one,
                "forecast_operations": [
                    {{"action": "update", "reviewed_forecast_ref": "forecast-a",
                      "corrected_forecast_ref": "forecast-x", "reason": "only a declared"}},
                ],
            }}))
            assert bad_undeclared.status is not AiIngestStatus.ACCEPTED
            assert counts() == before

            corrected_good = {{
                "schema_version": "2.1.0", "run_id": run.run_id, "source_id": source.source_id,
                "upstream_artifact_id": p05.artifact_ids[0], "upstream_prompt_id": "P05",
                "prompt_execution": prompt("P08"), "input_hash": p05.output_hash,
                "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
                "processing_status": "processed_with_forecasts",
                "forecasts": [forecast("a2"), forecast("b2")],
            }}
            good = ingest_ai_output(settings, write("p09-good.json", {{
                "schema_version": "2.1.0", "run_id": run.run_id, "source_id": source.source_id,
                "reviewed_artifact_id": p08.artifact_ids[0], "prompt_execution": prompt("P09"),
                "input_hash": p08.output_hash, "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
                "decision": "correct", "findings": [], "corrected_payload": corrected_good,
                "forecast_operations": [
                    {{"action": "update", "reviewed_forecast_ref": "forecast-a",
                      "corrected_forecast_ref": "forecast-a2", "reason": "fix a"}},
                    {{"action": "update", "reviewed_forecast_ref": "forecast-b",
                      "corrected_forecast_ref": "forecast-b2", "reason": "fix b"}},
                ],
            }}))
            assert good.status is AiIngestStatus.ACCEPTED, [i.message for i in good.issues]
            with sf() as session:
                actives = session.scalars(
                    select(ForecastIssuanceRecord).where(
                        ForecastIssuanceRecord.lifecycle_status == "active"
                    )
                ).all()
                assert {{a.local_ref for a in actives}} == {{"forecast-a2", "forecast-b2"}}
                old = session.scalars(
                    select(ForecastComponentRecord).where(
                        ForecastComponentRecord.local_ref == "component-a"
                    )
                ).first()
                old_id = old.forecast_component_id
            with sf() as session:
                gate = require_active_component_context(session, old_id)
                assert isinstance(gate, InactiveComponentError)

            with sf() as session:
                art = session.get(AiArtifactRecord, p08.artifact_ids[0])
                p08_hash = art.output_hash
            p11 = ingest_ai_output(settings, write("p11-old.json", {{
                "schema_version": "2.0.0", "run_id": run.run_id, "source_id": source.source_id,
                "forecast_component_id": old_id, "prompt_execution": prompt("P11"),
                "input_hash": p08_hash, "knowledge_cutoff": "2026-01-10T09:00:00+00:00",
                "resolution_status": "proposed",
                "candidates": [{{
                    "candidate_ref": "candidate-1", "rank": 1,
                    "canonical_name": "日経平均", "target_type": "index",
                    "mapping_method": "explicit", "confidence": 0.9,
                }}],
            }}))
            assert p11.status is not AiIngestStatus.ACCEPTED

            from analyst_forecast.application.evaluation import (
                INSTRUMENT_AUDIT_KEYS, build_coverage_audit, build_instrument_coverage_audit,
                evaluate_component,
            )
            from analyst_forecast.domain.market import MarketBar, MarketSeries
            from datetime import date as ddate
            from decimal import Decimal

            class WheelProvider:
                name = "wheel-fixture"
                def __init__(self, bars):
                    self._bars = bars
                def fetch(self, request):
                    return MarketSeries(
                        provider=self.name, symbol=request.symbol, currency=request.currency,
                        adjustment_type="split_adjusted_ohlc", frequency="daily",
                        retrieved_at=datetime(2026, 7, 20, 12, 0, tzinfo=UTC),
                        bars=tuple(self._bars),
                    )

            def _bar(day, o, c):
                po, pc = Decimal(o), Decimal(c)
                return MarketBar(
                    date=day, open=po, high=max(po, pc), low=min(po, pc), close=pc,
                    adjusted_open=po, adjusted_close=pc,
                )

            one_day_stats = build_instrument_coverage_audit(
                symbol="N225", currency="JPY", weight=1.0,
                requested_start=ddate(2026, 1, 13), requested_end=ddate(2026, 4, 13),
                input_bars=[_bar(ddate(2026, 1, 13), "100", "101")],
                adjustment_type="split_adjusted_ohlc",
            )
            assert one_day_stats["unique_valid_date_count"] == 1
            two_day_stats = build_instrument_coverage_audit(
                symbol="N225", currency="JPY", weight=1.0,
                requested_start=ddate(2026, 1, 13), requested_end=ddate(2026, 4, 13),
                input_bars=[
                    _bar(ddate(2026, 1, 13), "100", "100"),
                    _bar(ddate(2026, 1, 14), "100", "110"),
                ],
                adjustment_type="split_adjusted_ohlc",
            )
            assert two_day_stats["unique_valid_date_count"] == 2

            # Coverage audit builders from installed package
            sample = build_instrument_coverage_audit(
                symbol="N225", currency="JPY", weight=1.0,
                requested_start=ddate(2026, 1, 13), requested_end=ddate(2026, 1, 20),
                input_bars=[_bar(ddate(2026, 1, 13), "100", "101")],
                adjustment_type="split_adjusted_ohlc",
            )
            assert set(INSTRUMENT_AUDIT_KEYS) <= set(sample)
            assert sample["unique_valid_date_count"] == 1
            audit = build_coverage_audit(
                coverage_status="insufficient", reason_code="insufficient_trading_dates",
                requested_start=ddate(2026, 1, 13), requested_end=ddate(2026, 1, 20),
                effective_start=ddate(2026, 1, 13), effective_end=ddate(2026, 1, 20),
                evaluation_as_of=ddate(2026, 4, 13), method_version="direction-v2.0.0",
                series_kind="single", selected_start=None, selected_end=None,
                common_date_count=1, intersection_rule="single_symbol_trading_dates_v1",
                mapping_hash=None, instruments=[sample],
            )
            assert audit["schema_version"] == "1.0.0"
            assert audit["instruments"][0]["series_hash"]

            db = Path("mig.sqlite")
            upgrade_database(db, revision="0007")
            ts = "2026-07-20T00:00:00+00:00"
            hx = "a" * 64
            empty_json = "{{}}"
            instruments_json = '[{{"symbol":"N225","currency":"JPY","weight":1.0}}]'
            youtube_media = '["youtube"]'
            seed_sql = [
                f"INSERT INTO analysts (analyst_id, canonical_name, normalized_name, aliases, specialties, created_at, updated_at, version) VALUES ('A0001', '監査', '監査', '[]', '[]', '{{ts}}', '{{ts}}', 1)",
                f"INSERT INTO runs (run_id, analyst_id, period_start, period_end, evaluation_as_of, selected_media, focus_targets, ai_environment, model_configuration, status, run_path, created_at, updated_at, version) VALUES ('RUN-20260101-001', 'A0001', '2026-01-01', '2026-06-30', '2026-07-01', '{{youtube_media}}', '[]', '{{empty_json}}', '{{empty_json}}', 'active', 'runs/RUN-20260101-001', '{{ts}}', '{{ts}}', 1)",
                f"INSERT INTO sources (source_id, analyst_id, medium, url, title, recorded_at, published_at, retrieved_at, evidence_level, raw_file_path, raw_hash, acquisition_status, source_relation, created_at, updated_at, version) VALUES ('SRC-000001', 'A0001', 'youtube', 'https://example.invalid/v1', 'fixture', '{{ts}}', '{{ts}}', '{{ts}}', 'A', 'raw/a.txt', '{{hx}}', 'acquired', 'original', '{{ts}}', '{{ts}}', 1)",
                f"INSERT INTO forecast_groups (forecast_group_id, analyst_id, central_thesis, first_issued_at, latest_issued_at, current_stance, reaffirmation_count, revision_count, withdrawal_status, created_at, updated_at, version) VALUES ('FCG-000001', 'A0001', 'thesis', '{{ts}}', '{{ts}}', 'up', 0, 0, 'active', '{{ts}}', '{{ts}}', 1)",
                f"INSERT INTO run_sources (run_id, source_id, observed_medium, linked_at, processing_status) VALUES ('RUN-20260101-001', 'SRC-000001', 'youtube', '{{ts}}', 'accepted')",
                f"INSERT INTO prompt_executions (prompt_execution_id, run_id, prompt_id, prompt_version, environment, model, input_files, output_file, executed_at, validation_status, created_at, updated_at, version) VALUES ('PEX-000001', 'RUN-20260101-001', 'P08', '2.0.0', 'cursor', 'fixture', '[]', 'out.json', '{{ts}}', 'accepted', '{{ts}}', '{{ts}}', 1)",
                f"INSERT INTO ai_artifacts (ai_artifact_id, run_id, source_id, prompt_execution_id, prompt_id, schema_version, input_hash, output_hash, classified_file_path, classification, resolution_status, importance, payload, created_at, updated_at, version) VALUES ('AIF-000001', 'RUN-20260101-001', 'SRC-000001', 'PEX-000001', 'P08', '2.1.0', '{{hx}}', '{{hx}}', 'out.json', 'accepted', 'accepted', 'normal', '{{empty_json}}', '{{ts}}', '{{ts}}', 1)",
                f"INSERT INTO forecast_issuances (forecast_issuance_id, analyst_id, forecast_group_id, ai_artifact_id, source_id, local_ref, made_at, publicly_available_at, forecast_type, commitment_strength, evidence_level, extraction_confidence, human_readable_summary, relation_to_previous, current_status, created_at, updated_at, version) VALUES ('FCI-000001', 'A0001', 'FCG-000001', 'AIF-000001', 'SRC-000001', 'forecast-a', '{{ts}}', '{{ts}}', 'directional', 'explicit', 'A', 0.9, 'summary', 'initial', 'active', '{{ts}}', '{{ts}}', 1)",
                f"INSERT INTO forecast_evidence (forecast_evidence_id, forecast_issuance_id, source_id, quote, start_offset, end_offset, role, created_at, updated_at, version) VALUES ('FCE-000001', 'FCI-000001', 'SRC-000001', 'quote', 0, 5, 'prediction', '{{ts}}', '{{ts}}', 1)",
                f"INSERT INTO targets (target_id, raw_label, canonical_name, target_type, ticker, currency, aliases, created_at, updated_at, version) VALUES ('TGT-000001', '日経平均', '日経平均', 'index', 'N225', 'JPY', '[]', '{{ts}}', '{{ts}}', 1)",
                f"INSERT INTO target_mappings (target_mapping_id, target_id, mapping_method, evaluation_instruments, knowledge_cutoff, source_evidence, mapping_status, mapping_hash, locked_at, created_at, updated_at, version) VALUES ('TMP-000001', 'TGT-000001', 'manual', '{{instruments_json}}', '{{ts}}', 'fixture', 'verified', '{{hx}}', '{{ts}}', '{{ts}}', '{{ts}}', 1)",
                f"INSERT INTO forecast_components (forecast_component_id, forecast_issuance_id, local_ref, sequence_number, prediction_form, direction, time_source, raw_target_label, target_resolution_status, target_id, target_mapping_id, importance, created_at, updated_at, version) VALUES ('FCC-000001', 'FCI-000001', 'component-a', 1, 'period_direction', 'up', 'explicit', '日経平均', 'locked', 'TGT-000001', 'TMP-000001', 'normal', '{{ts}}', '{{ts}}', 1), ('FCC-000002', 'FCI-000001', 'component-b', 2, 'period_direction', 'up', 'explicit', 'TOPIX', 'pending', NULL, NULL, 'normal', '{{ts}}', '{{ts}}', 1)",
                f"INSERT INTO evaluations (evaluation_id, forecast_component_id, target_mapping_id, evaluation_method_version, evaluation_as_of, evaluation_status, created_at, updated_at, version) VALUES ('EVAL-000001', 'FCC-000001', 'TMP-000001', 'direction-v2.0.0', '2026-07-01', 'unevaluable', '{{ts}}', '{{ts}}', 1)",
            ]
            with sqlite3.connect(db) as conn:
                conn.execute("PRAGMA foreign_keys=ON")
                for stmt in seed_sql:
                    conn.execute(stmt)
                conn.commit()
                before_count = conn.execute("SELECT COUNT(*) FROM forecast_components").fetchone()[0]
                before_pk = conn.execute(
                    "SELECT forecast_component_id FROM forecast_components ORDER BY forecast_component_id"
                ).fetchall()
            upgrade_database(db)
            with sqlite3.connect(db) as conn:
                ver = conn.execute("SELECT version_num FROM alembic_version").fetchone()[0]
                assert ver.startswith("001"), ver
                assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
                after_count = conn.execute("SELECT COUNT(*) FROM forecast_components").fetchone()[0]
                after_pk = conn.execute(
                    "SELECT forecast_component_id FROM forecast_components ORDER BY forecast_component_id"
                ).fetchall()
                assert after_count == before_count == 2
                assert before_pk == after_pk
            print("WHEEL_VERTICAL_OK")
            """
        ),
        encoding="utf-8",
    )

    vertical = _run([str(py), str(script)], cwd=work)
    _assert_ok(vertical, label="installed vertical script")
    assert "WHEEL_VERTICAL_OK" in (vertical.stdout or "")
