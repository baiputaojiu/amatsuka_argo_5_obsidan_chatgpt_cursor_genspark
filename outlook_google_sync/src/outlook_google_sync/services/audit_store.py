"""Ch31: Audit / troubleshoot information persistence."""

import json
from datetime import UTC, datetime
from ..config.paths import runtime_dir


def save_audit(payload: dict) -> None:
    runtime_dir().mkdir(parents=True, exist_ok=True)
    payload.setdefault("timestamp", datetime.now(UTC).isoformat())
    path = runtime_dir() / "audit.json"

    history: list[dict] = []
    if path.exists():
        try:
            history = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(history, list):
                history = [history]
        except Exception:
            history = []

    history.append(payload)
    history = history[-50:]
    path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")


def load_audit() -> list[dict]:
    path = runtime_dir() / "audit.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
        return [data]
    except Exception:
        return []


def format_audit_summary() -> str:
    records = load_audit()
    if not records:
        return "監査記録なし"
    lines = []
    for rec in records[-10:]:
        ts = rec.get("timestamp", "?")[:19]
        mode = rec.get("mode", "?")
        c, u, d, f = rec.get("created", 0), rec.get("updated", 0), rec.get("deleted", 0), rec.get("failed", 0)
        lines.append(f"[{ts}] {mode}: 作成={c} 更新={u} 削除={d} 失敗={f}")
    return "\n".join(lines)
