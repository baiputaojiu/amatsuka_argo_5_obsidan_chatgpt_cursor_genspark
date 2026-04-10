"""Ch32: Config export/import — profile settings only, no secrets."""

from __future__ import annotations
import json
import logging
from pathlib import Path

from .settings_store import load_settings, save_settings
from ..models.config_schema import SCHEMA_VERSION

logger = logging.getLogger("outlook_google_sync")

FORBIDDEN_KEYS = {"token", "credentials", "refresh_token", "client_secret"}


def export_settings(dest_path: str) -> None:
    """Ch32.1: Export profile settings to JSON (no secrets)."""
    data = load_settings()
    clean = _strip_secrets(data)
    Path(dest_path).write_text(
        json.dumps(clean, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    logger.info("Settings exported to %s", dest_path)


def import_settings(src_path: str) -> tuple[bool, str]:
    """Ch32.3: Import profile settings from JSON.

    Returns (success, message).
    """
    try:
        raw = Path(src_path).read_text(encoding="utf-8")
        incoming = json.loads(raw)
    except Exception as exc:
        return False, f"インポート読込失敗: {exc}"

    incoming_version = incoming.get("schema_version", 0)
    if incoming_version > SCHEMA_VERSION:
        return False, f"非互換バージョン (imported={incoming_version}, current={SCHEMA_VERSION})"

    current = load_settings()
    # Merge profile data, preserving runtime state
    for key in ("runtime_state", "sync_metadata"):
        incoming.setdefault(key, current.get(key, {}))
    incoming["schema_version"] = SCHEMA_VERSION

    clean = _strip_secrets(incoming)
    save_settings(clean)
    logger.info("Settings imported from %s", src_path)
    return True, "インポート成功"


def _strip_secrets(data: dict) -> dict:
    """Ch32.2: Remove any secret-like keys."""
    out = {}
    for k, v in data.items():
        if any(fk in k.lower() for fk in FORBIDDEN_KEYS):
            continue
        if isinstance(v, dict):
            out[k] = _strip_secrets(v)
        else:
            out[k] = v
    return out
