"""Startup setup checklist evaluation (credentials / token / calendar)."""

from __future__ import annotations

from dataclasses import dataclass

from ..config.paths import credentials_path, token_path


@dataclass(frozen=True)
class SetupItem:
    id: str
    label: str
    done: bool
    hint: str


def evaluate_setup(state: dict | None = None) -> list[SetupItem]:
    """Return checklist rows for the current runtime / settings state."""
    state = state or {}
    cal_id = (state.get("calendar_id") or "").strip()
    creds_ok = credentials_path().exists()
    token_ok = token_path().exists()
    cal_ok = bool(cal_id)

    return [
        SetupItem(
            id="credentials",
            label="credentials.json の配置",
            done=creds_ok,
            hint=(
                "Google Cloud Console で OAuth クライアント（デスクトップ）を作成し、"
                "ダウンロードした JSON を credentials.json としてランタイムフォルダへ配置してください。"
            ),
        ),
        SetupItem(
            id="token",
            label="Google 認証（token.json）",
            done=token_ok,
            hint="「Google 認証」を実行し、ブラウザでアカウントを許可してください。",
        ),
        SetupItem(
            id="calendar",
            label="同期先カレンダー",
            done=cal_ok,
            hint="設定の Google タブでカレンダー一覧を取得し、対象を選択できます（未選択時は primary）。",
        ),
    ]


def is_setup_incomplete(state: dict | None = None) -> bool:
    """True when credentials or token is missing (calendar alone does not force the dialog)."""
    items = {item.id: item for item in evaluate_setup(state)}
    return (not items["credentials"].done) or (not items["token"].done)
