"""Build Google Calendar patch bodies for duplicate merge (Ch27 merge)."""

from __future__ import annotations

import copy
from typing import Literal

from ..connectors.google_calendar import delete_event, patch_event_merge, upsert_event
from .duplicate_repair import pick_managed_source_for_private


def _norm_loc(item: dict) -> str:
    loc = item.get("location")
    if not loc:
        return ""
    return str(loc).strip()


def _merge_descriptions(items: list[dict], mode: Literal["longer", "concat"]) -> str:
    texts: list[str] = []
    for it in items:
        d = (it.get("description") or "").strip()
        if d:
            texts.append(d)
    if not texts:
        return ""
    if mode == "longer":
        return max(texts, key=len)
    return "\n\n---\n\n".join(texts)


def _merged_location(winner: dict, group_items: list[dict]) -> str:
    w = _norm_loc(winner)
    if w:
        return w
    for it in group_items:
        n = _norm_loc(it)
        if n:
            return n
    return ""


def build_merged_body(
    winner: dict,
    *,
    description_mode: Literal["longer", "concat"],
    group_items: list[dict],
) -> dict:
    """Patch body for the surviving event. Attendees from winner; extendedProperties from managed row if any."""
    desc = _merge_descriptions(group_items, description_mode)
    loc = _merged_location(winner, group_items)
    managed = pick_managed_source_for_private(group_items)

    body: dict = {
        "summary": winner.get("summary") or "",
        "description": desc,
        "start": winner.get("start") or {},
        "end": winner.get("end") or {},
    }
    if loc:
        body["location"] = loc
    if "attendees" in winner:
        body["attendees"] = winner.get("attendees")
    for opt in ("colorId", "transparency", "visibility"):
        if winner.get(opt) is not None:
            body[opt] = winner[opt]

    if managed is not None:
        ext = managed.get("extendedProperties")
        if ext:
            body["extendedProperties"] = copy.deepcopy(ext)
    return body


def preview_merged_description(
    group_items: list[dict],
    description_mode: Literal["longer", "concat"],
) -> str:
    """Same description merge as build_merged_body for preview."""
    return _merge_descriptions(group_items, description_mode)


def preview_merged_location(winner: dict, group_items: list[dict]) -> str:
    return _merged_location(winner, group_items)


def execute_duplicate_merge(
    calendar_id: str,
    winner_id: str,
    group_items: list[dict],
    description_mode: Literal["longer", "concat"],
) -> None:
    """Patch winner with merged body, then delete other events in the group."""
    by_id = {str(it.get("id", "")): it for it in group_items if it.get("id")}
    winner = by_id.get(winner_id)
    if not winner:
        raise ValueError("winner not in group_items")
    body = build_merged_body(
        winner,
        description_mode=description_mode,
        group_items=group_items,
    )
    managed = pick_managed_source_for_private(group_items)
    if managed is not None:
        upsert_event(calendar_id, winner_id, body, existing_event=winner)
    else:
        patch_event_merge(calendar_id, winner_id, body)

    for it in group_items:
        eid = str(it.get("id", ""))
        if eid and eid != winner_id:
            delete_event(calendar_id, eid)
