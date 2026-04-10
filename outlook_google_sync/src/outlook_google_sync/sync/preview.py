"""Ch15: Preview sync (dry-run) with snapshot management."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from ..models.event import EventModel
from .conflict import has_conflict
from .delete_candidates import select_delete_candidates
from .diff_sync import filter_diff_targets
from .duplicate_repair import find_duplicates

STALE_THRESHOLD_MINUTES = 5


@dataclass
class PreviewAction:
    event: EventModel | None
    google_item: dict | None
    action: str  # create, update, delete, skip, merge, duplicate
    reason: str = ""


@dataclass
class PreviewSnapshot:
    actions: list[PreviewAction] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    # Convenience accessors
    @property
    def created(self) -> list[PreviewAction]:
        return [a for a in self.actions if a.action == "create"]

    @property
    def updated(self) -> list[PreviewAction]:
        return [a for a in self.actions if a.action == "update"]

    @property
    def skipped(self) -> list[PreviewAction]:
        return [a for a in self.actions if a.action == "skip"]

    @property
    def deleted(self) -> list[PreviewAction]:
        return [a for a in self.actions if a.action == "delete"]

    @property
    def merged(self) -> list[PreviewAction]:
        return [a for a in self.actions if a.action == "merge"]

    @property
    def duplicates(self) -> list[PreviewAction]:
        return [a for a in self.actions if a.action == "duplicate"]

    def is_stale(self) -> bool:
        """PRE-SNAP-03: stale if more than 5 minutes since generation."""
        elapsed = datetime.now(UTC) - self.generated_at
        return elapsed > timedelta(minutes=STALE_THRESHOLD_MINUTES)


def build_preview(
    events: list[EventModel],
    existing_by_key: dict[str, dict],
    fingerprints: dict[str, str],
    mode: str = "normal",
    range_start: datetime | None = None,
    range_end: datetime | None = None,
    time_zone: str | None = None,
) -> PreviewSnapshot:
    """Build preview snapshot (Ch15.1~15.3).

    Args:
        mode: "normal" for differential, "full" for full re-evaluation.
    """
    snap = PreviewSnapshot()

    if mode == "normal":
        targets = filter_diff_targets(events, fingerprints, existing_by_key)
        target_keys = {e.sync_key for e in targets}
        for e in events:
            if e.sync_key not in target_keys:
                snap.actions.append(PreviewAction(event=e, google_item=None, action="skip", reason="差分なし"))
    else:
        targets = list(events)

    for e in targets:
        google_item = existing_by_key.get(e.sync_key)
        if not google_item:
            snap.actions.append(PreviewAction(event=e, google_item=None, action="create", reason="新規"))
        else:
            body = e.to_google_body("full", time_zone=time_zone)
            if has_conflict(google_item, body):
                snap.actions.append(PreviewAction(event=e, google_item=google_item, action="update", reason="衝突検知（手編集あり）"))
            else:
                snap.actions.append(PreviewAction(event=e, google_item=google_item, action="update", reason="ソース変更"))

    # Delete candidates (Ch15.2)
    current_keys = [e.sync_key for e in events]
    del_candidates = select_delete_candidates(existing_by_key, current_keys, range_start, range_end)
    for item in del_candidates:
        snap.actions.append(PreviewAction(
            event=None, google_item=item, action="delete",
            reason="ソースに存在しない管理イベント",
        ))

    # Duplicate candidates (Ch15.2)
    dup_map = find_duplicates(list(existing_by_key.values()))
    for key, items in dup_map.items():
        for item in items[1:]:
            snap.actions.append(PreviewAction(
                event=None, google_item=item, action="duplicate",
                reason=f"sync_key重複: {key[:16]}",
            ))

    return snap
