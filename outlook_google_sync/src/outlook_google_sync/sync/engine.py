"""Ch14: Sync engine — orchestrates the full sync pipeline."""

from __future__ import annotations
import logging
from datetime import datetime
from typing import Callable, Optional

from ..connectors.google_calendar import (
    delete_event,
    detach_event,
    get_calendar_time_zone,
    list_managed_events,
    upsert_event,
)
from ..models.event import EventModel
from ..models.sync_result import SyncResult
from .conflict import has_conflict
from .delete_candidates import select_delete_candidates
from .diff_sync import filter_diff_targets
from .full_sync import full_targets
from .merge import merge_fields

logger = logging.getLogger("outlook_google_sync")


class SyncEngine:
    def __init__(self, log=None):
        self.logger = log or logger

    def run(
        self,
        calendar_id: str,
        events: list[EventModel],
        mode: str,
        previous_fingerprints: dict,
        detail_level: str = "full",
        conflict_policy: str = "overwrite",
        range_start: Optional[datetime] = None,
        range_end: Optional[datetime] = None,
        progress: Optional[Callable[[str], None]] = None,
        cancel_check: Optional[Callable[[], bool]] = None,
        on_error: Optional[Callable[[str], str]] = None,
    ) -> SyncResult:
        """Execute sync pipeline (Ch14.1).

        Args:
            on_error: callback(error_msg) -> "continue" | "stop"
        """
        r = SyncResult()
        if not events:
            return r

        if progress:
            progress("Google 既存イベント取得中...")

        cal_tz = get_calendar_time_zone(calendar_id)

        existing = list_managed_events(
            calendar_id,
            min(e.start for e in events),
            max(e.end for e in events),
        )

        # Step 3: diff or full targeting
        if mode == "full":
            targets = full_targets(events)
        else:
            targets = filter_diff_targets(events, previous_fingerprints, existing)
        r.skipped = max(0, len(events) - len(targets))

        if progress:
            progress(f"書込対象: {len(targets)} 件")

        # Steps 4-6: upsert with conflict detection
        for i, e in enumerate(targets):
            if cancel_check and cancel_check():
                self.logger.info("Sync cancelled by user")
                break

            body = e.to_google_body(detail_level, time_zone=cal_tz)
            google_item = existing.get(e.sync_key)
            event_id = google_item["id"] if google_item else None

            try:
                if google_item and has_conflict(google_item, body):
                    self._apply_conflict_policy(
                        calendar_id, e, body, google_item, conflict_policy, detail_level, r,
                    )
                else:
                    action, _ = upsert_event(
                        calendar_id, event_id, body, existing_event=google_item,
                    )
                    if action == "created":
                        r.created += 1
                    else:
                        r.updated += 1
            except Exception as exc:
                err_msg = f"{e.summary}: {exc}"
                r.errors.append(err_msg)
                r.failed += 1
                self.logger.error("Sync error: %s", err_msg)
                if on_error:
                    decision = on_error(err_msg)
                    if decision == "stop":
                        break

            if progress:
                progress(f"処理中 {i + 1}/{len(targets)}")

        # Step 8: collect delete candidates with R range (Ch26 DEL-02)
        all_current_keys = [e.sync_key for e in events]
        r.delete_candidates_list = select_delete_candidates(
            existing, all_current_keys, range_start, range_end,
        )
        r.deleted_candidates = len(r.delete_candidates_list)

        return r

    def _apply_conflict_policy(
        self,
        calendar_id: str,
        event: EventModel,
        body: dict,
        google_item: dict,
        policy: str,
        detail_level: str,
        result: SyncResult,
    ) -> None:
        """Ch17.2: Apply conflict resolution policy."""
        event_id = google_item["id"]

        if policy == "overwrite":
            upsert_event(calendar_id, event_id, body, existing_event=google_item)
            result.updated += 1
            self.logger.info("Conflict overwrite: %s", event.summary)

        elif policy == "detach_new":
            # Ch17.3: detach old, insert new with source sync_key
            detach_event(calendar_id, event_id)
            upsert_event(calendar_id, None, body)
            result.created += 1
            self.logger.info("Conflict detach_new: %s", event.summary)

        elif policy == "merge":
            merged = merge_fields(body, google_item)
            upsert_event(calendar_id, event_id, merged, existing_event=google_item)
            result.updated += 1
            result.merged += 1
            self.logger.info("Conflict merge: %s", event.summary)

        else:
            upsert_event(calendar_id, event_id, body, existing_event=google_item)
            result.updated += 1

    def execute_deletions(
        self,
        calendar_id: str,
        approved_items: list[dict],
        progress: Optional[Callable[[str], None]] = None,
    ) -> tuple[int, list[str]]:
        """Ch26.4: Execute approved deletions."""
        deleted = 0
        errors: list[str] = []
        for item in approved_items:
            try:
                delete_event(calendar_id, item["id"])
                deleted += 1
            except Exception as exc:
                errors.append(f"Delete failed: {item.get('summary', '?')}: {exc}")
            if progress:
                progress(f"削除 {deleted}/{len(approved_items)}")
        return deleted, errors
