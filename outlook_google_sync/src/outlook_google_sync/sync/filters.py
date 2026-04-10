"""Ch8: Event filtering based on FilterConfig."""

from __future__ import annotations
from typing import TYPE_CHECKING

from ..utils.text_utils import simple_partial_match

if TYPE_CHECKING:
    from ..models.event import EventModel
    from ..models.profile import FilterConfig


def apply_filters(events: list[EventModel], fc: FilterConfig, include_private: bool) -> list[EventModel]:
    """Return events that pass all filter criteria."""
    result = []
    for e in events:
        if fc.exclude_all_day and e.is_all_day:
            continue
        if fc.exclude_free and e.busy_status == 0:
            continue
        if fc.exclude_tentative and e.busy_status == 1:
            continue
        # Ch8.3-4 / Ch21.1: Private filter exclusion takes priority
        if e.is_private and (fc.exclude_private or not include_private):
            continue
        if not _pass_category_filter(e, fc):
            continue
        if fc.subject_keywords and simple_partial_match(e.summary, fc.subject_keywords):
            continue
        if fc.location_keywords and simple_partial_match(e.location, fc.location_keywords):
            continue
        result.append(e)
    return result


def _pass_category_filter(e: EventModel, fc: FilterConfig) -> bool:
    if fc.category_mode == "none" or not fc.category_list:
        return True
    event_cats = {c.strip().lower() for c in e.categories}
    filter_cats = {c.strip().lower() for c in fc.category_list}
    has_match = bool(event_cats & filter_cats)
    if fc.category_mode == "exclude":
        return not has_match
    if fc.category_mode == "include_only":
        return has_match
    return True
