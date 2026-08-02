"""Typed enums for previously stringly-typed values.

Phase 1 refactor: this module defines the authoritative set of valid values
for settings/state fields that used to be bare strings throughout the code
base (``"com"``, ``"overwrite"``, ``"normal"``, ``"title_only"`` ...).

Design notes
------------
- Each enum inherits from ``(str, Enum)`` so *existing* code doing
  ``if mode == "normal"`` or ``json.dumps(...)`` keeps working unchanged.
  ``InputMethod.COM == "com"`` evaluates to ``True``.
- ``StrEnum`` (3.11+) is avoided intentionally: the project targets
  Python 3.10+.
- Callers that want to coerce a user-supplied string to an enum member
  can use ``InputMethod(value)`` which raises ``ValueError`` on unknown
  values — useful for migrations and config validation.

Phase 2+ will progressively convert call sites from raw strings to these
enum members; Phase 1 only establishes the contract.
"""

from __future__ import annotations

from enum import Enum


class InputMethod(str, Enum):
    """GUI-selectable input source (see README §入力方法)."""

    COM = "com"
    ICS_MANUAL = "ics_manual"
    ICS_MACRO = "ics_macro"


class ReaderEngine(str, Enum):
    """Underlying reader used to parse the source.

    Distinct from ``InputMethod`` because ``ICS_MANUAL`` and ``ICS_MACRO``
    both map to the ``ICS`` reader.
    """

    OUTLOOK_COM = "outlook_com"
    ICS = "ics"


class DetailLevel(str, Enum):
    """How much of each event is sent to Google (SPEC Ch21)."""

    FULL = "full"
    TITLE_ONLY = "title_only"


class ConflictPolicy(str, Enum):
    """Behaviour when Google-side hand-edits are detected (SPEC Ch17)."""

    OVERWRITE = "overwrite"
    DETACH_NEW = "detach_new"
    MERGE = "merge"


class SyncMode(str, Enum):
    """Top-level sync operation mode (SPEC Ch14)."""

    NORMAL = "normal"
    FULL = "full"


class SyncKeyKind(str, Enum):
    """Whether ``sync_key`` was derived from the source primary id
    (EntryID / UID+RECURRENCE-ID) or a fallback hash (SPEC Ch11)."""

    PRIMARY = "primary"
    FALLBACK = "fallback"


class DuplicateRepairMode(str, Enum):
    """Grouping strategy for duplicate repair (SPEC Ch27)."""

    SYNC_KEY = "sync_key"
    CONTENT = "content"


class DescriptionMergeMode(str, Enum):
    """Per-group description handling during duplicate merge."""

    SKIP = "skip"
    LONGER = "longer"
    CONCAT = "concat"


class CategoryFilterMode(str, Enum):
    """``FilterConfig.category_mode`` (SPEC Ch8)."""

    NONE = "none"
    EXCLUDE = "exclude"
    INCLUDE_ONLY = "include_only"


class LogVerbosity(str, Enum):
    """Logger detail level (GUI setting)."""

    STANDARD = "standard"
    DETAILED = "detailed"


class MergePriority(str, Enum):
    """Field-level priority used by the ``merge`` conflict policy
    (SPEC Ch18)."""

    SOURCE = "source"
    GOOGLE = "google"


class BusyStatus(int, Enum):
    """Outlook ``BusyStatus`` enumeration as used in filters (Ch8)."""

    FREE = 0
    TENTATIVE = 1
    BUSY = 2
    OUT_OF_OFFICE = 3
