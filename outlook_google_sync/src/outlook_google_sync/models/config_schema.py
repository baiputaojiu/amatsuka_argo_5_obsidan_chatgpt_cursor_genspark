"""Backward-compatible re-export of ``SCHEMA_VERSION``.

The authoritative definition now lives in :mod:`outlook_google_sync.constants`.
This module is retained because existing imports
(``from outlook_google_sync.models.config_schema import SCHEMA_VERSION``) and
external tests rely on this path. New code should import from ``constants``.
"""

from ..constants import SCHEMA_VERSION

__all__ = ["SCHEMA_VERSION"]
