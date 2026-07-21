"""話者・著者名の正規化と分析対象者照合。"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Literal

VerifiedAttributionStatus = Literal[
    "target_confirmed",
    "uncertain",
    "not_target",
    "legacy_unknown",
    "needs_review",
]

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_person_name(value: str | None) -> str:
    """Unicode NFKC + 空白正規化 + casefold の決定的正規化。"""
    if value is None:
        return ""
    normalized = unicodedata.normalize("NFKC", value)
    normalized = _WHITESPACE_RE.sub(" ", normalized.strip())
    return normalized.casefold()


def analyst_name_set(analyst: Any) -> set[str]:
    """canonical_name / normalized_name / aliases の正規化集合。"""
    names = {
        normalize_person_name(getattr(analyst, "canonical_name", None)),
        normalize_person_name(getattr(analyst, "normalized_name", None)),
    }
    aliases = getattr(analyst, "aliases", None) or []
    for alias in aliases:
        names.add(normalize_person_name(str(alias)))
    names.discard("")
    return names


def person_matches_analyst(candidate: str | None, analyst: Any) -> bool:
    """曖昧一致は使わず、正規化後の exact match のみ。"""
    normalized = normalize_person_name(candidate)
    if not normalized:
        return False
    return normalized in analyst_name_set(analyst)


@dataclass(frozen=True)
class AttributionVerification:
    claimed_status: str
    verified_status: VerifiedAttributionStatus
    reason: str
    matched_speaker: str | None = None
    segment_refs: tuple[str, ...] = ()


def verify_forecast_attribution(
    *,
    claimed_status: str,
    statement_kind: str,
    speaker_candidate: str | None,
    upstream_segment_refs: list[str],
    segments_by_ref: dict[str, Any],
    analyst: Any,
) -> AttributionVerification:
    """AI申告とは別に、Pythonが検証した attribution を返す。"""
    refs = tuple(upstream_segment_refs)
    if statement_kind == "third_party_summary":
        return AttributionVerification(
            claimed_status=claimed_status,
            verified_status="not_target",
            reason="第三者要約は正式予想にしない",
            segment_refs=refs,
        )
    if claimed_status in {"not_target", "uncertain", "legacy_unknown"}:
        return AttributionVerification(
            claimed_status=claimed_status,
            verified_status=claimed_status,  # type: ignore[arg-type]
            reason=f"AI申告が{claimed_status}のため正式化しない",
            matched_speaker=speaker_candidate,
            segment_refs=refs,
        )
    if not refs:
        return AttributionVerification(
            claimed_status=claimed_status,
            verified_status="legacy_unknown",
            reason="upstream_segment_refsが空のため正式化しない",
            matched_speaker=speaker_candidate,
            segment_refs=refs,
        )

    speakers: list[str] = []
    for ref in refs:
        segment = segments_by_ref.get(ref)
        if segment is None:
            return AttributionVerification(
                claimed_status=claimed_status,
                verified_status="needs_review",
                reason=f"未知のsegment参照: {ref}",
                matched_speaker=speaker_candidate,
                segment_refs=refs,
            )
        status = getattr(segment, "speaker_status", None)
        candidate = getattr(segment, "speaker_candidate", None)
        if status in {"unknown", None} or not candidate:
            return AttributionVerification(
                claimed_status=claimed_status,
                verified_status="needs_review",
                reason="上流segmentの話者がunknownのためP06解決が必要",
                matched_speaker=candidate,
                segment_refs=refs,
            )
        speakers.append(str(candidate))

    unique_speakers = {normalize_person_name(item) for item in speakers}
    if len(unique_speakers) != 1:
        return AttributionVerification(
            claimed_status=claimed_status,
            verified_status="needs_review",
            reason="参照segmentの話者が一致しない",
            matched_speaker=speakers[0] if speakers else None,
            segment_refs=refs,
        )

    statement_speaker = speakers[0]
    if not person_matches_analyst(statement_speaker, analyst):
        return AttributionVerification(
            claimed_status=claimed_status,
            verified_status="not_target",
            reason="上流話者が分析対象者のcanonical/aliasと一致しない",
            matched_speaker=statement_speaker,
            segment_refs=refs,
        )

    if claimed_status == "target_confirmed":
        return AttributionVerification(
            claimed_status=claimed_status,
            verified_status="target_confirmed",
            reason="上流話者が分析対象者とNFKC exact一致",
            matched_speaker=statement_speaker,
            segment_refs=refs,
        )

    return AttributionVerification(
        claimed_status=claimed_status,
        verified_status="legacy_unknown",
        reason="検証可能な本人一致だが申告がtarget_confirmedではない",
        matched_speaker=statement_speaker,
        segment_refs=refs,
    )


def is_formal_verified(verification: AttributionVerification) -> bool:
    return verification.verified_status == "target_confirmed"
