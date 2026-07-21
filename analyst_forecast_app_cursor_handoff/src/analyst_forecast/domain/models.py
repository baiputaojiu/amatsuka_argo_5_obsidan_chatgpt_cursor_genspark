from enum import StrEnum


class Medium(StrEnum):
    YOUTUBE = "youtube"
    BLOG = "blog"
    X = "x"
    WEB = "web"


class Direction(StrEnum):
    UP = "up"
    DOWN = "down"
    FLAT = "flat"


class TimeSource(StrEnum):
    EXPLICIT = "explicit"
    INFERRED = "inferred"
    SYSTEM_DEFAULT = "system_default"
    NONE = "none"


class MappingStatus(StrEnum):
    UNRESOLVED = "unresolved"
    PROPOSED = "proposed"
    VERIFIED = "verified"
    CORRECTED = "corrected"
    MULTIPLE_PROXIES = "multiple_proxies"
    UNRESOLVABLE = "unresolvable"


class TargetResolutionStatus(StrEnum):
    """構成予想の対象解決段階（DB・workflow・NEXT_ACTIONSで共有）。"""

    PENDING = "pending"
    AWAITING_REVIEW = "awaiting_review"
    AWAITING_ADJUDICATION = "awaiting_adjudication"
    LOCKED = "locked"
    UNRESOLVABLE = "unresolvable"
    # legacy Schema 1.0.0 互換。v2の awaiting_review 相当として扱う。
    REVIEW_PENDING = "review_pending"
    # 旧実装の別名。awaiting_adjudication へ正規化する。
    NEEDS_ADJUDICATION = "needs_adjudication"
    # 旧実装がP11直後に書いた値。awaiting_review へ正規化する。
    PROPOSED = "proposed"


class ComponentResolutionStage(StrEnum):
    """workflowが次行動を決めるための正規化済み段階。"""

    NEED_P11 = "need_p11"
    NEED_P12 = "need_p12"
    NEED_P13 = "need_p13"
    READY_FOR_EVALUATION = "ready_for_evaluation"
    UNRESOLVABLE = "unresolvable"
    DONE = "done"


class EvaluationStatus(StrEnum):
    NOT_STARTED = "not_started"
    ACTIVE_ON_TRACK = "active_on_track"
    ACTIVE_OFF_TRACK = "active_off_track"
    ACTIVE_INDETERMINATE = "active_indeterminate"
    FULFILLED_EARLY = "fulfilled_early"
    FAILED_EARLY = "failed_early"
    EXPIRED_HIT = "expired_hit"
    EXPIRED_MISS = "expired_miss"
    CONDITION_PENDING = "condition_pending"
    WITHDRAWN = "withdrawn"
    SUPERSEDED = "superseded"
    UNEVALUABLE = "unevaluable"
