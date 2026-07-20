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
