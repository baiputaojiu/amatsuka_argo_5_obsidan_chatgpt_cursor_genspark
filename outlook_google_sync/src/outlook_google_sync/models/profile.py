from dataclasses import dataclass, field


@dataclass
class FilterConfig:
    """Ch8: exclusion filters."""
    exclude_all_day: bool = False
    exclude_free: bool = False
    exclude_tentative: bool = False
    exclude_private: bool = False
    category_mode: str = "none"  # "none" | "exclude" | "include_only"
    category_list: list[str] = field(default_factory=list)
    subject_keywords: list[str] = field(default_factory=list)
    location_keywords: list[str] = field(default_factory=list)


@dataclass
class MergeSettings:
    tolerance_minutes: int = 2
    description_priority: str = "source"  # "source" | "google"
    location_priority: str = "source"


@dataclass
class Profile:
    profile_id: str = "default"
    display_name: str = "Default"
    input_method: str = "com"
    detail_level: str = "full"
    include_private_appointments: bool = True
    default_date_range_mode: str = "relative_60"
    target_calendar_id: str = "primary"
    conflict_policy: str = "overwrite"
    merge_settings: MergeSettings = field(default_factory=MergeSettings)
    notification_enabled: bool = True
    log_verbosity: str = "standard"
    filter: FilterConfig = field(default_factory=FilterConfig)
    sync_metadata: dict = field(default_factory=lambda: {
        "last_success_utc": None,
        "last_run_mode": None,
        "last_D_start": None,
        "last_D_end": None,
        "last_conditions_hash": None,
        "per_source_fingerprint": {},
    })
