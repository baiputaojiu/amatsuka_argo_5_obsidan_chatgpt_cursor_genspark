from dataclasses import dataclass, field


@dataclass
class SyncResult:
    created: int = 0
    updated: int = 0
    deleted_candidates: int = 0
    deleted: int = 0
    skipped: int = 0
    merged: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)
    delete_candidates_list: list[dict] = field(default_factory=list)
