from __future__ import annotations

import hashlib
import importlib.resources
from pathlib import Path


def resource_root() -> Path:
    return Path(str(importlib.resources.files("analyst_forecast.resources")))


def read_text_resource(*parts: str) -> str:
    path = resource_root().joinpath(*parts)
    return path.read_text(encoding="utf-8")


def resource_sha256(*parts: str) -> str:
    content = read_text_resource(*parts).encode("utf-8")
    return hashlib.sha256(content).hexdigest()


def iter_resource_files(relative: str) -> list[tuple[str, Path]]:
    root = resource_root() / relative
    if not root.exists():
        return []
    results: list[tuple[str, Path]] = []
    for path in root.rglob("*"):
        if path.is_file():
            results.append((path.relative_to(root).as_posix(), path))
    return results
