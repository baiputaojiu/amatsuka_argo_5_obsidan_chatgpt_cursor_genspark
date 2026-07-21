from __future__ import annotations

import time
from pathlib import Path


def atomic_write_text(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding=encoding, newline="\n")
    last_error: OSError | None = None
    for _ in range(5):
        try:
            temporary.replace(path)
            return
        except PermissionError as error:
            last_error = error
            time.sleep(0.05)
            try:
                if path.exists():
                    path.unlink()
                temporary.replace(path)
                return
            except OSError as nested:
                last_error = nested
                time.sleep(0.05)
    if last_error is not None:
        raise last_error
