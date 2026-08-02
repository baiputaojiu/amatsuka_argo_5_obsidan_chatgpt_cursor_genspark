"""R4-045: packaged docs stay in sync with repo docs."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_r4_045_sync_packaged_docs_check() -> None:
    """R4-045: scripts/sync_packaged_docs.py --check exits 0."""
    script = PROJECT_ROOT / "scripts" / "sync_packaged_docs.py"
    assert script.is_file()
    result = subprocess.run(
        [sys.executable, str(script), "--check"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"docs sync drift detected:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
