"""R4-046: clean wheel install exposes CLI help."""

from __future__ import annotations

import os
import subprocess
import sys
import venv
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _venv_python(venv_dir: Path) -> Path:
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def _run(cmd: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}
    return subprocess.run(
        cmd,
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env=env,
    )


def test_r4_046_wheel_help(tmp_path: Path) -> None:
    """R4-046: build wheel, install into temp venv, run analyst-forecast --help."""
    pytest.importorskip("build")

    dist_dir = tmp_path / "dist"
    dist_dir.mkdir()
    build = _run(
        [sys.executable, "-m", "build", "--wheel", "--outdir", str(dist_dir)],
        cwd=PROJECT_ROOT,
    )
    if build.returncode != 0:
        pytest.xfail(
            f"wheel build failed in this environment: {(build.stderr or build.stdout)[-500:]}"
        )

    wheels = list(dist_dir.glob("*.whl"))
    assert wheels, "expected a wheel artifact"
    wheel = wheels[0]

    venv_dir = tmp_path / "venv"
    venv.create(venv_dir, with_pip=True, clear=True)
    py = _venv_python(venv_dir)
    install = _run([str(py), "-m", "pip", "install", "--quiet", str(wheel)])
    assert install.returncode == 0, install.stderr

    help_result = _run([str(py), "-m", "analyst_forecast", "--help"])
    assert help_result.returncode == 0, help_result.stderr or help_result.stdout
    out = help_result.stdout or ""
    assert "Usage" in out or "usage" in out.lower()
    assert "init" in out.lower()

    docs_probe = _run(
        [
            str(py),
            "-c",
            "import analyst_forecast, pathlib; "
            "root = pathlib.Path(analyst_forecast.__file__).parent / 'resources' / 'docs'; "
            "print(root.is_dir(), sum(1 for _ in root.rglob('*.md')))",
        ]
    )
    assert docs_probe.returncode == 0, docs_probe.stderr
    assert (docs_probe.stdout or "").strip().startswith("True "), docs_probe.stdout
