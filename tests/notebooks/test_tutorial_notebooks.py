"""Execution and hygiene tests of the tutorial notebooks in ``tutorials/``.

The tutorials teach the present API, so a notebook that no longer runs is a
broken tutorial. Until this pass most of their code sat in fenced markdown
blocks that never executed, and they drifted from the code without any test
failing. Every code example is now an executable cell, and this module runs
each notebook end to end.

The execution test starts ``jupyter nbconvert`` in a child process with a hard
time limit, so a hang fails the test instead of blocking the suite. The kernel
runs in ``tutorials/``, exactly as it does for a reader who opens the notebook
in Jupyter, and ``PYTHONPATH`` is removed from its environment: the notebooks
that import ``examples/`` must put the repository root on the path themselves.
The notebooks that train write their results below ``tutorials/``, in
directories the repository ignores. The test is marked ``notebook``; the
fishery tutorial starts its own Ray runtime inside the kernel. Run it with::

    WANDB_MODE=offline uv run python -m pytest -m notebook --no-cov

The hygiene tests are fast and run with the unit tests: every notebook is
committed without outputs and declares the kernel the execution test uses, so
a reader's kernel picker and the test agree.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TUTORIALS = REPO_ROOT / "tutorials"
NOTEBOOKS = sorted(TUTORIALS.glob("*.ipynb"))
KERNEL_NAME = "python3"
# The slowest tutorial trains a reduced bilevel fishery through Ray in about a
# minute; the limit leaves room for a slow CI runner without hiding a hang.
CHILD_TIMEOUT_S = 900


def _load(notebook: Path) -> dict:
    return json.loads(notebook.read_text(encoding="utf-8"))


def test_every_tutorial_is_collected():
    """The glob finds the five tutorials, so an empty list cannot pass silently."""
    assert len(NOTEBOOKS) == 5, [nb.name for nb in NOTEBOOKS]


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=[nb.stem for nb in NOTEBOOKS])
def test_tutorial_is_committed_without_outputs(notebook: Path):
    """No code cell carries outputs or an execution count."""
    dirty = [
        index
        for index, cell in enumerate(_load(notebook)["cells"])
        if cell["cell_type"] == "code"
        and (cell.get("outputs") or cell.get("execution_count") is not None)
    ]
    assert dirty == [], f"{notebook.name}: cells with outputs {dirty}"


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=[nb.stem for nb in NOTEBOOKS])
def test_tutorial_declares_the_python3_kernel(notebook: Path):
    """The kernel specification names the kernel the execution test starts."""
    kernelspec = _load(notebook).get("metadata", {}).get("kernelspec", {})
    assert kernelspec.get("name") == KERNEL_NAME, f"{notebook.name}: {kernelspec!r}"


@pytest.mark.notebook
@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=[nb.stem for nb in NOTEBOOKS])
def test_tutorial_executes(notebook: Path, tmp_path: Path):
    """The notebook runs from its first cell to its last without an error."""
    env = dict(os.environ, WANDB_MODE="offline")
    env.pop("PYTHONPATH", None)
    command = [
        sys.executable,
        "-m",
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        f"--ExecutePreprocessor.kernel_name={KERNEL_NAME}",
        f"--ExecutePreprocessor.timeout={CHILD_TIMEOUT_S}",
        "--output-dir",
        str(tmp_path),
        str(notebook),
    ]
    try:
        done = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=CHILD_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        tail = (exc.stderr or b"")[-2000:]
        pytest.fail(f"{notebook.name} exceeded {CHILD_TIMEOUT_S} s; stderr: {tail!r}")
    assert done.returncode == 0, done.stderr[-4000:]
    assert (tmp_path / notebook.name).is_file()
