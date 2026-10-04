"""End-to-end smoke test of the fishery debug script with the CSV reporter.

This is the port of the August ``integration/test_fishery_reporting.py``. The
August version built the configuration in-process and ran it inside the pytest
process, which is what hung on an earlier version of the code. This version
launches ``examples.bilevel_fishery.debug`` as a child process with a hard time
limit, so a hang fails the test instead of blocking the suite.

The script reads its command line at import time and writes its results below
the working directory (``results/<project>/...``), so the child runs in a
temporary directory. No test here starts Ray in the pytest process; the child
starts and stops its own local Ray runtime, which is why the test is marked
``integration`` and not ``unit``.

The run is a two-generation, two-fisherman, twenty-step fishery. The first
test checks that the run completes and reports its final summary. The second
test checks that the CSV reporter actually leaves ``.csv`` files behind; it is
expected to fail today because the reporter object is created (its directories
appear) but never receives any metrics.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CHILD_TIMEOUT_S = 180
SCRIPT_ARGS = (
    "--outer-iters",
    "2",
    "--train-iters",
    "2",
    "--num-agents",
    "2",
    "--horizon",
    "20",
    "--reporter",
    "csv",
)


@pytest.fixture(scope="module")
def csv_run(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Run the debug script once and share the result between the tests.

    Returns
    -------
    dict
        ``returncode``, ``output`` (stdout and stderr, concatenated) and
        ``workdir`` (the directory the child ran in).
    """
    workdir = tmp_path_factory.mktemp("debug_csv_smoke")
    env = dict(os.environ, WANDB_MODE="offline", PYTHONPATH=str(REPO_ROOT))
    # A fixed hash seed would make the child's world identifiers repeat across runs.
    env.pop("PYTHONHASHSEED", None)
    command = [sys.executable, "-m", "examples.bilevel_fishery.debug", *SCRIPT_ARGS]
    try:
        done = subprocess.run(
            command,
            cwd=workdir,
            env=env,
            capture_output=True,
            text=True,
            timeout=CHILD_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        tail = (exc.stdout or b"")[-500:]
        pytest.fail(f"debug script exceeded {CHILD_TIMEOUT_S} s; stdout tail: {tail!r}")
    return {
        "returncode": done.returncode,
        "output": done.stdout + "\n" + done.stderr,
        "workdir": workdir,
    }


@pytest.mark.integration
def test_debug_script_runs_to_completion(csv_run):
    assert csv_run["returncode"] == 0, csv_run["output"][-2000:]
    assert "Run finished | iters=2" in csv_run["output"]


@pytest.mark.integration
def test_debug_script_creates_the_reporter_directory(csv_run):
    project_dir = csv_run["workdir"] / "results" / "bilevel"
    assert project_dir.is_dir()
    assert any(project_dir.iterdir())


@pytest.mark.integration
@pytest.mark.xfail(
    strict=True, reason="the CSV reporter is created but no .csv file is ever written"
)
def test_debug_script_writes_csv_files(csv_run):
    csv_files = sorted((csv_run["workdir"] / "results").rglob("*.csv"))
    assert csv_files, "no CSV file under results/"
