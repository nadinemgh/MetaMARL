"""End-to-end smoke test of the fishery YAML CLI with the CSV reporter.

Launch the small experiment in a child process so Ray cannot hang pytest.
Write results in a temporary directory; leave the main YAML unchanged.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "examples" / "bilevel_fishery" / "config.yaml"
CHILD_TIMEOUT_S = 180
OVERRIDES = (
    "params.outer_iters=2",
    "params.train_iters=2",
    "params.num_agents=2",
    "params.horizon=20",
)


def csv_config(tmp_path: Path) -> Path:
    """Write a CSV-enabled copy of the existing experiment configuration."""

    data = yaml.safe_load(CONFIG.read_text())
    reporter = next(
        c["reporter"] for c in data["experiment"]["_calls_"] if "reporter" in c
    )
    reporter["config"] = {
        "_target_": "core.reporting.csv.CSVConfig",
        "project": "bilevel",
    }
    result = tmp_path / "fishery_csv.yaml"

    result.write_text(yaml.safe_dump(data, sort_keys=False))

    return result


@pytest.fixture(scope="module")
def csv_run(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Execute the small YAML fishery once in a child Ray process."""

    workdir = tmp_path_factory.mktemp("fishery_csv_smoke")
    config = csv_config(workdir)
    env = dict(os.environ, WANDB_MODE="offline", PYTHONPATH=str(REPO_ROOT))

    env.pop("PYTHONHASHSEED", None)

    try:
        done = subprocess.run(
            [sys.executable, "-m", "core.config.cli", "run", str(config), *OVERRIDES],
            cwd=workdir,
            env=env,
            capture_output=True,
            text=True,
            timeout=CHILD_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"Fishery YAML run exceeded {CHILD_TIMEOUT_S}s: {exc}")

    return {
        "returncode": done.returncode,
        "output": done.stdout + "\n" + done.stderr,
        "workdir": workdir,
    }


@pytest.mark.integration
def test_debug_script_runs_to_completion(csv_run):
    """The YAML CLI completes both ES generations."""

    assert csv_run["returncode"] == 0, csv_run["output"][-4000:]

    assert "Run finished | iters=2" in csv_run["output"]


@pytest.mark.integration
def test_debug_script_creates_the_reporter_directory(csv_run):
    project_dir = csv_run["workdir"] / "results" / "bilevel"

    assert project_dir.is_dir(), f"Missing reporter directory: {project_dir}"
    assert any(project_dir.iterdir())


@pytest.mark.integration
def test_debug_script_writes_csv_files(csv_run):
    csv_files = sorted((csv_run["workdir"] / "results").rglob("*.csv"))

    assert csv_files, f"No CSV files under {csv_run['workdir'] / 'results'}"


@pytest.mark.integration
def test_every_query_of_the_debug_script_renders(csv_run):
    # The reporter logs a failed query instead of raising; keep this check.
    assert "could not render query" not in csv_run["output"], csv_run["output"][-4000:]
