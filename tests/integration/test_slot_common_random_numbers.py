"""The slots of one policy seed differ only by their mechanism, end to end.

Run the fishery YAML in a child process with negligible ES spread (``1e-9``),
so four float32 candidates are identical. Their fitness must match exactly;
differences indicate that the slots do not share common random numbers.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "examples" / "bilevel_fishery" / "config.yaml"
CHILD_TIMEOUT_S = 180
NEGLIGIBLE_SIGMA = 1e-9
OVERRIDES = (
    "params.outer_iters=2",
    "params.train_iters=2",
    "params.num_agents=2",
    "params.horizon=20",
)
FITNESS_LINE = re.compile(
    r"\[ES\] BEFORE UPDATE \| gen=(\d+) .*"
    + r"\| population=(\[\[.*\]\]) \| fitness=\[([^\]]*)\]"
)


def config_with_negligible_spread(tmp_path: Path) -> Path:
    """Copy YAML with effectively identical float32 ES candidates."""

    data = yaml.safe_load(CONFIG.read_text())
    outer = next(
        c["regulator"] for c in data["experiment"]["_calls_"] if "regulator" in c
    )["_args_"][0]
    training = next(c["training"] for c in outer["_calls_"] if "training" in c)

    for field in ("sigma", "min_sigma", "max_sigma"):
        training[field] = NEGLIGIBLE_SIGMA

    config = tmp_path / "fishery_same_candidates.yaml"

    config.write_text(yaml.safe_dump(data, sort_keys=False))

    return config


@pytest.fixture(scope="module")
def negligible_spread_run(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Run two small generations with effectively identical candidates."""

    workdir = tmp_path_factory.mktemp("slot_common_random")
    config = config_with_negligible_spread(workdir)
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
        pytest.fail(
            f"Common-random-numbers experiment exceeded {CHILD_TIMEOUT_S}s: {exc}"
        )

    return {"returncode": done.returncode, "output": done.stdout + "\n" + done.stderr}


@pytest.mark.integration
def test_identical_candidates_of_one_seed_return_identical_fitness(
    negligible_spread_run: dict,
):
    assert negligible_spread_run["returncode"] == 0, negligible_spread_run["output"][
        -4000:
    ]

    generations = FITNESS_LINE.findall(negligible_spread_run["output"])

    assert [int(gen) for gen, _, _ in generations] == [0, 1], negligible_spread_run[
        "output"
    ][-4000:]

    for gen, population, values in generations:
        candidates = ast.literal_eval(population)
        fitness = [float(value) for value in values.split(",")]

        assert len(fitness) == 4
        assert candidates == [candidates[0]] * 4, f"generation {gen}: {candidates}"
        assert fitness == [fitness[0]] * 4, f"generation {gen}: {fitness}"
