"""The slots of one policy seed differ only by their mechanism, end to end.

The outer ES compares the fitness of the four slots of a generation, which
train their fishers against four candidate mechanisms under one policy seed. On
2026-10-05 the shrunk fishery returned four slot-dependent values (spread about
0.008) for four candidates that could not be told apart, because the output
layers and the exploration draws of each slot came from torch's global stream.
With seeded heads and keyed exploration draws, four identical candidates must
return four identical fitness values: nothing but the slot tells them apart.

The ES spread is forced to 1e-9, which float32 rounds away around the initial
mean of 0.5, so the four candidates of each generation are equal bit for bit.
The test used a spread of 1e-4 while the fishery searched only the quota, which
does not bind in that range. The debug script now also searches the restoration
subsidy, whose effect on the reward is continuous, and its quota has a soft
transition, so a spread of 1e-4 moves the fitness in its sixth decimal through
the mechanism alone. Measured on 2026-10-13 with that spread, generation 0 gave
2.3980112, 2.3980119, 2.3980114 and 2.3980107; with identical candidates the
four values are equal.

The run is the debug script in a child process (it starts its own Ray
runtime), with the ES spread forced by a small driver that patches
``ESConfig.training`` before calling the script's ``main``. The driver is code
passed with ``-c``, so the restart that ``ensure_hash_seed`` may perform
replays the patch.
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CHILD_TIMEOUT_S = 180
NEGLIGIBLE_SIGMA = 1e-9
SCRIPT_ARGS = [
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
]
DRIVER = f"""
from core.optimizers.es.config import ESConfig
from examples.bilevel_fishery import debug

original = ESConfig.training

def training(self, **kwargs):
    for key in ("sigma", "min_sigma", "max_sigma"):
        if key in kwargs:
            kwargs[key] = {NEGLIGIBLE_SIGMA!r}
    return original(self, **kwargs)

ESConfig.training = training
debug.main({SCRIPT_ARGS!r})
"""
FITNESS_LINE = re.compile(
    r"\[ES\] BEFORE UPDATE \| gen=(\d+) .*"
    + r"\| population=(\[\[.*\]\]) \| fitness=\[([^\]]*)\]"
)


@pytest.fixture(scope="module")
def negligible_spread_run(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """Run the shrunk fishery once with a negligible ES spread.

    Returns
    -------
    dict
        ``returncode`` and ``output`` (stdout and stderr, concatenated).
    """

    workdir = tmp_path_factory.mktemp("slot_common_random")
    env = dict(os.environ, WANDB_MODE="offline", PYTHONPATH=str(REPO_ROOT))

    env.pop("PYTHONHASHSEED", None)

    try:
        done = subprocess.run(
            [sys.executable, "-c", DRIVER],
            cwd=workdir,
            env=env,
            capture_output=True,
            text=True,
            timeout=CHILD_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        tail = (exc.stdout or b"")[-500:]

        pytest.fail(f"driver exceeded {CHILD_TIMEOUT_S} s; stdout tail: {tail!r}")

    return {"returncode": done.returncode, "output": done.stdout + "\n" + done.stderr}


@pytest.mark.integration
def test_identical_candidates_of_one_seed_return_identical_fitness(
    negligible_spread_run,
):
    assert negligible_spread_run["returncode"] == 0, negligible_spread_run["output"][
        -2000:
    ]

    generations = FITNESS_LINE.findall(negligible_spread_run["output"])

    assert [int(gen) for gen, _, _ in generations] == [0, 1]

    for gen, population, values in generations:
        candidates = ast.literal_eval(population)
        fitness = [float(value) for value in values.split(",")]

        assert len(fitness) == 4
        assert candidates == [candidates[0]] * 4, f"generation {gen}: {candidates}"
        assert fitness == [fitness[0]] * 4, f"generation {gen}: {fitness}"
