"""The slots of one policy seed differ only by their mechanism, end to end.

The outer ES compares the fitness of the four slots of a generation, which
train their fishers against four candidate quotas under one policy seed. On
2026-10-05 the shrunk fishery returned the same four slot-dependent values
(spread about 0.008) for every candidate below the binding range, because the
output layers and the exploration draws of each slot came from torch's global
stream. With seeded heads and keyed exploration draws, four candidates that
differ by a negligible spread must return four identical fitness values: the
quota does not bind there, so nothing but the slot could tell them apart.

The run is the debug script in a child process (it starts its own Ray
runtime), with the ES spread forced to 1e-4 by a small driver that patches
``ESConfig.training`` before calling the script's ``main``. The driver is code
passed with ``-c``, so the restart that ``ensure_hash_seed`` may perform
replays the patch.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CHILD_TIMEOUT_S = 180
NEGLIGIBLE_SIGMA = 1e-4
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
    r"\[ES\] BEFORE UPDATE \| gen=(\d+) .*\| fitness=\[([^\]]*)\]"
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
def test_slots_of_one_seed_return_identical_fitness_without_a_binding_quota(
    negligible_spread_run,
):
    assert negligible_spread_run["returncode"] == 0, negligible_spread_run["output"][
        -2000:
    ]

    generations = FITNESS_LINE.findall(negligible_spread_run["output"])
    assert [int(gen) for gen, _ in generations] == [0, 1]

    for gen, values in generations:
        fitness = [float(value) for value in values.split(",")]
        assert len(fitness) == 4
        assert fitness == [fitness[0]] * 4, f"generation {gen}: {fitness}"
