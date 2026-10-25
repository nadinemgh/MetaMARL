"""What the fishers observe, step by step, under the regulator of ``debug.py``.

The fisher, the quota and the incentive each write part of the nine-component
observation as a difference to what the step already holds. Observations are
per-step flows, so these differences must be taken against the current step:
taken against the previous one, the slots the regulator writes alternate
between their target and zero from one step to the next. These tests drive the
real fishery behind the RLlib adapter and check every slot at every step.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from gymnasium import spaces

from core.adaptors.ray.marl_env import RLlibMultiAgentEnvAdapter
from core.agents.base import AgentConfig
from core.envs import marl_regulated
from core.mechanism.algorithms.incentive import Incentive
from core.mechanism.algorithms.quota import FINE_AMOUNT, MAX_DEMAND_FRAC, Quota
from core.reporting.csv import CSVConfig
from core.utils import sigmoid
from examples.bilevel_fishery.metric_schema import FisheryMetricSchema
from examples.bilevel_fishery.regulated_env import (
    FishermanConfig,
    FisheryRegulatedEnv,
    FishingConfig,
    RestoreConfig,
)
from tests.integration.conftest import CAPACITY, ECOLOGY, REGULATOR, ScriptedWorld

QUOTA = 0.5
SUBSIDY = 0.3
QUOTA_WIDTH = 0.05
STEPS = 8
OBSERVATION_SIZE = 9  # the layout of examples/bilevel_fishery/debug.py
FLOAT32_ATOL = 1e-6


def allowed_fraction(level: float) -> float:
    """Allowance of the quota at a normalised stock, as in ``Quota``."""

    lower = sigmoid((0.0 - QUOTA) / QUOTA_WIDTH)
    upper = sigmoid((1.0 - QUOTA) / QUOTA_WIDTH)
    current = sigmoid((level - QUOTA) / QUOTA_WIDTH)

    return (current - lower) / (upper - lower)


@pytest.fixture
def adapter(tmp_path, monkeypatch, candidate_context):
    monkeypatch.setattr(marl_regulated.ray, "get", lambda ref, *a, **k: ref)

    unit = spaces.Box(0.0, 1.0, (1,), np.float32)
    unbounded = spaces.Box(-np.inf, np.inf, (1,), np.float32)
    fisherman = FishermanConfig(
        id="fisherman",
        policy_id="fisher_policy",
        mechanisms=(
            FishingConfig(id="harvest", action_space=unbounded),
            RestoreConfig(id="restore", action_space=unbounded),
        ),
        observation_space=spaces.Box(-np.inf, np.inf, (OBSERVATION_SIZE,), np.float32),
    )
    regulator = AgentConfig(
        id=REGULATOR,
        policy_id="quota_policy",
        mechanisms=(
            Quota(
                id="quota",
                action_space=unit,
                acts_on=("fisherman", "harvest"),
                obs_map={"resource_level": "fish"},
                quota_transition_width=QUOTA_WIDTH,
            ),
            Incentive(
                id="restoration_subsidy",
                action_space=unit,
                acts_on=("fisherman", "restore"),
            ),
        ),
    )
    candidate = {
        "quota": np.array([QUOTA], dtype=np.float32),
        "restoration_subsidy": np.array([SUBSIDY], dtype=np.float32),
    }
    env = FisheryRegulatedEnv(
        world=ScriptedWorld([candidate_context(candidate)]),
        opt_id="inner",
        env_name="fishery_test",
        horizon=STEPS + 1,
        agents_cfg_dict={
            f"fisherman:{i}": dataclasses.replace(fisherman, id=f"fisherman:{i}")
            for i in range(2)
        },
        leaders_cfg_dict={REGULATOR: regulator},
        mechanism_id=0,
        seed=0,
        policy_seed=0,
        mode="train",
        reporter_cfg=CSVConfig(project="fishery_test", output_dir=str(tmp_path)),
        queries=(),
        schema=FisheryMetricSchema,
        ecology_cfg=ECOLOGY,
    )

    return RLlibMultiAgentEnvAdapter(env)


def expected_observation(adapter) -> np.ndarray:
    """Observation every fisher should receive at the current step."""

    mdp = adapter._mdp
    level = float(mdp.state["fish"][mdp.t]) / CAPACITY
    harvest = float(mdp.state["harvest"][mdp.t - 1]) / CAPACITY if mdp.t else 0.0
    expected = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
    expected[0] = level
    expected[1] = allowed_fraction(level)
    expected[2] = harvest
    expected[3] = QUOTA
    expected[4] = MAX_DEMAND_FRAC
    expected[5] = FINE_AMOUNT
    expected[8] = SUBSIDY

    return expected


@pytest.mark.unit
def test_every_slot_holds_its_target_at_every_step(adapter):
    observations, _ = adapter.reset()
    rng = np.random.default_rng(0)

    for _ in range(STEPS):
        for aid in adapter.possible_agents:
            np.testing.assert_allclose(
                observations[aid], expected_observation(adapter), atol=FLOAT32_ATOL
            )

        actions = {
            aid: {
                "harvest": rng.normal(size=1).astype(np.float32),
                "restore": rng.normal(size=1).astype(np.float32),
            }
            for aid in adapter.possible_agents
        }
        observations, *_ = adapter.step(actions)
