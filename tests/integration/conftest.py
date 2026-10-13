"""Builders for the fishery integration tests.

Nothing here starts Ray. ``build_fishery`` assembles the real
``FisheryRegulatedEnv`` of the example behind the real RLlib adapter, with a
regulator that holds the mechanisms the test names, and replaces the ``World``
actor by an object that replays a script of candidates. The fixtures return
builders, so that each test chooses its mechanisms and its candidate.
"""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from gymnasium import spaces

from core.adaptors.ray.marl_env import RLlibMultiAgentEnvAdapter
from core.agents.base import AgentConfig
from core.envs import marl_regulated
from core.mechanism.config import MechanismConfig
from core.reporting.csv import CSVConfig
from core.world.context import MechanismContext, MechanismStatus
from examples.bilevel_fishery.metric_schema import FisheryMetricSchema
from examples.bilevel_fishery.regulated_env import (
    FishermanConfig,
    FisheryRegulatedEnv,
    FishingConfig,
    RestoreConfig,
)

CAPACITY = 5_000.0
"""Carrying capacity ``K`` of the shrunk fishery, in biomass units."""

ECOLOGY = {
    "r": 0.3,
    "K": CAPACITY,
    "p": 1.0,
    "B0": 4_000,
    "fish_init": 4_000,
    "sigma": 0.02,
    "initial_stock_log_sigma": 0.05,
    "unregulated_f_multiplier": 2.0,
}
"""Ecology of ``examples/bilevel_fishery/debug.py``, unchanged."""

OBSERVATION_SIZE = 5
REGULATOR = "fisheries_regulator"


class ScriptedWorld:
    """Stand-in for the ``World`` actor that replays a script of candidates.

    The real ``World.get_mechanism_by_id`` hands a training candidate out once
    and returns ``None`` afterwards; ``answers`` lists what successive fetches
    return, and every fetch after the last answer returns ``None``.
    """

    def __init__(self, answers: list[MechanismContext | None]) -> None:
        self.answers = list(answers)
        self.calls = 0
        self.get_mechanism_by_id = SimpleNamespace(remote=self._fetch)

    def _fetch(self, **kwargs: Any) -> MechanismContext | None:
        self.calls += 1
        return self.answers.pop(0) if self.answers else None


@pytest.fixture
def candidate_context():
    """Factory of the ``MechanismContext`` the regulator publishes."""

    def factory(mechanism: dict[str, np.ndarray]) -> MechanismContext:
        return MechanismContext(
            index=0,
            env_id=None,
            seed=0,
            status=MechanismStatus.train,
            mechanism=mechanism,
            metrics=None,
        )

    return factory


@pytest.fixture
def build_fishery(tmp_path, monkeypatch):
    """Factory of a shrunk fishery behind the RLlib adapter.

    Parameters of the returned builder: the regulator's mechanism
    configurations (positional), ``candidates`` (the script of the world),
    ``num_agents`` (default 2), ``horizon`` (default 20) and ``seed``.
    It returns a namespace with the ``adapter``, the ``env`` and the ``world``.
    """
    monkeypatch.setattr(marl_regulated.ray, "get", lambda ref, *a, **k: ref)

    def build(
        *regulator_mechanisms: MechanismConfig,
        candidates: list[MechanismContext | None],
        num_agents: int = 2,
        horizon: int = 20,
        seed: int = 0,
    ) -> SimpleNamespace:
        unbounded = spaces.Box(-np.inf, np.inf, (1,), np.float32)
        fisherman = FishermanConfig(
            id="fisherman",
            policy_id="fisher_policy",
            mechanisms=(
                FishingConfig(id="harvest", action_space=unbounded),
                RestoreConfig(id="restore", action_space=unbounded),
            ),
            observation_space=spaces.Box(
                -np.inf, np.inf, (OBSERVATION_SIZE,), np.float32
            ),
        )
        followers = {
            f"fisherman:{i}": dataclasses.replace(fisherman, id=f"fisherman:{i}")
            for i in range(num_agents)
        }
        regulator = AgentConfig(
            id=REGULATOR, policy_id="quota_policy", mechanisms=regulator_mechanisms
        )
        world = ScriptedWorld(candidates)
        env = FisheryRegulatedEnv(
            world=world,
            opt_id="inner",
            env_name="fishery_test",
            horizon=horizon,
            agents_cfg_dict=followers,
            leaders_cfg_dict={REGULATOR: regulator},
            mechanism_id=0,
            seed=seed,
            policy_seed=seed,
            mode="train",
            reporter_cfg=CSVConfig(project="fishery_test", output_dir=str(tmp_path)),
            queries=(),
            schema=FisheryMetricSchema,
            ecology_cfg=ECOLOGY,
        )
        return SimpleNamespace(
            adapter=RLlibMultiAgentEnvAdapter(env), env=env, world=world
        )

    return build
