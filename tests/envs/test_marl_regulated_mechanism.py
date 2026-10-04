"""``MultiAgentEnv.reset``: every episode carries the candidate mechanism.

``World.get_mechanism_by_id`` returns a training candidate only on its first
fetch; later fetches return ``None``. The environment must keep what it
fetched and give it to the leaders at every reset, and switch to a newer
candidate as soon as the World publishes one.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from core.envs import marl_regulated
from core.envs.marl_regulated import MultiAgentEnv
from core.envs.schema import EpisodeRolloutSchema
from core.mechanism.base import MDPState
from core.metrics.logger import MetricLogger
from core.world.context import MechanismStatus


class ScriptedWorld:
    """World stand-in whose ``get_mechanism_by_id`` replays fixed answers."""

    def __init__(self, answers: list) -> None:
        self.answers = list(answers)
        self.calls = 0
        self.get_mechanism_by_id = SimpleNamespace(remote=self._fetch)

    def _fetch(self, **kwargs):
        self.calls += 1
        return self.answers.pop(0)


def make_env(world: ScriptedWorld) -> MultiAgentEnv:
    """``MultiAgentEnv`` with only the attributes ``reset`` touches."""
    env = MultiAgentEnv.__new__(MultiAgentEnv)
    env.world = world
    env.logger = MetricLogger.from_schema(EpisodeRolloutSchema)
    env.mechanism_id = 0
    env.seed = env.policy_seed = 1
    env.mode = MechanismStatus.train
    env.lids = {"regulator"}
    env.leaders = {}
    env.followers = {}
    env.m = env.m_ctx = None
    env._using_default_mechanism = True
    return env


def quota_at_reset(env: MultiAgentEnv) -> float | None:
    mdp = env.reset(MDPState())
    if "regulator" not in mdp.actions:
        return None
    return float(np.asarray(mdp.actions["regulator"]["quota"][0]).item())


def candidate(quota: float):
    return SimpleNamespace(mechanism={"quota": np.array([quota], dtype=np.float32)})


@pytest.fixture(autouse=True)
def ray_get_is_identity(monkeypatch):
    monkeypatch.setattr(marl_regulated.ray, "get", lambda ref: ref)


@pytest.mark.unit
def test_later_episodes_keep_the_fetched_candidate():
    env = make_env(ScriptedWorld([candidate(0.3), None, None]))

    quotas = [quota_at_reset(env) for _ in range(3)]

    assert quotas == pytest.approx([0.3, 0.3, 0.3])


@pytest.mark.unit
def test_a_newly_published_candidate_replaces_the_kept_one():
    env = make_env(ScriptedWorld([candidate(0.3), None, candidate(0.7), None]))

    quotas = [quota_at_reset(env) for _ in range(4)]

    assert quotas == pytest.approx([0.3, 0.3, 0.7, 0.7])
    assert env.world.calls == 4


@pytest.mark.unit
def test_no_candidate_before_the_first_publication():
    env = make_env(ScriptedWorld([None, candidate(0.3)]))

    assert quota_at_reset(env) is None
    assert quota_at_reset(env) == pytest.approx(0.3)
