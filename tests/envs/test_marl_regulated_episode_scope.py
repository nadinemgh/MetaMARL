"""``MultiAgentEnv.reset``: the environment's metrics cover one episode only.

RLlib checks every environment it builds by resetting it and stepping it once
with a random action, outside any episode. The values that step logs must not
reach the first real episode, whose reduced metrics feed the regulator's
fitness, so ``reset`` starts every episode with an empty logger.
"""

import dataclasses
from types import SimpleNamespace

import pytest

from core.envs import marl_regulated
from core.envs.marl_regulated import MultiAgentEnv
from core.envs.schema import EpisodeRolloutSchema
from core.mechanism.base import MDPState
from core.metrics.logger import MetricLogger
from core.world.context import MechanismStatus


class SettableRewardAgent:
    """Follower whose reward at every step is ``self.value``."""

    def __init__(self, aid: str, reward: float) -> None:
        self.aid = aid
        self.value = reward

    def action(self, mdp: MDPState) -> MDPState:
        return mdp

    def reward(self, mdp: MDPState) -> MDPState:
        return MDPState(rewards={self.aid: self.value})

    def observation(self, mdp: MDPState) -> MDPState:
        return MDPState()


def make_env(follower: SettableRewardAgent) -> MultiAgentEnv:
    """``MultiAgentEnv`` with only the attributes ``reset`` and ``step`` touch."""
    env = MultiAgentEnv.__new__(MultiAgentEnv)
    env.world = SimpleNamespace(
        get_mechanism_by_id=SimpleNamespace(remote=lambda **kwargs: None)
    )
    env.logger = MetricLogger.from_schema(EpisodeRolloutSchema)
    env.mechanism_id = 0
    env.seed = env.policy_seed = 1
    env.mode = MechanismStatus.eval
    env.lids = set()
    env.followers = {follower.aid: follower}
    env.agents = dict(env.followers)
    env.horizon = None
    env.m = env.m_ctx = None
    env._using_default_mechanism = True
    env._t = 0
    env.transition = lambda mdp: dataclasses.replace(mdp, t=mdp.t + 1)
    return env


@pytest.fixture(autouse=True)
def ray_get_is_identity(monkeypatch):
    monkeypatch.setattr(marl_regulated.ray, "get", lambda ref: ref)


@pytest.mark.unit
def test_a_step_logged_before_reset_does_not_reach_the_episode():
    follower = SettableRewardAgent("fisherman", reward=100.0)
    env = make_env(follower)

    # The environment check: reset, then one step outside any episode.
    MultiAgentEnv.step(env, env.reset(MDPState(aids={"fisherman"})))

    follower.value = 1.0
    mdp = env.reset(MDPState(aids={"fisherman"}))
    for _ in range(2):
        mdp = MultiAgentEnv.step(env, mdp)

    reduced = env.logger.reduce()
    assert reduced.reward_mean == pytest.approx(1.0)
    assert reduced.reward_total == pytest.approx(2.0)
    assert reduced.reward_max == pytest.approx(1.0)
