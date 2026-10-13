"""``MultiAgentEnv.step``: per-step metrics pushed to the environment's logger."""

import dataclasses
from types import SimpleNamespace

import pytest

from core.envs.marl_regulated import MultiAgentEnv
from core.envs.schema import EpisodeRolloutSchema
from core.mechanism.base import MDPState
from core.metrics.logger import MetricLogger


class ConstantRewardAgent:
    """Agent whose reward at every step is a fixed number."""

    def __init__(self, aid: str, reward: float) -> None:
        self.aid = aid
        self.value = reward

    def action(self, mdp: MDPState) -> MDPState:
        return mdp

    def reward(self, mdp: MDPState) -> MDPState:
        return MDPState(rewards={self.aid: self.value})

    def observation(self, mdp: MDPState) -> MDPState:
        return MDPState()


def make_env(rewards: dict[str, float]):
    """Stand-in ``self`` exposing what ``MultiAgentEnv.step`` touches."""
    followers = {aid: ConstantRewardAgent(aid, r) for aid, r in rewards.items()}
    env = SimpleNamespace(
        agents=dict(followers),
        followers=followers,
        leaders={},
        logger=MetricLogger.from_schema(EpisodeRolloutSchema),
        horizon=None,
    )
    env.transition = lambda mdp: dataclasses.replace(mdp, t=mdp.t + 1)
    env.termination = lambda mdp: MultiAgentEnv.termination(env, mdp)
    return env


@pytest.mark.unit
def test_step_logs_the_mean_follower_reward():
    env = make_env({"fisherman_0": 1.0, "fisherman_1": 3.0})
    mdp = MDPState(aids=set(env.followers))

    for _ in range(3):
        mdp = MultiAgentEnv.step(env, mdp)

    # Each step's reward is the mean of the two fishermen's rewards, 2.0.
    reduced = env.logger.reduce()
    assert reduced.reward_mean == pytest.approx(2.0)
    assert reduced.reward_total == pytest.approx(6.0)
    assert reduced.reward_min == reduced.reward_max == pytest.approx(2.0)
    assert reduced.reward_terminal == pytest.approx(2.0)
