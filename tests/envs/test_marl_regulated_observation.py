"""``MultiAgentEnv.step``: each step's observation is the one the agent built."""

from types import SimpleNamespace

import numpy as np
import pytest

from core.envs.marl_regulated import MultiAgentEnv
from core.envs.schema import EpisodeRolloutSchema
from core.mechanism.base import MDPState
from core.metrics.logger import MetricLogger


class StockObserver:
    """Agent that observes the stock of the current step and earns nothing."""

    def __init__(self, aid: str) -> None:
        self.aid = aid

    def action(self, mdp: MDPState) -> MDPState:
        return mdp

    def reward(self, mdp: MDPState) -> MDPState:
        return MDPState(rewards={self.aid: 0.0})

    def observation(self, mdp: MDPState) -> MDPState:
        fish = mdp.state["fish"][mdp.t]
        return MDPState(obs={self.aid: np.asarray([fish, 0.0], dtype=np.float32)})


def make_env(stocks: list[float]):
    """Stand-in ``self`` whose transition replays ``stocks``."""
    followers = {"fisherman:0": StockObserver("fisherman:0")}
    env = SimpleNamespace(
        agents=dict(followers),
        followers=followers,
        leaders={},
        logger=MetricLogger.from_schema(EpisodeRolloutSchema),
        horizon=None,
        _t=0,
    )
    env.transition = lambda mdp: mdp.advance(state={"fish": stocks[mdp.t + 1]})
    env.termination = lambda mdp: MultiAgentEnv.termination(env, mdp)
    return env


@pytest.mark.unit
def test_observation_is_the_current_stock_not_a_running_sum():
    stocks = [0.8, 0.7, 0.6, 0.5]
    env = make_env(stocks)
    mdp = MDPState(aids=set(env.followers), state={"fish": stocks[0]})
    mdp = mdp.add([agent.observation(mdp) for agent in env.followers.values()])

    observed = [float(mdp.obs["fisherman:0"][mdp.t][0])]
    for _ in range(3):
        mdp = MultiAgentEnv.step(env, mdp)
        observed.append(float(mdp.obs["fisherman:0"][mdp.t][0]))

    assert observed == pytest.approx(stocks)
