"""``MultiAgentEnv``: what the followers observe at reset and after each step.

Each step's observation is the one the agent built for that step, plus what
the leaders' mechanisms contribute to it.
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


class PeerSignalLeader:
    """Leader whose mechanisms write ``t + 1`` into the second entry."""

    def action(self, mdp: MDPState) -> MDPState:
        return mdp

    def reward(self, mdp: MDPState) -> MDPState:
        return MDPState()

    def mechanism_observations(self, mdp: MDPState) -> list[MDPState]:
        signal = np.asarray([0.0, float(mdp.t + 1)], dtype=np.float32)
        return [MDPState(obs={"fisherman:0": signal})]


def make_env(stocks: list[float], leaders: dict | None = None):
    """Stand-in ``self`` whose transition replays ``stocks``."""
    followers = {"fisherman:0": StockObserver("fisherman:0")}
    leaders = leaders or {}
    env = SimpleNamespace(
        agents={**leaders, **followers},
        followers=followers,
        leaders=leaders,
        logger=MetricLogger.from_schema(EpisodeRolloutSchema),
        horizon=None,
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


@pytest.mark.unit
def test_step_adds_the_leaders_contribution_to_the_new_observation():
    stocks = [0.8, 0.7, 0.6]
    env = make_env(stocks, leaders={"regulator": PeerSignalLeader()})
    mdp = MDPState(aids=set(env.followers), state={"fish": stocks[0]})
    mdp = mdp.add([agent.observation(mdp) for agent in env.followers.values()])

    for _ in range(2):
        mdp = MultiAgentEnv.step(env, mdp)

    # The contribution is computed after the transition, on the new step.
    np.testing.assert_allclose(mdp.obs["fisherman:0"][1], [0.7, 2.0])
    np.testing.assert_allclose(mdp.obs["fisherman:0"][2], [0.6, 3.0])


@pytest.mark.unit
def test_reset_adds_the_leaders_contribution(monkeypatch):
    monkeypatch.setattr(marl_regulated.ray, "get", lambda ref: ref)
    env = MultiAgentEnv.__new__(MultiAgentEnv)
    env.world = SimpleNamespace(
        get_mechanism_by_id=SimpleNamespace(remote=lambda **kwargs: None)
    )
    env.logger = MetricLogger.from_schema(EpisodeRolloutSchema)
    env.mechanism_id = 0
    env.seed = env.policy_seed = 1
    env.mode = MechanismStatus.train
    env.lids = {"regulator"}
    env.leaders = {"regulator": PeerSignalLeader()}
    env.followers = {"fisherman:0": StockObserver("fisherman:0")}
    env.m = env.m_ctx = None
    env._using_default_mechanism = True

    mdp = env.reset(MDPState(state={"fish": 0.8}))

    np.testing.assert_allclose(mdp.obs["fisherman:0"][0], [0.8, 1.0])
