"""``MultiAgentEnv``: the observation has the size its follower declares.

The leaders' ``on_reset`` hooks run before the followers observe. A mechanism
that writes a residual of its own fixed size must therefore not decide the size
of the observation. The environment opens every follower's observation with
zeros of the declared shape, and these tests check that the declared size
survives a residual that is smaller, at reset and after a step.
"""

from typing import Any, ClassVar

import numpy as np
import pytest
from gymnasium import spaces

from core.agents.base import AgentConfig
from core.mechanism.base import MDPState, Mechanism
from core.mechanism.config import MechanismConfig

from .conftest import FEE_SPACE, UNBOUNDED, Fisher, FisherConfig, HarvestConfig

DECLARED_SIZE = 11
RESIDUAL_SIZE = 9
FOLLOWERS = ["f0", "f1"]
AIDS = set(FOLLOWERS)
OBSERVATION_SPACE = spaces.Box(
    low=-np.inf, high=np.inf, shape=(DECLARED_SIZE,), dtype=np.float32
)


class SizedFisher(Fisher):
    """Follower that fills slot 0 of an observation it finds at the current step.

    The residual it returns has the size of what the step already holds, which
    is how the fishery's fisher behaves: it does not know the declared size
    unless the environment has put an entry of that size there.
    """

    def observation(self, mdp: MDPState) -> MDPState:
        history = mdp.obs.data.get(self.id, [])
        current = (
            np.asarray(history[mdp.t], dtype=np.float32)
            if len(history) > mdp.t
            else np.zeros(RESIDUAL_SIZE, dtype=np.float32)
        )
        residual = np.zeros_like(current)
        residual[0] = mdp.state["stock"][mdp.t] - current[0]

        return MDPState(obs={self.id: residual})


class SizedFisherConfig(FisherConfig):
    agent_cls: ClassVar[type] = SizedFisher


class ResidualFee(Mechanism):
    """Leader mechanism that writes into the observation, with a nine-slot fallback.

    It takes the size of the entry the follower already holds at the current
    step and falls back to ``RESIDUAL_SIZE`` slots when there is none, as the
    fishery's quota and incentive mechanisms do.
    """

    def decode(self, mdp: MDPState, action: Any) -> float:
        return float(np.asarray(action).reshape(-1)[0])

    def apply(self, mdp: MDPState, action: float) -> MDPState:
        return MDPState()

    def on_reset(self, mdp: MDPState, action: Any = None) -> MDPState:
        def residual_for(aid: str) -> np.ndarray:
            history = mdp.obs.data.get(aid, [])
            size = len(history[mdp.t]) if len(history) > mdp.t else RESIDUAL_SIZE
            residual = np.zeros(size, dtype=np.float32)
            residual[1] = 0.0 if action is None else float(action)

            return residual

        return MDPState(obs={aid: residual_for(aid) for aid in mdp.aids})


class ResidualFeeConfig(MechanismConfig):
    mechanism_cls: ClassVar[type[Mechanism]] = ResidualFee


def make_sized_env(toy):
    leaders = {
        "regulator": AgentConfig(
            id="regulator",
            policy_id="regulator_policy",
            mechanisms=ResidualFeeConfig(action_space=FEE_SPACE, id="fee"),
        )
    }
    followers = {
        aid: SizedFisherConfig(
            id=aid,
            policy_id="fisher_policy",
            mechanisms=HarvestConfig(action_space=UNBOUNDED, id="harvest"),
            observation_space=OBSERVATION_SPACE,
        )
        for aid in FOLLOWERS
    }

    return toy.make_env(
        toy.ScriptedWorld([toy.candidate(0.25)]),
        agents_cfg_dict=followers,
        leaders_cfg_dict=leaders,
    )


@pytest.mark.unit
def test_a_smaller_leader_residual_does_not_shrink_the_declared_observation(
    toy, identity_ray_get
):
    env = make_sized_env(toy)

    mdp = env.reset(MDPState(aids=AIDS))

    for aid in FOLLOWERS:
        assert mdp.obs[aid][0].shape == (DECLARED_SIZE,)
        # Slot 0 is the stock, slot 1 the leader's fee, the rest stays zero.
        expected = np.zeros(DECLARED_SIZE, dtype=np.float32)
        expected[0], expected[1] = toy.initial_stock, 0.25
        np.testing.assert_allclose(mdp.obs[aid][0], expected)

    for amount in (0.1, 0.1):
        raw = toy.harvest_actions(FOLLOWERS, amount)
        decoded = {
            aid: {
                mid: env.followers[aid].mechanisms[mid].decode(mdp, a)
                for mid, a in acts.items()
            }
            for aid, acts in raw.items()
        }
        mdp.update(actions=decoded)
        mdp = env.step(mdp)

        for aid in FOLLOWERS:
            assert mdp.obs[aid][mdp.t].shape == (DECLARED_SIZE,)
            assert mdp.obs[aid][mdp.t][1] == pytest.approx(0.25)
