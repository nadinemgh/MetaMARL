"""``Agent.action(reset=True)``: what an agent's mechanisms add to observations.

Since 778383b and 82fd437 the ``observe`` hook is gone: a mechanism contributes
to the initial observations through ``on_reset``, which ``Agent.action`` calls
when the environment resets. It hands each mechanism the last action the agent
holds for it, or ``None`` when it holds none, so a mechanism that writes nothing
without an action leaves the observations alone before the regulator's
candidate reaches the environment.
"""

import numpy as np
import pytest
from gymnasium import spaces

from core.agents.base import Agent
from core.mechanism.base import MDPState, Mechanism

BOX = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)


class Silent(Mechanism):
    """Mechanism that keeps the default ``on_reset``."""

    def apply(self, mdp: MDPState, action) -> MDPState:
        return MDPState()


class Beacon(Silent):
    """Mechanism that writes a constant into every follower's observation."""

    def on_reset(self, mdp: MDPState, action=None) -> MDPState:
        if action is None:
            return MDPState()

        return MDPState(obs={aid: np.asarray([0.0, 1.0]) for aid in mdp.aids})


def make_regulator() -> Agent:
    mechanisms = {
        mid: cls(aid="regulator", id=mid, action_space=BOX, acts_on=("fisherman", "h"))
        for mid, cls in (("silent", Silent), ("beacon", Beacon))
    }

    return Agent(id="regulator", policy_id="p", mechanisms=mechanisms)


@pytest.mark.unit
def test_on_reset_contributes_nothing_by_default():
    mechanism = Silent(aid="regulator", id="silent", action_space=BOX, acts_on=None)
    assert mechanism.on_reset(MDPState(aids={"fisherman:0"})).obs.data == {}


@pytest.mark.unit
def test_mechanisms_the_agent_acts_with_contribute():
    mdp = MDPState(
        aids={"fisherman:0"},
        actions={"regulator": {"silent": np.array([0.5]), "beacon": np.array([0.5])}},
    )
    contribution = make_regulator().action(mdp, reset=True)

    assert contribution.obs.data == {
        "fisherman:0": [pytest.approx(np.array([0.0, 1.0]))]
    }


@pytest.mark.unit
def test_no_contribution_without_an_action_for_the_mechanism():
    regulator = make_regulator()
    no_action = MDPState(aids={"fisherman:0"})
    other_mechanism = MDPState(
        aids={"fisherman:0"}, actions={"regulator": {"silent": np.array([0.5])}}
    )
    assert regulator.action(no_action, reset=True).obs.data == {}
    assert regulator.action(other_mechanism, reset=True).obs.data == {}
