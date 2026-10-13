"""``Agent.mechanism_observations``: what an agent's mechanisms add to observations.

A mechanism contributes only while its agent holds an action for it, the same
rule ``Agent.action`` follows, so nothing is written before the regulator's
candidate reaches the environment.
"""

import numpy as np
import pytest
from gymnasium import spaces

from core.agents.base import Agent
from core.mechanism.base import MDPState, Mechanism

BOX = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)


class Silent(Mechanism):
    """Mechanism that keeps the default ``observe``."""

    def apply(self, mdp: MDPState, action) -> MDPState:
        return MDPState()


class Beacon(Silent):
    """Mechanism that writes a constant into every follower's observation."""

    def observe(self, mdp: MDPState) -> MDPState:
        return MDPState(obs={aid: np.asarray([0.0, 1.0]) for aid in mdp.aids})


def make_regulator() -> Agent:
    mechanisms = {
        mid: cls(aid="regulator", id=mid, action_space=BOX, acts_on=("fisherman", "h"))
        for mid, cls in (("silent", Silent), ("beacon", Beacon))
    }
    return Agent(id="regulator", policy_id="p", mechanisms=mechanisms)


@pytest.mark.unit
def test_observe_contributes_nothing_by_default():
    mechanism = Silent(aid="regulator", id="silent", action_space=BOX, acts_on=None)

    assert mechanism.observe(MDPState(aids={"fisherman:0"})).obs.data == {}


@pytest.mark.unit
def test_mechanisms_the_agent_acts_with_contribute():
    mdp = MDPState(
        aids={"fisherman:0"},
        actions={"regulator": {"silent": np.array([0.5]), "beacon": np.array([0.5])}},
    )

    contributions = make_regulator().mechanism_observations(mdp)

    assert [c.obs.data for c in contributions if c.obs.data] == [
        {"fisherman:0": [pytest.approx(np.array([0.0, 1.0]))]}
    ]


@pytest.mark.unit
def test_no_contribution_without_an_action_for_the_mechanism():
    regulator = make_regulator()
    no_action = MDPState(aids={"fisherman:0"})
    other_mechanism = MDPState(
        aids={"fisherman:0"}, actions={"regulator": {"silent": np.array([0.5])}}
    )

    assert regulator.mechanism_observations(no_action) == []
    assert all(
        c.obs.data == {} for c in regulator.mechanism_observations(other_mechanism)
    )
