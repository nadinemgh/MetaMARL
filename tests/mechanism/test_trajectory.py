"""``Trajectory`` and ``FlowTrajectory``: how unwritten timesteps are filled.

States are stocks: a timestep that no transition wrote keeps the previous
value, and a delta added at ``t`` is applied on top of it. Rewards are flows:
each timestep starts from zero, and deltas added at the same ``t`` (the
agent's utility, a penalty, a subsidy) are summed. Observations follow the
same rule as rewards: every agent rebuilds its observation at each step, and a
mechanism may add its own contribution to it.
"""

import numpy as np
import pytest

from core.mechanism.base import MDPState
from core.mechanism.types import FlowTrajectory, Trajectory


def _at(mdp: MDPState, t: int) -> MDPState:
    mdp.t = t
    return mdp


@pytest.mark.unit
def test_reward_of_a_step_does_not_include_earlier_steps():
    mdp = MDPState()
    for t, harvest in enumerate([0.5, 0.25, 0.75]):
        mdp = _at(mdp, t).add(MDPState(rewards={"a": harvest}))

    assert mdp.rewards["a"] == [0.5, 0.25, 0.75]


@pytest.mark.unit
def test_reward_deltas_at_the_same_step_are_summed():
    mdp = MDPState().add(MDPState(rewards={"a": 1.0}))
    mdp = _at(mdp, 1).add(
        [MDPState(rewards={"a": 1.0}), MDPState(rewards={"a": -0.25})]
    )

    assert mdp.rewards["a"] == [1.0, 0.75]


@pytest.mark.unit
def test_unwritten_reward_steps_are_zero():
    # "b" was not written at the two latest steps of the tree, and starts from
    # zero when it is written again.
    rewards = {"a": [2.0, 3.0, 4.0], "b": 1.0}
    mdp = _at(MDPState(rewards=rewards), 2).add(MDPState(rewards={"b": 5.0}))
    assert mdp.rewards["b"] == [1.0, 0, 5.0]
    assert mdp.rewards["a"] == [2.0, 3.0, 4.0]

    advanced = MDPState(rewards={"a": 2.0}).advance(rewards={})
    assert advanced.rewards["a"] == [2.0, 0]


@pytest.mark.unit
def test_state_still_carries_forward():
    mdp = _at(MDPState(state={"fish": 10.0}), 1).add(MDPState(state={"fish": -2.0}))

    assert mdp.state["fish"] == [10.0, 8.0]


@pytest.mark.unit
def test_mdp_state_wraps_each_field_in_its_trajectory_type():
    mdp = MDPState(state={"fish": 1.0}, rewards={"a": 1.0}, obs={"a": 1.0})

    assert type(mdp.state) is Trajectory
    assert type(mdp.actions) is Trajectory
    assert type(mdp.rewards) is FlowTrajectory
    assert type(mdp.obs) is FlowTrajectory
    assert type(mdp.add(MDPState(rewards={"a": 1.0})).rewards) is FlowTrajectory
    assert type(mdp.add(MDPState(obs={"a": 1.0})).obs) is FlowTrajectory
    assert type(MDPState().obs) is FlowTrajectory


@pytest.mark.unit
def test_observation_of_a_step_does_not_include_earlier_steps():
    mdp = MDPState()
    for t, fish_norm in enumerate([0.8, 0.7, 0.6]):
        mdp = _at(mdp, t).add(MDPState(obs={"a": np.array([fish_norm, 0.0])}))

    np.testing.assert_allclose(mdp.obs["a"], [[0.8, 0.0], [0.7, 0.0], [0.6, 0.0]])


@pytest.mark.unit
def test_observation_contributions_at_the_same_step_are_summed():
    own = MDPState(obs={"a": np.array([0.8, 0.0])})
    from_mechanism = MDPState(obs={"a": np.array([0.0, 0.3])})

    mdp = _at(MDPState().add(own), 1).add([own, from_mechanism])

    np.testing.assert_allclose(mdp.obs["a"][1], [0.8, 0.3])
