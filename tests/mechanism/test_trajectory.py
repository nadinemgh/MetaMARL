"""``Trajectory`` and ``MDPState``: explicit per-step flow initialization.

All MDPState fields use ``Trajectory`` and therefore carry unwritten values
forward. An environment explicitly resets rewards and observations at each
new timestep before adding contributions; deltas within that timestep sum.
"""

import numpy as np
import pytest

from core.mechanism.base import MDPState
from core.mechanism.types import Trajectory


def _at(mdp: MDPState, t: int) -> MDPState:
    mdp.t = t
    return mdp


@pytest.mark.unit
def test_reward_of_a_step_does_not_include_earlier_steps():
    mdp = MDPState()

    for t, harvest in enumerate([0.5, 0.25, 0.75]):
        mdp = _at(mdp, t)
        # Trajectory does not reset itself.
        # The environment initializes each new step's reward before adding it.
        mdp.update(rewards={"a": 0.0})
        mdp = mdp.add(MDPState(rewards={"a": harvest}))

    assert mdp.rewards["a"] == [0.5, 0.25, 0.75]


@pytest.mark.unit
def test_reward_deltas_at_the_same_step_are_summed():
    mdp = MDPState().add(MDPState(rewards={"a": 1.0}))
    mdp = _at(mdp, 1)
    mdp.update(rewards={"a": 0.0})
    mdp = mdp.add(
        [MDPState(rewards={"a": 1.0}), MDPState(rewards={"a": -0.25})]
    )

    assert mdp.rewards["a"] == [1.0, 0.75]


@pytest.mark.unit
def test_unwritten_rewards_carry_forward_until_explicitly_reset():
    # Trajectory fills gaps with the last value, even when storing rewards.
    mdp = _at(MDPState(rewards={"a": [2.0, 3.0, 4.0], "b": 1.0}), 2)
    mdp.update(rewards={"b": 0.0})
    mdp = mdp.add(MDPState(rewards={"b": 5.0}))

    assert mdp.rewards["b"] == [1.0, 1.0, 5.0]
    assert mdp.rewards["a"] == [2.0, 3.0, 4.0]

    # Explicit initialization replaces the value carried to the new timestep.
    advanced = MDPState(rewards={"a": 2.0}).advance(rewards={"a": 0.0})

    assert advanced.rewards["a"] == [2.0, 0.0]


@pytest.mark.unit
def test_state_still_carries_forward():
    mdp = _at(MDPState(state={"fish": 10.0}), 1).add(MDPState(state={"fish": -2.0}))

    assert mdp.state["fish"] == [10.0, 8.0]


@pytest.mark.unit
def test_mdp_state_wraps_each_field_in_its_trajectory_type():
    mdp = MDPState(state={"fish": 1.0}, rewards={"a": 1.0}, obs={"a": 1.0})

    assert type(mdp.state) is Trajectory
    assert type(mdp.actions) is Trajectory
    assert type(mdp.rewards) is Trajectory
    assert type(mdp.obs) is Trajectory

    assert type(mdp.add(MDPState(rewards={"a": 1.0})).rewards) is Trajectory
    assert type(mdp.add(MDPState(obs={"a": 1.0})).obs) is Trajectory
    assert type(MDPState().obs) is Trajectory


@pytest.mark.unit
def test_observation_of_a_step_does_not_include_earlier_steps():
    mdp = MDPState()

    for t, fish_norm in enumerate([0.8, 0.7, 0.6]):
        mdp = _at(mdp, t)
        # The environment initializes the current observation before mechanisms
        # and follower observations contribute their deltas.
        mdp.update(obs={"a": np.zeros(2, dtype=np.float32)})
        mdp = mdp.add(MDPState(obs={"a": np.array([fish_norm, 0.0])}))

    np.testing.assert_allclose(mdp.obs["a"], [[0.8, 0.0], [0.7, 0.0], [0.6, 0.0]])


@pytest.mark.unit
def test_observation_contributions_at_the_same_step_are_summed():
    own = MDPState(obs={"a": np.array([0.8, 0.0])})
    from_mechanism = MDPState(obs={"a": np.array([0.0, 0.3])})
    mdp = _at(MDPState().add(own), 1)
    mdp.update(obs={"a": np.zeros(2, dtype=np.float32)})
    mdp = mdp.add([own, from_mechanism])

    np.testing.assert_allclose(mdp.obs["a"][0], [0.8, 0.0])
    np.testing.assert_allclose(mdp.obs["a"][1], [0.8, 0.3])
