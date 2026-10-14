"""``IncentiveMechanism``: which agents are paid.

Complements ``test_subsidy.py`` with the case of a follower that holds no
action for the subsidised mechanism, which the mechanism must skip rather than
pay or fail on.
"""

import numpy as np
import pytest
from gymnasium import spaces

from core.mechanism.algorithms.incentive import MAX, Incentive
from core.mechanism.base import MDPState

REGULATOR = "regulator"


def half_effort_action() -> np.float32:
    """Decoded effort ``1/2``, as the adapter stores it in the state."""

    return np.float32(0.5)


@pytest.mark.unit
def test_agent_without_the_subsidised_mechanism_is_not_paid():
    mechanism = Incentive(
        id="subsidy",
        action_space=spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        acts_on=("fisherman", "restore"),
    ).build(REGULATOR)
    mdp = MDPState(
        actions={
            REGULATOR: {"subsidy": np.float32(1.0)},
            "fisherman:0": {"restore": half_effort_action()},
            "fisherman:1": {"harvest": half_effort_action()},
        }
    )
    delta = mechanism(mdp, mdp.actions[REGULATOR]["subsidy"][0])

    assert set(delta.rewards.data) == {"fisherman:0"}
    assert delta.rewards["fisherman:0"][0] == pytest.approx(MAX * 0.5, abs=1e-6)
