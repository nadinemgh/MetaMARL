"""``SubsidyMechanism``: which agents are paid.

Complements ``test_subsidy.py`` with the case of a follower that holds no
action for the subsidised mechanism, which the mechanism must skip rather than
pay or fail on.
"""

import numpy as np
import pytest
from gymnasium import spaces

from core.mechanism.algorithms.subsidy import MAX_SUBSIDY, Subsidy
from core.mechanism.base import MDPState

REGULATOR = "regulator"


def zero_effort_action() -> np.ndarray:
    """Policy output whose decoded effort is ``1/2``."""
    return np.array([0.0], dtype=np.float32)


@pytest.mark.unit
def test_agent_without_the_subsidised_mechanism_is_not_paid():
    mechanism = Subsidy(
        id="subsidy",
        action_space=spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        acts_on=("fisherman", "restore"),
        cost=0.0,
    ).build(REGULATOR)
    mdp = MDPState(
        actions={
            REGULATOR: {"subsidy": np.array([1.0], dtype=np.float32)},
            "fisherman:0": {"restore": zero_effort_action()},
            "fisherman:1": {"harvest": zero_effort_action()},
        }
    )

    delta = mechanism(mdp, mdp.actions[REGULATOR]["subsidy"][0])

    assert set(delta.rewards.data) == {"fisherman:0"}
    assert delta.rewards["fisherman:0"][0] == pytest.approx(MAX_SUBSIDY * 0.5, abs=1e-6)
