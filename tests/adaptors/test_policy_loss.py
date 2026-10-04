"""``get_policy_loss_if_present``: reading the policy loss from an RLlib result.

On the new API stack the losses live under ``learners/<module>/policy_loss``,
next to an ``__all_modules__`` entry without one. APPO's learner reports its
stats only once every 20 gradient updates; on the iterations in between the
module entries hold NaN, and the reader must answer NaN rather than average
them in.
"""

import math

import numpy as np
import pytest

from core.adaptors.ray.utils import get_policy_loss_if_present


@pytest.mark.unit
def test_new_stack_losses_are_averaged_over_modules():
    result = {
        "learners": {
            "__all_modules__": {"num_env_steps_trained": 1680},
            "fisher_policy_m0_s1": {"policy_loss": np.float64(-1.0)},
            "fisher_policy_m1_s1": {"policy_loss": np.float64(-2.0)},
        }
    }

    assert get_policy_loss_if_present(result) == pytest.approx(-1.5)


@pytest.mark.unit
def test_iteration_without_learner_report_gives_nan():
    result = {
        "learners": {
            "__all_modules__": {"num_env_steps_trained": float("nan")},
            "fisher_policy_m0_s1": {"policy_loss": float("nan")},
        }
    }

    assert math.isnan(get_policy_loss_if_present(result))
    assert math.isnan(get_policy_loss_if_present({}))


@pytest.mark.unit
def test_classic_layout_is_still_read():
    result = {"info": {"learner": {"p0": {"learner_stats": {"policy_loss": 0.5}}}}}

    assert get_policy_loss_if_present(result) == pytest.approx(0.5)
