"""``empty_action_space``: the action space of a mechanism with nothing to search."""

import numpy as np
import pytest
from gymnasium import spaces

from core.mechanism.config import empty_action_space


@pytest.mark.unit
def test_the_space_is_a_box_without_any_dimension():
    space = empty_action_space()

    assert isinstance(space, spaces.Box)
    assert space.shape == (0,)
    assert space.dtype == np.float32


@pytest.mark.unit
def test_the_bounds_hold_no_value():
    space = empty_action_space()

    assert space.low.size == 0
    assert space.high.size == 0
    assert space.bounded_below.size == 0


@pytest.mark.unit
def test_sampling_and_membership_use_an_empty_array():
    space = empty_action_space()
    sample = space.sample()

    assert sample.size == 0
    assert space.contains(sample)

    assert space.contains(np.zeros(0, dtype=np.float32))
    assert not space.contains(np.zeros(1, dtype=np.float32))


@pytest.mark.unit
def test_each_call_returns_a_new_space():
    assert empty_action_space() is not empty_action_space()
