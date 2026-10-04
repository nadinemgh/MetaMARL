"""Unit tests for the composition helpers of ``core.utils``.

``intersect`` narrows gymnasium spaces (boxes and dictionaries of boxes) when
mechanisms are composed, ``logical_or_dict`` merges termination flags, and
``add`` sums additive quantities, recursing into dictionaries. ``None`` always
means "this component contributes nothing". The scalar and smooth helpers of
the same module are covered in ``test_core_utils.py``. ``add`` has no caller in
``core`` today (``MDPState.add`` uses ``Trajectory.add``), but it is public.
"""

from __future__ import annotations

import numpy as np
import pytest
from gymnasium import spaces

from core.utils import add, intersect, logical_or_dict


def box(low, high, dtype=np.float32):
    return spaces.Box(low=np.array(low, dtype=dtype), high=np.array(high, dtype=dtype))


# --------------------------------------------------------------------------- #
# intersect
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_intersect_without_deltas_returns_the_base_unchanged():
    base = box([0.0], [1.0])

    assert intersect(base, []) is base
    assert intersect(base, [None, None]) is base
    assert intersect(None, [None]) is None


@pytest.mark.unit
def test_intersect_of_a_missing_base_with_one_delta_returns_the_delta():
    delta = box([0.0], [1.0])

    assert intersect(None, [delta]) is delta
    assert intersect(None, [None, delta]) is delta


@pytest.mark.unit
def test_intersect_of_boxes_takes_the_tightest_bounds():
    base = box([0.0, -5.0], [10.0, 5.0])

    result = intersect(
        base, [box([2.0, -1.0], [12.0, 3.0]), box([1.0, -2.0], [8.0, 9.0])]
    )

    assert isinstance(result, spaces.Box)
    np.testing.assert_array_equal(result.low, [2.0, -1.0])
    np.testing.assert_array_equal(result.high, [8.0, 3.0])
    assert result.dtype == base.dtype


@pytest.mark.unit
def test_intersect_does_not_mutate_its_inputs():
    base = box([0.0], [10.0])
    delta = box([4.0], [6.0])

    intersect(base, [delta])

    np.testing.assert_array_equal(base.low, [0.0])
    np.testing.assert_array_equal(base.high, [10.0])
    np.testing.assert_array_equal(delta.low, [4.0])


@pytest.mark.unit
def test_intersect_without_base_folds_the_first_delta_into_the_others():
    result = intersect(None, [box([0.0], [10.0]), box([3.0], [7.0])])

    np.testing.assert_array_equal(result.low, [3.0])
    np.testing.assert_array_equal(result.high, [7.0])


@pytest.mark.unit
def test_intersect_keeps_a_degenerate_box_with_equal_bounds():
    result = intersect(box([0.0], [5.0]), [box([5.0], [9.0])])

    np.testing.assert_array_equal(result.low, [5.0])
    np.testing.assert_array_equal(result.high, [5.0])


@pytest.mark.unit
def test_intersect_of_disjoint_boxes_is_an_error():
    with pytest.raises(ValueError, match="empty space"):
        intersect(box([0.0], [1.0]), [box([2.0], [3.0])])


@pytest.mark.unit
def test_intersect_of_dicts_merges_keys_and_narrows_shared_ones():
    base = spaces.Dict({"a": box([0.0], [10.0]), "b": box([0.0], [1.0])})
    delta = spaces.Dict({"a": box([2.0], [4.0]), "c": box([-1.0], [1.0])})

    result = intersect(base, [delta])

    assert isinstance(result, spaces.Dict)
    assert set(result.spaces) == {"a", "b", "c"}
    np.testing.assert_array_equal(result["a"].low, [2.0])
    np.testing.assert_array_equal(result["a"].high, [4.0])
    np.testing.assert_array_equal(result["b"].high, [1.0])
    np.testing.assert_array_equal(result["c"].low, [-1.0])


@pytest.mark.unit
def test_intersect_of_dicts_recurses_into_nested_dicts():
    base = spaces.Dict({"inner": spaces.Dict({"x": box([0.0], [10.0])})})
    delta = spaces.Dict({"inner": spaces.Dict({"x": box([1.0], [3.0])})})

    result = intersect(base, [delta])

    np.testing.assert_array_equal(result["inner"]["x"].low, [1.0])
    np.testing.assert_array_equal(result["inner"]["x"].high, [3.0])


@pytest.mark.unit
@pytest.mark.parametrize(
    "base, delta, message",
    [
        (spaces.Dict({"a": box([0.0], [1.0])}), box([0.0], [1.0]), "Dict with Box"),
        (box([0.0], [1.0]), spaces.Dict({"a": box([0.0], [1.0])}), "Box with Dict"),
        (box([0.0], [1.0]), spaces.Discrete(3), "Box with Discrete"),
    ],
)
def test_intersect_rejects_mixed_space_types(base, delta, message):
    with pytest.raises(TypeError, match=message):
        intersect(base, [delta])


@pytest.mark.unit
def test_intersect_rejects_unsupported_space_types():
    with pytest.raises(TypeError, match="Cannot intersect spaces of type Discrete"):
        intersect(spaces.Discrete(3), [spaces.Discrete(3)])


# --------------------------------------------------------------------------- #
# add
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_add_without_deltas_returns_the_base():
    assert add(5.0, []) == 5.0
    assert add(5.0, [None]) == 5.0
    assert add(None, []) is None


@pytest.mark.unit
def test_add_sums_scalars_and_arrays():
    assert add(1.0, [2.0, None, 3.5]) == 6.5
    np.testing.assert_array_equal(
        add(np.array([1.0, 2.0]), [np.array([10.0, 20.0])]), [11.0, 22.0]
    )


@pytest.mark.unit
def test_add_without_base_sums_the_deltas():
    assert add(None, [4.0]) == 4.0
    assert add(None, [1.0, 2.0, 3.0]) == 6.0


@pytest.mark.unit
def test_add_merges_dictionaries_key_by_key():
    base = {"a": 1.0, "b": 2.0}

    result = add(base, [{"a": 10.0, "c": 5.0}, {"a": 100.0}])

    assert result == {"a": 111.0, "b": 2.0, "c": 5.0}
    assert base == {"a": 1.0, "b": 2.0}


@pytest.mark.unit
def test_add_recurses_into_nested_dictionaries():
    result = add({"x": {"y": 1.0}}, [{"x": {"y": 2.0, "z": 3.0}}])

    assert result == {"x": {"y": 3.0, "z": 3.0}}


# --------------------------------------------------------------------------- #
# logical_or_dict
# --------------------------------------------------------------------------- #


@pytest.mark.unit
def test_logical_or_dict_combines_flags_per_key():
    result = logical_or_dict(
        {"a": False, "b": True}, [{"a": True, "c": False}, {"d": True}]
    )

    assert result == {"a": True, "b": True, "c": False, "d": True}


@pytest.mark.unit
def test_logical_or_dict_accepts_missing_entries():
    assert logical_or_dict(None, [None, {}, {"a": True}]) == {"a": True}
    assert logical_or_dict({"a": True}, []) == {"a": True}
    assert logical_or_dict(None, []) == {}


@pytest.mark.unit
def test_logical_or_dict_coerces_values_to_bool_and_does_not_mutate():
    base = {"a": 0}

    result = logical_or_dict(base, [{"a": 2}])

    assert result == {"a": True}
    assert result["a"] is True
    assert base == {"a": 0}
