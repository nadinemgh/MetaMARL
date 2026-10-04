"""Unit tests for ``core.utils``.

Covers the tolerant scalar conversions used by the reporting layer
(``to_float``, ``finite``, ``safe_ratio``, ``flatten_numeric``,
``sanitize_key``, ``is_mapping``, ``generate_uuid``) and the smooth reward
shaping helpers (``sigmoid``, ``smooth_positive``, ``smooth_min``,
``smooth_cap_01``, ``smooth_positive_zero_at_origin``). The docstring examples
of the module are replayed here as explicit assertions.
"""

from __future__ import annotations

import math
import uuid
from collections import OrderedDict

import numpy as np
import pytest

from core import utils
from core.utils import (
    EPS,
    finite,
    flatten_numeric,
    generate_uuid,
    is_mapping,
    safe_ratio,
    sanitize_key,
    sigmoid,
    smooth_cap_01,
    smooth_min,
    smooth_positive,
    smooth_positive_zero_at_origin,
    to_float,
)

# ---------------------------------------------------------------------------
# generate_uuid
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_generate_uuid_docstring_example():
    uid = generate_uuid({"a", "b"})
    assert uid not in {"a", "b"}
    uuid.UUID(uid)  # valid UUID string


@pytest.mark.unit
def test_generate_uuid_retries_on_collision(monkeypatch):
    taken = "00000000-0000-0000-0000-000000000000"
    fresh = "11111111-1111-1111-1111-111111111111"
    draws = iter([uuid.UUID(taken), uuid.UUID(fresh)])
    monkeypatch.setattr(utils.uuid, "uuid4", lambda: next(draws))
    assert generate_uuid({taken}) == fresh


@pytest.mark.unit
def test_generate_uuid_accepts_dict_keys_view():
    registry = {"x": 1}
    assert generate_uuid(registry.keys()) != "x"


# ---------------------------------------------------------------------------
# to_float / finite / safe_ratio
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_to_float_docstring_example():
    assert (to_float(np.float32(2.5)), to_float("abc")) == (2.5, None)


@pytest.mark.unit
@pytest.mark.parametrize(
    "value, expected",
    [
        (None, None),
        (3, 3.0),
        ("4.5", 4.5),
        (np.int64(7), 7.0),
        (np.array(2.0), 2.0),
        (True, 1.0),
        ([1, 2], None),
        (object(), None),
    ],
)
def test_to_float_cases(value, expected):
    assert to_float(value) == expected


@pytest.mark.unit
def test_to_float_passes_nan_and_inf_through():
    assert math.isnan(to_float(float("nan")))
    assert to_float(np.inf) == np.inf


@pytest.mark.unit
@pytest.mark.parametrize(
    "value, expected",
    [
        (1.5, 1.5),
        (np.float64(2.0), 2.0),
        (None, None),
        ("nope", None),
        (float("nan"), None),
        (float("inf"), None),
        (-np.inf, None),
    ],
)
def test_finite(value, expected):
    assert finite(value) == expected


@pytest.mark.unit
def test_safe_ratio_docstring_example():
    assert safe_ratio(1.0, 0.0) is None


@pytest.mark.unit
@pytest.mark.parametrize(
    "num, den, expected",
    [
        (1.0, 4.0, 0.25),
        (np.float32(3), 2, 1.5),
        (None, 1.0, None),
        (1.0, None, None),
        (float("nan"), 1.0, None),
        (1.0, float("inf"), None),
        ("a", 1.0, None),
        (0.0, 5.0, 0.0),
    ],
)
def test_safe_ratio(num, den, expected):
    assert safe_ratio(num, den) == expected


# ---------------------------------------------------------------------------
# sanitize_key / is_mapping / flatten_numeric
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sanitize_key_docstring_example():
    assert sanitize_key("info/stock level (t)") == "info_stock_level_t_"


@pytest.mark.unit
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("already_ok-1", "already_ok-1"),
        ("a  b", "a_b"),
        ("é/à", "_"),
        (123, "123"),
        ("", ""),
    ],
)
def test_sanitize_key_cases(raw, expected):
    assert sanitize_key(raw) == expected


@pytest.mark.unit
def test_is_mapping():
    assert is_mapping({})
    assert is_mapping(OrderedDict())
    assert not is_mapping([("a", 1)])
    assert not is_mapping("str")
    assert not is_mapping(None)


@pytest.mark.unit
def test_flatten_numeric_docstring_example():
    assert flatten_numeric([[1, 2], [3, 4]]) == [1.0, 2.0, 3.0, 4.0]


@pytest.mark.unit
def test_flatten_numeric_scalar_and_arrays():
    assert flatten_numeric(5) == [5.0]
    assert flatten_numeric(np.float32(1.5)) == [1.5]
    assert flatten_numeric(np.zeros((2, 3))) == [0.0] * 6
    assert flatten_numeric([]) == []
    out = flatten_numeric(np.arange(6).reshape(3, 2))
    assert out == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    assert all(isinstance(v, float) for v in out)


# ---------------------------------------------------------------------------
# Smooth helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_sigmoid_docstring_example():
    assert sigmoid(0.0) == 0.5


@pytest.mark.unit
def test_sigmoid_matches_reference_and_is_stable():
    for x in [-5.0, -0.3, 0.7, 4.0]:
        assert sigmoid(x) == pytest.approx(1.0 / (1.0 + math.exp(-x)))
    # No overflow warnings / nan for large magnitudes.
    assert sigmoid(1000.0) == 1.0
    assert sigmoid(-1000.0) == 0.0
    assert sigmoid(np.float32(2)) == pytest.approx(sigmoid(2.0))
    assert isinstance(sigmoid(1), float)


@pytest.mark.unit
def test_smooth_positive_properties():
    width = 0.5
    assert smooth_positive(0.0, width) == pytest.approx(width * math.log(2.0))
    assert smooth_positive(10.0, width) == pytest.approx(10.0, abs=1e-6)
    assert 0.0 < smooth_positive(-10.0, width) < 1e-6
    # Width is clamped to EPS: zero and negative widths behave like a hard hinge.
    assert smooth_positive(1.0, 0.0) == pytest.approx(1.0)
    assert smooth_positive(-1.0, -3.0) == 0.0
    assert smooth_positive(0.0, 0.0) == pytest.approx(EPS * math.log(2.0))


@pytest.mark.unit
def test_smooth_min_blends_between_operands():
    a, b = 3.0, 1.0
    out = smooth_min(a, b, width=0.1)
    assert min(a, b) <= out <= max(a, b)
    assert out == pytest.approx(b, abs=1e-6)  # a >> b leans to b
    assert smooth_min(b, a, width=0.1) == pytest.approx(
        b, abs=1e-6
    )  # a << b leans to a
    assert smooth_min(2.0, 2.0, width=1.0) == 2.0  # equal operands
    # Explicit formula check.
    w = sigmoid((a - b) / 1.0)
    assert smooth_min(a, b, width=1.0) == pytest.approx((1 - w) * a + w * b)
    # Clamped width.
    assert smooth_min(a, b, width=0.0) == pytest.approx(b)


@pytest.mark.unit
def test_smooth_cap_01():
    assert smooth_cap_01(0.0) == 0.0
    assert smooth_cap_01(1.0) == pytest.approx(1.0 - math.exp(-1.0))
    assert smooth_cap_01(50.0) == pytest.approx(1.0)
    assert smooth_cap_01(-1.0) < 0.0
    assert isinstance(smooth_cap_01(np.float32(0.5)), float)


@pytest.mark.unit
def test_smooth_positive_zero_at_origin():
    width = 0.25
    assert smooth_positive_zero_at_origin(0.0, width) == 0.0
    assert smooth_positive_zero_at_origin(-2.0, width) == 0.0
    expected = smooth_positive(3.0, width) - width * math.log(2.0)
    assert smooth_positive_zero_at_origin(3.0, width) == pytest.approx(expected)
    assert smooth_positive_zero_at_origin(3.0, width) < 3.0
    assert smooth_positive_zero_at_origin(1.0, 0.0) == pytest.approx(1.0)
