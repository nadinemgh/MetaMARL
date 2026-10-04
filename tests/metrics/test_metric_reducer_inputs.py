"""Metric reducers on the inputs that decide what a report shows.

A reduction window is the set of values pushed between two reductions. The
framework reduces the environment's logger once per episode, and an untouched
metric must then reduce to a defined value: ``None`` for the mean, minimum,
maximum and last value (a schema field is ``Optional``), ``0`` for the sum and
an empty list for a series. This file checks those empty windows, single
values, NaN and infinite values, integer inputs, the independence of
successive windows, and the protocol table of the factory.

Behaviour that the code does not settle is deliberately not asserted. In
particular ``MaxMetric`` and ``MinMetric`` return a result that depends on the
push order when a NaN is present (Python's ``max`` and ``min`` compare with
``>`` and ``<``, which are false against NaN), and the intended result is
undecided, so no test pins it.
"""

import math

import numpy as np
import pytest

from core.metrics.enums import ReduceProtocol
from core.metrics.metric.base import Metric
from core.metrics.metric.factory import MetricFactory
from core.metrics.metric.last import LastMetric
from core.metrics.metric.max import MaxMetric
from core.metrics.metric.mean import MeanMetric
from core.metrics.metric.min import MinMetric
from core.metrics.metric.series import SeriesMetric
from core.metrics.metric.sum import SumMetric

NAN = float("nan")
SCALAR_METRICS = [MeanMetric, MinMetric, MaxMetric, SumMetric, LastMetric]
NUMERIC_METRICS = [MeanMetric, MinMetric, MaxMetric, SumMetric]


def filled(cls, values):
    metric = cls()
    for value in values:
        metric.push(value)
    return metric


@pytest.mark.unit
@pytest.mark.parametrize(
    "cls, empty",
    [
        (MeanMetric, None),
        (MinMetric, None),
        (MaxMetric, None),
        (LastMetric, None),
        (SumMetric, 0),
        (SeriesMetric, []),
    ],
    ids=lambda v: v.__name__ if isinstance(v, type) else repr(v),
)
class TestEmptyWindow:
    """A window with no pushed value reduces to a defined, stable result."""

    def test_peek_and_reduce_agree_on_the_empty_value(self, cls, empty):
        metric = cls()
        assert metric.peek() == empty
        assert metric.reduce() == empty
        assert metric.reduce() == empty  # reducing twice does not change it

    def test_the_empty_value_has_the_documented_type(self, cls, empty):
        reduced = cls().reduce()
        assert type(reduced) is type(empty)

    def test_a_window_emptied_by_reduce_is_empty_again(self, cls, empty):
        metric = filled(cls, [1, 2, 3])
        metric.reduce()
        assert len(metric) == 0
        assert metric.peek() == empty

    def test_a_window_emptied_by_flush_is_empty_again(self, cls, empty):
        metric = filled(cls, [1, 2, 3])
        metric.flush()
        assert len(metric) == 0
        assert metric.reduce() == empty


@pytest.mark.unit
class TestSingleValue:
    @pytest.mark.parametrize(
        "cls, pushed, expected",
        [
            (MeanMetric, 3, 3.0),
            (MeanMetric, -2.5, -2.5),
            (MinMetric, 3, 3),
            (MinMetric, -2.5, -2.5),
            (MaxMetric, 3, 3),
            (MaxMetric, -2.5, -2.5),
            (SumMetric, 3, 3),
            (SumMetric, -2.5, -2.5),
            (LastMetric, 3, 3),
            (LastMetric, "label", "label"),
            (SeriesMetric, 3, [3]),
        ],
    )
    def test_single_value_reduces_to_itself(self, cls, pushed, expected):
        metric = filled(cls, [pushed])
        assert metric.peek() == expected
        assert metric.reduce() == expected
        assert len(metric) == 0

    def test_zero_is_a_value_not_an_empty_window(self):
        """``0`` must not be confused with the empty sentinel ``None``."""
        for cls in (MeanMetric, MinMetric, MaxMetric, LastMetric):
            assert filled(cls, [0]).reduce() == 0
            assert filled(cls, [0]).reduce() is not None


@pytest.mark.unit
class TestNumericTypes:
    def test_mean_of_integers_is_a_float(self):
        reduced = filled(MeanMetric, [1, 2]).reduce()
        assert reduced == 1.5 and isinstance(reduced, float)

    @pytest.mark.parametrize("cls", [SumMetric, MinMetric, MaxMetric])
    def test_integer_inputs_keep_the_integer_type(self, cls):
        reduced = filled(cls, [4, 2, 3]).reduce()
        assert isinstance(reduced, int) and not isinstance(reduced, bool)

    def test_numpy_float64_is_accepted_as_a_float(self):
        """``np.float64`` subclasses ``float``, so the type check lets it through."""
        metric = filled(MeanMetric, [np.float64(1.0), np.float64(3.0)])
        assert metric.reduce() == 2.0

    @pytest.mark.parametrize("cls", NUMERIC_METRICS)
    def test_rejected_value_leaves_the_window_untouched(self, cls):
        metric = filled(cls, [1.0, 2.0])
        with pytest.raises(TypeError, match=f"{cls.__name__} only accepts"):
            metric.push(None)
        assert metric.peek(compile=False) == [1.0, 2.0]


@pytest.mark.unit
class TestNonFiniteValues:
    @pytest.mark.parametrize("cls", SCALAR_METRICS + [SeriesMetric])
    def test_lone_nan_is_reported_as_nan(self, cls):
        reduced = filled(cls, [NAN]).reduce()
        values = reduced if isinstance(reduced, list) else [reduced]
        assert len(values) == 1 and math.isnan(values[0])

    @pytest.mark.parametrize("cls", [MeanMetric, SumMetric])
    @pytest.mark.parametrize("values", [[NAN, 1.0], [1.0, NAN], [1.0, NAN, 3.0]])
    def test_mean_and_sum_propagate_nan_whatever_the_position(self, cls, values):
        assert math.isnan(filled(cls, values).reduce())

    @pytest.mark.parametrize(
        "cls, expected",
        [(MeanMetric, math.inf), (SumMetric, math.inf), (MaxMetric, math.inf)],
    )
    def test_infinity_dominates(self, cls, expected):
        assert filled(cls, [1.0, math.inf]).reduce() == expected

    def test_minimum_with_a_negative_infinity(self):
        assert filled(MinMetric, [1.0, -math.inf]).reduce() == -math.inf

    def test_opposite_infinities_in_a_mean_give_nan(self):
        assert math.isnan(filled(MeanMetric, [math.inf, -math.inf]).reduce())

    def test_last_keeps_a_nan_only_when_it_is_the_last_value(self):
        assert filled(LastMetric, [1.0, NAN, 3.0]).reduce() == 3.0
        assert math.isnan(filled(LastMetric, [1.0, NAN]).reduce())

    def test_series_keeps_every_nan_in_place(self):
        reduced = filled(SeriesMetric, [1.0, NAN, 3.0]).reduce()
        assert reduced[0] == 1.0 and math.isnan(reduced[1]) and reduced[2] == 3.0


@pytest.mark.unit
class TestWindows:
    @pytest.mark.parametrize(
        "cls, first, second, expected",
        [
            (MeanMetric, [1.0, 3.0], [10.0], (2.0, 10.0)),
            (SumMetric, [1.0, 3.0], [10.0], (4.0, 10.0)),
            (MinMetric, [1.0, 3.0], [10.0], (1.0, 10.0)),
            (MaxMetric, [1.0, 3.0], [-10.0], (3.0, -10.0)),
            (LastMetric, [1, 3], [10], (3, 10)),
            (SeriesMetric, [1.0, 3.0], [10.0], ([1.0, 3.0], [10.0])),
        ],
    )
    def test_successive_windows_are_independent(self, cls, first, second, expected):
        metric = filled(cls, first)
        assert metric.reduce() == expected[0]
        for value in second:
            metric.push(value)
        assert metric.reduce() == expected[1]

    @pytest.mark.parametrize(
        "cls", [MeanMetric, MinMetric, MaxMetric, SumMetric, LastMetric, SeriesMetric]
    )
    def test_peek_does_not_open_a_new_window(self, cls):
        metric = filled(cls, [1.0, 2.0])
        before = metric.peek()
        assert metric.peek() == before
        assert len(metric) == 2

    @pytest.mark.parametrize("cls", [SeriesMetric] + SCALAR_METRICS)
    def test_peeked_history_is_a_copy(self, cls):
        metric = filled(cls, [1.0, 2.0])
        history = metric.peek(compile=False)
        history.append(99.0)
        assert metric.peek(compile=False) == [1.0, 2.0]

    def test_uncompiled_reduce_of_a_series_is_detached_from_the_source(self):
        metric = filled(SeriesMetric, [1.0, 2.0])
        kept = metric.reduce(compile=False)
        metric.push(5.0)
        assert kept.peek() == [1.0, 2.0]
        assert metric.peek() == [5.0]

    def test_empty_copy_has_the_same_class_and_no_values(self):
        for cls in SCALAR_METRICS + [SeriesMetric]:
            source = filled(cls, [1.0])
            copy = source.empty_copy()
            assert type(copy) is cls and len(copy) == 0 and len(source) == 1


@pytest.mark.unit
class TestConversions:
    @pytest.mark.parametrize("cls", [MeanMetric, MinMetric, MaxMetric, SumMetric])
    def test_float_and_int_use_the_compiled_value(self, cls):
        metric = filled(cls, [2, 4])
        assert float(metric) == float(metric.peek())
        assert int(metric) == int(metric.peek())

    def test_conversion_does_not_consume_the_window(self):
        metric = filled(MeanMetric, [2, 4])
        float(metric), int(metric)
        assert len(metric) == 2

    def test_converting_an_empty_sum_gives_zero(self):
        assert float(SumMetric()) == 0.0 and int(SumMetric()) == 0


@pytest.mark.unit
class TestFactoryTable:
    @pytest.mark.parametrize(
        "protocol, cls",
        [
            (ReduceProtocol.MEAN, MeanMetric),
            (ReduceProtocol.SUM, SumMetric),
            (ReduceProtocol.MAX, MaxMetric),
            (ReduceProtocol.MIN, MinMetric),
            (ReduceProtocol.LAST, LastMetric),
            (ReduceProtocol.SERIES, SeriesMetric),
        ],
    )
    def test_each_protocol_maps_to_its_own_class(self, protocol, cls):
        metric = MetricFactory.create(protocol)
        assert type(metric) is cls
        assert metric is not MetricFactory.create(
            protocol
        )  # a fresh instance each time

    def test_only_ema_is_unimplemented_and_the_error_names_it(self):
        unimplemented = []
        for protocol in ReduceProtocol:
            try:
                assert isinstance(MetricFactory.create(protocol), Metric)
            except NotImplementedError as error:
                assert repr(protocol.value) in str(error)
                unimplemented.append(protocol)
        assert unimplemented == [ReduceProtocol.EMA]

    def test_protocol_values_are_the_lowercase_names(self):
        assert {p.name: p.value for p in ReduceProtocol} == {
            "MEAN": "mean",
            "SUM": "sum",
            "MAX": "max",
            "MIN": "min",
            "LAST": "last",
            "EMA": "ema",
            "SERIES": "series",
        }
