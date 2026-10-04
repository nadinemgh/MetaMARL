"""Each ``Metric`` reducer: push, peek, reduce, flush and the empty-window values.

Ported from the August suite (all tests passed unchanged against the refactored
code). The type-check test now asserts the error message, which names the
rejecting metric and the offending type. Inputs that are empty, single, NaN or
integer-typed are covered in ``test_metric_reducer_inputs.py``.
"""

import pytest

from core.metrics.enums import ReduceProtocol
from core.metrics.metric.factory import MetricFactory
from core.metrics.metric.last import LastMetric
from core.metrics.metric.max import MaxMetric
from core.metrics.metric.mean import MeanMetric
from core.metrics.metric.min import MinMetric
from core.metrics.metric.series import SeriesMetric
from core.metrics.metric.sum import SumMetric

CASES = [
    (ReduceProtocol.MEAN, MeanMetric, [1.0, 2.0, 6.0], 3.0, None),
    (ReduceProtocol.SERIES, SeriesMetric, [1.0, 2.0, 6.0], [1.0, 2.0, 6.0], []),
    (ReduceProtocol.LAST, LastMetric, [1, 2, 6], 6, None),
    (ReduceProtocol.SUM, SumMetric, [1.0, 2.0, 6.0], 9.0, 0),
    (ReduceProtocol.MIN, MinMetric, [1.0, 2.0, 6.0], 1.0, None),
    (ReduceProtocol.MAX, MaxMetric, [1.0, 2.0, 6.0], 6.0, None),
]


@pytest.mark.unit
@pytest.mark.parametrize(
    "protocol, cls, values, reduced, empty", CASES, ids=[c[0].value for c in CASES]
)
def test_factory_push_peek_reduce(protocol, cls, values, reduced, empty):
    metric = MetricFactory.create(protocol)
    assert isinstance(metric, cls)
    assert metric.reduce() == empty  # empty reducer semantics

    for v in values:
        metric.push(v)
    assert len(metric) == len(values)
    assert metric.peek(compile=False) == values  # raw history
    assert metric.peek() == reduced  # non-destructive
    assert metric.peek() == reduced
    assert metric.reduce() == reduced  # destructive
    assert len(metric) == 0
    assert metric.reduce() == empty


@pytest.mark.unit
def test_reduce_without_compile_returns_metric_holding_the_value():
    m = MeanMetric()
    m.push(1.0), m.push(3.0)
    reduced = m.reduce(compile=False)
    assert isinstance(reduced, MeanMetric) and reduced.peek() == 2.0
    s = SeriesMetric()
    s.push(1.0)
    assert s.reduce(compile=False).peek() == [1.0]
    assert isinstance(MeanMetric().reduce(compile=False), MeanMetric)


@pytest.mark.unit
def test_numeric_metrics_reject_non_numbers():
    for m in (MeanMetric(), SumMetric(), MinMetric(), MaxMetric()):
        name = type(m).__name__
        with pytest.raises(
            TypeError, match=f"{name} only accepts int or float, got str"
        ):
            m.push("x")
        with pytest.raises(
            TypeError, match=f"{name} only accepts int or float, got bool"
        ):
            m.push(True)
        assert len(m) == 0  # a rejected value is not stored
    for m in (LastMetric(), SeriesMetric()):  # LAST and SERIES accept any primitive
        m.push("anything")
        m.push(True)
        assert m.peek(compile=False) == ["anything", True]


@pytest.mark.unit
def test_float_int_conversions_and_repr():
    m = MeanMetric()
    m.push(2), m.push(4)
    assert float(m) == 3.0 and int(m) == 3
    assert "MeanMetric" in repr(m)
    s = SeriesMetric()
    s.push(1.0)
    with pytest.raises(ValueError, match="to float"):
        float(s)
    assert isinstance(m.empty_copy(), MeanMetric) and len(m.empty_copy()) == 0


@pytest.mark.unit
def test_unimplemented_protocol():
    with pytest.raises(NotImplementedError):
        MetricFactory.create(ReduceProtocol.EMA)


@pytest.mark.unit
@pytest.mark.parametrize(
    "cls, values, reduced",
    [
        (LastMetric, [1, 5], 5),
        (MinMetric, [3.0, 1.0], 1.0),
        (MaxMetric, [3.0, 7.0], 7.0),
        (SumMetric, [3.0, 7.0], 10.0),
    ],
)
def test_reduce_uncompiled_keeps_a_metric_with_the_reduced_value(cls, values, reduced):
    m = cls()
    for v in values:
        m.push(v)
    kept = m.reduce(compile=False)
    assert isinstance(kept, cls) and kept.peek() == reduced and len(kept) == 1
    assert len(m) == 0
    empty = cls().reduce(compile=False)
    assert isinstance(empty, cls)
    assert len(empty) == (1 if cls is SumMetric else 0)  # an empty sum reduces to 0
    assert repr(kept).startswith(cls.__name__)


@pytest.mark.unit
@pytest.mark.parametrize("cls", [MeanMetric, MinMetric, MaxMetric, LastMetric])
def test_empty_compiled_peek_is_none_and_raw_peek_is_empty_list(cls):
    metric = cls()
    assert metric.peek() is None
    assert metric.peek(compile=False) == []
    assert repr(metric).startswith(cls.__name__)


@pytest.mark.unit
def test_int_conversion_rejects_series():
    s = SeriesMetric()
    s.push(2.0)
    with pytest.raises(ValueError, match="to int"):
        int(s)
    m = MaxMetric()
    m.push(2.7)
    assert int(m) == 2
