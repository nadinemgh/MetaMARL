"""Gap entries keep the children of a dynamic node aligned with its pushes.

A dynamic node (``dict[ID, MetricSchema]``) receives one dictionary per
``push_data`` call, and its ids can vary from one call to the next: an inner
iteration where an environment finished no episode has no entry for it. The
logger records a gap (``None``) in every leaf of a child that a push leaves out,
and back-fills a child created after earlier pushes with one gap per missed
push, so that entry ``i`` of every child belongs to push ``i``. The compiled
values skip the gaps; the raw history keeps them.
"""

from typing import Optional

import pytest
from pydantic import Field

from core.metrics.logger import MetricLogger
from core.metrics.metric.last import LastMetric
from core.metrics.metric.max import MaxMetric
from core.metrics.metric.mean import MeanMetric
from core.metrics.metric.min import MinMetric
from core.metrics.metric.series import SeriesMetric
from core.metrics.metric.sum import SumMetric
from core.metrics.schemas import MetricSchema


class Seed(MetricSchema):
    value: Optional[float] = None


class Mechanism(MetricSchema):
    by_seed: dict[str, Seed] = Field(default_factory=dict)


class Rollout(MetricSchema):
    by_mechanism: dict[str, Mechanism] = Field(default_factory=dict)


class RichRollout(Rollout):
    """Runtime subtype of ``Rollout``, to trigger a specialisation."""

    extra: Optional[float] = None


class Root(MetricSchema):
    rollout: Optional[Rollout] = None


def rollout(**mechanisms: dict[str, float]) -> Root:
    return Root(
        rollout=Rollout(
            by_mechanism={
                mid: Mechanism(by_seed={sid: Seed(value=v) for sid, v in seeds.items()})
                for mid, seeds in mechanisms.items()
            }
        )
    )


def values(logger: MetricLogger) -> dict[tuple[str, str], list]:
    peeked = logger.peek().rollout
    return {
        (mid, sid): seed.value
        for mid, mechanism in peeked.by_mechanism.items()
        for sid, seed in mechanism.by_seed.items()
    }


@pytest.mark.unit
class TestMetricGaps:
    @pytest.mark.parametrize(
        ("cls", "compiled"),
        [
            (MeanMetric, 2.0),
            (SumMetric, 4.0),
            (MinMetric, 1.0),
            (MaxMetric, 3.0),
            (LastMetric, 3.0),
            (SeriesMetric, [1.0, None, 3.0, None]),
        ],
    )
    def test_compiled_values_skip_gaps_and_history_keeps_them(self, cls, compiled):
        metric = cls()
        metric.push(1.0)
        metric.push_gap()
        metric.push(3.0)
        metric.push_gap()

        assert metric.peek(compile=False) == [1.0, None, 3.0, None]
        assert metric.peek() == compiled
        assert metric.reduce() == compiled
        assert len(metric) == 0

    @pytest.mark.parametrize(
        ("cls", "compiled"),
        [
            (MeanMetric, None),
            (SumMetric, 0),
            (MinMetric, None),
            (MaxMetric, None),
            (LastMetric, None),
        ],
    )
    def test_a_history_of_gaps_compiles_like_an_empty_one(self, cls, compiled):
        metric = cls()
        metric.push_gap()

        assert metric.peek() == compiled


@pytest.mark.unit
class TestLoggerGaps:
    def test_a_child_left_out_of_a_push_receives_a_gap(self):
        logger = MetricLogger.from_schema(Root)
        logger.push_data(rollout(m0={"s0": 1.0, "s1": 2.0}))
        logger.push_data(rollout(m0={"s0": 3.0}))

        assert values(logger) == {("m0", "s0"): [1.0, 3.0], ("m0", "s1"): [2.0, None]}

    def test_a_child_created_late_is_back_filled(self):
        logger = MetricLogger.from_schema(Root)
        logger.push_data(rollout(m0={"s0": 1.0}))
        logger.push_data(rollout(m0={"s0": 3.0, "s1": 4.0}))

        assert values(logger) == {("m0", "s0"): [1.0, 3.0], ("m0", "s1"): [None, 4.0]}

    def test_gaps_reach_the_dynamic_nodes_below_an_absent_child(self):
        logger = MetricLogger.from_schema(Root)
        logger.push_data(rollout(m0={"s0": 1.0}))
        logger.push_data(rollout(m1={"s0": 2.0}))
        logger.push_data(rollout(m0={"s0": 3.0, "s1": 4.0}))

        assert values(logger) == {
            ("m0", "s0"): [1.0, None, 3.0],
            ("m0", "s1"): [None, None, 4.0],
            ("m1", "s0"): [None, 2.0, None],
        }

    def test_a_branch_pushed_as_none_receives_no_gap(self):
        logger = MetricLogger.from_schema(Root)
        logger.push_data(rollout(m0={"s0": 1.0}))
        logger.push_data(Root(rollout=None))
        logger.push_data(rollout(m0={"s0": 3.0}))

        assert values(logger) == {("m0", "s0"): [1.0, 3.0]}

    def test_reduce_skips_gaps_and_starts_a_new_alignment(self):
        logger = MetricLogger.from_schema(Root)
        logger.push_data(rollout(m0={"s0": 1.0}))
        logger.push_data(rollout(m0={"s1": 4.0}))

        reduced = logger.reduce().rollout.by_mechanism["m0"].by_seed
        assert {sid: seed.value for sid, seed in reduced.items()} == {
            "s0": 1.0,
            "s1": 4.0,
        }

        logger.push_data(rollout(m0={"s2": 5.0}))
        assert values(logger)[("m0", "s2")] == [5.0]

    def test_reset_starts_a_new_alignment(self):
        logger = MetricLogger.from_schema(Root)
        logger.push_data(rollout(m0={"s0": 1.0}))
        logger.reset()
        logger.push_data(rollout(m0={"s1": 4.0}))

        assert values(logger) == {("m0", "s0"): [None], ("m0", "s1"): [4.0]}

    def test_a_specialisation_keeps_the_gaps_and_the_alignment(self):
        logger = MetricLogger.from_schema(Root)
        logger.push_data(rollout(m0={"s0": 1.0, "s1": 2.0}))
        logger.push_data(rollout(m0={"s0": 3.0}))
        logger.push_data(
            Root(
                rollout=RichRollout(
                    extra=0.5,
                    by_mechanism={
                        "m0": Mechanism(
                            by_seed={"s0": Seed(value=5.0), "s2": Seed(value=6.0)}
                        )
                    },
                )
            )
        )

        assert values(logger) == {
            ("m0", "s0"): [1.0, 3.0, 5.0],
            ("m0", "s1"): [2.0, None, None],
            ("m0", "s2"): [None, None, 6.0],
        }
