"""Sparse metric accumulation for a schema fixed at logger construction.

``MetricLogger`` records values that were actually reported. A ``push_data``
call is not an implicit timestep: missing dynamic IDs, optional branches, and
fields set to ``None`` do not append placeholder values. Dynamic IDs may be
created at runtime, but their declared value-schema types remain fixed.

Callers who need iteration-aligned observations are responsible for recording
the associated iteration or timestamp in their own schemas.
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
    snapshot = logger.peek().rollout

    return {
        (mid, sid): seed.value
        for mid, mechanism in snapshot.by_mechanism.items()
        for sid, seed in mechanism.by_seed.items()
    }


@pytest.mark.unit
class TestMetricAccumulation:
    @pytest.mark.parametrize(
        ("cls", "compiled"),
        [
            (MeanMetric, 2.0),
            (SumMetric, 4.0),
            (MinMetric, 1.0),
            (MaxMetric, 3.0),
            (LastMetric, 3.0),
            (SeriesMetric, [1.0, 3.0]),
        ],
    )
    def test_only_explicit_values_are_accumulated(self, cls, compiled):
        metric = cls()

        metric.push(1.0)
        metric.push(3.0)

        assert metric.peek(compile=False) == [1.0, 3.0]

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
    def test_empty_metric_uses_its_own_reduction_identity(self, cls, compiled):
        metric = cls()
        assert metric.peek(compile=False) == []

        assert metric.peek() == compiled


@pytest.mark.unit
class TestSparseDynamicLogging:
    def test_missing_child_does_not_append_a_placeholder(self):
        logger = MetricLogger.from_schema(Root)

        logger.push_data(rollout(m0={"s0": 1.0, "s1": 2.0}))
        logger.push_data(rollout(m0={"s0": 3.0}))
        assert values(logger) == {("m0", "s0"): [1.0, 3.0], ("m0", "s1"): [2.0]}

    def test_late_child_starts_its_history_at_first_observation(self):
        logger = MetricLogger.from_schema(Root)

        logger.push_data(rollout(m0={"s0": 1.0}))
        logger.push_data(rollout(m0={"s0": 3.0, "s1": 4.0}))
        assert values(logger) == {("m0", "s0"): [1.0, 3.0], ("m0", "s1"): [4.0]}

    def test_nested_dynamic_ids_accumulate_independently(self):
        logger = MetricLogger.from_schema(Root)

        logger.push_data(rollout(m0={"s0": 1.0}))
        logger.push_data(rollout(m1={"s0": 2.0}))
        logger.push_data(rollout(m0={"s0": 3.0, "s1": 4.0}))
        assert values(logger) == {
            ("m0", "s0"): [1.0, 3.0],
            ("m0", "s1"): [4.0],
            ("m1", "s0"): [2.0],
        }

    def test_absent_optional_subtree_does_not_change_histories(self):
        logger = MetricLogger.from_schema(Root)

        logger.push_data(rollout(m0={"s0": 1.0}))
        logger.push_data(Root(rollout=None))
        logger.push_data(rollout(m0={"s0": 3.0}))
        assert values(logger) == {("m0", "s0"): [1.0, 3.0]}

    def test_explicit_none_leaf_is_not_recorded(self):
        logger = MetricLogger.from_schema(Root)

        logger.push_data(rollout(m0={"s0": 1.0}))
        logger.push_data(
            Root(
                rollout=Rollout(
                    by_mechanism={"m0": Mechanism(by_seed={"s0": Seed(value=None)})}
                )
            )
        )
        logger.push_data(rollout(m0={"s0": 3.0}))
        assert values(logger) == {("m0", "s0"): [1.0, 3.0]}

    def test_empty_dynamic_dictionary_does_not_change_histories(self):
        logger = MetricLogger.from_schema(Root)

        logger.push_data(rollout(m0={"s0": 1.0}))
        logger.push_data(rollout())
        logger.push_data(rollout(m0={"s0": 3.0}))
        assert values(logger) == {("m0", "s0"): [1.0, 3.0]}

    def test_push_by_path_and_push_data_share_the_same_leaf(self):
        logger = MetricLogger.from_schema(Root)
        path = ("rollout", "by_mechanism", "m0", "by_seed", "s0", "value")

        logger.push(path, 1.0)
        logger.push_data(rollout(m0={"s0": 2.0}))
        logger.push(path, 3.0)
        assert values(logger) == {("m0", "s0"): [1.0, 2.0, 3.0]}
        assert logger.peek_value(path) == pytest.approx(2.0)

    def test_peek_does_not_consume_accumulated_values(self):
        logger = MetricLogger.from_schema(Root)

        logger.push_data(rollout(m0={"s0": 1.0}))
        assert values(logger) == {("m0", "s0"): [1.0]}
        assert values(logger) == {("m0", "s0"): [1.0]}
        logger.push_data(rollout(m0={"s0": 3.0}))
        assert values(logger) == {("m0", "s0"): [1.0, 3.0]}

    def test_reduce_aggregates_observed_values_and_resets_the_cycle(self):
        logger = MetricLogger.from_schema(Root)

        logger.push_data(rollout(m0={"s0": 1.0}))
        logger.push_data(rollout(m0={"s0": 3.0, "s1": 4.0}))

        reduced = logger.reduce().rollout.by_mechanism["m0"].by_seed

        assert reduced["s0"].value == pytest.approx(2.0)
        assert reduced["s1"].value == pytest.approx(4.0)
        assert values(logger) == {("m0", "s0"): [], ("m0", "s1"): []}
        logger.push_data(rollout(m0={"s0": 7.0}))
        assert values(logger) == {("m0", "s0"): [7.0], ("m0", "s1"): []}

    def test_reset_clears_all_values_without_removing_dynamic_ids(self):
        logger = MetricLogger.from_schema(Root)

        logger.push_data(rollout(m0={"s0": 1.0}))
        logger.reset()
        logger.push_data(rollout(m0={"s1": 4.0}))
        assert values(logger) == {("m0", "s0"): [], ("m0", "s1"): [4.0]}
