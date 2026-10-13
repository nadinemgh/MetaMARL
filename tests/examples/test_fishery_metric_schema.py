"""``FisheryMetricSchema``: each per-episode field reduces its step series as named."""

import pytest

from core.metrics.logger import MetricLogger
from examples.bilevel_fishery.metric_schema import FisheryMetricSchema

STEPS = [0.3, 0.5, 0.4]


@pytest.mark.unit
@pytest.mark.parametrize(
    "field, expected",
    [
        ("fish_norm_next_mean", 0.4),
        ("fish_norm_next_min", 0.3),
        ("fish_norm_next_max", 0.5),
        ("fish_norm_next_last", 0.4),
    ],
)
def test_fish_norm_reductions(field, expected):
    logger = MetricLogger.from_schema(FisheryMetricSchema)
    for value in STEPS:
        logger.push(key=(field,), value=value)

    assert getattr(logger.reduce(), field) == pytest.approx(expected)
