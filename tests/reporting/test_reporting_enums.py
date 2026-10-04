"""Reporting enumerations: the string values are part of the configuration format.

``ReporterType`` is copied from the August suite; ``Resolution`` (the time axis
names) is new.
"""

from __future__ import annotations

import pytest

from core.reporting.enums import ReporterType, Resolution

pytestmark = pytest.mark.unit


def test_reporter_type_members():
    assert {r.value for r in ReporterType} == {"wandb", "local"}
    assert ReporterType("wandb") is ReporterType.wandb


def test_reporter_type_rejects_unknown_values():
    with pytest.raises(ValueError, match="'csv' is not a valid ReporterType"):
        ReporterType("csv")


def test_resolution_values_name_the_time_axes():
    assert Resolution.env.value == "env_steps"
    assert Resolution.inner.value == "train_iters"
    assert Resolution.outer.value == "generation"
    assert Resolution("generation") is Resolution.outer
