"""Shared fixtures of the reporting tests.

The metric tree below is built by hand with pydantic constructors instead of a
``MetricLogger``: a populated schema is just a tree of lists, so the reporter
resolution can be tested without the logger. It has the three node kinds a
query walks through: a static sub-tree, a dynamic node keyed by mechanism id
(``by_mech``) and, below each mechanism, a second dynamic node keyed by seed
(``by_seed``).
"""

from __future__ import annotations

from typing import Any

import pytest

from core.metrics.schemas import MetricSchema
from core.reporting.base import PathResolution, Reporter


class SeedMetrics(MetricSchema):
    """Leaf node of the second dynamic level."""

    value: list[float] = []


class MechMetrics(MetricSchema):
    """One mechanism: two series and a dynamic node of seeds."""

    fitness: list[float] = []
    param: list[float] = []
    by_seed: dict[str, SeedMetrics] = {}


class RootMetrics(MetricSchema):
    """Root of the test tree; ``iter`` is a series here, as in a populated schema."""

    iter: list[int] = []  # type: ignore[assignment]
    loss: list[float] = []
    nested: list[list[float]] = []
    scalar: float = 0.0
    by_mech: dict[str, MechMetrics] = {}


class BackendlessReporter(Reporter):
    """Concrete ``Reporter`` that records what the backend hook receives."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, ...]] = []
        self.closed = False

    def _report(self, query, x, ys, errors, colors) -> None:
        self.calls.append((query, x, ys, errors, colors))

    def close(self) -> None:
        self.closed = True


class StubbedReporter(BackendlessReporter):
    """Reporter whose path resolution is replaced by canned results.

    The consistency checks of ``_resolve_query`` guard against resolutions
    that real paths cannot produce (an error group without its y series, for
    instance); canned resolutions reach them.
    """

    def __init__(self, resolutions: dict[tuple, PathResolution]) -> None:
        super().__init__()

        self._resolutions = resolutions

    def _resolve_path(self, path, metrics, **kwargs) -> PathResolution:
        return self._resolutions[tuple(path)]


@pytest.fixture
def reporting_metrics() -> RootMetrics:
    """Two mechanisms of three steps, the second one with two seeds."""

    return RootMetrics(
        iter=[0, 1, 2],
        loss=[3.0, 2.0, 1.0],
        nested=[[1.0], [2.0]],
        scalar=5.0,
        by_mech={
            "m0": MechMetrics(
                fitness=[10.0, 11.0, 12.0],
                param=[0.1, 0.2, 0.3],
                by_seed={
                    "s1": SeedMetrics(value=[1.0, 2.0, 3.0]),
                    "s2": SeedMetrics(value=[3.0, 4.0, 5.0]),
                },
            ),
            "m1": MechMetrics(
                fitness=[20.0, 21.0, 22.0],
                param=[0.4, 0.5, 0.6],
                by_seed={
                    "s1": SeedMetrics(value=[5.0, 6.0, 7.0]),
                    "s2": SeedMetrics(value=[7.0, 8.0, 9.0]),
                },
            ),
        },
    )


@pytest.fixture
def backendless_reporter() -> BackendlessReporter:
    return BackendlessReporter()


@pytest.fixture
def stubbed_reporter_class() -> type[StubbedReporter]:
    """Class of reporters whose resolution is canned (see ``StubbedReporter``)."""

    return StubbedReporter
