"""Shared pytest configuration.

Unit tests run without a Ray runtime. Tests that start one are marked
``integration`` (or ``notebook``, since every tutorial starts Ray) and share the
session-scoped ``ray_session`` fixture. The collection hook below runs those
tests after every other test; see its docstring for why the order matters.
"""

from __future__ import annotations

import pytest
import ray


@pytest.fixture(scope="session")
def ray_session():
    """Start a small local Ray runtime shared by the integration tests."""
    if not ray.is_initialized():
        ray.init(num_cpus=2, ignore_reinit_error=True, include_dashboard=False)
    yield
    ray.shutdown()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Run the tests that start a real Ray runtime after every other test.

    When a Ray actor class is exported to a live cluster it is serialized by
    value, and unpickling it back in the driver process makes cloudpickle
    rewrite the methods of the already-imported class with copies whose
    ``__globals__`` is a frozen snapshot of the module namespace. Any later
    ``monkeypatch.setattr(module, ...)`` in a unit test is then invisible to
    those methods. This was observed on an earlier version of the framework
    (``core.reporting.wandb.WandbReporter``, then a Ray actor). Keeping the
    ``integration`` and ``notebook`` items last removes the order dependency
    without touching the production classes.
    """
    ray_items = [
        item
        for item in items
        if item.get_closest_marker("integration") or item.get_closest_marker("notebook")
    ]
    others = [item for item in items if item not in ray_items]
    items[:] = others + ray_items
