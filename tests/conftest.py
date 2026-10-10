"""Shared pytest configuration.

Unit tests run without a Ray runtime: the Ray actors are replaced by fakes in
the per-directory ``conftest.py`` files. The tests marked ``integration`` start
the example scripts in child processes, which start their own Ray runtime; the
``notebook`` marker is reserved for the tutorials, which do the same through a
kernel. The collection hook below runs those tests after every other test; see
its docstring for why the order matters.
"""

from __future__ import annotations

import pytest


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
    without touching the production classes. No test starts Ray in the pytest
    process today, so the hook is a guard for the first one that does; it also
    reports unit failures before the slow child-process runs begin.
    """
    ray_items = [
        item
        for item in items
        if item.get_closest_marker("integration") or item.get_closest_marker("notebook")
    ]
    others = [item for item in items if item not in ray_items]
    items[:] = others + ray_items
