"""``ReporterConfig``: the serialisable factory shared by every backend.

The abstract class is exercised through a minimal subclass; the backend
configs (CSV, TensorBoard, W&B) have their own tests with their reporters.
"""

from __future__ import annotations

import pytest

from core.reporting.config import ReporterConfig

pytestmark = pytest.mark.unit


class _Config(ReporterConfig):
    def build(self):
        return None


def test_the_class_is_abstract():
    with pytest.raises(TypeError, match="abstract"):
        ReporterConfig(project="p")  # type: ignore[abstract]


def test_world_and_outer_iters_are_unset_until_the_optimizer_fills_them():
    config = _Config(project="p")

    assert config.project_name == "p"
    assert config.world is None and config.outer_iters is None


def test_world_and_outer_iters_setters():
    config = _Config(project="p")

    config.world = "w"
    config.outer_iters = 7

    assert config.world == "w" and config.outer_iters == 7


def test_copy_is_deep_and_independent():
    config = _Config(project="p")
    config.world = "w"
    config.outer_iters = 3
    config.extra = {"nested": [1]}  # type: ignore[attr-defined]

    clone = config.copy()
    clone.world = "other"
    clone.extra["nested"].append(2)  # type: ignore[attr-defined]

    assert clone is not config and isinstance(clone, _Config)
    assert clone.outer_iters == 3 and clone.project_name == "p"
    assert config.world == "w"
    assert config.extra == {"nested": [1]}  # type: ignore[attr-defined]
