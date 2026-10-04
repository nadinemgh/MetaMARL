"""``Optimizer`` base class: construction without a config and unset capacity."""

import pytest

from core.optimizers.base import Optimizer


class NoOpOptimizer(Optimizer):
    """Concrete ``Optimizer`` whose training does nothing."""

    def train(self) -> None:
        pass


@pytest.mark.unit
def test_optimizer_can_be_built_without_a_config():
    opt = NoOpOptimizer()

    assert opt.config is None
    assert opt.episodes is None
    assert opt.env is None


@pytest.mark.unit
def test_unset_batch_capacity_raises_a_clear_error():
    with pytest.raises(RuntimeError, match="batch_capacity not set"):
        NoOpOptimizer().batch_capacity
