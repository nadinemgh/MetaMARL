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


@pytest.mark.unit
def test_save_accepts_the_optional_checkpoint_directory_of_its_overrides(tmp_path):
    # ``RayOptimizer.save`` takes ``checkpoint_dir``; the base signature must
    # accept the same keyword so a caller can use any optimizer.
    opt = NoOpOptimizer()

    assert opt.save() is None
    assert opt.save(checkpoint_dir=tmp_path) is None
    assert opt.save(checkpoint_dir=str(tmp_path)) is None
