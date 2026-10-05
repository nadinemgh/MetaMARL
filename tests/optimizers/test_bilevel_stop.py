"""``BilevelOptimizer.train``: both levels are stopped when the run ends."""

from types import SimpleNamespace

import pytest

from core.optimizers.bilevel import BilevelOptimizer

SUMMARY = {
    "episodes": 1,
    "converged": False,
    "best_mechanism": [0.5],
    "best_fitness": 1.0,
}


class RecordingLevel:
    """Optimizer stand-in recording ``stop`` calls; ``train`` may raise."""

    def __init__(self, fails: bool = False) -> None:
        self.episodes = 1
        self.fails = fails
        self.stopped = 0

    def train(self) -> dict:
        if self.fails:
            raise RuntimeError("inner training crashed")
        return SUMMARY

    def stop(self) -> None:
        self.stopped += 1


def make_bilevel(outer: RecordingLevel, inner: RecordingLevel) -> BilevelOptimizer:
    config = SimpleNamespace(episodes=None, env=None, world_name="w")
    return BilevelOptimizer(config, outer=outer, inner=inner, reporter=None)


@pytest.mark.unit
def test_train_stops_both_levels():
    outer, inner = RecordingLevel(), RecordingLevel()

    assert make_bilevel(outer, inner).train() == SUMMARY
    assert (outer.stopped, inner.stopped) == (1, 1)


@pytest.mark.unit
def test_failed_train_still_stops_both_levels():
    outer, inner = RecordingLevel(fails=True), RecordingLevel()

    with pytest.raises(RuntimeError, match="inner training crashed"):
        make_bilevel(outer, inner).train()
    assert (outer.stopped, inner.stopped) == (1, 1)
