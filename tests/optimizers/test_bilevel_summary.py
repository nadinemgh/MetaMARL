"""``BilevelOptimizer``: construction, the summary of ``train`` and its shutdown order.

Stopping both levels when training ends or fails is covered by
``test_bilevel_stop.py``. This file adds what that one leaves out: the
attributes copied from the config, the summary handed back unchanged, the
order in which the levels are stopped before the primary reporter is closed
(also on failure), and the two log lines that frame a run.
"""

import logging
from types import SimpleNamespace

import pytest

from core.optimizers.bilevel import BilevelOptimizer

SUMMARY = {
    "episodes": 3,
    "converged": True,
    "best_mechanism": [0.25, 0.75],
    "best_fitness": 1.5,
    "population_history": ["generation"],
}


class OrderedLevel:
    """Level that appends ``(name, event)`` to a shared journal."""

    def __init__(self, name, journal, *, episodes=3, fails=False):
        self.name = name
        self.journal = journal
        self.episodes = episodes
        self.fails = fails

    def train(self):
        self.journal.append((self.name, "train"))
        if self.fails:
            raise RuntimeError("outer search failed")

        return SUMMARY

    def stop(self):
        self.journal.append((self.name, "stop"))


class OrderedReporter:
    def __init__(self, journal):
        self.journal = journal

    def close(self):
        self.journal.append(("reporter", "close"))


def config(**overrides):
    values = {"episodes": None, "env": None, "world_name": "w1", "output_dir": "out"}

    return SimpleNamespace(**{**values, **overrides})


@pytest.mark.unit
def test_constructor_copies_the_config_and_starts_empty():
    cfg = config()
    outer, inner, reporter = object(), object(), object()

    opt = BilevelOptimizer(cfg, outer=outer, inner=inner, reporter=reporter)

    assert opt.config is cfg
    assert (opt.outer, opt.inner, opt.reporting) == (outer, inner, reporter)
    assert (opt.world_name, opt.output_dir) == ("w1", "out")
    assert opt.converged is False
    assert opt.all_trajectories == []
    assert opt.population_history == []
    assert opt.es_metrics_history == []
    assert opt.env is None and opt.world is None


@pytest.mark.unit
def test_the_summary_of_the_outer_level_is_returned_unchanged():
    journal = []
    opt = BilevelOptimizer(
        config(),
        outer=OrderedLevel("outer", journal),
        inner=OrderedLevel("inner", journal),
        reporter=None,
    )

    assert opt.train() is SUMMARY


@pytest.mark.unit
def test_levels_are_stopped_outer_first_then_the_reporter_is_closed():
    journal = []
    opt = BilevelOptimizer(
        config(),
        outer=OrderedLevel("outer", journal),
        inner=OrderedLevel("inner", journal),
        reporter=OrderedReporter(journal),
    )

    opt.train()

    assert journal == [
        ("outer", "train"),
        ("outer", "stop"),
        ("inner", "stop"),
        ("reporter", "close"),
    ]


@pytest.mark.unit
def test_the_reporter_is_closed_even_when_training_fails():
    journal = []
    opt = BilevelOptimizer(
        config(),
        outer=OrderedLevel("outer", journal, fails=True),
        inner=OrderedLevel("inner", journal),
        reporter=OrderedReporter(journal),
    )

    with pytest.raises(RuntimeError, match="outer search failed"):
        opt.train()

    assert journal[-3:] == [("outer", "stop"), ("inner", "stop"), ("reporter", "close")]


@pytest.mark.unit
def test_a_missing_inner_level_is_skipped_when_stopping():
    journal = []
    opt = BilevelOptimizer(
        config(),
        outer=OrderedLevel("outer", journal),
        inner=None,
        reporter=OrderedReporter(journal),
    )

    opt.train()

    assert journal == [("outer", "train"), ("outer", "stop"), ("reporter", "close")]


@pytest.mark.unit
def test_the_run_is_framed_by_two_log_lines(caplog):
    journal = []
    opt = BilevelOptimizer(
        config(),
        outer=OrderedLevel("outer", journal, episodes=3),
        inner=None,
        reporter=None,
    )

    with caplog.at_level(logging.INFO, logger="core.optimizers.bilevel"):
        opt.train()

    start, finish = [record.getMessage() for record in caplog.records]
    assert start == "[Bilevel] Starting run | max_outer_iters=3 | world=w1"
    assert finish == (
        "[Bilevel] Run finished | iters=3 | converged=True | "
        + "mechanism=[0.25, 0.75] | best_fitness=1.5000"
    )
