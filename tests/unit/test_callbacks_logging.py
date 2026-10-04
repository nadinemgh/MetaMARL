"""Unit tests for ``log_and_report_episode_metrics`` (the episode-end hook).

The hook is checked with a real ``MetricLogger`` and a concrete ``Reporter``
that records what it is asked to draw: peek, report, reduce, then hand the
reduced schema to RLlib's ``MetricsLogger`` under ``("by_episode", <id>)``.
The episode-tagging hook and the evaluation function are covered in
``test_callbacks_tagging.py`` and ``test_callbacks_evaluation.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.callbacks import log_and_report_episode_metrics
from core.metrics.logger import MetricLogger
from core.metrics.schemas import MetricSchema
from core.reporting.base import Reporter
from core.reporting.query import Query


class ValueSchema(MetricSchema):
    """One leaf metric, averaged when reduced."""

    value: float = 0.0


class RecordingReporter(Reporter):
    """Reporter that keeps the resolved ``x`` and ``y`` of each report."""

    def __init__(self) -> None:
        super().__init__()
        self.reports: list[tuple] = []

    def _report(self, query, x, ys, errors, colors) -> None:
        self.reports.append((query.title, x, ys))

    def close(self) -> None:
        pass


def make_runner(values=(2.0,), iteration=1):
    """Return ``(env, env_runner)`` around a logger holding the given values."""
    logger = MetricLogger.from_schema(ValueSchema)

    for value in values:
        logger.push(("iter",), iteration)
        logger.push(("value",), value)

    reporter = RecordingReporter()
    reporter.add_query(Query(title="v", x=("iter",), y=("value",)))
    env = SimpleNamespace(logger=logger, reporter=reporter)
    env_runner = SimpleNamespace(
        env=SimpleNamespace(envs=[SimpleNamespace(unwrapped=env)])
    )

    return env, env_runner


def recording_metrics_logger():
    """RLlib ``MetricsLogger`` stand-in keeping every ``log_value`` call."""
    logged: list[dict] = []

    return SimpleNamespace(log_value=lambda **kw: logged.append(kw)), logged


@pytest.mark.unit
def test_episode_end_hook_reports_then_reduces():
    env, env_runner = make_runner(values=(2.0, 4.0))
    metrics_logger, logged = recording_metrics_logger()

    log_and_report_episode_metrics(
        episode=SimpleNamespace(id_="env=0|m=1|ps=2|ss=3|raw=abc"),
        env_runner=env_runner,
        env=None,
        env_index=0,
        metrics_logger=metrics_logger,
    )

    # The reporter received the raw (peeked) history, not the reduced mean.
    assert len(env.reporter.reports) == 1
    title, x, ys = env.reporter.reports[0]
    assert title == "v"
    assert x == {(): [1, 1]}
    assert ys[0] == {(): [2.0, 4.0]}

    (call,) = logged
    assert call["key"] == ("by_episode", "env=0|m=1|ps=2|ss=3")
    assert call["reduce"] == "item"
    assert call["value"].value == pytest.approx(3.0)
    assert call["value"].iter == 1


@pytest.mark.unit
def test_episode_end_hook_empties_the_env_logger():
    env, env_runner = make_runner(values=(2.0,))
    metrics_logger, _ = recording_metrics_logger()

    log_and_report_episode_metrics(
        episode=SimpleNamespace(id_="env=0|raw=abc"),
        env_runner=env_runner,
        env=None,
        env_index=0,
        metrics_logger=metrics_logger,
    )

    assert len(env.logger._refs[("value",)]) == 0


@pytest.mark.unit
def test_episode_end_hook_picks_the_sub_env_by_index_and_keeps_untagged_ids():
    env, _ = make_runner(values=(3.0,), iteration=4)
    other = SimpleNamespace(logger=None, reporter=None)
    env_runner = SimpleNamespace(
        env=SimpleNamespace(
            envs=[SimpleNamespace(unwrapped=other), SimpleNamespace(unwrapped=env)]
        )
    )
    metrics_logger, logged = recording_metrics_logger()

    log_and_report_episode_metrics(
        episode=SimpleNamespace(id_="plain-id"),
        env_runner=env_runner,
        env=None,
        env_index=1,
        metrics_logger=metrics_logger,
        extra="ignored",
    )

    # An ID without the ``|raw=`` marker is used unchanged.
    assert logged[0]["key"] == ("by_episode", "plain-id")
    assert logged[0]["value"].value == pytest.approx(3.0)
    assert logged[0]["value"].iter == 4
    assert len(env.reporter.reports) == 1
