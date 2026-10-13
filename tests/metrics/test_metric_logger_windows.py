"""``MetricLogger`` reduction windows: one episode, one ``iter``, empty windows.

The environment resets its logger at every episode and reduces it once at the
end of the episode (commit ``439249a``), so what a report shows is the
reduction of exactly one episode. These tests reproduce that call pattern on
the schemas of ``conftest.py``: per-step pushes, a ``reset`` at the start, a
``reduce`` at the end. They also cover the ``subtree_reduce`` override that the
Ray adaptor schema uses, ``compile`` and NaN values pushed through the logger.
"""

import math

import pytest
from pydantic import Field

from core.metrics.enums import ReduceProtocol
from core.metrics.logger import MetricLogger
from core.metrics.schemas import MetricSchema

MEAN = ("static", "mean_value")
ITER = ("static", "iter")
ROOT_ITER = ("iter",)


def run_episode(logger: MetricLogger, rewards: list[float]) -> None:
    """Push one reward and one step index per step, as the environment does."""
    for step, reward in enumerate(rewards):
        logger.push(ROOT_ITER, step)
        logger.push(MEAN, reward)
        logger.push(("static", "sum_value"), reward)


@pytest.mark.unit
class TestEmptyLogger:
    def test_reducing_an_untouched_logger_gives_the_empty_value_of_each_protocol(
        self, logger, metric_schemas
    ):
        *_, Root = metric_schemas
        reduced = logger.reduce()

        assert isinstance(reduced, Root)
        assert reduced.iter is None
        static = reduced.static
        assert static.mean_value is None
        assert static.default_value is None
        assert static.min_value is None
        assert static.max_value is None
        assert static.last_value is None
        assert static.iter is None
        assert static.sum_value == 0
        assert static.series_value == []
        assert reduced.group.by_id == {}  # no dynamic child was ever created

    def test_nested_open_slot_reduces_to_a_bare_schema(self, logger):
        reduced = logger.reduce()
        assert type(reduced.inner) is MetricSchema and reduced.inner.iter is None

    def test_compile_of_an_untouched_logger_is_a_plain_dict_with_empty_values(
        self, logger
    ):
        compiled = logger.compile()
        assert compiled["iter"] is None
        assert compiled["static"]["mean_value"] is None
        assert compiled["static"]["sum_value"] == 0
        assert compiled["static"]["series_value"] == []
        assert compiled["group"] == {"iter": None, "by_id": {}}

    def test_peek_of_an_untouched_logger_returns_raw_empty_histories(self, logger):
        peeked = logger.peek()
        assert peeked.static.mean_value == []
        assert peeked.static.sum_value == []
        assert peeked.iter == []


@pytest.mark.unit
class TestEpisodeWindow:
    def test_reduction_covers_the_whole_episode_and_iter_is_the_last_step(self, logger):
        run_episode(logger, [1.0, 2.0, 6.0])

        reduced = logger.reduce()

        assert reduced.static.mean_value == 3.0
        assert reduced.static.sum_value == 9.0
        assert reduced.iter == 2  # LAST: the final step index, not a count or a mean

    def test_a_second_reduce_without_new_pushes_is_empty(self, logger):
        run_episode(logger, [1.0, 2.0])
        logger.reduce()

        again = logger.reduce()

        assert again.static.mean_value is None
        assert again.static.sum_value == 0
        assert again.iter is None

    def test_peek_does_not_end_the_episode(self, logger):
        run_episode(logger, [1.0, 3.0])

        logger.peek()

        assert logger.reduce().static.mean_value == 2.0

    def test_consecutive_episodes_do_not_leak_into_each_other(self, logger):
        run_episode(logger, [1.0, 3.0, 5.0, 7.0, 9.0])
        first = logger.reduce()

        run_episode(logger, [10.0, 20.0])
        second = logger.reduce()

        assert (first.static.mean_value, first.iter) == (5.0, 4)
        assert (second.static.mean_value, second.iter) == (15.0, 1)

    def test_reset_discards_what_a_probe_step_pushed_before_the_episode(self, logger):
        """RLlib steps a new env once before the first episode; ``reset`` drops it."""
        logger.push(ROOT_ITER, 0)
        logger.push(MEAN, 1000.0)  # the probe step's value

        logger.reset()
        run_episode(logger, [2.0, 4.0])
        reduced = logger.reduce()

        assert reduced.static.mean_value == 3.0
        assert reduced.iter == 1

    def test_reset_then_reduce_gives_the_empty_window(self, logger):
        run_episode(logger, [1.0, 2.0])

        logger.reset()
        reduced = logger.reduce()

        assert reduced.static.mean_value is None
        assert reduced.static.sum_value == 0
        assert reduced.iter is None

    def test_reset_returns_nothing_and_peek_afterwards_is_empty(self, logger):
        run_episode(logger, [1.0])

        assert logger.reset() is None
        assert logger.peek().static.mean_value == []
        assert logger.peek().iter == []

    def test_reset_also_empties_dynamic_children_created_after_construction(
        self, logger
    ):
        path = ("group", "by_id", "agent_0", "mean_value")
        logger.push(path, 5.0)
        logger.push(("group", "by_id", "agent_0", "sum_value"), 5.0)

        logger.reset()

        assert logger.peek_value(path) is None
        assert logger.peek_value(("group", "by_id", "agent_0", "sum_value")) == 0

    def test_flush_ends_the_window_of_one_metric_only(self, logger):
        run_episode(logger, [1.0, 3.0])

        logger.flush(MEAN)
        reduced = logger.reduce()

        assert reduced.static.mean_value is None
        assert reduced.static.sum_value == 4.0

    def test_dynamic_child_with_no_push_in_the_episode_reduces_to_empty_values(
        self, logger
    ):
        logger.push(("group", "by_id", "a", "mean_value"), 2.0)
        logger.reduce()  # episode 1 ends; agent "a" stays materialised

        logger.push(("group", "by_id", "b", "mean_value"), 4.0)
        reduced = logger.reduce()

        assert reduced.group.by_id["b"].mean_value == 4.0
        assert reduced.group.by_id["a"].mean_value is None
        assert reduced.group.by_id["a"].sum_value == 0

    def test_compile_ends_the_window_like_reduce(self, logger):
        run_episode(logger, [1.0, 3.0])

        compiled = logger.compile()

        assert compiled["static"]["mean_value"] == 2.0
        assert compiled["iter"] == 1
        assert logger.compile()["static"]["mean_value"] is None

    def test_compile_serialises_a_runtime_subtype_with_its_extra_fields(
        self, logger, metric_schemas
    ):
        Leaf, Rich, Group, Root = metric_schemas
        logger.push_data(
            Root(static=Leaf(), group=Group(), inner=Rich(extra=2.0, mean_value=1.0))
        )

        compiled = logger.compile()

        assert compiled["inner"]["extra"] == 2.0
        assert compiled["inner"]["mean_value"] == 1.0


@pytest.mark.unit
class TestNonFiniteThroughTheLogger:
    def test_nan_pushed_to_a_mean_comes_out_as_nan(self, logger):
        logger.push(MEAN, 1.0)
        logger.push(MEAN, float("nan"))

        assert math.isnan(logger.reduce().static.mean_value)

    def test_nan_in_one_window_does_not_poison_the_next(self, logger):
        logger.push(MEAN, float("nan"))
        logger.reduce()

        logger.push(MEAN, 2.0)

        assert logger.reduce().static.mean_value == 2.0

    def test_nan_payload_is_pushed_by_push_data(self, logger, metric_schemas):
        Leaf, _, Group, Root = metric_schemas
        logger.push_data(
            Root(static=Leaf(mean_value=float("nan"), sum_value=1.0), group=Group())
        )

        reduced = logger.reduce()

        assert math.isnan(reduced.static.mean_value)
        assert reduced.static.sum_value == 1.0


@pytest.mark.unit
class TestSubtreeReduce:
    """``json_schema_extra={"subtree_reduce": ...}`` overrides every leaf below."""

    def test_forced_subtree_overrides_default_and_declared_protocols(
        self, subtree_logger
    ):
        for value in (1.0, 2.0, 6.0):
            subtree_logger.push(("forced", "a"), value)  # would be MEAN
            subtree_logger.push(("forced", "b"), value)  # would be SERIES

        reduced = subtree_logger.reduce()

        assert reduced.forced.a == 6.0
        assert reduced.forced.b == 6.0

    def test_override_propagates_to_schemas_nested_below_the_field(
        self, subtree_logger
    ):
        for value in (1.0, 5.0, 2.0):
            subtree_logger.push(("nested", "inner", "a"), value)
            subtree_logger.push(("nested", "inner", "b"), value)

        reduced = subtree_logger.reduce()

        assert reduced.nested.inner.a == 5.0  # MAX, not MEAN
        assert reduced.nested.inner.b == 5.0  # MAX, not SERIES

    def test_control_subtree_keeps_its_own_protocols(self, subtree_logger):
        for value in (1.0, 2.0, 6.0):
            subtree_logger.push(("control", "a"), value)
            subtree_logger.push(("control", "b"), value)

        reduced = subtree_logger.reduce()

        assert reduced.control.a == 3.0
        assert reduced.control.b == [1.0, 2.0, 6.0]

    def test_override_reaches_the_children_of_a_dynamic_node(self, subtree_logger):
        for value in (1.0, 2.0):
            subtree_logger.push(("forced_by_id", "x", "a"), value)
            subtree_logger.push(("forced_by_id", "x", "b"), value * 10)
        subtree_logger.push(("forced_by_id", "y", "a"), 5.0)

        reduced = subtree_logger.reduce()

        assert reduced.forced_by_id["x"].a == 3.0  # SUM, not MEAN
        assert reduced.forced_by_id["x"].b == 30.0  # SUM, not SERIES
        assert reduced.forced_by_id["y"].a == 5.0

    def test_override_is_inherited_by_a_runtime_subtype(
        self, subtree_logger, subtree_schemas
    ):
        SubLeaf, SubRoot = subtree_schemas

        class Wider(SubLeaf):
            c: float | None = Field(
                default=None, json_schema_extra={"reduce": ReduceProtocol.MIN}
            )

        for value in (4.0, 1.0, 2.0):
            subtree_logger.push_data(SubRoot(forced=Wider(a=value, b=value, c=value)))

        reduced = subtree_logger.reduce()

        assert type(reduced.forced) is Wider
        assert (reduced.forced.a, reduced.forced.b, reduced.forced.c) == (2.0, 2.0, 2.0)
        # without the override, c would be MIN = 1.0 and a would be MEAN = 7/3

    def test_override_is_inherited_by_a_dynamic_runtime_subtype(
        self, subtree_logger, subtree_schemas
    ):
        SubLeaf, SubRoot = subtree_schemas

        class Wider(SubLeaf):
            c: float | None = None

        for value in (1.0, 2.0, 3.0):
            subtree_logger.push_data(
                SubRoot(forced_by_id={"x": Wider(a=value, c=value)})
            )

        reduced = subtree_logger.reduce()

        assert reduced.forced_by_id["x"].a == 6.0
        assert reduced.forced_by_id["x"].c == 6.0
