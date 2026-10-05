"""``Reporter`` base class: query registry, path and query resolution.

Ported from the August ``test_query_and_reporter.py`` and ``test_reporter_edges.py``.
The wildcard ``"*"`` of that version is now the ``ReduceProtocol.SERIES`` token
(one group per dynamic id) and ``reduce="mean"`` is the ``ReduceProtocol.MEAN``
token inside the path. A resolution maps a *group*, a tuple of
``(junction, dynamic_id)`` pairs, to a list of values; the empty group ``()`` is
a path with no dynamic node. The metric tree comes from ``reporting_metrics``
in ``conftest.py``; tests that need a broken tree copy it and edit the copy.
"""

from __future__ import annotations

import logging
import math
from enum import Enum

import pytest

from core.metrics.enums import ReduceProtocol
from core.reporting.base import PathResolution, Reporter
from core.reporting.query import Query

SERIES = ReduceProtocol.SERIES
MEAN = ReduceProtocol.MEAN

M0 = (("by_mech", "m0"),)
M1 = (("by_mech", "m1"),)
M0_S1 = (("by_mech", "m0"), ("by_seed", "s1"))
M0_S2 = (("by_mech", "m0"), ("by_seed", "s2"))
M1_S1 = (("by_mech", "m1"), ("by_seed", "s1"))
M1_S2 = (("by_mech", "m1"), ("by_seed", "s2"))


@pytest.mark.unit
class TestSeriesLabels:
    """The label helpers of the base class, shared by every backend."""

    def test_path_name_drops_reduction_tokens_and_unwraps_enums(self):
        class Axis(Enum):
            TIME = "time"

        name = Reporter._path_name(("by_mech", SERIES, "by_seed", MEAN, Axis.TIME))

        assert name == "by_mech/by_seed/time"

    def test_series_label_appends_the_group(self):
        group = (("by_mech", "m0"), ("by_seed", "s1"))

        assert Reporter._series_label(("a", "b"), ()) == "a/b"
        assert (
            Reporter._series_label(("a", "b"), group) == "a/b [by_mech=m0, by_seed=s1]"
        )
        assert Reporter._series_label(("a", "b"), group, label="L") == (
            "L [by_mech=m0, by_seed=s1]"
        )

    def test_the_backends_inherit_the_helpers(self):
        from core.reporting.csv import CSVReporter
        from core.reporting.tensor_board import TensorBoardReporter
        from core.reporting.wandb import WandbReporter

        for backend in (CSVReporter, TensorBoardReporter, WandbReporter):
            assert "_path_name" not in vars(backend)
            assert "_series_label" not in vars(backend)


@pytest.mark.unit
class TestQueryRegistry:
    def test_queries_accumulate_in_order(self, backendless_reporter):
        first = Query(title="a", x=("iter",), y=("loss",))
        second = Query(title="b", x=("iter",), y=("loss",))

        backendless_reporter.add_query(first)
        backendless_reporter.add_query(second, first)

        assert backendless_reporter.queries == (first, second, first)

    def test_queries_are_not_shared_between_reporters(self, backendless_reporter):
        backendless_reporter.add_query(Query(title="a", x=("iter",), y=("loss",)))

        other = type(backendless_reporter)()

        assert other.queries == ()


@pytest.mark.unit
class TestResolvePathStatic:
    @pytest.mark.parametrize(
        ("path", "expected"),
        [
            (("iter",), [0, 1, 2]),
            (("loss",), [3.0, 2.0, 1.0]),
            (("by_mech", "m1", "fitness"), [20.0, 21.0, 22.0]),
            (("by_mech", "m0", "by_seed", "s2", "value"), [3.0, 4.0, 5.0]),
        ],
    )
    def test_concrete_path_resolves_to_the_empty_group(
        self, backendless_reporter, reporting_metrics, path, expected
    ):
        result = backendless_reporter._resolve_path(path, reporting_metrics)

        assert result == PathResolution(values={(): expected}, errors={})

    @pytest.mark.parametrize(
        ("path", "message"),
        [
            (("scalar",), r"does not point to a metric series: \('scalar',\)"),
            ((), "does not point to a metric series"),
            (("by_mech",), "does not point to a metric series"),
            (("by_mech", "m0"), "does not point to a metric series"),
            (("nope",), r"Unknown metric path: \('nope',\)"),
            (("by_mech", "zz", "fitness"), "Unknown metric path"),
            (("by_mech", "m0", "nope"), "Unknown metric path"),
        ],
    )
    def test_path_errors_name_the_path(
        self, backendless_reporter, reporting_metrics, path, message
    ):
        with pytest.raises(KeyError, match=message):
            backendless_reporter._resolve_path(path, reporting_metrics)

    def test_nested_series_requires_a_reduction(
        self, backendless_reporter, reporting_metrics
    ):
        with pytest.raises(ValueError, match=r"series reduction is required"):
            backendless_reporter._resolve_path(("nested",), reporting_metrics)

    def test_reduction_token_on_a_schema_is_a_type_error(
        self, backendless_reporter, reporting_metrics
    ):
        with pytest.raises(TypeError, match="cannot be applied to RootMetrics"):
            backendless_reporter._resolve_path((MEAN, "loss"), reporting_metrics)

    @pytest.mark.parametrize("token", [ReduceProtocol.SUM, ReduceProtocol.LAST])
    def test_unsupported_dictionary_reduction(
        self, backendless_reporter, reporting_metrics, token
    ):
        with pytest.raises(NotImplementedError, match=str(token)):
            backendless_reporter._resolve_path(
                ("by_mech", token, "fitness"), reporting_metrics
            )


@pytest.mark.unit
class TestResolvePathSeries:
    def test_series_expands_one_group_per_dynamic_id(
        self, backendless_reporter, reporting_metrics
    ):
        result = backendless_reporter._resolve_path(
            ("by_mech", SERIES, "fitness"), reporting_metrics
        )

        assert result.values == {M0: [10.0, 11.0, 12.0], M1: [20.0, 21.0, 22.0]}
        assert result.errors == {}

    def test_two_levels_of_series_chain_the_groups(
        self, backendless_reporter, reporting_metrics
    ):
        result = backendless_reporter._resolve_path(
            ("by_mech", SERIES, "by_seed", SERIES, "value"), reporting_metrics
        )

        assert list(result.values) == [M0_S1, M0_S2, M1_S1, M1_S2]
        assert result.values[M1_S2] == [7.0, 8.0, 9.0]

    def test_groups_come_out_sorted_whatever_the_insertion_order(
        self, backendless_reporter, reporting_metrics
    ):
        reporting_metrics.by_mech = dict(reversed(reporting_metrics.by_mech.items()))

        result = backendless_reporter._resolve_path(
            ("by_mech", SERIES, "fitness"), reporting_metrics
        )

        assert list(result.values) == [M0, M1]

    def test_series_over_an_empty_dynamic_node_is_empty(
        self, backendless_reporter, reporting_metrics
    ):
        reporting_metrics.by_mech = {}

        result = backendless_reporter._resolve_path(
            ("by_mech", SERIES, "fitness"), reporting_metrics
        )

        assert result == PathResolution(values={}, errors={})


@pytest.mark.unit
class TestResolvePathMean:
    def test_mean_over_the_dynamic_node_is_pointwise(
        self, backendless_reporter, reporting_metrics
    ):
        result = backendless_reporter._resolve_path(
            ("by_mech", MEAN, "fitness"), reporting_metrics
        )

        assert result.values == {(): [15.0, 16.0, 17.0]}
        assert result.errors == {}

    def test_mean_over_seeds_keeps_one_group_per_mechanism(
        self, backendless_reporter, reporting_metrics
    ):
        result = backendless_reporter._resolve_path(
            ("by_mech", SERIES, "by_seed", MEAN, "value"), reporting_metrics
        )

        assert result.values == {M0: [2.0, 3.0, 4.0], M1: [6.0, 7.0, 8.0]}

    def test_std_is_captured_at_the_reduction_named_by_error_path(
        self, backendless_reporter, reporting_metrics
    ):
        result = backendless_reporter._resolve_path(
            ("by_mech", MEAN, "fitness"),
            reporting_metrics,
            error="std",
            error_path=("by_mech",),
        )

        assert result.values == {(): [15.0, 16.0, 17.0]}
        assert result.errors == {(): [5.0, 5.0, 5.0]}

    def test_std_per_mechanism_over_seeds(
        self, backendless_reporter, reporting_metrics
    ):
        result = backendless_reporter._resolve_path(
            ("by_mech", SERIES, "by_seed", MEAN, "value"),
            reporting_metrics,
            error="std",
            error_path=("by_mech", SERIES, "by_seed"),
        )

        assert result.values == {M0: [2.0, 3.0, 4.0], M1: [6.0, 7.0, 8.0]}
        assert result.errors == {M0: [1.0, 1.0, 1.0], M1: [1.0, 1.0, 1.0]}

    def test_error_path_elsewhere_captures_nothing(
        self, backendless_reporter, reporting_metrics
    ):
        result = backendless_reporter._resolve_path(
            ("by_mech", MEAN, "fitness"),
            reporting_metrics,
            error="std",
            error_path=("elsewhere",),
        )

        assert result.errors == {}

    def test_mean_over_an_empty_dynamic_node_is_empty(
        self, backendless_reporter, reporting_metrics
    ):
        reporting_metrics.by_mech = {}

        result = backendless_reporter._resolve_path(
            ("by_mech", MEAN, "fitness"), reporting_metrics
        )

        assert result == PathResolution(values={}, errors={})

    def test_mean_keeps_the_sorted_order_of_the_series_groups(
        self, backendless_reporter
    ):
        # Twenty-six groups: an order taken from a set of string tuples, which
        # depends on PYTHONHASHSEED, matches the sorted one by chance with a
        # probability of 1/26!.
        policies = [chr(ord("a") + index) for index in range(26)]
        metrics = {
            "by_seed": {
                seed: {"by_policy": {policy: {"v": [1.0]} for policy in policies}}
                for seed in ("s1", "s2")
            }
        }

        result = backendless_reporter._resolve_path(
            ("by_seed", MEAN, "by_policy", SERIES, "v"), metrics
        )

        assert [group[-1][1] for group in result.values] == policies

    def test_mean_over_branches_with_different_series_groups(
        self, backendless_reporter, reporting_metrics
    ):
        del reporting_metrics.by_mech["m1"].by_seed["s2"]

        with pytest.raises(ValueError, match="different SERIES groups"):
            backendless_reporter._resolve_path(
                ("by_mech", MEAN, "by_seed", SERIES, "value"), reporting_metrics
            )

    def test_mean_over_series_of_different_lengths(
        self, backendless_reporter, reporting_metrics
    ):
        reporting_metrics.by_mech["m1"].fitness.append(23.0)

        with pytest.raises(ValueError, match=r"different lengths: \[3, 4\]"):
            backendless_reporter._resolve_path(
                ("by_mech", MEAN, "fitness"), reporting_metrics
            )

    def test_mean_and_std_skip_the_gap_entries(
        self, backendless_reporter, reporting_metrics
    ):
        # A gap (None) is a push in which the logger had no value for that id.
        reporting_metrics.by_mech["m0"].fitness = [10.0, None, 12.0]

        result = backendless_reporter._resolve_path(
            ("by_mech", MEAN, "fitness"),
            reporting_metrics,
            error="std",
            error_path=("by_mech",),
        )

        assert result.values == {(): [15.0, 21.0, 17.0]}
        assert result.errors == {(): [5.0, 0.0, 5.0]}

    def test_a_point_where_every_branch_is_a_gap_is_nan(
        self, backendless_reporter, reporting_metrics
    ):
        reporting_metrics.by_mech["m0"].fitness = [10.0, None, 12.0]
        reporting_metrics.by_mech["m1"].fitness = [20.0, None, 22.0]

        result = backendless_reporter._resolve_path(
            ("by_mech", MEAN, "fitness"),
            reporting_metrics,
            error="std",
            error_path=("by_mech",),
        )

        mean, std = result.values[()], result.errors[()]
        assert (mean[0], mean[2]) == (15.0, 17.0) and math.isnan(mean[1])
        assert (std[0], std[2]) == (5.0, 5.0) and math.isnan(std[1])

    def test_a_nan_value_still_propagates_through_the_mean(
        self, backendless_reporter, reporting_metrics
    ):
        # NaN is a value (a statistic RLlib did not report), not a gap.
        reporting_metrics.by_mech["m0"].fitness = [10.0, float("nan"), 12.0]

        result = backendless_reporter._resolve_path(
            ("by_mech", MEAN, "fitness"), reporting_metrics
        )

        assert math.isnan(result.values[()][1])

    def test_error_below_another_mean_is_rejected(
        self, backendless_reporter, reporting_metrics
    ):
        with pytest.raises(ValueError, match="nested below another MEAN"):
            backendless_reporter._resolve_path(
                ("by_mech", MEAN, "by_seed", MEAN, "value"),
                reporting_metrics,
                error="std",
                error_path=("by_mech", MEAN, "by_seed"),
            )


@pytest.mark.unit
class TestResolveQuery:
    def test_static_x_with_several_y_paths(
        self, backendless_reporter, reporting_metrics
    ):
        query = Query(
            title="t", x=("iter",), y=(("loss",), ("by_mech", "m0", "fitness"))
        )

        xs, yss, error_yss, colors = backendless_reporter._resolve_query(
            reporting_metrics, query
        )

        assert xs == {(): [0, 1, 2]}
        assert yss == [{(): [3.0, 2.0, 1.0]}, {(): [10.0, 11.0, 12.0]}]
        assert error_yss == [{}, {}]
        assert colors is None

    def test_dynamic_x_and_y_share_their_groups(
        self, backendless_reporter, reporting_metrics
    ):
        query = Query(
            title="t", x=("by_mech", SERIES, "param"), y=("by_mech", SERIES, "fitness")
        )

        xs, yss, _, _ = backendless_reporter._resolve_query(reporting_metrics, query)

        assert xs == {M0: [0.1, 0.2, 0.3], M1: [0.4, 0.5, 0.6]}
        assert set(yss[0]) == set(xs)

    def test_errors_are_returned_per_y_path(
        self, backendless_reporter, reporting_metrics
    ):
        query = Query(
            title="t",
            x=("iter",),
            y=(("by_mech", MEAN, "fitness"), ("loss",)),
            error="std",
            error_path=("by_mech",),
        )

        _, yss, error_yss, _ = backendless_reporter._resolve_query(
            reporting_metrics, query
        )

        assert yss[0] == {(): [15.0, 16.0, 17.0]}
        assert error_yss == [{(): [5.0, 5.0, 5.0]}, {}]

    def test_color_series_are_resolved_with_the_same_machinery(
        self, backendless_reporter, reporting_metrics
    ):
        shared = Query(
            title="t", x=("iter",), y=("by_mech", SERIES, "fitness"), color=("loss",)
        )
        per_group = Query(
            title="t",
            x=("iter",),
            y=("by_mech", SERIES, "fitness"),
            color=("by_mech", SERIES, "param"),
        )

        *_, shared_colors = backendless_reporter._resolve_query(
            reporting_metrics, shared
        )
        *_, group_colors = backendless_reporter._resolve_query(
            reporting_metrics, per_group
        )

        assert shared_colors == {(): [3.0, 2.0, 1.0]}
        assert group_colors == {M0: [0.1, 0.2, 0.3], M1: [0.4, 0.5, 0.6]}

    def test_static_x_length_mismatch_names_x_y_and_group(
        self, backendless_reporter, reporting_metrics
    ):
        reporting_metrics.by_mech["m1"].fitness.append(23.0)
        query = Query(title="t", x=("iter",), y=("by_mech", SERIES, "fitness"))

        with pytest.raises(ValueError, match="must have equal length") as exc:
            backendless_reporter._resolve_query(reporting_metrics, query)

        message = str(exc.value)
        assert "x=('iter',) (3)" in message
        assert "group=(('by_mech', 'm1'),) (4)" in message

    def test_dynamic_x_and_y_groups_must_match(
        self, backendless_reporter, reporting_metrics
    ):
        query = Query(
            title="t", x=("by_mech", SERIES, "param"), y=("loss",)
        )  # y has only the empty group

        with pytest.raises(ValueError, match="groups do not match"):
            backendless_reporter._resolve_query(reporting_metrics, query)

    def test_dynamic_group_length_mismatch(
        self, backendless_reporter, reporting_metrics
    ):
        reporting_metrics.by_mech["m0"].param.append(0.4)
        query = Query(
            title="t", x=("by_mech", SERIES, "param"), y=("by_mech", SERIES, "fitness")
        )

        with pytest.raises(
            ValueError, match=r"equal length for group .*m0.*: x=4, y=3"
        ):
            backendless_reporter._resolve_query(reporting_metrics, query)

    def test_color_length_mismatch(self, backendless_reporter, reporting_metrics):
        reporting_metrics.loss.pop()
        query = Query(
            title="t", x=("iter",), y=("by_mech", SERIES, "fitness"), color=("loss",)
        )

        with pytest.raises(ValueError, match=r"color=2, y=3"):
            backendless_reporter._resolve_query(reporting_metrics, query)

    def test_color_without_a_series_for_the_group(
        self, backendless_reporter, reporting_metrics
    ):
        query = Query(
            title="t",
            x=("iter",),
            y=("by_mech", SERIES, "fitness"),
            color=("by_mech", SERIES, "by_seed", SERIES, "value"),
        )

        with pytest.raises(ValueError, match="No color series exists for group"):
            backendless_reporter._resolve_query(reporting_metrics, query)

    def test_error_group_without_a_y_series(self, stubbed_reporter_class):
        reporter = stubbed_reporter_class(
            {
                ("x",): PathResolution(values={(): [0, 1]}, errors={}),
                ("e", MEAN, "y"): PathResolution(
                    values={(): [1.0, 2.0]}, errors={M0: [0.1, 0.1]}
                ),
            }
        )
        query = Query(
            title="t", x=("x",), y=("e", MEAN, "y"), error="std", error_path=("e",)
        )

        with pytest.raises(ValueError, match="has no corresponding y series"):
            reporter._resolve_query(object(), query)

    def test_error_series_length_must_match_y(self, stubbed_reporter_class):
        reporter = stubbed_reporter_class(
            {
                ("x",): PathResolution(values={(): [0, 1]}, errors={}),
                ("e", MEAN, "y"): PathResolution(
                    values={(): [1.0, 2.0]}, errors={(): [0.1]}
                ),
            }
        )
        query = Query(
            title="t", x=("x",), y=("e", MEAN, "y"), error="std", error_path=("e",)
        )

        with pytest.raises(ValueError, match="Error series must have the same length"):
            reporter._resolve_query(object(), query)


@pytest.mark.unit
class TestReport:
    def test_every_query_is_resolved_and_forwarded_in_order(
        self, backendless_reporter, reporting_metrics
    ):
        first = Query(title="a", x=("iter",), y=("loss",))
        second = Query(
            title="b",
            x=("iter",),
            y=("by_mech", MEAN, "fitness"),
            error="std",
            error_path=("by_mech",),
        )
        backendless_reporter.add_query(first, second)

        backendless_reporter.report(reporting_metrics)

        (q1, x1, ys1, errors1, colors1), (q2, _, ys2, errors2, _) = (
            backendless_reporter.calls
        )
        assert (q1, q2) == (first, second)
        assert x1 == {(): [0, 1, 2]}
        assert ys1 == [{(): [3.0, 2.0, 1.0]}] and errors1 == [{}] and colors1 is None
        assert ys2 == [{(): [15.0, 16.0, 17.0]}]
        assert errors2 == [{(): [5.0, 5.0, 5.0]}]

    def test_report_does_not_mutate_the_metrics(
        self, backendless_reporter, reporting_metrics
    ):
        before = reporting_metrics.model_dump()
        backendless_reporter.add_query(
            Query(title="a", x=("iter",), y=("by_mech", MEAN, "fitness"))
        )

        backendless_reporter.report(reporting_metrics)

        assert reporting_metrics.model_dump() == before

    def test_no_query_means_no_backend_call(
        self, backendless_reporter, reporting_metrics
    ):
        backendless_reporter.report(reporting_metrics)

        assert backendless_reporter.calls == []

    def test_a_failing_query_is_logged_and_the_others_are_rendered(
        self, backendless_reporter, reporting_metrics, caplog
    ):
        # Reporting runs at the end of every training iteration: a stale query
        # must not abort the run nor hide the queries that come after it.
        good = Query(title="good", x=("iter",), y=("loss",))
        backendless_reporter.add_query(
            Query(title="bad", x=("iter",), y=("nope",)), good
        )

        with caplog.at_level(logging.ERROR, logger="core.reporting.base"):
            backendless_reporter.report(reporting_metrics)

        assert [call[0] for call in backendless_reporter.calls] == [good]
        (record,) = caplog.records
        assert "'bad'" in record.getMessage()
        assert record.exc_info is not None
        assert "Unknown metric path" in str(record.exc_info[1])

    def test_a_failing_backend_is_logged_and_the_others_are_rendered(
        self, backendless_reporter, reporting_metrics, caplog, monkeypatch
    ):
        first = Query(title="first", x=("iter",), y=("loss",))
        second = Query(title="second", x=("iter",), y=("loss",))
        backendless_reporter.add_query(first, second)
        rendered = []

        def flaky_report(query, *args):
            if query is first:
                raise OSError("disk full")
            rendered.append(query)

        monkeypatch.setattr(backendless_reporter, "_report", flaky_report)

        with caplog.at_level(logging.ERROR, logger="core.reporting.base"):
            backendless_reporter.report(reporting_metrics)

        assert rendered == [second]
        (record,) = caplog.records
        assert "'first'" in record.getMessage()
        assert isinstance(record.exc_info[1], OSError)

    def test_backend_hooks_are_abstract(self):
        with pytest.raises(TypeError, match="abstract"):
            Reporter()  # type: ignore[abstract]
