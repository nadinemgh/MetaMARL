"""``CSVReporter`` and ``CSVConfig``: the long-form file written for each query.

New tests (the August CSV tests used the removed ``Series`` contract). Every
test reads the file back and compares rows, not only its existence. All output
goes under pytest's ``tmp_path``. A row is ``(query, x, series, value, error,
color)``; ``series`` is the legend label or the y path joined by ``/`` followed
by ``[junction=id, ...]`` for a dynamic group; error and color cells are empty
when the query has none. The reporter is driven both through ``report`` (path
resolution included) and through ``_report`` with hand-made resolutions, to
reach the consistency errors that a real resolution cannot produce.
"""

from __future__ import annotations

import csv
from enum import Enum

import pytest

from core.metrics.enums import ReduceProtocol
from core.reporting.csv import CSVConfig, CSVReporter
from core.reporting.query import Query

SERIES = ReduceProtocol.SERIES
MEAN = ReduceProtocol.MEAN
HEADER = ["query", "x", "series", "value", "error", "color"]

pytestmark = pytest.mark.unit


def read_rows(reporter: CSVReporter, query: Query) -> list[list[str]]:
    with reporter.path_for(query).open(newline="", encoding="utf-8") as file:
        return list(csv.reader(file))


@pytest.fixture
def reporter(tmp_path) -> CSVReporter:
    return CSVReporter(output_dir=tmp_path / "csv")


class TestConfig:
    def test_build_nests_project_and_world_label(self, tmp_path):
        config = CSVConfig(project="proj", output_dir=str(tmp_path))
        config.world = "w"

        labelled = config.build(label="env0")
        unlabelled = config.build()

        assert isinstance(labelled, CSVReporter)
        assert labelled.output_dir == tmp_path / "proj" / "w-env0"
        assert unlabelled.output_dir == tmp_path / "proj" / "w"

    def test_building_creates_the_directory(self, tmp_path):
        config = CSVConfig(project="proj", output_dir=str(tmp_path / "deep" / "out"))
        config.world = "w"

        reporter = config.build(label="a")

        assert reporter.output_dir.is_dir()

    def test_default_output_dir_is_relative_results(self):
        assert CSVConfig(project="p").output_dir.as_posix() == "results"

    def test_copy_keeps_the_output_dir(self, tmp_path):
        config = CSVConfig(project="p", output_dir=str(tmp_path))
        config.world = "w"

        assert config.copy().output_dir == tmp_path


class TestFileNaming:
    def test_title_is_sanitised_into_the_file_name(self, reporter):
        query = Query(title="Fitness (mean ± std)", x=("iter",), y=("loss",))

        assert reporter.path_for(query) == reporter.output_dir / "Fitness_mean_std_.csv"

    def test_path_name_drops_reduction_tokens_and_unwraps_enums(self):
        class Axis(Enum):
            TIME = "time"

        name = CSVReporter._path_name(("by_mech", SERIES, "by_seed", MEAN, Axis.TIME))

        assert name == "by_mech/by_seed/time"

    def test_series_label_appends_the_group(self):
        group = (("by_mech", "m0"), ("by_seed", "s1"))

        assert CSVReporter._series_label(("a", "b"), ()) == "a/b"
        assert (
            CSVReporter._series_label(("a", "b"), group)
            == "a/b [by_mech=m0, by_seed=s1]"
        )
        assert CSVReporter._series_label(("a", "b"), group, label="L") == (
            "L [by_mech=m0, by_seed=s1]"
        )


class TestReportedRows:
    def test_static_series_rows(self, reporter, reporting_metrics):
        query = Query(
            title="Loss", x=("iter",), y=(("loss",), ("by_mech", "m1", "fitness"))
        )
        reporter.add_query(query)

        reporter.report(reporting_metrics)

        assert read_rows(reporter, query) == [
            HEADER,
            ["Loss", "0", "loss", "3.0", "", ""],
            ["Loss", "1", "loss", "2.0", "", ""],
            ["Loss", "2", "loss", "1.0", "", ""],
            ["Loss", "0", "by_mech/m1/fitness", "20.0", "", ""],
            ["Loss", "1", "by_mech/m1/fitness", "21.0", "", ""],
            ["Loss", "2", "by_mech/m1/fitness", "22.0", "", ""],
        ]

    def test_dynamic_groups_become_labelled_series(self, reporter, reporting_metrics):
        query = Query(
            title="By mech",
            x=("by_mech", SERIES, "param"),
            y=("by_mech", SERIES, "fitness"),
        )
        reporter.add_query(query)

        reporter.report(reporting_metrics)

        rows = read_rows(reporter, query)
        assert rows[0] == HEADER
        assert rows[1] == [
            "By mech",
            "0.1",
            "by_mech/fitness [by_mech=m0]",
            "10.0",
            "",
            "",
        ]
        assert rows[6] == [
            "By mech",
            "0.6",
            "by_mech/fitness [by_mech=m1]",
            "22.0",
            "",
            "",
        ]
        assert len(rows) == 7

    def test_legend_labels_replace_the_path_name(self, reporter, reporting_metrics):
        query = Query(
            title="Labelled",
            x=("iter",),
            y=(("loss",), ("by_mech", SERIES, "fitness")),
            legend_labels=("Loss", "Fitness"),
        )
        reporter.add_query(query)

        reporter.report(reporting_metrics)

        series = [row[2] for row in read_rows(reporter, query)[1:]]
        assert series == ["Loss"] * 3 + [
            "Fitness [by_mech=m0]",
            "Fitness [by_mech=m0]",
            "Fitness [by_mech=m0]",
            "Fitness [by_mech=m1]",
            "Fitness [by_mech=m1]",
            "Fitness [by_mech=m1]",
        ]

    def test_error_column_holds_the_pointwise_std(self, reporter, reporting_metrics):
        query = Query(
            title="Mean",
            x=("iter",),
            y=("by_mech", MEAN, "fitness"),
            error="std",
            error_path=("by_mech",),
        )
        reporter.add_query(query)

        reporter.report(reporting_metrics)

        assert read_rows(reporter, query) == [
            HEADER,
            ["Mean", "0", "by_mech/fitness", "15.0", "5.0", ""],
            ["Mean", "1", "by_mech/fitness", "16.0", "5.0", ""],
            ["Mean", "2", "by_mech/fitness", "17.0", "5.0", ""],
        ]

    def test_color_column_holds_the_shared_color_series(
        self, reporter, reporting_metrics
    ):
        query = Query(
            title="Colored",
            x=("by_mech", SERIES, "param"),
            y=("by_mech", SERIES, "fitness"),
            color=("iter",),
        )
        reporter.add_query(query)

        reporter.report(reporting_metrics)

        rows = read_rows(reporter, query)[1:]
        assert [row[5] for row in rows] == ["0", "1", "2", "0", "1", "2"]
        assert [row[4] for row in rows] == [""] * 6

    def test_color_column_follows_the_group_when_the_color_is_dynamic(
        self, reporter, reporting_metrics
    ):
        query = Query(
            title="Per group",
            x=("iter",),
            y=("by_mech", SERIES, "fitness"),
            color=("by_mech", SERIES, "param"),
        )
        reporter.add_query(query)

        reporter.report(reporting_metrics)

        rows = read_rows(reporter, query)[1:]
        assert [row[5] for row in rows] == ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6"]

    def test_reporting_again_rewrites_the_file(self, reporter, reporting_metrics):
        query = Query(title="Loss", x=("iter",), y=("loss",))
        reporter.add_query(query)

        reporter.report(reporting_metrics)
        reporting_metrics.loss = [9.0, 8.0, 7.0]
        reporter.report(reporting_metrics)

        rows = read_rows(reporter, query)
        assert [row[3] for row in rows] == ["value", "9.0", "8.0", "7.0"]

    def test_each_query_gets_its_own_file(self, reporter, reporting_metrics):
        first = Query(title="First", x=("iter",), y=("loss",))
        second = Query(title="Second", x=("iter",), y=("by_mech", "m0", "param"))
        reporter.add_query(first, second)

        reporter.report(reporting_metrics)

        assert sorted(p.name for p in reporter.output_dir.iterdir()) == [
            "First.csv",
            "Second.csv",
        ]
        assert read_rows(reporter, second)[1][3] == "0.1"

    def test_a_query_without_series_writes_no_file(self, reporter, reporting_metrics):
        reporting_metrics.by_mech = {}
        query = Query(title="Empty", x=("iter",), y=("by_mech", SERIES, "fitness"))
        reporter.add_query(query)

        reporter.report(reporting_metrics)

        assert list(reporter.output_dir.iterdir()) == []

    def test_unicode_values_survive_the_round_trip(self, reporter):
        query = Query(
            title="Unicode", x=("t",), y=("v",), legend_labels=("rendement ± écart",)
        )

        reporter._report(query, {(): [1]}, [{(): [2.5]}], [{}], None)

        assert read_rows(reporter, query)[1][2] == "rendement ± écart"

    def test_close_leaves_the_files_in_place(self, reporter, reporting_metrics):
        query = Query(title="Loss", x=("iter",), y=("loss",))
        reporter.add_query(query)
        reporter.report(reporting_metrics)

        assert reporter.close() is None

        assert reporter.path_for(query).exists()


class TestInconsistentResolutions:
    def test_missing_x_group_is_an_error(self, reporter):
        query = Query(title="t", x=("x",), y=("y",))
        x = {(("d", "a"),): [0, 1]}
        ys = [{(("d", "b"),): [1.0, 2.0]}]

        with pytest.raises(ValueError, match="No x series exists for group"):
            reporter._report(query, x, ys, [{}], None)

    def test_missing_color_group_is_an_error(self, reporter):
        query = Query(title="t", x=("x",), y=("y",), color=("c",))
        group_b = (("d", "b"),)
        colors = {(("d", "a"),): [0.0, 1.0]}

        with pytest.raises(ValueError, match="No color series exists for group"):
            reporter._report(query, {(): [0, 1]}, [{group_b: [1.0, 2.0]}], [{}], colors)

    def test_series_of_different_lengths_are_rejected_not_truncated(self, reporter):
        query = Query(title="t", x=("x",), y=("y",))

        with pytest.raises(ValueError, match="zip"):
            reporter._report(query, {(): [0, 1, 2]}, [{(): [1.0, 2.0]}], [{}], None)
