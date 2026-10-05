"""``TensorBoardReporter`` and ``TensorBoardConfig``: tags, steps, writer lifecycle.

New tests (the August ones used the removed ``Series`` contract). Two layers:
a recording fake writer injected in ``_writer`` pins the exact ``add_scalar``
calls without touching the disk, and a few tests let the real
``torch.utils.tensorboard.SummaryWriter`` write event files under ``tmp_path``
and read them back with tensorboard's ``EventAccumulator``. Tags are
``<sanitised title>/<sanitised series label>``, a standard-deviation series is
logged under the same tag plus ``/std``, the step is the integer x value, and
the colour path of a query is ignored with an info message.
"""

from __future__ import annotations

import logging
import sys
from enum import Enum

import pytest

from core.metrics.enums import ReduceProtocol
from core.reporting.query import Query
from core.reporting.tensor_board import TensorBoardConfig, TensorBoardReporter

SERIES = ReduceProtocol.SERIES
MEAN = ReduceProtocol.MEAN

pytestmark = pytest.mark.unit


class FakeWriter:
    """Records the calls of ``SummaryWriter`` that the reporter uses."""

    def __init__(self) -> None:
        self.scalars: list[tuple[str, float, int]] = []
        self.flushes = 0
        self.closed = False

    def add_scalar(self, *, tag, scalar_value, global_step) -> None:
        self.scalars.append((tag, scalar_value, global_step))

    def flush(self) -> None:
        self.flushes += 1

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_reporter(tmp_path) -> tuple[TensorBoardReporter, FakeWriter]:
    reporter = TensorBoardReporter(log_dir=tmp_path / "tb")
    writer = FakeWriter()
    reporter._writer = writer  # type: ignore[assignment]

    return reporter, writer


def read_scalars(log_dir) -> dict[str, list[tuple[int, float]]]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(str(log_dir))
    accumulator.Reload()

    return {
        tag: [(event.step, event.value) for event in accumulator.Scalars(tag)]
        for tag in accumulator.Tags()["scalars"]
    }


class TestConfig:
    def test_build_nests_project_and_world_label(self, tmp_path):
        config = TensorBoardConfig(project="proj", log_dir=str(tmp_path))
        config.world = "w"

        assert config.build(label="env0").log_dir == tmp_path / "proj" / "w-env0"
        assert config.build().log_dir == tmp_path / "proj" / "w"

    def test_building_is_lazy(self, tmp_path):
        config = TensorBoardConfig(project="proj", log_dir=str(tmp_path / "runs"))
        config.world = "w"

        reporter = config.build(label="a")

        assert isinstance(reporter, TensorBoardReporter)
        assert reporter._writer is None
        assert not (tmp_path / "runs").exists()

    def test_default_log_dir(self):
        assert TensorBoardConfig(project="p").log_dir.as_posix() == "runs"


class TestLabels:
    def test_path_name_drops_reduction_tokens_and_unwraps_enums(self):
        class Axis(Enum):
            TIME = "time"

        name = TensorBoardReporter._path_name(("a", SERIES, "b", MEAN, Axis.TIME))

        assert name == "a/b/time"

    def test_series_label_appends_the_group(self):
        group = (("by_mech", "m0"),)

        assert TensorBoardReporter._series_label(("a",), ()) == "a"
        assert TensorBoardReporter._series_label(("a",), group) == "a [by_mech=m0]"
        assert (
            TensorBoardReporter._series_label(("a",), group, label="L")
            == "L [by_mech=m0]"
        )


class TestScalarsWithAFakeWriter:
    def test_static_series_tag_value_and_step(self, fake_reporter, reporting_metrics):
        reporter, writer = fake_reporter
        reporter.add_query(Query(title="Loss curve", x=("iter",), y=("loss",)))

        reporter.report(reporting_metrics)

        assert writer.scalars == [
            ("Loss_curve/loss", 3.0, 0),
            ("Loss_curve/loss", 2.0, 1),
            ("Loss_curve/loss", 1.0, 2),
        ]
        assert writer.flushes == 1

    def test_dynamic_groups_get_one_tag_each(self, fake_reporter, reporting_metrics):
        reporter, writer = fake_reporter
        reporter.add_query(
            Query(title="Fit", x=("iter",), y=("by_mech", SERIES, "fitness"))
        )

        reporter.report(reporting_metrics)

        by_tag = {}
        for tag, value, step in writer.scalars:
            by_tag.setdefault(tag, []).append((step, value))
        assert by_tag == {
            "Fit/by_mech_fitness_by_mech_m0_": [(0, 10.0), (1, 11.0), (2, 12.0)],
            "Fit/by_mech_fitness_by_mech_m1_": [(0, 20.0), (1, 21.0), (2, 22.0)],
        }

    def test_legend_labels_name_the_tags(self, fake_reporter, reporting_metrics):
        reporter, writer = fake_reporter
        reporter.add_query(
            Query(
                title="T",
                x=("iter",),
                y=(("loss",), ("by_mech", "m0", "fitness")),
                legend_labels=("my loss", "my fitness"),
            )
        )

        reporter.report(reporting_metrics)

        assert {tag for tag, _, _ in writer.scalars} == {"T/my_loss", "T/my_fitness"}

    def test_std_is_logged_under_a_std_suffix(self, fake_reporter, reporting_metrics):
        reporter, writer = fake_reporter
        reporter.add_query(
            Query(
                title="Mean",
                x=("iter",),
                y=("by_mech", MEAN, "fitness"),
                error="std",
                error_path=("by_mech",),
            )
        )

        reporter.report(reporting_metrics)

        assert writer.scalars == [
            ("Mean/by_mech_fitness", 15.0, 0),
            ("Mean/by_mech_fitness", 16.0, 1),
            ("Mean/by_mech_fitness", 17.0, 2),
            ("Mean/by_mech_fitness/std", 5.0, 0),
            ("Mean/by_mech_fitness/std", 5.0, 1),
            ("Mean/by_mech_fitness/std", 5.0, 2),
        ]

    def test_dynamic_x_axis_is_matched_by_group(self, fake_reporter):
        reporter, writer = fake_reporter
        query = Query(title="t", x=("x",), y=("y",))
        group = (("d", "a"),)

        reporter._report(query, {group: [4, 5]}, [{group: [1.0, 2.0]}], [{}], None)

        assert [step for _, _, step in writer.scalars] == [4, 5]

    def test_missing_x_group_is_an_error(self, fake_reporter):
        reporter, _ = fake_reporter
        query = Query(title="t", x=("x",), y=("y",))

        with pytest.raises(ValueError, match="No x series exists for group"):
            reporter._report(
                query, {(("d", "a"),): [0]}, [{(("d", "b"),): [1.0]}], [{}], None
            )

    def test_empty_resolution_creates_no_writer(self, tmp_path):
        reporter = TensorBoardReporter(log_dir=tmp_path / "tb")
        query = Query(title="t", x=("x",), y=("y",))

        reporter._report(query, {(): [0]}, [{}], [{}], None)

        assert reporter._writer is None
        assert not (tmp_path / "tb").exists()

    def test_color_is_ignored_with_an_info_message(
        self, fake_reporter, reporting_metrics, caplog
    ):
        reporter, writer = fake_reporter
        reporter.add_query(
            Query(title="Colored", x=("iter",), y=("loss",), color=("iter",))
        )

        with caplog.at_level(logging.INFO, logger="core.reporting.tensor_board"):
            reporter.report(reporting_metrics)

        assert len(writer.scalars) == 3
        assert "ignores the color path ('iter',) of query 'Colored'" in caplog.text

    def test_close_closes_and_forgets_the_writer(self, fake_reporter):
        reporter, writer = fake_reporter

        reporter.close()
        reporter.close()

        assert writer.closed and reporter._writer is None

    def test_close_without_a_writer_is_a_no_op(self, tmp_path):
        reporter = TensorBoardReporter(log_dir=tmp_path / "tb")

        reporter.close()

        assert reporter._writer is None and not (tmp_path / "tb").exists()


class TestEachPointIsWrittenOnce:
    """A report receives the whole history again; only new points are written."""

    @pytest.fixture
    def history(self, reporting_metrics):
        def make(loss):
            return reporting_metrics.model_copy(
                update={"iter": list(range(len(loss))), "loss": loss}
            )

        return make

    def test_an_unchanged_history_writes_nothing_the_second_time(
        self, fake_reporter, history
    ):
        reporter, writer = fake_reporter
        reporter.add_query(Query(title="L", x=("iter",), y=("loss",)))
        metrics = history([3.0, 2.0, 1.0])

        reporter.report(metrics)
        reporter.report(metrics)

        assert writer.scalars == [
            ("L/loss", 3.0, 0),
            ("L/loss", 2.0, 1),
            ("L/loss", 1.0, 2),
        ]

    def test_a_growing_history_writes_only_the_new_points(self, fake_reporter, history):
        reporter, writer = fake_reporter
        reporter.add_query(Query(title="L", x=("iter",), y=("loss",)))

        reporter.report(history([3.0, 2.0]))
        reporter.report(history([3.0, 2.0, 1.0]))
        reporter.report(history([3.0, 2.0, 1.0, 0.5]))

        assert writer.scalars == [
            ("L/loss", 3.0, 0),
            ("L/loss", 2.0, 1),
            ("L/loss", 1.0, 2),
            ("L/loss", 0.5, 3),
        ]

    def test_a_cleared_history_is_written_again_from_its_start(
        self, fake_reporter, history
    ):
        reporter, writer = fake_reporter
        reporter.add_query(Query(title="L", x=("iter",), y=("loss",)))

        reporter.report(history([3.0, 2.0]))
        reporter.report(history([7.0, 6.0]))  # same length, new values
        reporter.report(history([5.0]))  # shorter

        assert writer.scalars == [
            ("L/loss", 3.0, 0),
            ("L/loss", 2.0, 1),
            ("L/loss", 7.0, 0),
            ("L/loss", 6.0, 1),
            ("L/loss", 5.0, 0),
        ]

    def test_nan_points_are_not_rewritten(self, fake_reporter, history):
        reporter, writer = fake_reporter
        reporter.add_query(Query(title="L", x=("iter",), y=("loss",)))
        metrics = history([float("nan"), 1.0])

        reporter.report(metrics)
        reporter.report(metrics)

        assert len(writer.scalars) == 2

    def test_each_group_and_the_std_series_are_tracked_separately(
        self, fake_reporter, reporting_metrics
    ):
        reporter, writer = fake_reporter
        reporter.add_query(
            Query(
                title="Mean",
                x=("iter",),
                y=("by_mech", MEAN, "fitness"),
                error="std",
                error_path=("by_mech",),
            ),
            Query(title="Fit", x=("iter",), y=("by_mech", SERIES, "fitness")),
        )

        reporter.report(reporting_metrics)
        written = len(writer.scalars)
        reporter.report(reporting_metrics)

        assert written == 3 + 3 + 3 + 3 and len(writer.scalars) == written

    def test_the_state_survives_close(self, fake_reporter, history):
        reporter, writer = fake_reporter
        reporter.add_query(Query(title="L", x=("iter",), y=("loss",)))
        metrics = history([3.0, 2.0])

        reporter.report(metrics)
        reporter.close()
        reporter._writer = writer  # stands for the writer created by the next report
        reporter.report(metrics)

        assert len(writer.scalars) == 2

    def test_a_rejected_x_writes_no_partial_series(self, fake_reporter):
        reporter, writer = fake_reporter
        query = Query(title="t", x=("x",), y=("y",))

        with pytest.raises(TypeError, match="integer-valued"):
            reporter._report(query, {(): [0, 0.5]}, [{(): [1.0, 2.0]}], [{}], None)

        assert writer.scalars == []

    def test_event_files_hold_each_step_once(self, tmp_path, history):
        reporter = TensorBoardReporter(log_dir=tmp_path / "tb")
        reporter.add_query(Query(title="L", x=("iter",), y=("loss",)))

        reporter.report(history([3.0, 2.0]))
        reporter.report(history([3.0, 2.0, 1.0]))
        reporter.report(history([3.0, 2.0, 1.0]))
        reporter.close()

        assert read_scalars(tmp_path / "tb")["L/loss"] == [(0, 3.0), (1, 2.0), (2, 1.0)]


class TestStep:
    QUERY = Query(title="t", x=("iter",), y=("loss",))

    @pytest.mark.parametrize(("value", "step"), [(3, 3), (4.0, 4), (True, 1)])
    def test_integer_valued_x_is_accepted(self, value, step):
        assert TensorBoardReporter._step(value, self.QUERY) == step

    @pytest.mark.parametrize("value", [0.5, float("nan"), "abc", None, [1]])
    def test_non_integer_x_is_a_type_error(self, value):
        with pytest.raises(TypeError, match="integer-valued") as exc:
            TensorBoardReporter._step(value, self.QUERY)

        assert f"{value!r}" in str(exc.value)
        assert "('iter',)" in str(exc.value)

    def test_a_fractional_x_stops_the_report(self, fake_reporter):
        reporter, _ = fake_reporter

        with pytest.raises(TypeError, match="integer-valued"):
            reporter._report(self.QUERY, {(): [0.5]}, [{(): [1.0]}], [{}], None)


class TestRealEventFiles:
    def test_scalars_are_written_and_readable(self, tmp_path, reporting_metrics):
        config = TensorBoardConfig(project="p", log_dir=str(tmp_path))
        config.world = "w"
        reporter = config.build(label="x")
        reporter.add_query(
            Query(
                title="Mean",
                x=("iter",),
                y=("by_mech", MEAN, "fitness"),
                error="std",
                error_path=("by_mech",),
            )
        )

        reporter.report(reporting_metrics)
        reporter.close()

        scalars = read_scalars(tmp_path / "p" / "w-x")
        assert scalars["Mean/by_mech_fitness"] == [(0, 15.0), (1, 16.0), (2, 17.0)]
        assert scalars["Mean/by_mech_fitness/std"] == [(0, 5.0), (1, 5.0), (2, 5.0)]

    def test_writer_is_created_once_and_reused(self, tmp_path, reporting_metrics):
        reporter = TensorBoardReporter(log_dir=tmp_path / "tb")
        reporter.add_query(Query(title="L", x=("iter",), y=("loss",)))

        reporter.report(reporting_metrics)
        writer = reporter._writer
        reporter.report(reporting_metrics)

        assert writer is not None and reporter._writer is writer
        reporter.close()
        assert reporter._writer is None

    def test_a_new_writer_is_created_after_close(self, tmp_path, reporting_metrics):
        reporter = TensorBoardReporter(log_dir=tmp_path / "tb")
        reporter.add_query(Query(title="L", x=("iter",), y=("loss",)))

        reporter.report(reporting_metrics)
        reporter.close()
        reporter.report(reporting_metrics)
        reporter.close()

        assert len(list((tmp_path / "tb").glob("events.out.tfevents.*"))) >= 1
        assert read_scalars(tmp_path / "tb")["L/loss"][0] == (0, 3.0)

    def test_missing_tensorboard_package_gives_an_actionable_error(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setitem(sys.modules, "torch.utils.tensorboard", None)
        reporter = TensorBoardReporter(log_dir=tmp_path / "tb")

        with pytest.raises(ImportError, match="uv sync --extra tensorboard"):
            reporter._get_writer()

        assert reporter._writer is None
