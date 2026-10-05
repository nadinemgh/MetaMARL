"""``WandbReporter`` and ``WandbConfig`` against a fake ``wandb`` module.

New tests (the August ones used the removed ``Series`` contract and the
``Table`` parallel-coordinates view). Nothing here touches the network or
starts a run: the ``wandb`` name inside ``core.reporting.wandb`` is replaced by
a namespace whose ``init`` returns a recording run, so the tests can check the
arguments of ``init``, the lifecycle (lazy start, ``finish`` on close, restart
after close) and the Plotly figure that is logged: traces, error bands, colour
axis, dash patterns and layout. The figure builder is also called directly,
because it is a pure function of the resolved series.

Not covered: a real ``wandb.init`` against a server or an offline run
directory, which would test the wandb library rather than this reporter.
"""

from __future__ import annotations

from enum import Enum
from types import SimpleNamespace

import plotly.graph_objects as go
import plotly.io as pio
import pytest

from core.metrics.enums import ReduceProtocol
from core.reporting.query import Query
from core.reporting.wandb import WandbConfig, WandbReporter

SERIES = ReduceProtocol.SERIES
MEAN = ReduceProtocol.MEAN
M0 = (("by_mech", "m0"),)
M1 = (("by_mech", "m1"),)

pytestmark = pytest.mark.unit


class FakeRun:
    """Stands for a ``wandb`` run: records what is logged and finished."""

    def __init__(self) -> None:
        self.logged: list[dict] = []
        self.finished = 0

    def log(self, payload: dict) -> None:
        self.logged.append(payload)

    def finish(self) -> None:
        self.finished += 1


@pytest.fixture
def fake_wandb(monkeypatch) -> SimpleNamespace:
    """Replace the ``wandb`` module used by the reporter; expose its calls."""

    state = SimpleNamespace(
        init_calls=[], settings_calls=[], runs=[], return_none=False
    )

    def init(**kwargs):
        state.init_calls.append(kwargs)

        if state.return_none:
            return None

        run = FakeRun()
        state.runs.append(run)

        return run

    def settings(**kwargs):
        state.settings_calls.append(kwargs)

        return {"settings": kwargs}

    monkeypatch.setattr(
        "core.reporting.wandb.wandb", SimpleNamespace(init=init, Settings=settings)
    )

    return state


@pytest.fixture
def wandb_reporter(fake_wandb) -> WandbReporter:
    return WandbReporter(
        project="proj",
        name="w-x",
        run_id="abc",
        group="w",
        config={"outer_iters": 4},
        settings={"quiet": True},
    )


def logged_figure(state: SimpleNamespace, key: str):
    (payload,) = state.runs[0].logged

    return payload[key]


class TestConfig:
    def test_default_settings_silence_the_wandb_side_channels(self):
        config = WandbConfig(project="p")

        assert config.settings == {
            "x_disable_stats": True,
            "x_disable_meta": True,
            "quiet": True,
            "max_end_of_run_summary_metrics": 0,
            "max_end_of_run_history_metrics": 0,
        }

    def test_settings_overrides(self):
        config = WandbConfig(project="p", quiet=False, x_disable_meta=False)

        assert config.settings["quiet"] is False
        assert config.settings["x_disable_meta"] is False

    def test_an_unknown_argument_is_rejected_by_name(self):
        with pytest.raises(TypeError, match="queit"):
            WandbConfig(project="p", queit=True)

    def test_build_describes_the_run_without_starting_it(self, fake_wandb):
        config = WandbConfig(project="proj")
        config.world = "w"
        config.outer_iters = 5

        reporter = config.build(label="env0")

        assert isinstance(reporter, WandbReporter)
        assert reporter._project == "proj" and reporter._group == "w"
        assert reporter._name == "w-env0"
        assert reporter._config == {"outer_iters": 5, "world_name": "w"}
        assert reporter._settings == config.settings
        assert reporter._run is None and fake_wandb.init_calls == []

    def test_build_without_label_names_the_run_after_the_world(self):
        config = WandbConfig(project="proj")
        config.world = "w"

        assert config.build()._name == "w"

    def test_every_build_gets_a_fresh_run_id(self):
        config = WandbConfig(project="proj")
        config.world = "w"

        first, second = config.build(), config.build()

        assert first._run_id != second._run_id
        assert len(first._run_id) == 32 and int(first._run_id, 16) >= 0


class TestRunLifecycle:
    def test_init_is_lazy_and_receives_the_run_description(
        self, fake_wandb, wandb_reporter
    ):
        assert fake_wandb.init_calls == []

        wandb_reporter._init_run()

        (call,) = fake_wandb.init_calls
        assert call == {
            "project": "proj",
            "id": "abc",
            "group": "w",
            "name": "w-x",
            "config": {"outer_iters": 4},
            "reinit": "create_new",
            "settings": {"settings": {"quiet": True}},
        }

    def test_init_happens_once(self, fake_wandb, wandb_reporter):
        wandb_reporter._init_run()
        wandb_reporter._init_run()

        assert len(fake_wandb.init_calls) == 1

    def test_missing_config_and_settings_become_empty_dicts(self, fake_wandb):
        reporter = WandbReporter(project="p", name="n", run_id="i", group="g")

        reporter._init_run()

        assert fake_wandb.init_calls[0]["config"] == {}
        assert fake_wandb.settings_calls == [{}]

    def test_close_finishes_the_run_and_a_later_report_restarts_it(
        self, fake_wandb, wandb_reporter, reporting_metrics
    ):
        wandb_reporter.add_query(Query(title="L", x=("iter",), y=("loss",)))
        wandb_reporter.report(reporting_metrics)

        wandb_reporter.close()
        wandb_reporter.report(reporting_metrics)

        assert fake_wandb.runs[0].finished == 1
        assert len(fake_wandb.init_calls) == 2
        assert len(fake_wandb.runs[1].logged) == 1

    def test_close_without_a_run_is_a_no_op(self, fake_wandb, wandb_reporter):
        wandb_reporter.close()

        assert fake_wandb.init_calls == [] and wandb_reporter._run is None

    def test_close_twice_finishes_once(self, fake_wandb, wandb_reporter):
        wandb_reporter._init_run()

        wandb_reporter.close()
        wandb_reporter.close()

        assert fake_wandb.runs[0].finished == 1

    def test_failed_initialisation_is_reported(self, fake_wandb, wandb_reporter):
        fake_wandb.return_none = True
        query = Query(title="t", x=("x",), y=("y",))

        with pytest.raises(RuntimeError, match="failed to initialize"):
            wandb_reporter._report(query, {(): [0]}, [{(): [1.0]}], [{}], None)


class TestLabels:
    def test_path_name_drops_reduction_tokens_and_unwraps_enums(self):
        class Axis(Enum):
            TIME = "time"

        assert WandbReporter._path_name(("a", SERIES, "b", Axis.TIME)) == "a/b/time"

    def test_series_label_appends_the_group(self):
        assert WandbReporter._series_label(("a",), M0) == "a [by_mech=m0]"
        assert WandbReporter._series_label(("a",), (), label="L") == "L"


class TestLoggedFigure:
    def test_report_logs_one_figure_under_the_sanitised_title(
        self, fake_wandb, wandb_reporter, reporting_metrics
    ):
        wandb_reporter.add_query(
            Query(title="Loss (train)", x=("iter",), y=("loss",), y_label="loss")
        )

        wandb_reporter.report(reporting_metrics)

        fig = logged_figure(fake_wandb, "plots/Loss_train_")
        (trace,) = fig.data
        assert trace.name == "loss" and trace.mode == "lines+markers"
        assert list(trace.x) == [0, 1, 2] and list(trace.y) == [3.0, 2.0, 1.0]
        assert fig.layout.title.text == "Loss (train)"
        assert fig.layout.xaxis.title.text == "iter"
        assert fig.layout.yaxis.title.text == "loss"
        assert fig.layout.hovermode == "x unified"
        assert fig.layout.height == 650
        assert fig.layout.template == pio.templates["plotly_white"]

    def test_axis_labels_default_and_override(
        self, fake_wandb, wandb_reporter, reporting_metrics
    ):
        wandb_reporter.add_query(
            Query(title="d", x=("iter",), y=("loss",)),
            Query(title="o", x=("iter",), y=("loss",), x_label="step", y_label="J"),
        )

        wandb_reporter.report(reporting_metrics)

        default, override = (p for p in fake_wandb.runs[0].logged)
        assert default["plots/d"].layout.xaxis.title.text == "iter"
        assert default["plots/d"].layout.yaxis.title.text == "value"
        assert override["plots/o"].layout.xaxis.title.text == "step"
        assert override["plots/o"].layout.yaxis.title.text == "J"

    def test_marker_only_modes_switch_to_closest_hover(
        self, fake_wandb, wandb_reporter, reporting_metrics
    ):
        wandb_reporter.add_query(
            Query(title="m", x=("iter",), y=("loss",), plot_modes=("markers",)),
            Query(title="l", x=("iter",), y=("loss",), plot_modes=("lines",)),
        )

        wandb_reporter.report(reporting_metrics)

        markers, lines = fake_wandb.runs[0].logged
        assert markers["plots/m"].layout.hovermode == "closest"
        assert markers["plots/m"].data[0].mode == "markers"
        assert lines["plots/l"].layout.hovermode == "x unified"
        assert lines["plots/l"].data[0].mode == "lines"

    def test_empty_resolution_starts_the_run_but_logs_nothing(
        self, fake_wandb, wandb_reporter
    ):
        query = Query(title="t", x=("x",), y=("y",))

        wandb_reporter._report(query, {(): [0]}, [{}], [{}], None)

        assert len(fake_wandb.runs) == 1 and fake_wandb.runs[0].logged == []

    def test_color_query_widens_legend_margin_and_adds_a_color_bar(
        self, fake_wandb, wandb_reporter, reporting_metrics
    ):
        wandb_reporter.add_query(
            Query(
                title="c",
                x=("by_mech", SERIES, "param"),
                y=("by_mech", SERIES, "fitness"),
                color=("iter",),
                color_label="step",
                colorscale="Viridis",
            )
        )

        wandb_reporter.report(reporting_metrics)

        fig = logged_figure(fake_wandb, "plots/c")
        assert fig.layout.legend.x == 1.15 and fig.layout.margin.r == 300
        assert fig.layout.coloraxis.colorbar.title.text == "step"
        assert (fig.layout.coloraxis.cmin, fig.layout.coloraxis.cmax) == (0.0, 2.0)

    def test_plain_query_uses_the_narrow_legend_margin(
        self, fake_wandb, wandb_reporter, reporting_metrics
    ):
        wandb_reporter.add_query(Query(title="p", x=("iter",), y=("loss",)))

        wandb_reporter.report(reporting_metrics)

        fig = logged_figure(fake_wandb, "plots/p")
        assert fig.layout.legend.x == 1.02 and fig.layout.margin.r == 220


class TestSeriesFigure:
    def build(self, query, xs, yss, error_yss=None, colors=None):
        return WandbReporter._series_figure(
            query, xs, yss, error_yss or [{} for _ in yss], colors
        )

    def test_one_trace_per_group_with_group_in_the_name(self):
        query = Query(title="t", x=("x",), y=("y",))
        fig = self.build(query, {(): [0, 1]}, [{M0: [1.0, 2.0], M1: [3.0, 4.0]}])

        assert [t.name for t in fig.data] == ["y [by_mech=m0]", "y [by_mech=m1]"]
        assert [t.line.dash for t in fig.data] == ["solid", "dash"]
        assert all(list(t.x) == [0, 1] for t in fig.data)

    def test_dash_patterns_cycle_through_four_styles(self):
        query = Query(title="t", x=("x",), y=("y",))
        groups = {(("d", str(i)),): [1.0] for i in range(5)}

        fig = self.build(query, {(): [0]}, [groups])

        assert [t.line.dash for t in fig.data] == [
            "solid",
            "dash",
            "dot",
            "dashdot",
            "solid",
        ]

    def test_group_labels_can_be_hidden_from_the_legend(self):
        query = Query(title="t", x=("x",), y=("y",), show_group_labels=False)
        fig = self.build(query, {(): [0]}, [{M0: [1.0], M1: [2.0]}])

        assert [t.name for t in fig.data] == ["y", "y"]
        assert [t.showlegend for t in fig.data] == [True, False]

    def test_each_y_path_has_its_own_color_label_and_mode(self):
        query = Query(
            title="t",
            x=("x",),
            y=(("a",), ("b",)),
            legend_labels=("A", "B"),
            plot_modes=("lines", "markers"),
        )

        fig = self.build(query, {(): [0]}, [{(): [1.0]}, {(): [2.0]}])

        assert [t.name for t in fig.data] == ["A", "B"]
        assert [t.mode for t in fig.data] == ["lines", "markers"]
        assert fig.data[0].line.color != fig.data[1].line.color

    def test_dynamic_x_is_looked_up_by_group(self):
        query = Query(title="t", x=("x",), y=("y",))
        fig = self.build(
            query, {M0: [5, 6], M1: [7, 8]}, [{M0: [1.0, 2.0], M1: [3.0, 4.0]}]
        )

        assert [list(t.x) for t in fig.data] == [[5, 6], [7, 8]]

    def test_missing_x_group_is_an_error(self):
        query = Query(title="t", x=("x",), y=("y",))

        with pytest.raises(ValueError, match="No x series exists for group"):
            self.build(query, {M0: [0]}, [{M1: [1.0]}])

    def test_error_band_is_a_closed_polygon_before_the_line(self):
        query = Query(title="t", x=("x",), y=("y",))
        fig = self.build(
            query, {(): [1, 2, 3]}, [{(): [2.0, 3.0, 4.0]}], [{(): [1.0, 1.0, 1.0]}]
        )

        band, line = fig.data
        assert band.name == "y ±1 std" and band.fill == "toself"
        assert band.showlegend is False and band.legendgroup == line.legendgroup == "y"
        assert list(band.x) == [1, 2, 3, 3, 2, 1]
        assert list(band.y) == [3.0, 4.0, 5.0, 3.0, 2.0, 1.0]
        assert band.line.color == line.line.color
        assert list(line.y) == [2.0, 3.0, 4.0]

    def test_error_length_mismatch_is_an_error(self):
        query = Query(title="t", x=("x",), y=("y",))

        with pytest.raises(ValueError, match="Error series length does not match"):
            self.build(query, {(): [1, 2]}, [{(): [2.0, 3.0]}], [{(): [1.0]}])

    def test_shared_color_series_colors_the_markers_on_one_axis(self):
        query = Query(title="t", x=("x",), y=("y",), color=("c",), colorscale="Cividis")
        fig = self.build(
            query,
            {(): [0, 1]},
            [{M0: [1.0, 2.0], M1: [3.0, 4.0]}],
            colors={(): [10.0, 30.0]},
        )

        assert all(t.marker.coloraxis == "coloraxis" for t in fig.data)
        assert all(list(t.marker.color) == [10.0, 30.0] for t in fig.data)
        axis = fig.layout.coloraxis
        assert (axis.cmin, axis.cmax) == (10.0, 30.0)
        assert axis.colorbar.title.text == "c"
        assert (
            axis.colorscale
            == go.Layout(coloraxis_colorscale="Cividis").coloraxis.colorscale
        )

    def test_per_group_colors_span_the_range_over_all_groups(self):
        query = Query(title="t", x=("x",), y=("y",), color=("c",), color_label="J")
        fig = self.build(
            query,
            {(): [0, 1]},
            [{M0: [1.0, 2.0], M1: [3.0, 4.0]}],
            colors={M0: [1.0, 2.0], M1: [5.0, 9.0]},
        )

        assert [list(t.marker.color) for t in fig.data] == [[1.0, 2.0], [5.0, 9.0]]
        assert (fig.layout.coloraxis.cmin, fig.layout.coloraxis.cmax) == (1.0, 9.0)
        assert fig.layout.coloraxis.colorbar.title.text == "J"

    def test_uncolored_marker_uses_the_path_color(self):
        query = Query(title="t", x=("x",), y=("y",))
        fig = self.build(query, {(): [0]}, [{(): [1.0]}])

        assert fig.data[0].marker.color == fig.data[0].line.color
        assert fig.data[0].marker.coloraxis is None

    def test_empty_color_series_is_an_error(self):
        query = Query(title="t", x=("x",), y=("y",), color=("c",))

        with pytest.raises(ValueError, match="Color path resolved to an empty series"):
            self.build(query, {(): [0]}, [{(): [1.0]}], colors={})

    def test_missing_color_group_is_an_error(self):
        query = Query(title="t", x=("x",), y=("y",), color=("c",))

        with pytest.raises(ValueError, match="No color series exists for group"):
            self.build(query, {(): [0]}, [{M1: [1.0]}], colors={M0: [0.0]})
