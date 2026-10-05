"""Weights & Biases reporter: one Plotly figure per query, logged to a run.

Every :class:`~core.reporting.query.Query` becomes one Plotly figure logged
under ``plots/<sanitised title>``: one trace per y series and dynamic group,
drawn with the query's plot modes, plus a translucent band for a requested
standard deviation. A query with a ``color`` path colours the points of every
trace on one shared colour axis (the query's ``colorscale``, or Plotly's
default when none is given).

The module is the Weights & Biases backend of the reporting layer:
:class:`WandbConfig` is the serialisable factory carried by the optimizer and
environment configs, and :class:`WandbReporter` owns one W&B run. The run is
started on the first report, never when the reporter is built.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Optional

import numpy as np
import plotly.graph_objects as go
from plotly.colors import qualitative

import wandb
from core.metrics.enums import ReduceProtocol
from core.reporting.base import Group, Reporter, Resolved
from core.reporting.config import ReporterConfig
from core.reporting.query import Path, Query
from core.utils import sanitize_key


class WandbConfig(ReporterConfig):
    """Configuration of a :class:`WandbReporter`.

    The reporters it builds each start their own W&B run, named
    ``<world>[-<label>]`` and grouped by world, so the runs of one experiment
    appear together. ``world`` and ``outer_iters`` are filled in by the
    optimizer that owns the config and are recorded in the run configuration.
    The keyword arguments after ``project`` become the W&B client settings.

    Parameters
    ----------
    project : str
        Name of the W&B project that receives the runs.
    x_disable_stats : bool or None, default True
        W&B setting ``x_disable_stats``: do not collect system metrics.
    x_disable_meta : bool or None, default True
        W&B setting ``x_disable_meta``: do not collect system metadata.
    quiet : bool or None, default True
        W&B setting ``quiet``: suppress the non-essential output of the client.
    max_end_of_run_summary_metrics : int or None, default 0
        W&B setting: maximum number of summary metrics displayed at the end of a
        run.
    max_end_of_run_history_metrics : int or None, default 0
        W&B setting: maximum number of history sparklines displayed at the end of
        a run.

    Raises
    ------
    TypeError
        If an argument other than the ones above is given (Python rejects the
        unknown keyword and names it).

    Attributes
    ----------
    settings : dict[str, Any]
        The five W&B settings above, passed to ``wandb.Settings`` when a run
        starts.

    When to use: when you want the curves of a run in the W&B web interface,
    grouped per world; it needs a W&B login, or offline mode
    (``WANDB_MODE=offline``).

    Examples
    --------
    Building a reporter does not start a run:

    >>> config = WandbConfig(project="fishery", quiet=False)
    >>> config.world = "lake"
    >>> config.outer_iters = 50
    >>> reporter = config.build(label="ES")
    >>> isinstance(reporter, WandbReporter)
    True
    >>> config.settings["quiet"]
    False

    An unknown argument is an error, not silently dropped:

    >>> WandbConfig(project="fishery", queit=True)
    Traceback (most recent call last):
        ...
    TypeError: WandbConfig.__init__() got an unexpected keyword argument 'queit'
    """

    def __init__(
        self,
        *,
        project: str,
        x_disable_stats: Optional[bool] = True,
        x_disable_meta: Optional[bool] = True,
        quiet: Optional[bool] = True,
        max_end_of_run_summary_metrics: Optional[int] = 0,
        max_end_of_run_history_metrics: Optional[int] = 0,
    ) -> None:
        super().__init__(project=project)

        self.settings = {
            "x_disable_stats": x_disable_stats,
            "x_disable_meta": x_disable_meta,
            "quiet": quiet,
            "max_end_of_run_summary_metrics": max_end_of_run_summary_metrics,
            "max_end_of_run_history_metrics": max_end_of_run_history_metrics,
        }

    def build(self, *, label: Optional[str] = None) -> WandbReporter:
        """Create a :class:`WandbReporter` with a fresh random run id.

        The run is named ``<world>-<label>`` (``<world>`` without a label),
        grouped by world, and its configuration records ``outer_iters`` and
        ``world_name``. No run is started yet.

        Parameters
        ----------
        label : str or None, default None
            Suffix identifying the owner (an optimizer class or an
            environment id).

        Returns
        -------
        WandbReporter
            A reporter that starts its W&B run on the first report.
        """

        name = f"{self.world}-{label}" if label is not None else self.world

        return WandbReporter(
            project=self.project_name,
            run_id=uuid.uuid4().hex,
            group=self.world,
            name=name,
            config={"outer_iters": self.outer_iters, "world_name": self.world},
            settings=self.settings,
        )


class WandbReporter(Reporter):
    """Reporter rendering each query as a Plotly figure logged to one W&B run.

    The run is created lazily on the first report call, so that building a
    reporter never touches the network. Each query is drawn as one figure
    logged under ``plots/<sanitised title>``: every y path and dynamic group
    is a trace, the dash pattern distinguishes groups, a standard deviation
    is a filled band of plus and minus one deviation, and a colour path
    colours the markers on a shared colour axis. A query that resolves to no
    series logs nothing and does not start the run. If ``wandb.init`` returns
    no run, the report raises ``RuntimeError``, which
    :meth:`~core.reporting.base.Reporter.report` logs.

    Parameters
    ----------
    project : str
        W&B project name.
    name : str
        Display name of the run.
    run_id : str
        Identifier of the run (a fresh UUID hex string when built by
        :class:`WandbConfig`).
    group : str
        W&B group of the run (the world name when built by
        :class:`WandbConfig`).
    config : dict[str, Any] or None, default None
        Run configuration recorded by W&B; an empty dictionary when ``None``.
    settings : dict[str, Any] or None, default None
        Keyword arguments of ``wandb.Settings``; defaults when ``None``.

    When to use: through :class:`WandbConfig`, to get interactive Plotly
    figures per query in the W&B interface.

    Examples
    --------
    Constructing the reporter is free of network access:

    >>> reporter = WandbReporter(
    ...     project="fishery", name="lake-ES", run_id="a1b2", group="lake"
    ... )

    Reporting starts a W&B run, so the example is not run here:

    >>> from core.metrics.schemas import MetricSchema
    >>> from core.reporting.query import Query
    >>> class Run(MetricSchema):
    ...     iter: list[int] = []
    ...     loss: list[float] = []
    >>> reporter.add_query(Query(title="Loss", x=("iter",), y=("loss",)))
    >>> reporter.report(Run(iter=[0, 1], loss=[3.0, 2.0]))  # doctest: +SKIP
    >>> reporter.close()
    """

    def __init__(
        self,
        *,
        project: str,
        name: str,
        run_id: str,
        group: str,
        config: Optional[dict[str, Any]] = None,
        settings: Optional[dict[str, Any]] = None,
    ) -> None:
        self._defined_prefixes: set[str] = set()
        self._run: wandb = None
        self._project = project
        self._name = name
        self._run_id = run_id
        self._group = group
        self._config = config
        self._settings = settings

    def _init_run(self):
        if self._run is None:
            self._run = wandb.init(
                project=self._project,
                id=self._run_id,
                group=self._group,
                name=self._name,
                config=self._config or {},
                reinit="create_new",
                settings=wandb.Settings(**(self._settings or {})),
            )

    @staticmethod
    def _path_name(path: Path) -> str:
        return "/".join(
            str(token.value) if isinstance(token, Enum) else token
            for token in path
            if not isinstance(token, ReduceProtocol)
        )

    @classmethod
    def _series_label(
        cls, path: Path, group: Group, label: Optional[str] = None
    ) -> str:
        name = label if label is not None else cls._path_name(path)

        if not group:
            return name

        group_name = ", ".join(
            f"{junction}={dynamic_id}" for junction, dynamic_id in group
        )

        return f"{name} [{group_name}]"

    @classmethod
    def _series_figure(
        cls,
        query: Query,
        xs: Resolved,
        yss: list[Resolved],
        error_yss: list[Resolved],
        color_values: Resolved | None,
    ) -> go.Figure:
        fig = go.Figure()
        palette = qualitative.Plotly
        dashes = ("solid", "dash", "dot", "dashdot")
        groups = list(dict.fromkeys(group for ys in yss for group in ys))
        group_dashes = {
            group: dashes[i % len(dashes)] for i, group in enumerate(groups)
        }
        labels = (
            query.legend_labels
            if query.legend_labels is not None
            else (None,) * len(query.y_paths)
        )
        modes = (
            query.plot_modes
            if query.plot_modes is not None
            else ("lines+markers",) * len(query.y_paths)
        )

        if color_values is not None:
            flattened_colors = [
                float(value) for values in color_values.values() for value in values
            ]

            if not flattened_colors:
                raise ValueError("Color path resolved to an empty series.")

            coloraxis: dict[str, Any] = {
                "cmin": min(flattened_colors),
                "cmax": max(flattened_colors),
                "colorbar": {
                    "title": {
                        "text": (
                            query.color_label
                            if query.color_label is not None
                            else cls._path_name(query.color)
                        )
                    }
                },
            }

            if query.colorscale is not None:
                coloraxis["colorscale"] = query.colorscale

            fig.update_layout(coloraxis=coloraxis)

        for path_index, (path, ys, errors, path_label, mode) in enumerate(
            zip(query.y_paths, yss, error_yss, labels, modes)
        ):
            path_color = palette[path_index % len(palette)]

            for group_index, (group, values) in enumerate(ys.items()):
                if () in xs:
                    x = xs[()]
                else:
                    try:
                        x = xs[group]
                    except KeyError:
                        raise ValueError(
                            f"No x series exists for group {group}."
                        ) from None

                label = cls._series_label(
                    path, group if query.show_group_labels else (), label=path_label
                )

                if group in errors:
                    y = np.asarray(values, dtype=np.float64)
                    std = np.asarray(errors[group], dtype=np.float64)

                    if len(y) != len(std):
                        raise ValueError(
                            f"Error series length does not match y for {group}."
                        )

                    upper = y + std
                    lower = y - std

                    fig.add_trace(
                        go.Scatter(
                            x=(list(x) + list(x)[::-1]),
                            y=(upper.tolist() + lower[::-1].tolist()),
                            mode="lines",
                            fill="toself",
                            fillcolor=path_color,
                            opacity=0.15,
                            line=dict(width=0, color=path_color),
                            name=f"{label} ±1 std",
                            hoverinfo="skip",
                            showlegend=False,
                            legendgroup=label,
                        )
                    )

                marker: dict[str, Any] = {"color": path_color}

                if color_values is not None:
                    if () in color_values:
                        point_colors = color_values[()]
                    else:
                        try:
                            point_colors = color_values[group]
                        except KeyError:
                            raise ValueError(
                                f"No color series exists for group {group}."
                            ) from None

                    marker = {"color": point_colors, "coloraxis": "coloraxis"}

                fig.add_trace(
                    go.Scatter(
                        x=x,
                        y=values,
                        mode=mode,
                        name=label,
                        showlegend=(query.show_group_labels or group_index == 0),
                        legendgroup=label,
                        line=dict(color=path_color, dash=group_dashes[group]),
                        marker=marker,
                    )
                )

        return fig

    def _report(
        self,
        query: Query,
        x: Resolved,
        ys: list[Resolved],
        errors: list[Resolved],
        colors: Resolved | None,
    ) -> None:
        if not any(ys):
            return

        self._init_run()

        if self._run is None:
            raise RuntimeError("W&B run failed to initialize.")

        fig = self._series_figure(
            query=query, xs=x, yss=ys, error_yss=errors, color_values=colors
        )
        x_name = (
            query.x_label if query.x_label is not None else self._path_name(query.x)
        )
        y_name = query.y_label if query.y_label is not None else "value"

        fig.update_layout(
            title=query.title,
            xaxis_title=x_name,
            yaxis_title=y_name,
            hovermode=(
                "closest"
                if query.plot_modes is not None
                and any(mode == "markers" for mode in query.plot_modes)
                else "x unified"
            ),
            template="plotly_white",
            height=650,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=(1.15 if colors is not None else 1.02),
            ),
            margin=dict(r=(300 if colors is not None else 220)),
        )
        fig.update_xaxes(rangeslider_visible=False)

        plot_name = sanitize_key(query.title)

        self._run.log({f"plots/{plot_name}": fig})

    def close(self) -> None:
        """Finish the W&B run if one was started.

        A later report calls ``wandb.init`` again with the same run id.
        """

        if self._run is not None:
            self._run.finish()

            self._run = None
