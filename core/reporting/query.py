"""Declarative selection of metric series to report.

A :class:`Query` names an x path and one or several y paths inside a
``MetricSchema`` tree, plus the labels and decorations a backend may use. Paths
are tuples of field names. At a *dynamic* node (``dict[ID, MetricSchema]``) a
path continues with a :class:`~core.metrics.enums.ReduceProtocol` token:
``SERIES`` keeps one series per runtime key (in sorted key order) and ``MEAN``
averages the keys pointwise. A reporter resolves each query against the
populated schema of an iteration and renders the result, so a query is the
only place where the shape of a plot is decided::

    Query(title="Fish biomass by mechanism",
          x=("iter",),
          y=("train", "rollout", "by_mechanism", ReduceProtocol.MEAN, "fish_norm"),
          error="std", error_path=("train", "rollout", "by_mechanism"))
"""

from dataclasses import dataclass
from typing import Literal, Optional, TypeAlias, cast

from core.metrics.enums import ReduceProtocol

# Field names and reduction tokens leading to one series of a metric schema.
Path: TypeAlias = tuple[str | ReduceProtocol, ...]
# How a backend draws one y series.
PlotMode: TypeAlias = Literal["lines", "markers", "lines+markers"]


@dataclass(frozen=True, slots=True)
class Query:
    """Immutable description of one plot over a ``MetricSchema``.

    A query selects an x series and one or more y series, optionally colours
    the points by a third series and draws a standard-deviation band around a
    mean. Construction validates the combination of fields and raises
    ``ValueError`` for an inconsistent one, so a malformed query fails when it
    is declared rather than at the end of an iteration.

    Attributes
    ----------
    title : str
        Title of the plot; backends also derive file and tag names from it.
    x : Path
        Path of the x series (for example ``("iter",)``). Must not be empty.
    y : Path or tuple[Path, ...]
        One y path, or a tuple of paths drawn together. Must not be empty.
    x_label : str or None, default None
        X-axis title of the Weights & Biases plot; falls back to the x path.
    y_label : str or None, default None
        Y-axis title of the Weights & Biases plot; falls back to ``"value"``.
    legend_labels : tuple[str, ...] or None, default None
        One series name per y path, used by every backend; the y path joined
        by ``/`` is used when absent.
    plot_modes : tuple[PlotMode, ...] or None, default None
        One drawing mode per y path, used by the Weights & Biases backend
        (``"lines+markers"`` when absent); the other backends ignore it.
    show_group_labels : bool, default True
        Whether the Weights & Biases legend names each dynamic group (a series
        per mechanism or seed) or shows one entry per y path.
    color : Path or None, default None
        Path of a series that colours the points of every y series (drawn by
        the Weights & Biases backend, ignored by TensorBoard and written as a
        column by the CSV backend).
    color_label : str or None, default None
        Title of the Weights & Biases colour bar; requires ``color``.
    colorscale : str or None, default None
        Name of a Plotly colour scale; requires ``color``.
    error : {"none", "std"}, default "none"
        ``"std"`` adds the standard deviation across the branches averaged by
        the ``MEAN`` token that follows ``error_path``.
    error_path : Path or None, default None
        Path of the dynamic node whose ``MEAN`` reduction supplies the
        standard deviation; required with ``error="std"`` and rejected
        without it. It must be followed by ``ReduceProtocol.MEAN`` in at
        least one y path.

    Raises
    ------
    ValueError
        If ``x`` or ``y`` is empty; if ``legend_labels`` or ``plot_modes`` do
        not have one entry per y path; if ``color_label`` or ``colorscale`` is
        given without ``color``; if ``error`` is neither ``"none"`` nor
        ``"std"``, if ``error_path`` is missing or unexpected, ends on a
        reduction token or is not followed by ``MEAN`` in a y path.

    When to use: to declare, next to the metric schema it reads, which series an
    optimizer or an environment reports (a fitness curve over outer
    iterations, a scatter of candidates coloured by iteration, a mean with a
    standard-deviation band over seeds).

    Examples
    --------
    A curve with one series per mechanism:

    >>> from core.metrics.enums import ReduceProtocol
    >>> query = Query(
    ...     title="Fitness",
    ...     x=("iter",),
    ...     y=("by_mechanism", ReduceProtocol.SERIES, "fitness"),
    ... )
    >>> query.y_paths
    (('by_mechanism', <ReduceProtocol.SERIES: 'series'>, 'fitness'),)

    The mean over mechanisms with its standard deviation:

    >>> band = Query(
    ...     title="Mean fitness",
    ...     x=("iter",),
    ...     y=("by_mechanism", ReduceProtocol.MEAN, "fitness"),
    ...     error="std",
    ...     error_path=("by_mechanism",),
    ... )
    >>> band.error_path
    ('by_mechanism',)

    A band without a ``MEAN`` to take it over is rejected:

    >>> Query(
    ...     title="Bad",
    ...     x=("iter",),
    ...     y=("by_mechanism", ReduceProtocol.SERIES, "fitness"),
    ...     error="std",
    ...     error_path=("by_mechanism",),
    ... )  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    ValueError: error_path ('by_mechanism',) must be followed by ReduceProtocol.MEAN...
    """

    title: str
    x: Path
    y: Path | tuple[Path, ...]
    x_label: Optional[str] = None
    y_label: Optional[str] = None
    legend_labels: Optional[tuple[str, ...]] = None
    plot_modes: Optional[tuple[PlotMode, ...]] = None
    show_group_labels: bool = True
    color: Optional[Path] = None
    color_label: Optional[str] = None
    colorscale: Optional[str] = None
    error: Literal["none", "std"] = "none"
    error_path: Optional[Path] = None

    def __post_init__(self) -> None:
        if not self.x:
            raise ValueError("Query x path cannot be empty.")

        if not self.y:
            raise ValueError("Query y path cannot be empty.")

        if self.legend_labels is not None and len(self.legend_labels) != len(
            self.y_paths
        ):
            raise ValueError("legend_labels must have the same length as y paths.")

        if self.plot_modes is not None and len(self.plot_modes) != len(self.y_paths):
            raise ValueError("plot_modes must have the same length as y paths.")

        if self.color is None:
            if self.color_label is not None:
                raise ValueError("color_label requires a color path.")

            if self.colorscale is not None:
                raise ValueError("colorscale requires a color path.")

        if self.error not in ("none", "std"):
            raise ValueError(f"Unsupported error statistic: {self.error!r}.")

        if self.error == "none":
            if self.error_path is not None:
                raise ValueError("error_path requires an error statistic.")

            return

        if not self.error_path:
            raise ValueError(f"error={self.error!r} requires error_path.")

        if isinstance(self.error_path[-1], ReduceProtocol):
            raise ValueError(
                "error_path must point to the dynamic dimension, not its reduction "
                + "operator."
            )

        target = self.error_path + (ReduceProtocol.MEAN,)

        if not any(path[: len(target)] == target for path in self.y_paths):
            raise ValueError(
                f"error_path {self.error_path} must be followed by "
                + "ReduceProtocol.MEAN in one of the query's y paths."
            )

    @property
    def y_paths(self) -> tuple[Path, ...]:
        """Return the y paths as a tuple, wrapping a single path.

        ``y`` may be one path or a tuple of paths; this property always returns
        a tuple of paths, which is what the reporters iterate over.
        """

        if self.y and isinstance(self.y[0], tuple):
            return cast(tuple[Path, ...], self.y)

        return (cast(Path, self.y),)
