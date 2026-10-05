"""TensorBoard reporter.

Each resolved y series is logged as one scalar tag indexed by the integer x
axis. Standard-deviation series are logged under a ``/std`` suffix.

TensorBoard scalar plots do not support per-point colour, so query ``color``
values are ignored.

The module is the TensorBoard backend of the reporting layer:
:class:`TensorBoardConfig` is the serialisable factory carried by the optimizer
and environment configs, and :class:`TensorBoardReporter` writes event files
that ``tensorboard --logdir`` can display. The ``tensorboard`` package is an
optional extra, imported only when the first scalar is written.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from core.metrics.metric.base import PrimitiveType
from core.reporting.base import Reporter, Resolved
from core.reporting.config import ReporterConfig
from core.reporting.query import Query
from core.utils import sanitize_key

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


def _same_point(
    first: tuple[int, PrimitiveType], second: tuple[int, PrimitiveType]
) -> bool:
    """Return whether two ``(step, value)`` points are equal, NaN included."""

    (step_a, value_a), (step_b, value_b) = first, second
    # ``x != x`` is true only for NaN, which never equals itself.
    both_nan = value_a != value_a and value_b != value_b

    return step_a == step_b and (value_a == value_b or both_nan)


class TensorBoardConfig(ReporterConfig):
    """Configuration of a :class:`TensorBoardReporter`.

    The reporters it builds write event files under
    ``<log_dir>/<project>/<world>[-<label>]``. ``world`` is filled in by the
    optimizer that owns the config, and ``label`` is the owner (an optimizer
    class or an environment id), so that two owners never share a run
    directory.

    Parameters
    ----------
    project : str
        Name of the project; the first directory level under ``log_dir``.
    log_dir : str, default "runs"
        Root directory of the event files, relative to the working directory
        unless absolute.

    Attributes
    ----------
    log_dir : pathlib.Path
        Root directory as a path.

    When to use: when you want to follow the curves of a run live with
    TensorBoard, without a Weights & Biases account.

    Examples
    --------
    Building a reporter touches neither the disk nor the ``tensorboard``
    package; the event directory is only created by the first report:

    >>> config = TensorBoardConfig(project="fishery", log_dir="runs")
    >>> config.world = "lake"
    >>> reporter = config.build(label="env0")
    >>> reporter.log_dir.as_posix()
    'runs/fishery/lake-env0'
    """

    def __init__(self, *, project: str, log_dir: str = "runs") -> None:
        super().__init__(project=project)

        self.log_dir = Path(log_dir)

    def build(self, *, label: Optional[str] = None) -> TensorBoardReporter:
        """Create a :class:`TensorBoardReporter` for one owner.

        Parameters
        ----------
        label : str or None, default None
            Suffix identifying the owner; the run directory is named
            ``<world>-<label>``, or ``<world>`` without a label. ``world``
            must have been set beforehand.

        Returns
        -------
        TensorBoardReporter
            A reporter writing under ``<log_dir>/<project>/<name>``.
        """

        name = f"{self.world}-{label}" if label is not None else self.world

        return TensorBoardReporter(log_dir=self.log_dir / self.project_name / name)


class TensorBoardReporter(Reporter):
    """Reporter logging each resolved series as TensorBoard scalars.

    Every y series becomes the scalar tag ``<sanitised title>/<sanitised
    series label>``, where the label is the query's legend label or the y path
    joined by ``/``, followed by ``[junction=id, ...]`` for a dynamic group.
    The step of a point is its integer x value, so the x path must resolve to
    integers (a non-integer value raises ``TypeError``, which
    :meth:`~core.reporting.base.Reporter.report` logs). A standard-deviation
    series is written under the same tag plus ``/std``. The colour path of a
    query is ignored with an info message.

    A report receives the whole history of each series, so the reporter keeps
    the points it has written for each tag and writes only the new ones: a
    history that grows between two reports adds its new points, an unchanged one
    adds nothing, and a history that was cleared (it no longer starts with the
    points written before) is written again from its first point. The record
    survives :meth:`close`. The ``SummaryWriter`` is created on the
    first non-empty report and released by :meth:`close`; if the
    ``tensorboard`` package is missing, that first report raises
    ``ImportError`` (which :meth:`~core.reporting.base.Reporter.report` logs).

    Parameters
    ----------
    log_dir : pathlib.Path
        Directory of the event files; created by the ``SummaryWriter`` on the
        first report.

    When to use: for live curves of a long run viewed with TensorBoard. Prefer
    the CSV reporter when you want the values themselves in a table.

    Examples
    --------
    >>> from pathlib import Path
    >>> reporter = TensorBoardReporter(log_dir=Path("runs") / "demo")
    >>> reporter.log_dir.as_posix()
    'runs/demo'

    Reporting needs the ``tensorboard`` extra and writes event files, so the
    example is not run here:

    >>> from core.metrics.schemas import MetricSchema
    >>> class Run(MetricSchema):
    ...     iter: list[int] = []
    ...     loss: list[float] = []
    >>> reporter.add_query(Query(title="Loss", x=("iter",), y=("loss",)))
    >>> reporter.report(Run(iter=[0, 1], loss=[3.0, 2.0]))  # doctest: +SKIP
    >>> reporter.close()
    """

    def __init__(self, *, log_dir: Path) -> None:
        self._log_dir = Path(log_dir)
        self._writer: SummaryWriter | None = None
        # Points already written, per tag, as (step, value) pairs.
        self._written: dict[str, list[tuple[int, PrimitiveType]]] = {}

    @property
    def log_dir(self) -> Path:
        """Return the directory the event files are written to."""

        return self._log_dir

    def _get_writer(self) -> SummaryWriter:
        if self._writer is None:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as e:
                raise ImportError(
                    "TensorBoardReporter needs the 'tensorboard' package: "
                    + "uv sync --extra tensorboard"
                ) from e

            self._writer = SummaryWriter(log_dir=str(self._log_dir))

        return self._writer

    @staticmethod
    def _step(x_value: PrimitiveType, query: Query) -> int:
        try:
            step = int(x_value)
        except (TypeError, ValueError) as e:
            raise TypeError(
                "TensorBoard x-axis must be integer-valued: "
                + f"{query.x} contains {x_value!r}."
            ) from e

        if step != x_value:
            raise TypeError(
                "TensorBoard x-axis must be integer-valued: "
                + f"{query.x} contains {x_value!r}."
            )

        return step

    def _write_new_points(
        self,
        writer: SummaryWriter,
        tag: str,
        x_values: list[PrimitiveType],
        y_values: list[PrimitiveType],
        query: Query,
    ) -> None:
        """Write the points of one series that the event files do not hold yet.

        A report receives the whole history again, so the points written by the
        previous report open the new series. When the new series starts with
        exactly those points, only the rest is written; otherwise the history
        was cleared since (the optimizers reduce it between rounds, the
        environments between episodes) and the series is written from its first
        point. All steps are converted before any point is written, so a
        non-integer x leaves the files untouched.
        """

        points = [
            (self._step(x_value, query), y_value)
            for x_value, y_value in zip(x_values, y_values, strict=True)
        ]
        written = self._written.get(tag, [])

        continues = len(written) <= len(points) and all(
            _same_point(old, new) for old, new in zip(written, points)
        )

        for step, value in points[len(written) if continues else 0 :]:
            writer.add_scalar(tag=tag, scalar_value=value, global_step=step)

        self._written[tag] = points

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

        if colors is not None:
            logger.info(
                "TensorBoardReporter ignores the color path %s "
                + "of query %r because TensorBoard scalars do not "
                + "support per-point colour.",
                query.color,
                query.title,
            )

        writer = self._get_writer()
        title = sanitize_key(query.title)
        labels = (
            query.legend_labels
            if query.legend_labels is not None
            else (None,) * len(query.y_paths)
        )

        for path, resolved_y, resolved_errors, path_label in zip(
            query.y_paths, ys, errors, labels, strict=True
        ):
            for group, y_values in resolved_y.items():
                if () in x:
                    x_values = x[()]
                else:
                    try:
                        x_values = x[group]
                    except KeyError:
                        raise ValueError(
                            f"No x series exists for group {group}."
                        ) from None

                label = self._series_label(path, group, label=path_label)
                tag = f"{title}/{sanitize_key(label)}"

                self._write_new_points(writer, tag, x_values, y_values, query)

                if group in resolved_errors:
                    self._write_new_points(
                        writer, f"{tag}/std", x_values, resolved_errors[group], query
                    )

        writer.flush()

    def close(self) -> None:
        """Close the ``SummaryWriter`` if one was created.

        A later report creates a new writer on the same directory.
        """

        if self._writer is not None:
            self._writer.close()

            self._writer = None
