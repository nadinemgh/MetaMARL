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
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from core.metrics.enums import ReduceProtocol
from core.metrics.metric.base import PrimitiveType
from core.reporting.base import Group, Reporter, Resolved
from core.reporting.config import ReporterConfig
from core.reporting.query import Path as QueryPath
from core.reporting.query import Query
from core.utils import sanitize_key

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


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

    Each report call writes every point of every series again, so the event
    file accumulates repeated steps. The ``SummaryWriter`` is created on the
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
    def _path_name(path: QueryPath) -> str:
        return "/".join(
            str(token.value) if isinstance(token, Enum) else token
            for token in path
            if not isinstance(token, ReduceProtocol)
        )

    @classmethod
    def _series_label(
        cls, path: QueryPath, group: Group, label: Optional[str] = None
    ) -> str:
        name = label if label is not None else cls._path_name(path)

        if not group:
            return name

        group_name = ", ".join(
            f"{junction}={dynamic_id}" for junction, dynamic_id in group
        )

        return f"{name} [{group_name}]"

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

                for x_value, y_value in zip(x_values, y_values, strict=True):
                    writer.add_scalar(
                        tag=tag,
                        scalar_value=y_value,
                        global_step=self._step(x_value, query),
                    )

                if group in resolved_errors:
                    for x_value, error_value in zip(
                        x_values, resolved_errors[group], strict=True
                    ):
                        writer.add_scalar(
                            tag=f"{tag}/std",
                            scalar_value=error_value,
                            global_step=self._step(x_value, query),
                        )

        writer.flush()

    def close(self) -> None:
        """Close the ``SummaryWriter`` if one was created.

        A later report creates a new writer on the same directory.
        """

        if self._writer is not None:
            self._writer.close()

            self._writer = None
