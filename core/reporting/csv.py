"""CSV reporter: one long-form file per query, rewritten on every report.

Each row is:

    (query, x, series, value, error, color)

``series`` is the resolved y-series label. ``error`` contains the pointwise
standard deviation when requested by the query, and ``color`` contains the
resolved per-point color value when a color path is configured. Missing
error/color values are written as empty cells.

The module is the dependency-free local backend of the reporting layer:
:class:`CSVConfig` is the serialisable factory carried by the optimizer and
environment configs, and :class:`CSVReporter` writes the files that a notebook
or a spreadsheet can load. A query whose y paths resolve to no series writes
no file.
"""

from __future__ import annotations

import csv
from enum import Enum
from pathlib import Path
from typing import Optional

from core.metrics.enums import ReduceProtocol
from core.reporting.base import Group, Reporter, Resolved
from core.reporting.config import ReporterConfig
from core.reporting.query import Path as QueryPath
from core.reporting.query import Query
from core.utils import sanitize_key


class CSVConfig(ReporterConfig):
    """Configuration of a :class:`CSVReporter`.

    The reporters it builds write under
    ``<output_dir>/<project>/<world>[-<label>]``. ``world`` is filled in by the
    optimizer that owns the config, and ``label`` is the owner (an optimizer
    class or an environment id), so that two owners never write to the same
    directory.

    Parameters
    ----------
    project : str
        Name of the project; the first directory level under ``output_dir``.
    output_dir : str, default "results"
        Root directory of the CSV files, relative to the working directory
        unless absolute.

    Attributes
    ----------
    output_dir : pathlib.Path
        Root directory as a path.

    When to use: in the YAML or Python configuration of an experiment when you
    want local, plain-text results without Weights & Biases or TensorBoard.

    Examples
    --------
    >>> import tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     config = CSVConfig(project="fishery", output_dir=tmp)
    ...     config.world = "lake"
    ...     reporter = config.build(label="env0")
    ...     reporter.output_dir.relative_to(tmp).as_posix()
    'fishery/lake-env0'
    """

    def __init__(self, *, project: str, output_dir: str = "results") -> None:
        super().__init__(project=project)

        self.output_dir = Path(output_dir)

    def build(self, *, label: Optional[str] = None) -> CSVReporter:
        """Create a :class:`CSVReporter` for one owner, creating its directory.

        Parameters
        ----------
        label : str or None, default None
            Suffix identifying the owner; the directory is named
            ``<world>-<label>``, or ``<world>`` without a label. ``world``
            must have been set beforehand.

        Returns
        -------
        CSVReporter
            A reporter writing under ``<output_dir>/<project>/<name>``.
        """

        name = f"{self.world}-{label}" if label is not None else self.world

        return CSVReporter(output_dir=self.output_dir / self.project_name / name)


class CSVReporter(Reporter):
    """Reporter writing one long-form CSV file per query.

    On every :meth:`~core.reporting.base.Reporter.report` call, each query
    is written to ``<output_dir>/<sanitised title>.csv``, replacing the
    previous file: the file always holds the full history of the latest
    schema it was given, not an increment. The columns are those of
    :attr:`HEADER`. A query whose y paths resolve to no series (an empty
    dynamic node) leaves the previous file, if any, untouched. The output
    directory is created when the reporter is built.

    Parameters
    ----------
    output_dir : pathlib.Path
        Directory receiving the files; created with its parents if missing.

    Attributes
    ----------
    HEADER : tuple[str, ...]
        Column names of every file:
        ``("query", "x", "series", "value", "error", "color")``.

    When to use: for results you want to load with pandas or open in a
    spreadsheet, or as a backend that needs no network or extra package.

    Examples
    --------
    >>> import tempfile
    >>> from core.metrics.schemas import MetricSchema
    >>> class Run(MetricSchema):
    ...     iter: list[int] = []
    ...     loss: list[float] = []
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     reporter = CSVReporter(output_dir=Path(tmp) / "csv")
    ...     query = Query(title="Loss", x=("iter",), y=("loss",))
    ...     reporter.add_query(query)
    ...     reporter.report(Run(iter=[0, 1], loss=[3.0, 2.0]))
    ...     print(reporter.path_for(query).name)
    ...     print(reporter.path_for(query).read_text(), end="")
    Loss.csv
    query,x,series,value,error,color
    Loss,0,loss,3.0,,
    Loss,1,loss,2.0,,
    """

    HEADER = ("query", "x", "series", "value", "error", "color")

    def __init__(self, *, output_dir: Path) -> None:
        self._output_dir = Path(output_dir)

        self._output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def output_dir(self) -> Path:
        """Return the directory the CSV files are written to."""

        return self._output_dir

    def path_for(self, query: Query) -> Path:
        """Return the file a query is written to.

        The name is the query title with every run of characters outside
        ``[0-9a-zA-Z_-]`` replaced by one underscore, plus ``.csv``. Two
        queries whose titles sanitise to the same name share a file, and the
        later one overwrites the earlier.

        Parameters
        ----------
        query : Query
            The query whose file is wanted.

        Returns
        -------
        pathlib.Path
            ``output_dir / "<sanitised title>.csv"``.
        """

        return self._output_dir / f"{sanitize_key(query.title)}.csv"

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

        labels = (
            query.legend_labels
            if query.legend_labels is not None
            else (None,) * len(query.y_paths)
        )

        with self.path_for(query).open("w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)

            writer.writerow(self.HEADER)

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

                    error_values = resolved_errors.get(group)

                    if error_values is None:
                        error_values = [""] * len(y_values)

                    color_values = None

                    if colors is not None:
                        if () in colors:
                            color_values = colors[()]
                        else:
                            try:
                                color_values = colors[group]
                            except KeyError:
                                raise ValueError(
                                    f"No color series exists for group {group}."
                                ) from None

                    if color_values is None:
                        color_values = [""] * len(y_values)

                    label = self._series_label(path, group, label=path_label)

                    for x_value, y_value, error_value, color_value in zip(
                        x_values, y_values, error_values, color_values, strict=True
                    ):
                        writer.writerow(
                            (
                                query.title,
                                x_value,
                                label,
                                y_value,
                                error_value,
                                color_value,
                            )
                        )

    def close(self) -> None:
        """Do nothing: the files are closed after every report."""
