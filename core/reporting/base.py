"""Reporter base class: query resolution shared by every reporting backend.

A reporter receives a populated ``MetricSchema`` at the end of an iteration,
resolves each of its :class:`~core.reporting.query.Query` objects against it into
plain lists of numbers, and hands them to a backend hook (``_report``) that
draws or writes them. This module holds the part that does not depend on the
backend: the path walk (:meth:`Reporter._resolve_path`), the consistency checks
between x, y, error and colour series (:meth:`Reporter._resolve_query`) and the
per-query error isolation of :meth:`Reporter.report`. The CSV, TensorBoard and
Weights & Biases reporters subclass :class:`Reporter`; the optimizers and the
environments own one reporter each and call ``report`` on it.

A *group* identifies one series inside a query that expands dynamic nodes: it
is a tuple of ``(junction, dynamic_id)`` pairs, for example
``(("by_mechanism", "quota"), ("by_seed", "3"))``, and the empty tuple ``()``
when the path crosses no dynamic node.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Optional, TypeAlias

import numpy as np

from core.metrics.enums import ReduceProtocol
from core.metrics.metric.base import PrimitiveType
from core.metrics.schemas import MetricSchema
from core.reporting.query import Path, Query

# One series of a query: ``(junction, dynamic_id)`` pairs, ``()`` if no dynamic node.
Group: TypeAlias = tuple[tuple[str, str], ...]
# Resolved series of one path: one list of values per group.
Resolved: TypeAlias = dict[Group, list[PrimitiveType]]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PathResolution:
    """Result of walking one query path through a metric schema.

    Both fields map a group (see the module docstring) to a list of values with
    one entry per x point. ``errors`` is empty unless the path crosses the
    ``MEAN`` reduction named by the query's ``error_path`` and the query asks
    for a standard deviation.

    Attributes
    ----------
    values : dict[tuple[tuple[str, str], ...], list[int | float | bool | str]]
        The resolved series of the path, one list per group.
    errors : dict[tuple[tuple[str, str], ...], list[int | float | bool | str]]
        The pointwise standard deviation across the averaged branches, one list
        per group, with the same length as the matching ``values`` list. Empty
        when no error was requested or captured.

    When to use: this is the return type of :meth:`Reporter._resolve_path`; you
    only build one by hand when you test a reporter backend with canned
    resolutions.

    Examples
    --------
    >>> resolution = PathResolution(values={(): [1.0, 2.0]}, errors={})
    >>> resolution.values[()]
    [1.0, 2.0]
    """

    values: Resolved
    errors: Resolved


class Reporter(ABC):
    """Base class of the reporters: query resolution plus a backend hook.

    A reporter receives populated ``MetricSchema`` objects, resolves each of its
    registered queries against them and passes the resolved series to the
    backend-specific ``_report`` hook, which a subclass implements to draw or
    write them. A subclass must also implement ``close``. Queries are added
    after construction with :meth:`add_query`; resolution walks the object
    passed to :meth:`report`.

    Reporters are built by a :class:`~core.reporting.config.ReporterConfig`
    rather than directly, one per owner (an optimizer or an environment).

    Attributes
    ----------
    queries : tuple[Query, ...]
        The registered queries, in registration order (read-only property).

    When to use: subclass it to add a reporting backend. To report with an
    existing backend, build a reporter from its config and call
    :meth:`report` once per iteration.

    Examples
    --------
    A backend that prints what it receives; the metric tree is a populated
    schema whose leaves are lists:

    >>> from core.metrics.schemas import MetricSchema
    >>> class Run(MetricSchema):
    ...     iter: list[int] = []
    ...     loss: list[float] = []
    >>> class PrintReporter(Reporter):
    ...     def _report(self, query, x, ys, errors, colors):
    ...         print(query.title, x, ys)
    ...     def close(self):
    ...         pass
    >>> reporter = PrintReporter()
    >>> reporter.add_query(Query(title="Loss", x=("iter",), y=("loss",)))
    >>> reporter.report(Run(iter=[0, 1, 2], loss=[3.0, 2.0, 1.0]))
    Loss {(): [0, 1, 2]} [{(): [3.0, 2.0, 1.0]}]
    """

    _queries: tuple[Query, ...] = ()

    @property
    def queries(self) -> tuple[Query, ...]:
        """Return the queries registered with this reporter, in order."""

        return self._queries

    def add_query(self, *queries: Query) -> None:
        """Register one or more reporting queries.

        Queries are appended to the existing ones and rendered in registration
        order on every :meth:`report` call. The same query may be registered
        twice; it is then rendered twice.

        Parameters
        ----------
        *queries : Query
            Queries to register with this reporter.
        """

        self._queries += queries

    @staticmethod
    def _path_name(path: Path) -> str:
        """Return a path as ``/``-joined names, without reduction tokens.

        ``ReduceProtocol`` tokens are dropped and an ``Enum`` token is replaced
        by its value, so ``("by_mech", SERIES, "fitness")`` is named
        ``"by_mech/fitness"``. The backends use it as the default label of a
        series and of an axis.
        """

        return "/".join(
            str(token.value) if isinstance(token, Enum) else token
            for token in path
            if not isinstance(token, ReduceProtocol)
        )

    @classmethod
    def _series_label(
        cls, path: Path, group: Group, label: Optional[str] = None
    ) -> str:
        """Return the label of one series of a query.

        The label is ``label`` when given, else the name of ``path``; for a
        dynamic group it is followed by ``[junction=id, ...]``.
        """

        name = label if label is not None else cls._path_name(path)

        if not group:
            return name

        group_name = ", ".join(
            f"{junction}={dynamic_id}" for junction, dynamic_id in group
        )

        return f"{name} [{group_name}]"

    def _resolve_path(
        self,
        path: Path,
        metrics: MetricSchema | dict | list[Any],
        *,
        index: int = 0,
        group: Group = (),
        junction: str | None = None,
        error: Literal["none", "std"] = "none",
        error_path: Path | None = None,
    ) -> PathResolution:
        """Walk ``path`` through ``metrics`` and return the series it selects.

        Each token of ``path`` is a field name (an attribute of a schema, a key
        of a dynamic node) or a :class:`~core.metrics.enums.ReduceProtocol`.
        ``SERIES`` on a dynamic node expands one group per runtime id, in
        sorted order of the ids; ``MEAN`` averages the branches pointwise. The
        walk ends when ``path`` is exhausted, where ``metrics`` must be a flat
        list of values. ``index``, ``group`` and ``junction`` are the
        recursion state and keep their defaults on the first call.

        Parameters
        ----------
        path : Path
            Tokens to follow, starting at the root of ``metrics``.
        metrics : MetricSchema or dict or list
            The node the walk is currently at.
        index : int, default 0
            Position of the next token in ``path``.
        group : Group, default ()
            Dynamic ids crossed so far.
        junction : str or None, default None
            Name of the field holding the dynamic node about to be expanded.
        error : {"none", "std"}, default "none"
            Whether to capture the standard deviation of a ``MEAN`` reduction.
        error_path : Path or None, default None
            The path prefix that ends just before the ``MEAN`` token whose
            standard deviation is captured.

        Returns
        -------
        PathResolution
            The series of every group, and their errors when captured.

        Raises
        ------
        KeyError
            If a name is unknown or the path stops before a series.
        ValueError
            If the path ends on a nested series, or ``MEAN`` meets branches
            with different groups or lengths, or an error is requested below
            a second ``MEAN``.
        TypeError
            If a reduction token is applied to a schema rather than a dynamic
            node.
        NotImplementedError
            If a dynamic node meets a reduction other than ``SERIES`` or
            ``MEAN``.
        """

        if index >= len(path):
            if not isinstance(metrics, list):
                raise KeyError(f"Path does not point to a metric series: {path}")

            if any(isinstance(value, list) for value in metrics):
                raise ValueError(
                    "Path resolves to a nested metric series. "
                    + f"A series reduction is required: {path}"
                )

            return PathResolution(values={group: metrics}, errors={})

        token = path[index]

        if isinstance(metrics, dict):
            if token == ReduceProtocol.SERIES:
                values: Resolved = {}
                errors: Resolved = {}

                for dynamic_id, child in sorted(
                    metrics.items(), key=lambda item: str(item[0])
                ):
                    child_group = group + ((junction or "dict", str(dynamic_id)),)
                    child_result = self._resolve_path(
                        path=path,
                        metrics=child,
                        index=index + 1,
                        group=child_group,
                        error=error,
                        error_path=error_path,
                    )

                    values.update(child_result.values)
                    errors.update(child_result.errors)

                return PathResolution(values=values, errors=errors)

            if token == ReduceProtocol.MEAN:
                branches = [
                    self._resolve_path(
                        path=path,
                        metrics=child,
                        index=index + 1,
                        group=group,
                        error=error,
                        error_path=error_path,
                    )
                    for child in metrics.values()
                ]

                if not branches:
                    return PathResolution(values={}, errors={})

                groups = set(branches[0].values)

                if any(set(branch.values) != groups for branch in branches[1:]):
                    raise ValueError(
                        "Cannot compute mean across branches with different "
                        + "SERIES groups."
                    )

                reduction_path = path[:index]
                capture_error = error != "none" and error_path == reduction_path

                if not capture_error and any(branch.errors for branch in branches):
                    raise ValueError(
                        "error_path is nested below another MEAN reduction. "
                        + "Error propagation through additional reductions is not "
                        + "defined."
                    )

                reduced: Resolved = {}
                errors: Resolved = {}

                for branch_group in groups:
                    series = [branch.values[branch_group] for branch in branches]
                    lengths = {len(values) for values in series}

                    if len(lengths) != 1:
                        raise ValueError(
                            "Cannot compute pointwise mean over series with "
                            + "different lengths: "
                            + f"{sorted(lengths)}."
                        )

                    values = np.asarray(series, dtype=np.float64)
                    reduced[branch_group] = np.mean(values, axis=0).tolist()

                    if capture_error:
                        if error == "std":
                            errors[branch_group] = np.std(values, axis=0).tolist()

                return PathResolution(values=reduced, errors=errors)

            if isinstance(token, ReduceProtocol):
                raise NotImplementedError(
                    f"Dictionary query reduction {token} is not supported."
                )

            try:
                child = metrics[token]
            except KeyError:
                raise KeyError(f"Unknown metric path: {path}") from None

            return self._resolve_path(
                path=path,
                metrics=child,
                index=index + 1,
                group=group,
                error=error,
                error_path=error_path,
            )

        """MetricSchema"""

        if isinstance(token, ReduceProtocol):
            raise TypeError(
                f"{token} cannot be applied to {type(metrics).__name__} in path {path}."
            )

        try:
            child = getattr(metrics, token)
        except AttributeError:
            raise KeyError(f"Unknown metric path: {path}") from None

        return self._resolve_path(
            path=path,
            metrics=child,
            index=index + 1,
            group=group,
            junction=token if isinstance(child, dict) else None,
            error=error,
            error_path=error_path,
        )

    def _resolve_query(
        self, metrics: MetricSchema, query: Query
    ) -> tuple[Resolved, list[Resolved], list[Resolved], Resolved | None]:
        """Resolve a query against a populated metric schema.

        Returns ``(x, ys, errors, colors)`` where ``ys`` and ``errors`` hold
        one resolution per y path, and checks that every series of a group has
        the length of its x series. Raises ``ValueError`` when they do not.
        """

        x_result = self._resolve_path(path=query.x, metrics=metrics)
        xs = x_result.values
        y_results = [
            self._resolve_path(
                path=path,
                metrics=metrics,
                error=query.error,
                error_path=query.error_path,
            )
            for path in query.y_paths
        ]
        yss = [result.values for result in y_results]
        error_yss = [result.errors for result in y_results]

        colors: Resolved | None = None

        if query.color is not None:
            colors = self._resolve_path(path=query.color, metrics=metrics).values

        for path, ys, errors in zip(query.y_paths, yss, error_yss):
            if set(xs) == {()}:
                x = xs[()]

                for group, y in ys.items():
                    if len(x) != len(y):
                        raise ValueError(
                            "Query series must have equal length: "
                            + f"x={query.x} ({len(x)}), "
                            + f"y={path}, group={group} ({len(y)})."
                        )
            else:
                if set(xs) != set(ys):
                    raise ValueError(
                        "Dynamic x and y groups do not match: "
                        + f"x={set(xs)}, y={set(ys)}."
                    )

                for group in xs:
                    if len(xs[group]) != len(ys[group]):
                        raise ValueError(
                            "Query series must have equal "
                            + f"length for group {group}: x={len(xs[group])}, "
                            + f"y={len(ys[group])}."
                        )

            for group, values in errors.items():
                if group not in ys:
                    raise ValueError(
                        f"Error group {group} has no corresponding y series."
                    )

                if len(values) != len(ys[group]):
                    raise ValueError(
                        "Error series must have the same length as y for group "
                        + f"{group}."
                    )

            if colors is not None:
                for group, y in ys.items():
                    if () in colors:
                        color_values = colors[()]
                    else:
                        try:
                            color_values = colors[group]
                        except KeyError:
                            raise ValueError(
                                f"No color series exists for group {group}."
                            ) from None

                    if len(color_values) != len(y):
                        raise ValueError(
                            "Color series must have the same length as y for group "
                            + f"{group}: "
                            + f"color={len(color_values)}, y={len(y)}."
                        )

        return (xs, yss, error_yss, colors)

    @abstractmethod
    def _report(
        self,
        query: Query,
        x: Resolved,
        ys: list[Resolved],
        errors: list[Resolved],
        colors: Resolved | None,
    ) -> None:
        """Report one resolved query using the concrete reporting backend.

        Parameters
        ----------
        query : Query
            Query defining how the resolved values should be represented.
        x : Resolved
            Resolved values of the query's x path, one list per group.
        ys : list[Resolved]
            Resolved values of each y path, in the order of ``query.y_paths``.
        errors : list[Resolved]
            Standard-deviation series of each y path (empty mappings when none).
        colors : Resolved or None
            Resolved values of the colour path, or ``None`` without one.
        """

        ...

    def report(self, metrics: MetricSchema) -> None:
        """Report all applicable configured views for a metric schema.

        Each configured query is resolved against ``metrics`` and the resolved
        series are forwarded to the backend-specific reporting implementation.

        Each query is rendered on its own: an error raised while resolving or
        rendering one query is logged with its traceback and the remaining
        queries are still rendered. Reporting runs at the end of every
        training iteration, so a single stale or malformed query must not
        abort the run that called it.

        Parameters
        ----------
        metrics : MetricSchema
            Populated metric schema to report, for example the result of
            ``MetricLogger.peek()``. Its leaves are lists of values.
        """

        for query in self._queries:
            try:
                x, ys, errors, colors = self._resolve_query(metrics, query)

                self._report(query, x, ys, errors, colors)
            except Exception:
                logger.exception(
                    "%s could not render query %r; the other queries are "
                    + "still rendered.",
                    type(self).__name__,
                    query.title,
                )

    @abstractmethod
    def close(self) -> None:
        """Release the backend resources held by this reporter."""

        ...
