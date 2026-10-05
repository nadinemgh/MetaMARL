"""Metric keeping the full history of pushed values (``SERIES``)."""

from __future__ import annotations

from core.annotations import override
from core.metrics.metric.base import Metric, PrimitiveType


class SeriesMetric(Metric):
    """Metric keeping every pushed value in order.

    ``peek`` returns a copy of the whole history as a list whatever ``compile``
    is. ``reduce`` returns the history and clears it; with ``compile`` false
    it wraps the history in a new ``SeriesMetric``. The class is also the
    base of the scalar metrics, which reuse its ``values`` list and only
    override the reduction.

    A gap (``None``, appended by :meth:`push_gap`) marks a push of the logger in
    which this leaf had no value, so that entry ``i`` of every leaf below a
    dynamic node belongs to the same push. The history keeps the gaps; the
    scalar metrics skip them when they compile.

    Attributes
    ----------
    values : list[int or float or bool or str or None]
        The pushed values, in order, with ``None`` for a gap. Any primitive
        type is accepted.

    When to use: for values you want to plot against an x axis (a curve per
    iteration, the fitness of every candidate); reporters expect series
    leaves. Use :class:`~core.metrics.metric.last.LastMetric` for a counter
    and :class:`~core.metrics.metric.mean.MeanMetric` for a value averaged
    over an iteration.

    Examples
    --------
    >>> metric = SeriesMetric()
    >>> metric.push(0.5)
    >>> metric.push(0.7)
    >>> metric.peek()
    [0.5, 0.7]
    >>> metric.reduce()
    [0.5, 0.7]
    >>> len(metric)
    0
    >>> metric.push(0.5)
    >>> metric.push_gap()
    >>> metric.peek(), metric.present_values()
    ([0.5, None], [0.5])
    """

    def __init__(self) -> None:
        self.values: list[PrimitiveType | None] = []

    def __len__(self) -> int:
        return len(self.values)

    def __repr__(self) -> str:
        return f"SeriesMetric(len={len(self)})"

    def push(self, value: PrimitiveType) -> None:
        """Append ``value`` to the history without type checking.

        Parameters
        ----------
        value : int or float or bool or str
            The value to record.
        """

        self.values.append(value)

    @override(Metric)
    def push_gap(self) -> None:
        """Append a gap (``None``): this push had no value for the leaf."""

        self.values.append(None)

    def present_values(self) -> list[PrimitiveType]:
        """Return the pushed values without the gaps, in order.

        Returns
        -------
        list[int or float or bool or str]
            A new list of the values that are not gaps.
        """

        return [value for value in self.values if value is not None]

    def peek(self, compile: bool = True) -> list[PrimitiveType | None]:
        """Return a copy of the history; ``compile`` is ignored.

        Parameters
        ----------
        compile : bool, default True
            Ignored: the history is returned either way.

        Returns
        -------
        list[int or float or bool or str or None]
            A new list of the pushed values, gaps included; changing it does
            not change the metric.
        """

        return list(self.values)

    def reduce(self, compile: bool = True) -> list[PrimitiveType | None] | SeriesMetric:
        """Return the history and clear it.

        With ``compile`` false the history is returned inside a new
        ``SeriesMetric`` instead of a plain list.

        Parameters
        ----------
        compile : bool, default True
            If true, return the history as a list; if false, as a new
            ``SeriesMetric``.

        Returns
        -------
        list or SeriesMetric
            The history that was held before the call.
        """

        values = list(self.values)

        self.flush()

        if compile:
            return values

        metric = SeriesMetric()
        metric.values = values

        return metric

    def flush(self) -> None:
        """Clear the history in place."""

        self.values.clear()
