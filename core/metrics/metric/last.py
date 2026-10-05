"""Metric reducing to the last pushed value (``LAST``)."""

from __future__ import annotations

from core.metrics.metric.base import PrimitiveType
from core.metrics.metric.series import SeriesMetric


class LastMetric(SeriesMetric):
    """Metric reducing to the most recently pushed value.

    Any primitive type is accepted; ``peek`` returns ``None`` while empty.

    When to use: for counters and state where only the latest value matters
    (``iter``, a generation index, the current best fitness).

    Examples
    --------
    >>> metric = LastMetric()
    >>> metric.push(3)
    >>> metric.push(7)
    >>> metric.peek()
    7
    >>> metric.reduce()
    7
    >>> metric.peek() is None
    True
    """

    def peek(self, compile: bool = True) -> PrimitiveType | list[PrimitiveType] | None:
        """Return the last value (``None`` when empty).

        When ``compile`` is false, the history is returned instead.

        Parameters
        ----------
        compile : bool, default True
            If true, the most recent value; if false, a copy of the history.

        Returns
        -------
        int or float or bool or str or list or None
            The last pushed value, ``None`` while empty, or the history.
        """

        if not compile:
            return list(self.values)

        if not self.values:
            return None

        return self.values[-1]

    def reduce(self, compile: bool = True) -> PrimitiveType | LastMetric | None:
        """Return the last value and clear the history.

        With ``compile`` false a new ``LastMetric`` holding only that value is
        returned instead. An empty metric returns ``None``, or a new empty
        ``LastMetric`` when ``compile`` is false.

        Parameters
        ----------
        compile : bool, default True
            If true, return the last value; if false, a new ``LastMetric``.

        Returns
        -------
        int or float or bool or str or LastMetric or None
            The last value (or a metric holding it); ``None`` when empty and
            ``compile`` is true.
        """

        if not self.values:
            return None if compile else LastMetric()

        last = self.peek(compile=True)

        self.flush()

        if compile:
            return last

        metric = LastMetric()
        metric.values = [last]

        return metric

    def __repr__(self) -> str:
        return f"LastMetric({self.peek()}; len={len(self)})"
