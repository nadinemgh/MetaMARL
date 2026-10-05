"""Metric reducing to the maximum of numeric values (``MAX``)."""

from __future__ import annotations

from core.annotations import override
from core.metrics.metric.base import PrimitiveType
from core.metrics.metric.series import SeriesMetric


class MaxMetric(SeriesMetric):
    """Metric reducing to the maximum of the pushed numbers.

    Only ``int`` and ``float`` are accepted (``bool`` is rejected); ``peek``
    returns ``None`` while empty.

    When to use: for peak values over an iteration, such as the best reward of a
    batch of episodes.

    Examples
    --------
    >>> metric = MaxMetric()
    >>> metric.push(2.0)
    >>> metric.push(5)
    >>> metric.push(3.5)
    >>> metric.peek()
    5
    >>> metric.reduce()
    5
    >>> metric.peek() is None
    True
    """

    @override(SeriesMetric)
    def push(self, value: PrimitiveType) -> PrimitiveType:
        """Append a number.

        Booleans and non-numeric values are rejected with ``TypeError``.

        Parameters
        ----------
        value : int or float
            The number to record.

        Raises
        ------
        TypeError
            If ``value`` is a ``bool`` or not an ``int`` or ``float``.
        """

        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"MaxMetric only accepts int or float, got {type(value).__name__}."
            )

        self.values.append(value)

    def peek(self, compile: bool = True) -> PrimitiveType | list[PrimitiveType] | None:
        """Return the maximum (``None`` when empty).

        When ``compile`` is false, the history is returned instead.

        Parameters
        ----------
        compile : bool, default True
            If true, the maximum; if false, a copy of the history.

        Returns
        -------
        int or float or list or None
            The largest pushed value, ``None`` while empty, or the history.
        """

        if not compile:
            return list(self.values)

        if not self.values:
            return None

        return max(self.values)

    def reduce(self, compile: bool = True) -> PrimitiveType | MaxMetric | None:
        """Return the maximum and clear the history.

        With ``compile`` false a new ``MaxMetric`` holding only that value is
        returned instead. An empty metric returns ``None``, or a new empty
        ``MaxMetric`` when ``compile`` is false.

        Parameters
        ----------
        compile : bool, default True
            If true, return the maximum; if false, a new ``MaxMetric``.

        Returns
        -------
        int or float or MaxMetric or None
            The maximum (or a metric holding it); ``None`` when empty and
            ``compile`` is true.
        """

        if not self.values:
            return None if compile else MaxMetric()

        max = self.peek(compile=True)

        self.flush()

        if compile:
            return max

        metric = MaxMetric()
        metric.values = [max]

        return metric

    def __repr__(self) -> str:
        return f"MaxMetric({self.peek()}; len={len(self)})"
