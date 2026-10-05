"""Metric reducing to the minimum of numeric values (``MIN``)."""

from __future__ import annotations

from core.annotations import override
from core.metrics.metric.base import PrimitiveType
from core.metrics.metric.series import SeriesMetric


class MinMetric(SeriesMetric):
    """Metric reducing to the minimum of the pushed numbers.

    Only ``int`` and ``float`` are accepted (``bool`` is rejected); ``peek``
    returns ``None`` while empty.

    When to use: for floor values over an iteration, such as the lowest stock observed.

    Examples
    --------
    >>> metric = MinMetric()
    >>> metric.push(2.0)
    >>> metric.push(5)
    >>> metric.push(3.5)
    >>> metric.peek()
    2.0
    >>> metric.reduce()
    2.0
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
                f"MinMetric only accepts int or float, got {type(value).__name__}."
            )

        self.values.append(value)

    def peek(self, compile: bool = True) -> PrimitiveType | list[PrimitiveType] | None:
        """Return the minimum (``None`` when empty).

        When ``compile`` is false, the history is returned instead.

        Parameters
        ----------
        compile : bool, default True
            If true, the minimum; if false, a copy of the history.

        Returns
        -------
        int or float or list or None
            The smallest pushed value, ``None`` while empty, or the history.
        """

        if not compile:
            return list(self.values)

        if not self.values:
            return None

        return min(self.values)

    def reduce(self, compile: bool = True) -> PrimitiveType | MinMetric | None:
        """Return the minimum and clear the history.

        With ``compile`` false a new ``MinMetric`` holding only that value is
        returned instead. An empty metric returns ``None``, or a new empty
        ``MinMetric`` when ``compile`` is false.

        Parameters
        ----------
        compile : bool, default True
            If true, return the minimum; if false, a new ``MinMetric``.

        Returns
        -------
        int or float or MinMetric or None
            The minimum (or a metric holding it); ``None`` when empty and
            ``compile`` is true.
        """

        if not self.values:
            return None if compile else MinMetric()

        min = self.peek(compile=True)

        self.flush()

        if compile:
            return min

        metric = MinMetric()
        metric.values = [min]

        return metric

    def __repr__(self) -> str:
        return f"MinMetric({self.peek()}; len={len(self)})"
