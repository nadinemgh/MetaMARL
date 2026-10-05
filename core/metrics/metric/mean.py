"""Metric reducing to the arithmetic mean of numeric values (``MEAN``)."""

from __future__ import annotations

from core.annotations import override
from core.metrics.metric.base import PrimitiveType
from core.metrics.metric.series import SeriesMetric


class MeanMetric(SeriesMetric):
    """Metric reducing to the arithmetic mean of the pushed numbers.

    Only ``int`` and ``float`` are accepted (``bool`` is rejected); ``peek``
    returns ``None`` while empty.

    When to use: the default protocol, for per-step quantities averaged over an
    iteration (rewards, catches, losses).

    Examples
    --------
    >>> metric = MeanMetric()
    >>> metric.push(1.0)
    >>> metric.push(4)
    >>> metric.peek()
    2.5
    >>> metric.reduce()
    2.5
    >>> metric.peek() is None
    True
    """

    @override(SeriesMetric)
    def push(self, value: PrimitiveType) -> None:
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
                f"MeanMetric only accepts int or float, got {type(value).__name__}."
            )

        self.values.append(value)

    def peek(self, compile: bool = True) -> float | list[float]:
        """Return the arithmetic mean (``None`` when empty).

        When ``compile`` is false, the history is returned instead.

        Parameters
        ----------
        compile : bool, default True
            If true, the mean; if false, a copy of the history.

        Returns
        -------
        float or list[float] or None
            The mean of the pushed values, ``None`` while empty, or the
            history.
        """

        if not compile:
            return list(self.values)

        if not self.values:
            return None

        return sum(self.values) / len(self.values)

    def reduce(self, compile: bool = True) -> float | MeanMetric:
        """Return the arithmetic mean and clear the history.

        With ``compile`` false a new ``MeanMetric`` holding only that value is
        returned instead. An empty metric returns ``None``, or a new empty
        ``MeanMetric`` when ``compile`` is false.

        Parameters
        ----------
        compile : bool, default True
            If true, return the mean; if false, a new ``MeanMetric``.

        Returns
        -------
        float or MeanMetric or None
            The mean (or a metric holding it); ``None`` when empty and
            ``compile`` is true.
        """

        if not self.values:
            return None if compile else MeanMetric()

        mean = self.peek(compile=True)

        self.flush()

        if compile:
            return mean

        metric = MeanMetric()
        metric.values = [mean]

        return metric

    def __repr__(self) -> str:
        return f"MeanMetric({self.peek()}; len={len(self)})"
