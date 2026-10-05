"""Metric reducing to the sum of numeric values (``SUM``, empty -> 0)."""

from __future__ import annotations

from core.annotations import override
from core.metrics.metric.base import PrimitiveType, as_number
from core.metrics.metric.series import SeriesMetric


class SumMetric(SeriesMetric):
    """Metric reducing to the sum of the pushed numbers.

    Only numbers are accepted (``int``, ``float`` and NumPy numeric scalars;
    ``bool`` and ``np.bool_`` are rejected). Unlike the other scalar metrics an
    empty sum compiles to ``0``, not ``None``.

    When to use: for quantities accumulated over an iteration (total catch,
    number of episodes, environment steps).

    Examples
    --------
    >>> metric = SumMetric()
    >>> metric.peek()
    0
    >>> metric.push(2)
    >>> metric.push(1.5)
    >>> metric.reduce()
    3.5
    >>> metric.peek()
    0
    """

    @override(SeriesMetric)
    def push(self, value: PrimitiveType) -> PrimitiveType:
        """Append a number.

        Booleans and non-numeric values are rejected with ``TypeError``.

        Parameters
        ----------
        value : int or float
            The number to record; a NumPy numeric scalar is stored as the
            equivalent Python number.

        Raises
        ------
        TypeError
            If ``value`` is a ``bool`` (or ``np.bool_``) or not a number.
        """

        self.values.append(as_number(value, "SumMetric"))

    def peek(self, compile: bool = True) -> PrimitiveType | list[PrimitiveType]:
        """Return the sum (``0`` when empty).

        When ``compile`` is false, the history is returned instead.

        Parameters
        ----------
        compile : bool, default True
            If true, the sum; if false, a copy of the history.

        Returns
        -------
        int or float or list
            The sum of the pushed values (``0`` while empty), or the history.
        """

        if not compile:
            return list(self.values)

        return sum(self.values)

    def reduce(self, compile: bool = True) -> PrimitiveType | SumMetric:
        """Return the sum and clear the history.

        With ``compile`` false a new ``SumMetric`` holding only the sum is
        returned instead.

        Parameters
        ----------
        compile : bool, default True
            If true, return the sum; if false, a new ``SumMetric``.

        Returns
        -------
        int or float or SumMetric
            The sum (``0`` when empty), or a metric holding it.
        """

        sum = self.peek(compile=True)

        self.flush()

        if compile:
            return sum

        metric = SumMetric()
        metric.values = [sum]

        return metric

    def __repr__(self) -> str:
        return f"SumMetric({self.peek()}; len={len(self)})"
