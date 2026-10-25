"""Metric reducing to the maximum of numeric values (``MAX``)."""

from __future__ import annotations

import logging
import math

from core.annotations import override
from core.metrics.metric.base import PrimitiveType, as_number
from core.metrics.metric.series import SeriesMetric


class MaxMetric(SeriesMetric):
    """Metric reducing to the maximum of the pushed numbers.

    Only numbers are accepted (``int``, ``float`` and NumPy numeric scalars;
    ``bool`` and ``np.bool_`` are rejected). NaNs are logged and ignored
    during aggregation. A window holding only NaNs reduces to NaN.

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
    def push(self, value: PrimitiveType) -> None:
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

        self.values.append(as_number(value, "MaxMetric"))

    def peek(self, compile: bool = True) -> int | float | list[int | float] | None:
        """Return the maximum (``None`` when empty).

        When ``compile`` is false, the history is returned instead.

        Parameters
        ----------
        compile : bool, default True
            If true, the maximum; if false, a copy of the history.

        Returns
        -------
        int or float or list or None
            The largest pushed value, ignoring NaNs (NaN when every value is NaN),
            ``None`` while empty, or the history.
        """

        if not compile:
            return list(self.values)

        if not self.values:
            return None

        # TODO (nadine) Add a user-configurable NaN policy (ignore or propagate).
        values = [
            v for v in self.values if not (isinstance(v, float) and math.isnan(v))
        ]

        if len(values) != len(self.values):
            logging.getLogger(__name__).error(
                "MaxMetric: NaN ignored during aggregation."
            )

        return max(values) if values else math.nan

    def reduce(self, compile: bool = True) -> int | float | MaxMetric | None:
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
