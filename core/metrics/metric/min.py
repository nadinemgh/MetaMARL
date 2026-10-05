"""Metric reducing to the minimum of numeric values (``MIN``)."""

from __future__ import annotations

import math

from core.annotations import override
from core.metrics.metric.base import PrimitiveType, as_number
from core.metrics.metric.series import SeriesMetric


class MinMetric(SeriesMetric):
    """Metric reducing to the minimum of the pushed numbers.

    Only numbers are accepted (``int``, ``float`` and NumPy numeric scalars;
    ``bool`` and ``np.bool_`` are rejected); ``peek``
    returns ``None`` while empty. A NaN is ignored, because the Ray adaptor
    logs NaN for the learner statistics RLlib did not report and a missing
    value has no rank; the result therefore does not depend on the push order.
    A window holding only NaN reduces to NaN.

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

    A NaN is skipped whatever its position:

    >>> metric.push(float("nan"))
    >>> metric.push(4.0)
    >>> metric.peek()
    4.0
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

        self.values.append(as_number(value, "MinMetric"))

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
            The smallest pushed value, ignoring NaN (NaN when every value is NaN),
            ``None`` while empty, or the history.
        """

        if not compile:
            return list(self.values)

        if not self.values:
            return None

        # NaN is how the Ray adaptor marks a statistic that RLlib did not report,
        # so it carries no ordering information. Python's min() compares with
        # ``<``, which is false against NaN, and would then return
        # NaN or a number depending on the push order. NaN values are skipped
        # instead; a window holding nothing but NaN stays NaN.
        valid = [value for value in self.values if not math.isnan(value)]

        if not valid:
            return self.values[0]

        return min(valid)

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
