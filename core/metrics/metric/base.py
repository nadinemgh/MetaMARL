"""Abstract accumulator behind every metric schema leaf.

A ``Metric`` receives raw values through ``push``, exposes them without
side effects through ``peek`` and collapses them through ``reduce`` according
to its protocol (mean, sum, ...). ``compile=False`` returns the raw history or a
copy of the metric instead of the compiled scalar.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Self, TypeAlias, Union

# NOTE this is restrictive can be relaxed in the future
PrimitiveType: TypeAlias = Union[int, float, bool, str]


class Metric(ABC):
    """Accumulator of primitive values behind one schema field.

    Each subclass implements one reduction protocol (see
    :class:`~core.metrics.enums.ReduceProtocol`). ``float()`` and ``int()``
    on a metric return its compiled value and raise ``ValueError`` when that
    value is a list. The class is abstract: ``len``, ``push``, ``peek``,
    ``reduce`` and ``flush`` are implemented by the subclasses.

    When to use: you rarely instantiate a metric yourself; the logger creates
    one per schema field through
    :class:`~core.metrics.metric.factory.MetricFactory`. Subclass it to add a
    reduction, and register that reduction in the factory.

    Examples
    --------
    The concrete :class:`~core.metrics.metric.mean.MeanMetric` shows the
    contract:

    >>> from core.metrics.metric.mean import MeanMetric
    >>> metric = MeanMetric()
    >>> metric.push(1.0)
    >>> metric.push(3.0)
    >>> len(metric), float(metric)
    (2, 2.0)
    >>> metric.empty_copy()
    MeanMetric(None; len=0)
    """

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of values currently held."""

        ...

    def __float__(self):
        value = self.peek(compile=True)

        if isinstance(value, (list)):  # , tuple, deque
            raise ValueError(f"Can not convert {self} to float.")

        return float(value)

    def __int__(self):
        value = self.peek(compile=True)

        if isinstance(value, (list)):
            raise ValueError(f"Can not convert {self} to int.")

        return int(value)

    def empty_copy(self) -> Self:
        """Return a fresh, empty metric of the same class.

        Returns
        -------
        Metric
            A new instance created with no argument; the values of this metric
            are not copied.
        """

        return type(self)()

    @abstractmethod
    def peek(self, compile: bool = True) -> Union[PrimitiveType, list[PrimitiveType]]:
        """Return the reduction of the values without altering them.

        Users can call this method to look at the reduced value(s) of the
        current history; the history is left as it is.

        Parameters
        ----------
        compile : bool, default True
            If true, the result is the compiled value of the protocol (a mean,
            a sum, ...); if false, the raw history as a list.

        Returns
        -------
        int or float or bool or str or list
            The compiled value, or the history when ``compile`` is false.
            Subclasses return ``None`` for the compiled value of an empty
            history, except the sum, which is ``0``.
        """

    @abstractmethod
    def reduce(
        self, compile: bool = True
    ) -> Union[PrimitiveType, list[PrimitiveType], Metric]:
        """Reduce the values, clear the history and return the result.

        Users do not normally call this method: the logger calls it on every
        leaf when it reduces, which also starts a new accumulation cycle.
        What the reduction returns depends on the subclass: most reduce to a
        single value, the series metric returns its history.

        Parameters
        ----------
        compile : bool, default True
            If true, the result is the compiled value. If false, the result is
            a new metric of the same class holding the reduced value(s).

        Returns
        -------
        int or float or bool or str or list or Metric
            The compiled value, or a new metric holding it when ``compile`` is
            false.
        """

    @abstractmethod
    def push(self, value: PrimitiveType) -> None:
        """Append one value to the internal history.

        Parameters
        ----------
        value : int or float or bool or str
            The value to record. Subclasses may reject some types.
        """

        ...

    @abstractmethod
    def flush(self) -> None:
        """Discard every accumulated value, leaving the metric empty."""

    ...
