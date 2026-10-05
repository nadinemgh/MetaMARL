"""Map a :class:`ReduceProtocol` to its :class:`Metric` implementation."""

from core.metrics.enums import ReduceProtocol
from core.metrics.metric.base import Metric
from core.metrics.metric.last import LastMetric
from core.metrics.metric.max import MaxMetric
from core.metrics.metric.mean import MeanMetric
from core.metrics.metric.min import MinMetric
from core.metrics.metric.series import SeriesMetric
from core.metrics.metric.sum import SumMetric


class MetricFactory:
    """Instantiate the :class:`Metric` matching a :class:`ReduceProtocol`.

    The logger asks the factory for one metric per leaf of a schema, using the
    protocol declared on the field. ``MEAN``, ``SERIES``, ``LAST``, ``MAX``,
    ``MIN`` and ``SUM`` are implemented; ``EMA`` is not.

    When to use: when you build metrics by hand, or when you add a reduction
    and need it created from its protocol.

    Examples
    --------
    >>> MetricFactory.create(ReduceProtocol.MAX)
    MaxMetric(None; len=0)
    >>> MetricFactory.create(ReduceProtocol.EMA)
    Traceback (most recent call last):
        ...
    NotImplementedError: Reduce protocol 'ema' is not implemented.
    """

    @staticmethod
    def create(protocol: ReduceProtocol) -> Metric:
        """Return a new, empty metric for ``protocol``.

        Parameters
        ----------
        protocol : ReduceProtocol
            The reduction the metric implements.

        Returns
        -------
        Metric
            A new metric of the matching class, holding no value.

        Raises
        ------
        NotImplementedError
            For protocols without an implementation (currently ``EMA``).
        """

        match protocol:
            case ReduceProtocol.MEAN:
                return MeanMetric()

            case ReduceProtocol.SERIES:
                return SeriesMetric()

            case ReduceProtocol.LAST:
                return LastMetric()

            case ReduceProtocol.MAX:
                return MaxMetric()

            case ReduceProtocol.MIN:
                return MinMetric()

            case ReduceProtocol.SUM:
                return SumMetric()

            case _:
                raise NotImplementedError(
                    f"Reduce protocol {protocol.value!r} is not implemented."
                )
