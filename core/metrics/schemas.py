"""Root of the metric schema hierarchy.

Every logged structure derives from :class:`MetricSchema`; it carries the
``iter`` counter (last value kept) shared by all loggers. The schema is only a
declaration: :class:`~core.metrics.logger.MetricLogger` accumulates the values
and returns populated instances of it, which the reporters then resolve their
queries against.
"""

from typing import Optional

from pydantic import BaseModel, Field

from core.metrics.enums import ReduceProtocol


class MetricSchema(BaseModel):
    """Base class of every logged structure.

    Subclasses declare the metrics of one component as pydantic fields: a
    primitive field is a leaf reduced according to its ``reduce`` extra
    (``MEAN`` by default), a nested ``MetricSchema`` is a static sub-tree and
    a ``dict[ID, MetricSchema]`` is a dynamic node populated at runtime.
    ``iter`` is the only field shared by all schemas: the index of the
    iteration that produced the values (an integer counter, no unit), reduced
    with ``LAST`` so a populated schema always reports the most recent one.

    Attributes
    ----------
    iter : int or None
        Index of the iteration that produced the values; ``None`` by default.
        A schema returned by ``MetricLogger.peek`` holds the list of every
        pushed value in each leaf, including this one.

    When to use: subclass it to declare the metrics of one component (an
    environment, an optimizer, a policy) before logging them.

    Examples
    --------
    >>> class Rollout(MetricSchema):
    ...     reward: Optional[float] = None
    ...     steps: Optional[int] = Field(
    ...         default=None, json_schema_extra={"reduce": ReduceProtocol.SUM}
    ...     )
    >>> Rollout(iter=3, reward=1.5)
    Rollout(iter=3, reward=1.5, steps=None)
    """

    iter: Optional[int] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.LAST}
    )
