"""Metric schema logged by ``CartpoleRegulatedEnv`` during every episode.

``CartpoleMetricSchema`` extends the generic episode schema with three
series of the cart-pole state. The environment pushes one value per step and
the inner optimizer reduces each field to one value per episode. The episode
return itself needs no extra field: it is the ``reward_total`` of the generic
schema (one reward unit per step played), and the regulator environment reads
``reward_mean`` from it.
"""

from typing import Optional

from pydantic import Field

from core.envs.schema import EpisodeRolloutSchema
from core.metrics.enums import ReduceProtocol


class CartpoleMetricSchema(EpisodeRolloutSchema):
    """Cart-pole metrics, one value per step, reduced per episode.

    Attributes
    ----------
    cart_position : float or None
        Cart position after the step (metres), averaged over the episode. The
        track ends at ``+-2.4``.
    pole_angle : float or None
        Pole angle after the step (radians, positive when the pole leans to the
        right), averaged over the episode. A mean near zero does not mean the
        pole stayed upright, since the sign alternates; see the next field.
    pole_angle_abs_max : float or None
        Largest absolute pole angle of the episode (radians). The episode
        terminates when it exceeds about 0.2095 rad (12 degrees).

    The fields inherit the episode identity, the reward statistics and the empty
    ``by_agent`` mapping of ``EpisodeRolloutSchema``.

    When to use: as the ``schema`` of the inner environment
    (``APPOptimizerConfig().environment(schema=CartpoleMetricSchema)``), so that
    the environment logger and the regulator share one set of field names.

    Examples
    --------
    >>> from core.metrics.logger import MetricLogger
    >>> logger = MetricLogger.from_schema(CartpoleMetricSchema)
    >>> for angle in (0.02, 0.05, 0.03):
    ...     logger.push(key=("pole_angle_abs_max",), value=angle)
    >>> logger.reduce().pole_angle_abs_max
    0.05
    """

    cart_position: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    pole_angle: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    pole_angle_abs_max: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MAX}
    )
