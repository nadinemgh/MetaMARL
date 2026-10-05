"""Metric schema logged by ``FisheryRegulatedEnv`` during every episode.

``FisheryMetricSchema`` extends the generic episode schema with the stock
dynamics (biomass, growth, harvest, reference points) and
``FisheryAgentMetricSchema`` with per-fisher harvest fields. The regulated
environment pushes one value per step into the stock-level fields, and the
inner optimizer reduces each field to one value per episode according to its
declared reduction. The regulator environment reads the reduced
``fish_norm_next_mean``, ``H_realized`` and ``MSY`` values back to compute a
candidate's fitness. Several fields are declared here but not pushed by the
fishery environment of this package; each class says which.
"""

from typing import Optional, TypeAlias

from pydantic import Field

from core.envs.schema import AgentEnvStepSchema, EpisodeRolloutSchema
from core.metrics.enums import ReduceProtocol

AgentID: TypeAlias = str


class FisheryAgentMetricSchema(AgentEnvStepSchema):
    """Per-fisher harvest metrics, one value per step.

    ``FisheryRegulatedEnv`` does not push these fields: they are declared for
    environments that log the harvest request of each fisher, and they stay
    ``None`` (an empty series) in the fishery of this package.

    Attributes
    ----------
    requested_harvest : float or None
        Harvest the fisher asked for (biomass units per step).
    delivered_harvest : float or None
        Harvest actually delivered after regulation (biomass units per step).
    requested_frac : float or None
        Requested harvest as a fraction of the fisher's maximal request, in
        ``[0, 1]``.
    quota_violation : float or None
        Dimensionless violation measure of the quota.
    quota_penalty : float or None
        Penalty the quota subtracts from the reward (reward units).
    risk_penalty : float or None
        Penalty for risky harvests subtracted from the reward (reward units).

    All fields are reduced with ``MEAN`` and inherit the generic per-step
    fields of ``AgentEnvStepSchema``.

    When to use: as the value type of the ``by_agent`` mapping of
    ``FisheryMetricSchema``; it is rarely built by hand.

    Examples
    --------
    >>> FisheryAgentMetricSchema(requested_frac=0.25).requested_frac
    0.25
    """

    requested_harvest: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    delivered_harvest: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    requested_frac: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    quota_violation: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    quota_penalty: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    risk_penalty: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )


class FisheryMetricSchema(EpisodeRolloutSchema):
    """Stock-level fishery metrics, one value per step, reduced per episode.

    Besides the fields below, the schema inherits the episode identity, the
    reward statistics and the empty ``by_agent`` mapping of
    ``EpisodeRolloutSchema``. ``FisheryRegulatedEnv`` pushes ``B_msy``,
    ``MSY``, ``F_msy``, ``fish_stock``, ``fish_stock_next``, the four
    ``fish_norm_next_*`` fields, ``growth``, ``growth_noise``, ``H_realized``
    and ``total_usage_norm`` at every step. It does not push ``quota_stress``,
    ``allowed_harvest``, ``H_attempted``, ``fish_norm`` or any ``by_agent``
    value, so those stay ``None`` or empty.

    Attributes
    ----------
    quota_stress : float or None
        Allowed harvest fraction of the quota in force (dimensionless). Not
        pushed.
    allowed_harvest : float or None
        Total harvest the quota permits (biomass units per step). Not pushed.
    fish_stock : float or None
        Biomass read by the transition at the current step, after the
        fishers' harvest and restoration of that step (biomass units).
    growth : float or None
        Surplus production plus process noise computed by the transition
        (biomass units per step). Restoration is not part of it.
    growth_noise : float or None
        The process-noise part of ``growth`` (biomass units per step).
    H_attempted : float or None
        Total requested harvest (biomass units per step). Not pushed.
    H_realized : float or None
        Logged harvest of the step (biomass units per step). It is obtained as
        the difference between the stock one time index earlier and the current
        stock, capped at the available biomass, and is ``0`` at the first
        step. Because that difference also contains the previous step's growth
        and the restoration, it is not the sum of the fishers' catches and it
        can be negative.
    total_usage_norm : float or None
        ``H_realized`` divided by the carrying capacity ``K`` (dimensionless).
    B_msy : float or None
        Biomass at maximum sustainable yield (biomass units), a constant of
        the ecology.
    MSY : float or None
        Maximum sustainable yield (biomass units per step), a constant of the
        ecology.
    F_msy : float or None
        Harvest rate at maximum sustainable yield, ``MSY / B_msy`` (per step),
        a constant of the ecology.
    fish_stock_next : float or None
        Biomass after the transition (biomass units).
    fish_norm : float or None
        Normalized biomass (fraction of ``K``). Not pushed.
    fish_norm_next_mean, fish_norm_next_min, fish_norm_next_max : float or None
        Mean, minimum and maximum over the episode of the normalized biomass
        after the transition (fraction of ``K``). Required fields with no
        default.
    fish_norm_next_last : float or None
        Normalized biomass after the last transition of the episode
        (fraction of ``K``). Required, with no default.
    by_agent : dict[str, FisheryAgentMetricSchema]
        Per-fisher metrics keyed by agent id; empty in this fishery.

    All the stock-level fields are reduced with ``MEAN`` except the four
    ``fish_norm_next_*`` fields, which use ``MEAN``, ``MIN``, ``MAX`` and
    ``LAST``.

    When to use: as the ``schema`` of the inner environment
    (``APPOptimizerConfig().environment(schema=FisheryMetricSchema)``), so that
    the environment logger and the regulator share one set of field names.

    Examples
    --------
    >>> from core.metrics.logger import MetricLogger
    >>> logger = MetricLogger.from_schema(FisheryMetricSchema)
    >>> for biomass in (0.3, 0.5, 0.4):
    ...     logger.push(key=("fish_norm_next_min",), value=biomass)
    >>> logger.reduce().fish_norm_next_min
    0.3
    """

    quota_stress: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    allowed_harvest: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    fish_stock: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    growth: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    growth_noise: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    H_attempted: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    H_realized: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    total_usage_norm: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    B_msy: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    MSY: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    F_msy: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    fish_stock_next: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    fish_norm: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    fish_norm_next_mean: Optional[float] = Field(
        json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    fish_norm_next_min: Optional[float] = Field(
        json_schema_extra={"reduce": ReduceProtocol.MIN}
    )
    fish_norm_next_max: Optional[float] = Field(
        json_schema_extra={"reduce": ReduceProtocol.MAX}
    )
    fish_norm_next_last: Optional[float] = Field(
        json_schema_extra={"reduce": ReduceProtocol.LAST}
    )

    by_agent: dict[AgentID, FisheryAgentMetricSchema] = Field(default_factory=dict)
