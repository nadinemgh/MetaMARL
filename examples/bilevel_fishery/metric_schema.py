"""Metric schema logged by ``FisheryRegulatedEnv`` during every episode.

``FisheryMetricSchema`` extends the generic episode schema with the stock
dynamics (biomass, growth, harvest, reference points) and
``FisheryAgentMetricSchema`` with per-fisher harvest fields. The regulated
environment pushes one value per step into the stock-level fields, and the
inner optimizer reduces each field to one value per episode according to its
declared reduction. The regulator environment reads the reduced
``fish_norm_next_series``, ``H_realized_series``, ``reward_series`` and ``MSY``
values back to compute a candidate's fitness: the three ``*_series`` fields
keep every step of the episode (the ``SERIES`` reduction), so that the fitness
can be computed on the last steps of each episode.
"""

from typing import Optional, TypeAlias

from pydantic import Field

from core.envs.schema import AgentEnvStepSchema, EpisodeRolloutSchema
from core.metrics.enums import ReduceProtocol

AgentID: TypeAlias = str


class FisheryAgentMetricSchema(AgentEnvStepSchema):
    """Per-fisher harvest metrics, one value per step.

    ``FisheryRegulatedEnv`` pushes ``requested_harvest`` and
    ``delivered_harvest`` for every fisher at every step.

    Attributes
    ----------
    requested_harvest : float or None
        Harvest the fisher asked for, after the regulation of the leaders'
        mechanisms (biomass units per step).
    delivered_harvest : float or None
        Harvest the fisher actually received: its request, or its pro-rata
        share of the stock when the total request exceeds the stock (biomass
        units per step).

    All fields are reduced with ``MEAN`` and inherit the generic per-step
    fields of ``AgentEnvStepSchema``.

    When to use: as the value type of the ``by_agent`` mapping of
    ``FisheryMetricSchema``; it is rarely built by hand.

    Examples
    --------
    >>> FisheryAgentMetricSchema(requested_harvest=25.0).requested_harvest
    25.0
    """

    requested_harvest: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    delivered_harvest: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )


class FisheryMetricSchema(EpisodeRolloutSchema):
    """Stock-level fishery metrics, one value per step, reduced per episode.

    Besides the fields below, the schema inherits the episode identity, the
    reward statistics and the empty ``by_agent`` mapping of
    ``EpisodeRolloutSchema``. ``FisheryRegulatedEnv`` pushes ``B_msy``,
    ``MSY``, ``F_msy``, ``fish_stock``, ``fish_stock_next``, the four
    ``fish_norm_next_*`` fields, ``growth``, ``growth_noise``, ``H_attempted``,
    ``H_realized``, ``total_usage_norm``, the three ``*_series`` fields and the
    two harvest fields of each fisher in ``by_agent`` at every step.

    Attributes
    ----------
    fish_stock : float or None
        Biomass ``B(t)`` the step starts with, before the catches and the
        restoration of that step (biomass units).
    growth : float or None
        Surplus production ``g(B(t))`` plus process noise computed by the
        transition (biomass units per step). Restoration is not part of it.
    growth_noise : float or None
        The process-noise part of ``growth`` (biomass units per step).
    H_attempted : float or None
        Total harvest the fishers requested (biomass units per step).
    H_realized : float or None
        Total catch ``C(t)`` actually delivered to the fishers (biomass units
        per step): the total request, or the whole stock ``B(t)`` when the
        request exceeds it. It is the sum of the per-fisher
        ``delivered_harvest`` and is never negative.
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
        Biomass ``B(t + 1)`` after the transition (biomass units).
    fish_norm_next_mean, fish_norm_next_min, fish_norm_next_max : float or None
        Mean, minimum and maximum over the episode of the normalized biomass
        after the transition (fraction of ``K``). Required fields with no
        default.
    fish_norm_next_last : float or None
        Normalized biomass after the last transition of the episode
        (fraction of ``K``). Required, with no default.
    fish_norm_next_series : list[float] or None
        Normalized biomass after the transition at every step of the episode
        (fraction of ``K``), kept as a series. The regulator environment
        computes the biomass part of the fitness on its last steps.
    H_realized_series : list[float] or None
        ``H_realized`` at every step of the episode (biomass units per step),
        kept as a series.
    reward_series : list[float] or None
        Mean over the fishers of the reward they received at every step of the
        episode (reward units), kept as a series.
    by_agent : dict[str, FisheryAgentMetricSchema]
        Per-fisher metrics keyed by agent id (requested and delivered harvest
        in this fishery).

    All the stock-level fields are reduced with ``MEAN`` except the four
    ``fish_norm_next_*`` fields, which use ``MEAN``, ``MIN``, ``MAX`` and
    ``LAST``, and the three ``*_series`` fields, which keep the whole series.
    The reduced value of a ``*_series`` field of an episode is the list of its
    steps; the inner optimizer's logger holds one such list per episode it
    received, so a peeked leaf is a list of lists.

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

    fish_norm_next_series: Optional[list[float]] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.SERIES}
    )
    H_realized_series: Optional[list[float]] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.SERIES}
    )
    reward_series: Optional[list[float]] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.SERIES}
    )

    by_agent: dict[AgentID, FisheryAgentMetricSchema] = Field(default_factory=dict)
