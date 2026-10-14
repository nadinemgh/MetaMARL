"""Metric schema logged by ``WaterRegulatedEnv`` during every episode.

``WaterMetricSchema`` extends the generic episode schema with the reservoir,
the weather, the crop model and the quota quantities of the irrigation game.
The environment pushes one value per step (the farm-level quantities are
averaged over the farms first) and the inner optimizer reduces each field to
one value per episode. The regulator environment reads the reduced
``reservoir_level_norm_series`` together with its no-withdrawal baseline
``baseline_reservoir_level_norm_series``, which keep every step of the episode,
to measure how far the irrigation moved the lake from its natural level, and
``reward_mean`` of the generic schema for the economic part of the fitness.
"""

from typing import Optional

from pydantic import Field

from core.envs.schema import EpisodeRolloutSchema
from core.metrics.enums import ReduceProtocol


class WaterMetricSchema(EpisodeRolloutSchema):
    """Irrigation-game metrics, one value per step, reduced per episode.

    Attributes
    ----------
    reservoir_stage_m : float or None
        Reservoir stage after the step (metres).
    reservoir_level_norm : float or None
        Filled fraction of the reservoir after the step.
    streamflow_m3s : float or None
        Reservoir inflow after the step (cubic metres per second).
    outflow_m3s : float or None
        Release downstream after the step (cubic metres per second).
    precip_mm_day : float or None
        Rain of the day reached by the step (millimetres per day).
    temp_c : float or None
        Monthly mean air temperature of the day reached (degrees Celsius).
    release_pressure : float or None
        Share of the inflow the reservoir releases, in [0, 1].
    residence_time_days : float or None
        Residence time of the reservoir after the step (days): the stored volume
        (filled fraction times the capacity) over the release.
    total_usage_m3s : float or None
        Total irrigation withdrawal of the step (cubic metres per second).
    crop_kc : float or None
        Crop coefficient of the day played (zero off season).
    eto_mm_day : float or None
        Reference evapotranspiration of the day played (millimetres per
        day).
    etcrop_mm_day : float or None
        Crop evapotranspiration of the day played (millimetres per day).
    deficit_mm_day : float or None
        Crop water deficit of the day played, net of rain (millimetres per
        day).
    full_required_m3_day : float or None
        Water that covers the whole deficit on a farm (cubic metres per
        day).
    crop_satisfaction : float or None
        Mean over the farms of the crop satisfaction, in [0, 1].
    requested_m3_day : float or None
        Mean over the farms of the volume requested (cubic metres per day).
    allowed_m3_day : float or None
        Volume the quota allows a farm (cubic metres per day).
    delivered_m3_day : float or None
        Mean over the farms of the volume delivered (cubic metres per day).
    quota_violation_m3 : float or None
        Mean over the farms of the requested volume above the quota (cubic
        metres per day).
    requested_frac : float or None
        Mean over the farms of the requested share of the deficit.
    quota_penalty : float or None
        Mean over the farms of the fine (reward units).
    flow_penalty : float or None
        Mean over the farms of the release-risk penalty (reward units).
    quota_stress : float or None
        How far the reservoir is above the protected level, in [0, 1].
    gauge_02ga041_m3s : float or None
        Modelled flow at gauge 02GA041 (cubic metres per second); Raven
        only.
    gauge_02ga041_observed_m3s : float or None
        Observed flow at gauge 02GA041 (cubic metres per second); Raven
        only.
    gauge_02ga014_m3s : float or None
        Modelled flow at gauge 02GA014 (cubic metres per second); Raven
        only.
    gauge_02ga014_observed_m3s : float or None
        Observed flow at gauge 02GA014 (cubic metres per second); Raven
        only.
    gauge_west_montrose_m3s : float or None
        Modelled flow at West Montrose (cubic metres per second); Raven
        only.
    gauge_west_montrose_observed_m3s : float or None
        Observed flow at West Montrose (cubic metres per second); Raven
        only.
    reservoir_level_norm_min : float or None
        Lowest filled fraction of the reservoir reached in the episode.
        Reduced with ``MIN``.
    min_demand_frac : float or None
        Minimum demand fraction of the policy in force (constant over an
        episode). Reduced with ``LAST``.
    max_demand_frac : float or None
        Maximum demand fraction of the policy in force (constant over an
        episode). Reduced with ``LAST``.
    reservoir_level_norm_series : list[float] or None
        Filled fraction of the reservoir at every step, kept as a series.
        Reduced with ``SERIES``.
    baseline_reservoir_level_norm_series : list[float] or None
        Filled fraction of the reservoir that receives no withdrawal, at every
        step. Reduced with ``SERIES``.
    streamflow_m3s_series : list[float] or None
        Reservoir inflow at every step (cubic metres per second), kept as a
        series. Reduced with ``SERIES``.
    baseline_streamflow_m3s_series : list[float] or None
        Inflow of the reservoir that receives no withdrawal, at every step.
        Reduced with ``SERIES``.
    outflow_m3s_series : list[float] or None
        Release at every step (cubic metres per second), kept as a series.
        Reduced with ``SERIES``.
    baseline_outflow_m3s_series : list[float] or None
        Release of the reservoir that receives no withdrawal, at every step.
        Reduced with ``SERIES``.

    The fields inherit the episode identity, the reward statistics and the empty
    ``by_agent`` mapping of ``EpisodeRolloutSchema``. The gauge fields stay
    ``None`` with the surrogate lake, which has no gauges. The reduced value of
    a ``*_series`` field of an episode is the list of its steps; the inner
    optimizer's logger holds one such list per episode it received, so a peeked
    leaf is a list of lists.

    When to use: as the ``schema`` of the inner environment
    (``APPOptimizerConfig().environment(schema=WaterMetricSchema)``), so that the
    environment logger and the regulator share one set of field names.

    Examples
    --------
    >>> from core.metrics.logger import MetricLogger
    >>> logger = MetricLogger.from_schema(WaterMetricSchema)
    >>> for level in (0.9, 0.7, 0.8):
    ...     logger.push(key=("reservoir_level_norm_min",), value=level)
    ...     logger.push(key=("outflow_m3s_series",), value=level * 10)
    >>> reduced = logger.reduce()
    >>> reduced.reservoir_level_norm_min
    0.7
    >>> [round(value, 2) for value in reduced.outflow_m3s_series]
    [9.0, 7.0, 8.0]
    >>> reduced.gauge_west_montrose_m3s is None
    True
    """

    reservoir_stage_m: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    reservoir_level_norm: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    streamflow_m3s: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    outflow_m3s: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    precip_mm_day: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    temp_c: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    release_pressure: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    residence_time_days: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    total_usage_m3s: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    crop_kc: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    eto_mm_day: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    etcrop_mm_day: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    deficit_mm_day: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    full_required_m3_day: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    crop_satisfaction: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    requested_m3_day: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    allowed_m3_day: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    delivered_m3_day: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    quota_violation_m3: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    requested_frac: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    quota_penalty: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    flow_penalty: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    quota_stress: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    gauge_02ga041_m3s: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    gauge_02ga041_observed_m3s: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    gauge_02ga014_m3s: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    gauge_02ga014_observed_m3s: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    gauge_west_montrose_m3s: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    gauge_west_montrose_observed_m3s: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MEAN}
    )
    reservoir_level_norm_min: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.MIN}
    )
    min_demand_frac: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.LAST}
    )
    max_demand_frac: Optional[float] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.LAST}
    )
    reservoir_level_norm_series: Optional[list[float]] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.SERIES}
    )
    baseline_reservoir_level_norm_series: Optional[list[float]] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.SERIES}
    )
    streamflow_m3s_series: Optional[list[float]] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.SERIES}
    )
    baseline_streamflow_m3s_series: Optional[list[float]] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.SERIES}
    )
    outflow_m3s_series: Optional[list[float]] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.SERIES}
    )
    baseline_outflow_m3s_series: Optional[list[float]] = Field(
        default=None, json_schema_extra={"reduce": ReduceProtocol.SERIES}
    )
