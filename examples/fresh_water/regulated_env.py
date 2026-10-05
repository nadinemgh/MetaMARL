"""Irrigation from a shared reservoir: farms, a water policy, a lake model.

A crowd of corn farms draws irrigation water from one reservoir (Belwood Lake,
in the Grand River basin, in the first version of the example). Every day each
farm chooses the fraction ``f`` in ``[0, 1]`` of its crop water deficit that it
requests. The regulator's water policy (see :mod:`examples.fresh_water.mechanism`)
caps the request with a quota that depends on how full the reservoir is, fines
the part of the request above the quota and penalizes a large request when the
river depends on the reservoir's release.

The crop model is a daily water balance in the Blaney-Criddle form (Doorenbos &
Pruitt, 1977). With ``T`` the monthly mean temperature in degrees Celsius and
``p`` the mean daily share of the annual daytime hours at 45 degrees north
(``P_BY_MONTH_45N``), the reference evapotranspiration of a day is

    ETo = max(0, p * (0.46 * T + 8))   (millimetres per day),

the crop evapotranspiration is ``ETc = Kc * ETo`` with a crop coefficient ``Kc``
that depends on the days after planting (``CORN_GRAIN_KC``: 0.40 up to day 30,
0.80 up to day 70, 1.15 up to day 110, 0.70 up to day 150 and zero afterwards),
and the deficit is ``max(0, ETc - rain)``. A farm of area ``A`` square metres
needs ``ETc / 1000 * A`` cubic metres a day, of which the rain brings
``rain / 1000 * A`` and the irrigation must bring ``deficit / 1000 * A``
(``full_required_m3_day``). The coefficients and the stage lengths follow the
spirit of FAO-56 (Allen et al., 1998) but are the simplified values of the first
version of the example, not the entries of its tables.

A farm that requests ``f * full_required`` receives ``min(request, quota)``, and
its crop satisfaction is

    satisfaction = min(1, (delivered + rain water) / crop water need),

or 1 when the crop needs no water. The reward of a farm is its satisfaction
minus the policy's penalty (the penalty is added by the leader's mechanism). The
total delivered volume is withdrawn from the reservoir, and the lake model
(:mod:`examples.fresh_water.hydrology`) returns the next stage, the flows and the
rain. The episode starts on a planting day drawn uniformly between day 121 and
day 273 of the year (May to September) and lasts ``horizon`` days.

The observation of a farm has twelve entries: the filled fraction of the
reservoir, the release pressure (the share of the inflow that is released), the
allowed fraction at the current level (filled in by the policy), the total
volume withdrawn the previous day divided by the largest volume all the farms
can need in a day (``WaterRegulatedEnv.max_daily_need_m3_day``, so that the
entry lies in [0, 1] like the others) and the eight normalized rules (filled in
by the policy).

References
----------
Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998). Crop
evapotranspiration: Guidelines for computing crop water requirements. FAO
Irrigation and Drainage Paper 56. https://www.fao.org/4/x0490e/x0490e00.htm

Doorenbos, J., & Pruitt, W. O. (1977). Guidelines for predicting crop water
requirements. FAO Irrigation and Drainage Paper 24.
"""

import logging
from datetime import datetime, timedelta
from typing import Any, ClassVar, NamedTuple, Optional

import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType

from core.agents.base import Agent, AgentConfig
from core.envs.hooks import reset, transition
from core.envs.marl_regulated import MultiAgentEnv
from core.mechanism.base import MDPState, Mechanism
from core.mechanism.config import MechanismConfig
from examples.fresh_water.hydrology import (
    RAVEN_ORIGIN,
    SECONDS_PER_DAY,
    TEMP_C_BY_MONTH,
    LakeModel,
    LakeReading,
    RavenLake,
    SurrogateLake,
    estimate_temp_c,
)
from examples.fresh_water.mechanism import (
    DEFAULT_RULES,
    IRRIGATE_ID,
    OBSERVATION_SIZE,
    WATER_POLICY_ID,
    irrigation_fraction,
    quota_stress,
    violation_signal,
)

logger = logging.getLogger(__name__)

EPS = 1e-8

CORN_GRAIN_KC = {
    "initial": 0.40,
    "development": 0.80,
    "mid": 1.15,
    "late": 0.70,
    "offseason": 0.0,
}
"""Crop coefficient of corn for grain, by growth stage."""

P_BY_MONTH_45N = {
    1: 0.20,
    2: 0.23,
    3: 0.27,
    4: 0.30,
    5: 0.34,
    6: 0.35,
    7: 0.34,
    8: 0.32,
    9: 0.28,
    10: 0.24,
    11: 0.21,
    12: 0.20,
}
"""Mean daily share of the annual daytime hours at 45 degrees north, by month."""

DEFAULT_ECOLOGY = {
    "full_stage_m": 420.41,
    "max_depth_m": 11.0,
    "lake_area_m2": 5756935.89615,
    "max_farm_area_m2": 1_000_000.0,
}
"""Reservoir and farm constants of the Belwood Lake setup of the first version."""


def reference_evapotranspiration(temp_c: float, month: int) -> float:
    """Return the reference evapotranspiration of a day (Blaney-Criddle form).

    Parameters
    ----------
    temp_c : float
        Monthly mean air temperature (degrees Celsius).
    month : int
        Month of the day, 1 to 12; it selects the daylight share in
        ``P_BY_MONTH_45N``.

    Returns
    -------
    float
        ``max(0, p * (0.46 * T + 8))`` in millimetres per day.

    When to use: it is the ``ETo`` of :func:`crop_demand` and of the largest
    daily crop need ``PEAK_CROP_NEED_MM_DAY``.

    Examples
    --------
    >>> round(reference_evapotranspiration(23.0, 7), 4)
    6.3172
    """
    return max(0.0, P_BY_MONTH_45N[month] * (0.46 * temp_c + 8.0))


PEAK_ETO_MM_DAY = max(
    reference_evapotranspiration(TEMP_C_BY_MONTH[month], month)
    for month in P_BY_MONTH_45N
)
"""Largest reference evapotranspiration of the year (millimetres per day).

It is reached in July, with 23 degrees and a daylight share of 0.34: 6.3172.
"""

PEAK_CROP_NEED_MM_DAY = max(CORN_GRAIN_KC.values()) * PEAK_ETO_MM_DAY
"""Largest crop evapotranspiration a day can have (millimetres per day).

The peak crop coefficient (1.15, mid season) times ``PEAK_ETO_MM_DAY``. The two
peaks need not fall on the same day, so this is an upper bound of the crop need
``ETc`` of :func:`crop_demand`; it is about 7.265.
"""

PLANTING_DAY_OF_YEAR = (121, 274)
"""Half-open range of the planting day of the year (May 1 to September 30)."""

SURROGATE_KEYS = (
    "base_inflow_m3s",
    "seasonal_amplitude",
    "inflow_log_sigma",
    "release_fraction",
    "rain_probability",
    "rain_mean_mm",
    "initial_level_range",
)
"""Keys of ``ecology_cfg`` that configure the surrogate lake."""

IRRIGATE_SPACE = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
"""Action space of a farm: the requested share of its crop deficit."""


def crop_stage(days_after_planting: int) -> str:
    """Return the growth stage of corn ``days_after_planting`` days after sowing.

    Parameters
    ----------
    days_after_planting : int
        Days since the planting date; negative before planting.

    Returns
    -------
    str
        ``"initial"`` up to day 30, ``"development"`` up to 70, ``"mid"`` up to
        110, ``"late"`` up to 150, and ``"offseason"`` before planting and after
        day 150.

    When to use: to look up a crop coefficient in ``CORN_GRAIN_KC``.

    Examples
    --------
    >>> stages = [crop_stage(d) for d in (-1, 0, 30, 31, 70, 110, 150, 151)]
    >>> stages[:4]
    ['offseason', 'initial', 'initial', 'development']
    >>> stages[4:]
    ['development', 'mid', 'late', 'offseason']
    """
    if days_after_planting < 0:
        return "offseason"
    if days_after_planting <= 30:
        return "initial"
    if days_after_planting <= 70:
        return "development"
    if days_after_planting <= 110:
        return "mid"
    if days_after_planting <= 150:
        return "late"
    return "offseason"


class CropDemand(NamedTuple):
    """Water demand of a farm on one day.

    Attributes
    ----------
    stage : str
        Growth stage, a key of ``CORN_GRAIN_KC``.
    kc : float
        Crop coefficient of the stage.
    temp_c : float
        Monthly mean air temperature (degrees Celsius).
    eto_mm_day, etcrop_mm_day, deficit_mm_day : float
        Reference evapotranspiration, crop evapotranspiration and crop deficit
        net of rain (millimetres per day).
    precip_mm_day : float
        Rain of the day (millimetres per day).
    full_required_m3_day : float
        Irrigation volume that covers the deficit of the whole farm area (cubic
        metres per day).
    crop_water_need_m3_day : float
        Water the crop needs, rain included (cubic metres per day).
    precip_water_m3_day : float
        Rain that falls on the farm area (cubic metres per day).

    When to use: it is what :func:`crop_demand` returns.

    Examples
    --------
    >>> CropDemand("mid", 1.15, 20.0, 5.0, 5.75, 0.0, 5.75, 0.0, 0.0, 0.0).stage
    'mid'
    """

    stage: str
    kc: float
    temp_c: float
    eto_mm_day: float
    etcrop_mm_day: float
    precip_mm_day: float
    deficit_mm_day: float
    full_required_m3_day: float
    crop_water_need_m3_day: float
    precip_water_m3_day: float


def crop_demand(
    date: datetime, days_after_planting: int, precip_mm_day: float, farm_area_m2: float
) -> CropDemand:
    """Compute the water demand of a farm on ``date``.

    Parameters
    ----------
    date : datetime
        The day; its month sets the temperature and the daylight share.
    days_after_planting : int
        Days since the planting date; negative before planting.
    precip_mm_day : float
        Rain of the day (millimetres per day).
    farm_area_m2 : float
        Area of a farm (square metres).

    Returns
    -------
    CropDemand
        Evapotranspiration, deficit and the volumes for the whole farm.

    When to use: once per simulated day, to know what the farms can ask for.

    Examples
    --------
    On July 15, 40 days after planting, the temperature is 23 degrees and the
    daylight share 0.34, so ``ETo = 0.34 * (0.46 * 23 + 8) = 6.3...`` and the
    coefficient of the development stage is 0.80:

    >>> from datetime import datetime
    >>> demand = crop_demand(datetime(1980, 7, 15), 40, 1.0, 1_000_000.0)
    >>> demand.stage, round(demand.eto_mm_day, 4), round(demand.deficit_mm_day, 4)
    ('development', 6.3172, 4.0538)
    >>> round(demand.full_required_m3_day, 1)
    4053.8
    """
    temp_c = estimate_temp_c(date)
    stage = crop_stage(days_after_planting)
    kc = CORN_GRAIN_KC[stage]
    eto = reference_evapotranspiration(temp_c, date.month)
    etcrop = eto * kc
    deficit = max(0.0, etcrop - precip_mm_day)
    return CropDemand(
        stage=stage,
        kc=kc,
        temp_c=temp_c,
        eto_mm_day=eto,
        etcrop_mm_day=etcrop,
        precip_mm_day=precip_mm_day,
        deficit_mm_day=deficit,
        full_required_m3_day=deficit / 1000.0 * farm_area_m2,
        crop_water_need_m3_day=etcrop / 1000.0 * farm_area_m2,
        precip_water_m3_day=precip_mm_day / 1000.0 * farm_area_m2,
    )


def residence_time_days(storage_m3: float, outflow_m3s: float) -> float:
    """Return the residence time of the reservoir, in days.

    The residence time of a reservoir is its stored volume divided by the
    flow that leaves it, the time the outflow would take to empty it. The outflow
    (the release downstream, and the spill when there is one) is the one that
    carries water out of the reservoir, so it is the flow used here and not the
    inflow.

    Parameters
    ----------
    storage_m3 : float
        Water stored in the reservoir (cubic metres); a negative value is read
        as an empty reservoir.
    outflow_m3s : float
        Flow released downstream (cubic metres per second).

    Returns
    -------
    float
        ``max(0, storage) / max(EPS, outflow) / 86400`` days. A zero outflow is
        guarded by ``EPS``, which gives a very large residence time rather than
        a division by zero.

    When to use: once per simulated day, for the ``residence_time_days`` metric.

    Examples
    --------
    A reservoir of 6.048e7 cubic metres that releases 10 cubic metres per second
    empties in 70 days:

    >>> round(residence_time_days(6.048e7, 10.0), 6)
    70.0
    """
    return max(0.0, storage_m3) / max(EPS, outflow_m3s) / SECONDS_PER_DAY


class Utilizer(Agent):
    """Farm agent: sees the reservoir and is paid its crop satisfaction.

    The agent holds the ``Irrigate`` mechanism (built from ``UtilizerConfig``).
    Its ``observation`` fills the four state entries of the observation; the
    allowed fraction (entry 2) and the normalized rules (entries 4 to 11) are
    added by the regulator's ``WaterPolicy``. The reward is not overridden: the
    crop satisfaction comes from the transition of
    :class:`WaterRegulatedEnv`, which knows the delivered volume, and the
    penalty from the policy.

    When to use: build it through ``UtilizerConfig``, which the environment does
    for you; instantiate it directly only to test the observation.

    Examples
    --------
    >>> from core.mechanism.base import MDPState
    >>> farm = Utilizer(id="utilizer:0", policy_id="utilizer_policy", mechanisms={})
    >>> mdp = MDPState(
    ...     state={
    ...         "reservoir_level_norm": 0.9,
    ...         "release_pressure": 0.5,
    ...         "usage_norm": 0.25,
    ...     }
    ... )
    >>> farm.observation(mdp).obs["utilizer:0"][0].tolist()[:4]
    [0.8999999761581421, 0.5, 0.0, 0.25]
    """

    def observation(self, mdp: MDPState) -> MDPState:
        """Build this farm's observation at the current step.

        Parameters
        ----------
        mdp : MDPState
            Shared state with ``reservoir_level_norm`` (filled fraction),
            ``release_pressure`` (share of the inflow released) and
            ``usage_norm`` (total volume withdrawn the previous day over the
            largest volume the farms can need in a day, dimensionless, about in
            [0, 1]) at ``mdp.t``.

        Returns
        -------
        MDPState
            An MDP whose ``obs`` holds, for this agent, the ``float32`` vector of
            ``OBSERVATION_SIZE`` entries ``[level, release pressure, 0, usage,
            0, ..., 0]``.
        """
        observation = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
        observation[0] = mdp.state["reservoir_level_norm"][mdp.t]
        observation[1] = mdp.state["release_pressure"][mdp.t]
        observation[3] = mdp.state["usage_norm"][mdp.t]
        return MDPState(obs={self.id: observation})


class UtilizerConfig(AgentConfig):
    """Configuration that builds ``Utilizer`` agents.

    A frozen dataclass inherited from :class:`core.agents.base.AgentConfig`;
    only ``agent_cls`` changes.

    Attributes
    ----------
    policy_id : str
        Identifier of the RL policy that controls the farms.
    mechanisms : MechanismConfig or tuple[MechanismConfig, ...]
        Mechanisms each farm holds, one ``IrrigateConfig`` (id ``"irrigate"``).
    id : str or None
        Agent type; instances are named ``"<id>:<index>"``. The regulator's
        policy targets the type ``"utilizer"``.
    count : int
        Number of farms built from this configuration.
    shared_policy : bool
        Whether the farms share one policy (default ``True``).
    observation_space : gymnasium.Space or None
        Observation space of a farm, a ``Box`` of shape ``(12,)``.

    When to use: in the ``agents`` of the inner optimizer configuration, once
    for the whole crowd of farms.

    Examples
    --------
    >>> config = UtilizerConfig(
    ...     id="utilizer",
    ...     policy_id="utilizer_policy",
    ...     mechanisms=(IrrigateConfig(id=IRRIGATE_ID, action_space=IRRIGATE_SPACE),),
    ... )
    >>> sorted(config.build().mechanisms)
    ['irrigate']
    """

    agent_cls: ClassVar[type[Agent]] = Utilizer


class Irrigate(Mechanism):
    """Irrigation request of a farm, a share of its crop deficit.

    The mechanism only clips the policy output to ``[0, 1]``. It contributes no
    residual: the delivered volume, the withdrawal and the reward are computed by
    the transition of :class:`WaterRegulatedEnv`, which reads the decoded request
    from ``mdp.actions``, and the penalty by the regulator's ``WaterPolicy``.

    When to use: as the ``irrigate`` mechanism of ``UtilizerConfig``, through
    ``IrrigateConfig``.

    Examples
    --------
    >>> import numpy as np
    >>> from core.mechanism.base import MDPState
    >>> irrigate = IrrigateConfig(id=IRRIGATE_ID, action_space=IRRIGATE_SPACE).build(
    ...     "utilizer:0"
    ... )
    >>> irrigate.decode(None, np.asarray([1.3], dtype=np.float32))
    1.0
    >>> irrigate.apply(MDPState(), 0.5).state.data
    {}
    """

    def decode(self, mdp: MDPState, action: ActType) -> float:
        """Clip the raw policy output to a request fraction.

        Parameters
        ----------
        mdp : MDPState
            Unused.
        action : ActType
            Raw policy output, an array of shape ``(1,)`` or a float.

        Returns
        -------
        float
            The requested share of the crop deficit, in ``[0, 1]``.
        """
        return irrigation_fraction(action)

    def apply(self, mdp: MDPState, action: ActType) -> MDPState:
        """Return an empty residual: the transition delivers the water.

        Parameters
        ----------
        mdp : MDPState
            Unused.
        action : ActType
            Decoded request, unused here.

        Returns
        -------
        MDPState
            A state with no change.
        """
        return MDPState()


class IrrigateConfig(MechanismConfig):
    """Configuration that builds the ``Irrigate`` mechanism.

    A frozen dataclass inherited from
    :class:`core.mechanism.config.MechanismConfig`; only ``mechanism_cls``
    changes.

    Attributes
    ----------
    action_space : gymnasium.spaces.Box
        ``IRRIGATE_SPACE``, ``Box(0, 1, (1,), float32)``.
    id : str or None
        Mechanism identifier, ``"irrigate"``; the regulator targets it in
        ``acts_on``.
    acts_on : tuple[str, str] or None
        Unused (default ``None``).
    obs_map : dict[str, str] or None
        Unused (default ``None``).
    default : numpy.ndarray or None
        Unused (default ``None``).

    When to use: in the ``mechanisms`` of ``UtilizerConfig``.

    Examples
    --------
    >>> type(IrrigateConfig(id="irrigate", action_space=IRRIGATE_SPACE).build("u:0"))
    <class 'examples.fresh_water.regulated_env.Irrigate'>
    """

    mechanism_cls: ClassVar[type[Mechanism]] = Irrigate


class WaterRegulatedEnv(MultiAgentEnv):
    """Irrigation game that the inner level of the bilevel loop trains on.

    The environment owns the lake model, the crop calendar and the logging. Its
    reset hook, ``reset_water``, draws the planting day and starts the lake; its
    transition hook, ``step_water``, delivers the water, pays the crop
    satisfaction, withdraws the volume from the lake and reads the next day. The
    regulator's ``WaterPolicy`` (a leader mechanism) sets the rules in force and
    the penalty; without a candidate the environment uses ``DEFAULT_RULES``.

    Parameters
    ----------
    ecology_cfg : dict, optional
        Constants of the reservoir and the farms, read with these keys:
        ``full_stage_m`` (420.41), ``max_depth_m`` (11.0), ``lake_area_m2``
        (5756935.89615) and ``max_farm_area_m2`` (1000000.0, the area of every
        farm); and, for the surrogate lake only, the keys listed in
        ``SURROGATE_KEYS`` (see :class:`examples.fresh_water.hydrology.
        SurrogateLake`).
    hydrology : str, optional
        ``"surrogate"`` (default) for the built-in reservoir, or ``"raven"`` for
        the external Raven model.
    raven_cwd : str or Path, optional
        Raven model directory; required with ``hydrology="raven"``.
    raven_cmd : str or Path, optional
        Raven executable; required with ``hydrology="raven"``.
    raven_work_dir : str or Path, optional
        Where the Raven run directories are created (default
        ``<raven_cwd>/.cache/prepared_runs``).
    **kwargs : Any
        Forwarded to :class:`core.envs.marl_regulated.MultiAgentEnv`: ``world``,
        ``mechanism_id``, ``horizon``, ``agents_cfg_dict``, ``leaders_cfg_dict``,
        ``seed``, ``schema`` (:class:`examples.fresh_water.metric_schema.
        WaterMetricSchema`; the environment logs through it and cannot reset
        without one), ``reporter_cfg`` and the other options of the base class.

    Attributes
    ----------
    lake : LakeModel
        The lake model in use.
    max_farm_area_m2 : float
        Area of every farm (square metres).
    capacity_m3 : float
        Volume of the full reservoir (cubic metres): ``lake_area_m2 *
        max_depth_m`` of ``ecology_cfg``, for both lake models. It converts the
        filled fraction into the stored volume of the residence time; the
        reservoir is taken as a vertical-walled tank, as ``level_norm`` does.
    max_daily_need_m3_day : float
        Largest volume all the farms can need in one day (cubic metres per
        day): ``number of farms * max_farm_area_m2 * PEAK_CROP_NEED_MM_DAY /
        1000``. It is computed once at construction. A farm is never delivered
        more than its crop need, so the total delivered volume divided by this
        constant, the entry 3 of the observation, lies in [0, 1]. With 500 farms
        of one million square metres it is about 3.632e6.

    Raises
    ------
    ValueError
        If ``hydrology`` is unknown, or is ``"raven"`` without ``raven_cwd`` and
        ``raven_cmd``.

    When to use: as the ``env`` of the inner (society) optimizer of the
    fresh-water experiment, with ``UtilizerConfig`` agents and the
    ``WaterPolicyConfig`` leader.

    Examples
    --------
    The surrogate lake needs no external file:

    >>> from examples.fresh_water.metric_schema import WaterMetricSchema
    >>> env = WaterRegulatedEnv(
    ...     world=None,
    ...     mechanism_id=0,
    ...     seed=0,
    ...     agents_cfg_dict={},
    ...     schema=WaterMetricSchema,
    ... )
    >>> type(env.lake).__name__, env.max_farm_area_m2
    ('SurrogateLake', 1000000.0)
    """

    def __init__(
        self,
        *,
        ecology_cfg: Optional[dict[str, Any]] = None,
        hydrology: str = "surrogate",
        raven_cwd: Any = None,
        raven_cmd: Any = None,
        raven_work_dir: Any = None,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        ecology = {**DEFAULT_ECOLOGY, **(ecology_cfg or {})}
        self.max_farm_area_m2 = float(ecology["max_farm_area_m2"])
        self.capacity_m3 = float(ecology["lake_area_m2"]) * float(
            ecology["max_depth_m"]
        )
        self.max_daily_need_m3_day = (
            len(self.followers) * self.max_farm_area_m2 * PEAK_CROP_NEED_MM_DAY / 1000.0
        )
        lake_args = dict(
            full_stage_m=float(ecology["full_stage_m"]),
            max_depth_m=float(ecology["max_depth_m"]),
        )

        if hydrology == "surrogate":
            extra = {key: ecology[key] for key in SURROGATE_KEYS if key in ecology}
            self.lake: LakeModel = SurrogateLake(
                lake_area_m2=float(ecology["lake_area_m2"]), **lake_args, **extra
            )
        elif hydrology == "raven":
            if raven_cwd is None or raven_cmd is None:
                raise ValueError(
                    "hydrology='raven' needs raven_cwd (the model directory) and "
                    + "raven_cmd (the executable)."
                )
            self.lake = RavenLake(
                raven_cwd=raven_cwd,
                raven_cmd=raven_cmd,
                key=f"m_{self.mechanism_id}_seed_{self.seed}",
                work_dir=raven_work_dir,
                **lake_args,
            )
        else:
            raise ValueError(
                f"hydrology must be 'surrogate' or 'raven', got {hydrology!r}."
            )

        self._planting_date = RAVEN_ORIGIN
        self._date = RAVEN_ORIGIN
        self._crop: Optional[CropDemand] = None
        self._reading: Optional[LakeReading] = None

    def _state_entries(
        self, reading: LakeReading, crop: CropDemand
    ) -> dict[str, float]:
        """Numbers of the shared state that describe a day."""
        level_norm = self.lake.level_norm(reading.stage_m)
        release_pressure = float(
            np.clip(reading.outflow_m3s / max(EPS, reading.inflow_m3s), 0.0, 1.0)
        )
        return {
            "reservoir_level_norm": level_norm,
            "release_pressure": release_pressure,
            "full_required_m3_day": crop.full_required_m3_day,
            "crop_water_need_m3_day": crop.crop_water_need_m3_day,
            "precip_water_m3_day": crop.precip_water_m3_day,
        }

    def _demand(self, date: datetime, reading: LakeReading) -> CropDemand:
        days_after_planting = (date - self._planting_date).days
        return crop_demand(
            date, days_after_planting, reading.precip_mm_day, self.max_farm_area_m2
        )

    @reset
    def reset_water(self, mdp: MDPState) -> MDPState:
        """Draw the planting day, start the lake and expose the first day.

        The planting day is drawn uniformly from ``PLANTING_DAY_OF_YEAR`` with
        the environment's seeded generator, counting from January 1 of 1980, and
        the episode starts that day. The lake model then returns the initial
        state, with no withdrawal so far.

        Parameters
        ----------
        mdp : MDPState
            State being reset; only used to satisfy the hook signature.

        Returns
        -------
        MDPState
            A residual with ``state``: ``reservoir_level_norm``,
            ``release_pressure``, ``full_required_m3_day``,
            ``crop_water_need_m3_day``, ``precip_water_m3_day``,
            ``usage_m3_day`` (zero) and ``usage_norm`` (zero).

        Examples
        --------
        >>> from core.mechanism.base import MDPState
        >>> from examples.fresh_water.metric_schema import WaterMetricSchema
        >>> env = WaterRegulatedEnv(
        ...     world=None,
        ...     mechanism_id=0,
        ...     seed=0,
        ...     agents_cfg_dict={},
        ...     schema=WaterMetricSchema,
        ... )
        >>> residual = env.reset_water(MDPState())
        >>> residual.state["usage_m3_day"]
        [0.0]
        >>> 121 <= (env._planting_date - env._planting_date.replace(
        ...     month=1, day=1)).days + 1 <= 274
        True
        """
        low, high = PLANTING_DAY_OF_YEAR
        planting_day_of_year = int(self.rng.integers(low=low, high=high))
        self._planting_date = RAVEN_ORIGIN + timedelta(days=planting_day_of_year)
        self._date = self._planting_date

        reading = self.lake.start(self._date, self.rng)
        self._reading = reading
        self._crop = self._demand(self._date, reading)
        entries = self._state_entries(reading, self._crop)
        return MDPState(state={**entries, "usage_m3_day": 0.0, "usage_norm": 0.0})

    def _rules(self, mdp: MDPState) -> np.ndarray:
        """Return the rules in force: the leader's decoded action or the defaults."""
        for leader_id in self.lids:
            history = mdp.actions.data.get(leader_id, {}).get(WATER_POLICY_ID)
            if history:
                return np.asarray(history[min(mdp.t, len(history) - 1)], np.float64)
        return DEFAULT_RULES

    @transition
    def step_water(self, mdp: MDPState) -> MDPState:
        """Deliver the water, pay the crop satisfaction and advance one day.

        Every farm's decoded request is turned into a volume
        ``request * full_required``; the delivered volume is the smaller of the
        request and the quota of the rules in force at the current level, and
        the crop satisfaction is ``min(1, (delivered + rain water) / need)``, or
        1 when the crop needs no water. The satisfaction is added to the farm's
        reward at this step, on top of the penalty the policy has already
        recorded. The total delivered volume, as a flow, is withdrawn from the
        lake for the day and the lake returns the next day's reading.

        The method pushes the crop quantities of the day played and the lake and
        weather quantities of the day reached to the metric logger, as listed in
        :class:`examples.fresh_water.metric_schema.WaterMetricSchema`.

        Parameters
        ----------
        mdp : MDPState
            Shared state with the farms' ``irrigate`` actions at ``mdp.t`` and
            the day's entries written by the reset or the previous step.

        Returns
        -------
        MDPState
            The state advanced by one day, with the next day's entries appended
            and the delivered volume as ``usage_m3_day`` and, divided by
            ``max_daily_need_m3_day``, as ``usage_norm``.

        Examples
        --------
        One farm requests everything. With no leader, the default rules apply
        and the quota is large when the reservoir is nearly full, so the whole
        request is delivered and the satisfaction is the share of the crop need
        that irrigation and rain cover:

        >>> from core.mechanism.base import MDPState
        >>> from examples.fresh_water.metric_schema import WaterMetricSchema
        >>> farm = UtilizerConfig(
        ...     id="utilizer",
        ...     policy_id="utilizer_policy",
        ...     mechanisms=(
        ...         IrrigateConfig(id="irrigate", action_space=IRRIGATE_SPACE),
        ...     ),
        ... )
        >>> env = WaterRegulatedEnv(
        ...     world=None,
        ...     mechanism_id=0,
        ...     seed=0,
        ...     agents_cfg_dict={"utilizer:0": farm},
        ...     schema=WaterMetricSchema,
        ... )
        >>> mdp = MDPState().add(env.reset_water(MDPState()))
        >>> mdp = mdp.add(MDPState(actions={"utilizer:0": {"irrigate": [1.0]}}))
        >>> after = env.step_water(mdp=mdp)
        >>> after.t
        1
        >>> 0.0 <= after.rewards["utilizer:0"][0] <= 1.0
        True
        """
        t = mdp.t
        rules = self._rules(mdp)
        level_norm = float(mdp.state["reservoir_level_norm"][t])
        release_pressure = float(mdp.state["release_pressure"][t])
        full_required = float(mdp.state["full_required_m3_day"][t])
        need = float(mdp.state["crop_water_need_m3_day"][t])
        precip_water = float(mdp.state["precip_water_m3_day"][t])

        satisfaction: dict[str, float] = {}
        signals = []
        for aid in self.followers:
            signal = violation_signal(
                irrigation_fraction(mdp.actions[aid][IRRIGATE_ID][t]),
                full_required_m3_day=full_required,
                level_norm=level_norm,
                release_pressure=release_pressure,
                rules=rules,
            )
            signals.append(signal)
            if need <= EPS:
                satisfaction[aid] = 1.0
            else:
                satisfaction[aid] = min(
                    1.0, (signal.delivered_m3_day + precip_water) / need
                )

        total_usage_m3_day = sum(s.delivered_m3_day for s in signals)
        total_usage_m3s = total_usage_m3_day / SECONDS_PER_DAY
        crop = self._crop
        self._log_played_day(
            crop, satisfaction, signals, level_norm, rules, total_usage_m3s
        )

        self._date = self._date + timedelta(days=1)
        reading = self.lake.advance(self._date, total_usage_m3s, self.rng)
        self._reading = reading
        self._crop = self._demand(self._date, reading)
        entries = self._state_entries(reading, self._crop)
        self._log_reached_day(reading, entries)

        mdp = mdp.add(MDPState(rewards=satisfaction))
        usage_norm = total_usage_m3_day / max(EPS, self.max_daily_need_m3_day)
        return mdp.advance(
            state={
                **entries,
                "usage_m3_day": total_usage_m3_day,
                "usage_norm": usage_norm,
            }
        )

    def _log_played_day(
        self,
        crop: CropDemand,
        satisfaction: dict[str, float],
        signals: list,
        level_norm: float,
        rules: np.ndarray,
        total_usage_m3s: float,
    ) -> None:
        """Push the quantities of the day that was just played."""
        if self.logger is None:
            return

        def mean(values: list[float]) -> float:
            return float(np.mean(values)) if values else 0.0

        push = self.logger.push
        push(key=("total_usage_m3s",), value=total_usage_m3s)
        push(key=("crop_kc",), value=crop.kc)
        push(key=("eto_mm_day",), value=crop.eto_mm_day)
        push(key=("etcrop_mm_day",), value=crop.etcrop_mm_day)
        push(key=("deficit_mm_day",), value=crop.deficit_mm_day)
        push(key=("full_required_m3_day",), value=crop.full_required_m3_day)
        push(key=("crop_satisfaction",), value=mean(list(satisfaction.values())))
        push(
            key=("requested_m3_day",), value=mean([s.requested_m3_day for s in signals])
        )
        push(key=("allowed_m3_day",), value=mean([s.allowed_m3_day for s in signals]))
        push(
            key=("delivered_m3_day",), value=mean([s.delivered_m3_day for s in signals])
        )
        push(
            key=("quota_violation_m3",),
            value=mean([s.quota_violation_m3_day for s in signals]),
        )
        push(key=("requested_frac",), value=mean([s.requested_frac for s in signals]))
        push(key=("quota_penalty",), value=mean([s.quota_penalty for s in signals]))
        push(key=("flow_penalty",), value=mean([s.flow_penalty for s in signals]))
        push(key=("quota_stress",), value=quota_stress(level_norm, float(rules[0])))
        push(key=("min_demand_frac",), value=float(rules[1]))
        push(key=("max_demand_frac",), value=float(rules[2]))

    def _log_reached_day(self, reading: LakeReading, entries: dict[str, float]) -> None:
        """Push the lake and weather quantities of the day that was reached."""
        if self.logger is None:
            return

        push = self.logger.push
        level_norm = entries["reservoir_level_norm"]
        push(key=("reservoir_stage_m",), value=reading.stage_m)
        push(key=("reservoir_level_norm",), value=level_norm)
        push(key=("reservoir_level_norm_min",), value=level_norm)
        push(key=("reservoir_level_norm_series",), value=level_norm)
        push(
            key=("baseline_reservoir_level_norm_series",),
            value=self.lake.level_norm(reading.baseline_stage_m),
        )
        push(key=("streamflow_m3s",), value=reading.inflow_m3s)
        push(key=("outflow_m3s",), value=reading.outflow_m3s)
        push(key=("precip_mm_day",), value=reading.precip_mm_day)
        push(key=("temp_c",), value=estimate_temp_c(self._date))
        push(key=("release_pressure",), value=entries["release_pressure"])
        push(
            key=("residence_time_days",),
            value=residence_time_days(
                level_norm * self.capacity_m3, reading.outflow_m3s
            ),
        )
        push(key=("streamflow_m3s_series",), value=reading.inflow_m3s)
        push(key=("baseline_streamflow_m3s_series",), value=reading.baseline_inflow_m3s)
        push(key=("outflow_m3s_series",), value=reading.outflow_m3s)
        push(key=("baseline_outflow_m3s_series",), value=reading.baseline_outflow_m3s)
        for name, value in reading.gauges.items():
            push(key=(name,), value=value)
