"""The regulator's water policy: a demand quota and two penalties, eight numbers.

The regulator of the fresh-water example searches one vector ``u`` in
``[0, 1]^8``. :func:`decode_rules` maps it to eight physical *rules*, in the
order of ``RULE_NAMES``:

- ``fixed_quota``: ``0.6 + 0.35 u`` (default 0.85);
- ``min_demand_frac``: ``0.35 u`` (default 0.05);
- ``max_demand_frac``: ``0.35 + 0.65 u`` (default 1.0);
- ``fine_amount``: ``0.1 u`` (default 0.05);
- ``risk_penalty_scale``: ``u`` (default 0.5);
- ``risk_penalty_power``: ``1 + 4 u`` (default 2.0);
- ``under_irrigation_penalty_scale``: ``u`` (default 0.25);
- ``max_farm_area_m2``: ``1e5 + u (2e7 - 1e5)`` (default 500000).

``max_demand_frac`` is raised to ``min_demand_frac`` when the draw puts it
below, so that the quota curve stays monotonic. The rules
``under_irrigation_penalty_scale`` and ``max_farm_area_m2`` are searched but
have no effect in the dynamics, exactly as in the first version of the example,
which read the farm area from its own configuration and never used the
under-irrigation penalty.

With ``L`` the filled fraction of the reservoir and ``q`` the fixed quota, the
policy allows each farm the fraction of its crop water deficit

    allowed = min_demand + s * (max_demand - min_demand),
    s = clip((L - q) / (1 - q), 0, 1),

so that a reservoir at or below ``q`` leaves only the minimum and a full one
leaves the maximum. A farm that requests the fraction ``f`` of its deficit and
asks for more than it is allowed pays a fine, and a farm that requests a large
fraction while the river depends on the release pays a risk penalty:

    quota_penalty = min(1, fine_amount * excess / deficit),
    flow_penalty = risk_penalty_scale * release_pressure * f ** risk_penalty_power,
    penalty = min(1, quota_penalty + flow_penalty),

where ``excess`` is the requested volume above the allowed one and
``release_pressure`` the share of the inflow that the reservoir releases. The
reward of a farm is its crop satisfaction minus this penalty. These rules are
the design of the example and are not taken from a published model.
"""

from typing import ClassVar, NamedTuple

import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType

from core.mechanism.base import MDPState, Mechanism
from core.mechanism.config import MechanismConfig

EPS = 1e-8

RULE_NAMES = (
    "fixed_quota",
    "min_demand_frac",
    "max_demand_frac",
    "fine_amount",
    "risk_penalty_scale",
    "risk_penalty_power",
    "under_irrigation_penalty_scale",
    "max_farm_area_m2",
)
"""Names of the eight rules, in the order of the searched vector."""

DEFAULT_RULES = np.asarray([0.85, 0.05, 1.0, 0.05, 0.5, 2.0, 0.25, 500_000.0])
"""Rules in force when no candidate has reached the environment yet."""

OBSERVATION_SIZE = 4 + len(RULE_NAMES)
"""Length of a farm's observation: four state entries and the normalized rules."""

WATER_POLICY_ID = "water_policy"
"""Mechanism identifier of the policy, the key of the regulator's action."""

IRRIGATE_ID = "irrigate"
"""Mechanism identifier of a farm's irrigation request."""

RULES_SPACE = spaces.Box(low=0.0, high=1.0, shape=(len(RULE_NAMES),), dtype=np.float32)
"""Search space of the policy: the eight normalized rules in ``[0, 1]``."""

_FARM_AREA_RANGE = (100_000.0, 20_000_000.0)


def decode_rules(u: np.ndarray | list[float]) -> np.ndarray:
    """Map the normalized search vector to the eight physical rules.

    Parameters
    ----------
    u : array-like, shape (8,)
        Normalized rules; values are clipped to ``[0, 1]`` first.

    Returns
    -------
    numpy.ndarray, shape (8,)
        The rules of ``RULE_NAMES``, as ``float64``.

    Raises
    ------
    ValueError
        If ``u`` does not hold eight values.

    When to use: to read what a regulator candidate means; the mechanism and the
    environment call it for you.

    Examples
    --------
    >>> rules = decode_rules(np.full(8, 0.5))
    >>> dict(zip(RULE_NAMES, rules.round(4).tolist()))["fixed_quota"]
    0.775
    >>> float(decode_rules(np.zeros(8))[2])  # at least the minimum
    0.35
    """
    u = np.clip(np.asarray(u, dtype=np.float64).reshape(-1), 0.0, 1.0)
    if u.size != len(RULE_NAMES):
        raise ValueError(f"Expected {len(RULE_NAMES)} normalized rules, got {u.size}.")

    min_demand = 0.35 * u[1]
    area_low, area_high = _FARM_AREA_RANGE
    return np.asarray(
        [
            0.6 + 0.35 * u[0],
            min_demand,
            max(0.35 + 0.65 * u[2], min_demand),
            0.10 * u[3],
            u[4],
            1.0 + 4.0 * u[5],
            u[6],
            area_low + u[7] * (area_high - area_low),
        ]
    )


def encode_rules(rules: np.ndarray | list[float]) -> np.ndarray:
    """Map physical rules back to the normalized search vector.

    Parameters
    ----------
    rules : array-like, shape (8,)
        The rules of ``RULE_NAMES``.

    Returns
    -------
    numpy.ndarray, shape (8,)
        The normalized vector, as ``float32``. It is the inverse of
        :func:`decode_rules` except where ``max_demand_frac`` was raised to the
        minimum.

    When to use: to turn the default rules into a starting candidate.

    Examples
    --------
    >>> encode_rules(DEFAULT_RULES).astype(float).round(4).tolist()
    [0.7143, 0.1429, 1.0, 0.5, 0.5, 0.25, 0.25, 0.0201]
    """
    r = np.asarray(rules, dtype=np.float64).reshape(-1)
    area_low, area_high = _FARM_AREA_RANGE
    return np.asarray(
        [
            (r[0] - 0.6) / 0.35,
            r[1] / 0.35,
            (r[2] - 0.35) / 0.65,
            r[3] / 0.10,
            r[4],
            (r[5] - 1.0) / 4.0,
            r[6],
            (r[7] - area_low) / (area_high - area_low),
        ],
        dtype=np.float32,
    )


def quota_stress(level_norm: float, fixed_quota: float) -> float:
    """Return how far the reservoir is above the protected level.

    Parameters
    ----------
    level_norm : float
        Filled fraction of the reservoir.
    fixed_quota : float
        Protected fraction ``q`` of the policy.

    Returns
    -------
    float
        ``clip((L - q) / (1 - q), 0, 1)``: ``0`` at or below the protected level
        and ``1`` for a full reservoir.

    When to use: it is the weight that interpolates the allowed demand.

    Examples
    --------
    >>> round(quota_stress(0.9, 0.85), 4)
    0.3333
    >>> quota_stress(0.5, 0.85)
    0.0
    """
    return float(
        np.clip((level_norm - fixed_quota) / max(EPS, 1.0 - fixed_quota), 0.0, 1.0)
    )


def allowed_fraction(level_norm: float, rules: np.ndarray) -> float:
    """Return the share of its crop deficit a farm may take.

    Parameters
    ----------
    level_norm : float
        Filled fraction of the reservoir.
    rules : numpy.ndarray, shape (8,)
        Decoded rules.

    Returns
    -------
    float
        ``min_demand + stress * (max_demand - min_demand)``.

    When to use: for the quota the policy enforces at the current level; the
    observation and the penalty use it.

    Examples
    --------
    >>> round(allowed_fraction(0.9, DEFAULT_RULES), 4)
    0.3667
    >>> allowed_fraction(1.0, DEFAULT_RULES)
    1.0
    """
    stress = quota_stress(level_norm, float(rules[0]))
    return float(rules[1] + stress * (rules[2] - rules[1]))


def irrigation_fraction(action: ActType) -> float:
    """Read a farm's irrigation request as a fraction in ``[0, 1]``.

    Parameters
    ----------
    action : ActType
        Policy output: a float or an array whose first element is read.

    Returns
    -------
    float
        The first element, clipped to ``[0, 1]``.

    When to use: wherever the raw irrigation action is turned into a request.

    Examples
    --------
    >>> irrigation_fraction(np.asarray([1.4], dtype=np.float32))
    1.0
    >>> irrigation_fraction(0.25)
    0.25
    """
    return float(np.clip(np.asarray(action, dtype=np.float64).reshape(-1)[0], 0.0, 1.0))


class Violation(NamedTuple):
    """Outcome of a request measured against the policy.

    Attributes
    ----------
    requested_m3_day : float
        Volume requested (cubic metres per day).
    allowed_m3_day : float
        Volume the quota allows (cubic metres per day).
    delivered_m3_day : float
        ``min(requested, allowed)``.
    quota_violation_m3_day : float
        Requested volume above the allowed one.
    requested_frac : float
        Request as a fraction of the crop deficit.
    quota_penalty, flow_penalty, total_penalty : float
        The two penalties and their capped sum, in reward units.

    When to use: it is what :func:`violation_signal` returns.

    Examples
    --------
    >>> Violation(1.0, 2.0, 1.0, 0.0, 0.5, 0.0, 0.1, 0.1).total_penalty
    0.1
    """

    requested_m3_day: float
    allowed_m3_day: float
    delivered_m3_day: float
    quota_violation_m3_day: float
    requested_frac: float
    quota_penalty: float
    flow_penalty: float
    total_penalty: float


def violation_signal(
    fraction: float,
    *,
    full_required_m3_day: float,
    level_norm: float,
    release_pressure: float,
    rules: np.ndarray,
) -> Violation:
    """Measure one farm's request against the quota and the release risk.

    Parameters
    ----------
    fraction : float
        Requested share of the crop deficit, in ``[0, 1]``.
    full_required_m3_day : float
        Water the whole farm area needs to cover the deficit (cubic metres per
        day).
    level_norm : float
        Filled fraction of the reservoir.
    release_pressure : float
        Share of the inflow the reservoir releases, in ``[0, 1]``.
    rules : numpy.ndarray, shape (8,)
        Decoded rules.

    Returns
    -------
    Violation
        Volumes, the request fraction and the penalties.

    When to use: to price a request; the policy mechanism calls it for the
    reward and the environment calls it to log the same quantities.

    Examples
    --------
    >>> v = violation_signal(
    ...     0.8,
    ...     full_required_m3_day=1000.0,
    ...     level_norm=0.9,
    ...     release_pressure=0.5,
    ...     rules=DEFAULT_RULES,
    ... )
    >>> round(v.allowed_m3_day, 3), round(v.quota_penalty, 6), round(v.flow_penalty, 3)
    (366.667, 0.021667, 0.16)
    """
    requested = fraction * full_required_m3_day
    allowed = allowed_fraction(level_norm, rules) * full_required_m3_day
    delivered = min(requested, allowed)
    violation = max(0.0, requested - allowed)
    quota_penalty = min(
        1.0, float(rules[3]) * (violation / max(EPS, full_required_m3_day))
    )
    requested_frac = requested / max(EPS, full_required_m3_day)
    flow_penalty = (
        float(rules[4]) * release_pressure * requested_frac ** float(rules[5])
    )
    return Violation(
        requested_m3_day=requested,
        allowed_m3_day=allowed,
        delivered_m3_day=delivered,
        quota_violation_m3_day=violation,
        requested_frac=requested_frac,
        quota_penalty=quota_penalty,
        flow_penalty=flow_penalty,
        total_penalty=min(1.0, quota_penalty + flow_penalty),
    )


class WaterPolicy(Mechanism):
    """Leader mechanism: prices the farms' requests and tells them the rules.

    The mechanism is the regulator's action. ``decode`` turns the searched
    vector into the physical rules. ``apply`` runs before the farms' actions are
    played: it reads the request each targeted farm has recorded for the current
    step (clipped to ``[0, 1]``), measures it with :func:`violation_signal` and
    returns the negative of the total penalty as the farm's reward. The crop
    satisfaction is added by the environment's transition. ``observe`` fills the
    entries of the farms' observations that the farm leaves at zero: the
    allowed fraction at the current level (entry 2) and the normalized rules
    (entries 4 to 11).

    The mechanism needs ``acts_on=("utilizer", "irrigate")`` (the agent type and
    the mechanism it targets).

    When to use: as the only mechanism of the regulator agent of the fresh-water
    experiment, through ``WaterPolicyConfig``.

    Examples
    --------
    >>> policy = WaterPolicyConfig(
    ...     id=WATER_POLICY_ID,
    ...     action_space=RULES_SPACE,
    ...     acts_on=("utilizer", IRRIGATE_ID),
    ... ).build("regulator")
    >>> mdp = MDPState(
    ...     state={
    ...         "reservoir_level_norm": 0.9,
    ...         "release_pressure": 0.5,
    ...         "full_required_m3_day": 1000.0,
    ...     },
    ...     actions={"utilizer:0": {IRRIGATE_ID: 0.8}},
    ... )
    >>> round(policy.apply(mdp, DEFAULT_RULES).rewards["utilizer:0"][0], 6)
    -0.181667
    """

    def decode(self, mdp: MDPState, action: ActType) -> np.ndarray:
        """Turn the searched vector into the eight physical rules.

        Parameters
        ----------
        mdp : MDPState
            Unused.
        action : ActType
            Normalized rules, eight values.

        Returns
        -------
        numpy.ndarray
            The rules, see :func:`decode_rules`.
        """
        return decode_rules(action)

    def apply(self, mdp: MDPState, action: ActType) -> MDPState:
        """Return the penalty each targeted farm receives at this step.

        Parameters
        ----------
        mdp : MDPState
            Shared state: ``reservoir_level_norm``, ``release_pressure`` and
            ``full_required_m3_day`` at ``mdp.t`` in ``state``, and the farms'
            requests in ``actions``.
        action : ActType
            Decoded rules, as returned by :meth:`decode`.

        Returns
        -------
        MDPState
            A residual whose ``rewards`` hold, for every targeted farm that has
            a request, minus its total penalty.

        Raises
        ------
        ValueError
            If ``acts_on`` is missing.
        """
        if self.acts_on is None:
            raise ValueError("WaterPolicy requires `acts_on`.")

        target_agent, target_mechanism = self.acts_on
        t = mdp.t
        level_norm = float(mdp.state["reservoir_level_norm"][t])
        release_pressure = float(mdp.state["release_pressure"][t])
        full_required = float(mdp.state["full_required_m3_day"][t])
        rules = np.asarray(action, dtype=np.float64)

        rewards = {}
        for aid, actions in (mdp.actions.data or {}).items():
            if not str(aid).startswith(f"{target_agent}:"):
                continue
            if target_mechanism not in actions:
                continue
            violation = violation_signal(
                irrigation_fraction(actions[target_mechanism][t]),
                full_required_m3_day=full_required,
                level_norm=level_norm,
                release_pressure=release_pressure,
                rules=rules,
            )
            rewards[aid] = -violation.total_penalty
        return MDPState(rewards=rewards)

    def observe(self, mdp: MDPState) -> MDPState:
        """Return the policy's contribution to the farms' observations.

        Parameters
        ----------
        mdp : MDPState
            Shared state after the transition or the reset: the agents in
            ``aids``, the level at ``mdp.t`` in ``state`` and the candidate in
            ``raw_actions``.

        Returns
        -------
        MDPState
            A residual whose ``obs`` hold, for every targeted farm, a
            ``float32`` vector of ``OBSERVATION_SIZE`` entries: the allowed
            fraction at index 2 and the normalized rules at indices 4 to 11,
            zero elsewhere. Empty when no candidate is held.

        Raises
        ------
        ValueError
            If ``acts_on`` is missing.
        """
        if self.acts_on is None:
            raise ValueError("WaterPolicy requires `acts_on`.")

        history = mdp.raw_actions.data.get(self.aid, {}).get(self.id)
        if not history:
            return MDPState()

        # The candidate is carried forward: the last recorded value is in force.
        normalized = np.clip(
            np.asarray(history[min(mdp.t, len(history) - 1)], dtype=np.float64),
            0.0,
            1.0,
        )
        rules = self.decode(mdp, normalized)
        level_norm = float(mdp.state["reservoir_level_norm"][mdp.t])
        contribution = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
        contribution[2] = allowed_fraction(level_norm, rules)
        contribution[4:] = normalized

        target_agent = self.acts_on[0]
        farms = sorted(
            aid for aid in mdp.aids if str(aid).startswith(f"{target_agent}:")
        )
        return MDPState(obs={aid: contribution.copy() for aid in farms})


class WaterPolicyConfig(MechanismConfig):
    """Configuration that builds the ``WaterPolicy`` mechanism.

    A frozen dataclass inherited from
    :class:`core.mechanism.config.MechanismConfig`; only ``mechanism_cls``
    changes.

    Attributes
    ----------
    action_space : gymnasium.spaces.Box
        ``RULES_SPACE``, eight coordinates in ``[0, 1]``.
    id : str or None
        Mechanism identifier, ``WATER_POLICY_ID``. The ES optimizer names the
        searched parameters ``water_policy[0]`` to ``water_policy[7]``, in the
        order of ``RULE_NAMES``.
    acts_on : tuple[str, str] or None
        ``("utilizer", "irrigate")``: the farms and the request it prices.
    obs_map : dict[str, str] or None
        Unused (default ``None``).
    default : numpy.ndarray or None
        Unused (default ``None``).

    When to use: in the ``mechanisms`` of the regulator ``AgentConfig`` of the ES
    optimizer.

    Examples
    --------
    >>> config = WaterPolicyConfig(
    ...     id=WATER_POLICY_ID,
    ...     action_space=RULES_SPACE,
    ...     acts_on=("utilizer", IRRIGATE_ID),
    ... )
    >>> type(config.build("regulator")).__name__
    'WaterPolicy'
    """

    mechanism_cls: ClassVar[type[Mechanism]] = WaterPolicy
