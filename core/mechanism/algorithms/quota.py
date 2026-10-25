"""Smooth quota capping the action that the targeted agents request.

The regulator searches a quota ``q`` in ``[0, 1]``. With ``b`` the resource
level normalised by the carrying capacity, the mechanism computes the fraction
of the effort that the quota allows,

    allowed = (s((b - q) / w) - s(-q / w)) / (s((1 - q) / w) - s(-q / w)),

where ``s`` is the logistic function and ``w`` the quota transition width. The
request of a targeted agent is read as an effort fraction ``f`` in ``[0, 1]``:
the Ray adapter decodes the raw policy output before the regulator acts, so the
mechanism works directly in fraction space. For every targeted agent it returns
three residuals. The action residual is ``-excess``, where the excess is a
smooth hinge of ``f - allowed`` of width ``usage_transition_width``, so the
delivered fraction is ``f`` minus the excess. The reward residual is a fine,
``-min(1, FINE_AMOUNT * violation + risk)``, where the violation is a second
smooth hinge of ``f - allowed`` of width ``violation_transition_width`` and the
risk term is currently switched off (its scale is zero). The observation
residual moves the allowed fraction into slot 1 and the quota into slot 3 of the
agent's observation, so that the followers can see what the regulator allows.
At reset, :meth:`QuotaMechanism.on_reset` writes the same two slots together
with the fixed parameters of the fine into the slots 4 to 7.

The action cut is a transitional device: the code marks it with a TODO, and the
intention is to remove it and let the reward penalty alone discourage requests
above the allowance. Until then the delivered action and the reward both depend
on the excess.

The cap is a modelling choice of the fishery benchmark, a logistic smoothing
that keeps the delivered action differentiable in the quota and the stock. It is
not taken from a published model.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from core.annotations import override
from core.mechanism.base import ActType, MDPState, Mechanism
from core.mechanism.config import MechanismConfig
from core.utils import sigmoid, smooth_positive_zero_at_origin

RISK_PENALTY_SCALE = 0.0
RISK_PENALTY_POWER = 1.0
FINE_AMOUNT = 0.2
MAX_DEMAND_FRAC = 1.0
EPS = 1e-8


# TODO what if two mechanisms interfere by requiring context from each other ?
# for example a penalty based on how much quota is violated ?


class QuotaMechanism(Mechanism):
    """Cap the effort requested by the targeted agents with a smooth quota.

    The quota is the regulator's action, read in ``[0, 1]``. It is compared with
    the normalised resource level to give the fraction of effort allowed (see the
    module docstring for the formula), and every targeted agent whose request
    exceeds it is cut back. The request is the effort fraction ``f`` in
    ``[0, 1]`` that the adapter has already decoded from the policy output; the
    delivered effort is ``f`` minus the smooth excess over the allowance. When
    the stock is empty the allowance is ``0``, and when it is full the allowance
    is ``1``. The mechanism returns residuals, so nothing is modified in place:
    a cut on the request, a fine on the reward, and a correction of slots 1 and
    3 of the observation of each targeted agent. The agent must hold an
    observation, which the mechanism reads to build the correction.

    Parameters
    ----------
    quota_transition_width : float
        Width ``w`` of the logistic transition of the quota, positive, in
        normalised resource units (fraction of the carrying capacity). Default
        ``0.03``.
    usage_transition_width : float
        Width of the smooth excess over the allowance that is cut from the
        request, positive, in fraction of the effort. Default ``0.005``.
    violation_transition_width : float
        Width of the smooth violation that the fine is proportional to,
        positive, in fraction of the effort. Default ``0.03``.
    **kwargs
        Forwarded to :class:`core.mechanism.base.Mechanism`: ``action_space``
        (a ``Box`` of shape ``(1,)`` in ``[0, 1]``), ``acts_on`` and ``obs_map``
        are expected. ``obs_map`` must map ``"resource_level"`` to the state key
        of the resource, which is divided by ``mdp.params["K"]``.

    Raises
    ------
    ValueError
        If one of the widths is not positive.

    Notes
    -----
    When to use: the regulator should limit what the agents can take according
    to the state of the resource, in contrast with :class:`IncentiveMechanism`,
    which prices the action, or :class:`ThresholdPenaltyMechanism`, which
    sanctions every agent alike. The constructor of the mechanism does not check
    ``acts_on`` and ``obs_map``; ``apply`` raises a ``ValueError`` when they are
    missing.

    Two quotas held by the same regulator and aimed at the same target compose
    additively, like every residual of :meth:`core.agents.base.Agent.action`.
    Each quota reads the same request and returns the full cut and the full fine
    for its own allowance, so the cuts are summed, the fines are summed, and the
    delivered effort falls below what the stricter quota alone would deliver.
    The observation residuals are summed as well: each is the difference
    between its own allowance and the same current observation, so with two
    quotas slots 1 and 3 no longer show either of them. To cap a target by the
    stricter of two rules, compute that rule in one mechanism. The mechanism
    writes no state entry; the allowed fraction reaches the agents only through
    their observation.

    The residual action is a float in fraction space, and the action cut is
    meant to be replaced by the reward penalty later (see the TODO in
    :meth:`apply`).

    Examples
    --------
    >>> import numpy as np
    >>> from gymnasium import spaces
    >>> from core.mechanism.base import MDPState
    >>> mechanism = Quota(
    ...     id="quota",
    ...     action_space=spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
    ...     acts_on=("fisherman", "harvest"),
    ...     obs_map={"resource_level": "fish"},
    ... ).build("regulator")
    >>> mdp = MDPState(
    ...     params={"K": 1000.0},
    ...     state={"fish": 500.0},
    ...     actions={
    ...         "regulator": {"quota": 0.5},
    ...         "fisherman:0": {"harvest": 0.9},  # decoded request
    ...     },
    ...     obs={"fisherman:0": np.zeros(9, dtype=np.float32)},
    ... )
    >>> delta = mechanism(mdp, 0.5)
    >>> cut = delta.actions["fisherman:0"]["harvest"][0]
    >>> round(0.9 + cut, 3)  # half the stock: about half the effort
    0.503
    >>> round(delta.rewards["fisherman:0"][0], 3)  # the fine on the excess
    -0.076
    >>> delta.obs["fisherman:0"][0][[1, 3]]  # allowance and quota, as residual
    array([0.5, 0.5], dtype=float32)
    """

    # Fixed algorithmic parameters.
    quota_transition_width: float = 0.03
    usage_transition_width: float = 0.005
    violation_transition_width: float = 0.03

    def __init__(
        self,
        *,
        quota_transition_width: float = 0.03,
        usage_transition_width: float = 0.005,
        violation_transition_width: float = 0.03,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.quota_transition_width = quota_transition_width
        self.usage_transition_width = usage_transition_width
        self.violation_transition_width = violation_transition_width

        if self.quota_transition_width <= 0:
            raise ValueError("quota_transition_width must be > 0.")

        if self.usage_transition_width <= 0:
            raise ValueError("usage_transition_width must be > 0.")

        if self.violation_transition_width <= 0:
            raise ValueError("violation_transition_width must be > 0.")

    @override(Mechanism)
    def decode(self, mdp: MDPState, action: ActType) -> ActType:
        """Decode the regulator action into a quota parameter in [0, 1].

        The first component of ``action`` is clipped to ``[0, 1]``. The map is
        idempotent: a quota already in ``[0, 1]`` is returned unchanged.

        Parameters
        ----------
        mdp : MDPState
            Shared state; not used.
        action : ActType
            Raw action of the regulator, array-like with at least one element.

        Returns
        -------
        float
            Quota in ``[0, 1]``.
        """

        value = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])

        return float(np.clip(value, 0.0, 1.0))

    @override(Mechanism)
    def on_reset(self, mdp: MDPState, action: ActType) -> MDPState:
        """Write the quota and the fine parameters into the targeted observations.

        The "legacy attributes" of the observation are the slots 1 (allowed
        fraction), 3 (fixed quota), 4 (``MAX_DEMAND_FRAC``), 5 (``FINE_AMOUNT``),
        6 (``RISK_PENALTY_SCALE``) and 7 (risk penalty power, set to ``0``). The
        residual moves each of them from the value the agent's observation
        already holds at ``mdp.t`` to its target, and leaves the other slots
        alone. Observations are per-step flows, so a step the agent has not
        written yet holds ``0`` and the residual is the target itself, whatever
        the previous step held.

        Parameters
        ----------
        mdp : MDPState
            Shared state at initialization; the resource is read at ``mdp.t``.
        action : float or None
            Decoded quota in ``[0, 1]``, or ``None`` when the regulator has no
            candidate yet.

        Returns
        -------
        MDPState
            Residual holding an observation of size 9 for every agent whose
            identifier starts with the targeted type and a colon, or an empty
            residual when ``action`` is ``None``.

        Raises
        ------
        ValueError
            If ``acts_on`` is missing, or if ``obs_map`` does not map
            ``"resource_level"``, once a quota is given.
        """

        if action is None:
            return MDPState()

        if self.acts_on is None:
            raise ValueError("QuotaMechanism requires `acts_on`.")

        if self.obs_map is None or "resource_level" not in self.obs_map:
            raise ValueError("QuotaMechanism requires obs_map['resource_level'].")

        resource_level = mdp.state[self.obs_map["resource_level"]][mdp.t] / max(
            mdp.params["K"], EPS
        )
        width = max(self.quota_transition_width, EPS)
        lower = sigmoid((0.0 - action) / width)
        upper = sigmoid((1.0 - action) / width)
        current = sigmoid((resource_level - action) / width)
        allowed_frac = (current - lower) / max(upper - lower, EPS)

        # TODO (nadine) Allowed frac should be added to value AFTER transition
        # TODO (nadine) temp legacy attributes
        target = np.asarray(
            [
                0.0,  # fish_norm
                allowed_frac,  # effective_quota
                0.0,  # tot_usage_norm
                action,  # fixed_quota
                MAX_DEMAND_FRAC,
                FINE_AMOUNT,
                RISK_PENALTY_SCALE,
                0.0,  # risk_penalty_power
                0.0,  # restoration_subsidy
            ],
            dtype=np.float32,
        )
        indices = [1, 3, 4, 5, 6, 7]
        obs = {}

        for aid in mdp.aids:
            if not str(aid).startswith(f"{self.acts_on[0]}:"):
                continue

            history = mdp.obs.data.get(aid, [])
            existing = (
                np.asarray(history[-1], dtype=np.float32)
                if history
                else np.zeros_like(target)
            )
            delta = np.zeros_like(target)
            delta[indices] = target[indices] - existing[indices]
            obs[aid] = delta

        return MDPState(obs=obs)

    @override(Mechanism)
    def apply(self, mdp: MDPState, action: ActType) -> MDPState:
        """Return the cut, the fine and the observation of each targeted agent.

        Parameters
        ----------
        mdp : MDPState
            Shared state; the resource is read at ``mdp.t``, and so are the
            decoded requests and the observations of the targeted agents.
        action : float
            Quota in ``[0, 1]`` returned by :meth:`decode`.

        Returns
        -------
        MDPState
            Residual with, for each targeted agent that holds an action for the
            targeted mechanism, three entries. In ``actions``, the float
            ``-excess``, the cut in effort fraction that moves the request to
            the delivered effort. In ``rewards``, the float
            ``-min(1, FINE_AMOUNT * violation + risk)``, which is ``0`` when
            the request is at or below the allowance. In ``obs``, a ``float32``
            array of the size of the current observation, zero except for slot
            1, which moves to the allowed fraction, and slot 3, which moves to
            the quota. No state entry is written.

        Raises
        ------
        ValueError
            If ``acts_on`` is missing, or if ``obs_map`` does not map
            ``"resource_level"``.

        Notes
        -----
        A targeted agent that holds no observation at ``mdp.t`` yet is treated
        as observing zeros, as in :meth:`on_reset`, so its observation residual
        moves slots 1 and 3 from zero to the allowed fraction and the quota.
        """

        if self.acts_on is None:
            raise ValueError("QuotaMechanism requires `acts_on`.")

        if self.obs_map is None or "resource_level" not in self.obs_map:
            raise ValueError("QuotaMechanism requires obs_map['resource_level'].")

        target_agent, target_mechanism = self.acts_on
        resource_level = mdp.state[self.obs_map["resource_level"]][mdp.t] / max(
            mdp.params["K"], EPS
        )
        width = max(self.quota_transition_width, EPS)
        lower = sigmoid((0.0 - action) / width)
        upper = sigmoid((1.0 - action) / width)
        current = sigmoid((resource_level - action) / width)
        allowed_frac = (current - lower) / max(upper - lower, EPS)
        da = {}
        do = {}
        dr = {}

        for aid, a in (mdp.actions.data or {}).items():
            if not str(aid).startswith(f"{target_agent}:"):
                continue

            if target_mechanism not in a:
                continue

            harvest = float(np.asarray(a[target_mechanism][mdp.t]).item())
            e = smooth_positive_zero_at_origin(
                harvest - allowed_frac, self.usage_transition_width
            )
            da[aid] = {target_mechanism: -float(e)}  # TODO (nadine) apply dr not da

            # quota-violation penalty
            violation_frac = smooth_positive_zero_at_origin(
                harvest - allowed_frac, self.violation_transition_width
            )
            quota_penalty = min(1.0, FINE_AMOUNT * violation_frac)
            delivered_frac = float(np.clip(harvest - e, 0.0, 1.0))
            risk_penalty = (
                RISK_PENALTY_SCALE
                * max(0.0, 1.0 - resource_level)
                * delivered_frac**RISK_PENALTY_POWER
            )
            dr[aid] = -min(1.0, quota_penalty + risk_penalty)

            # Observation residual - Assuming obs the same for all target agents
            current_obs = np.asarray(mdp.obs[aid][mdp.t], dtype=np.float32)
            o = np.zeros_like(current_obs)
            o[1] = allowed_frac - current_obs[1]
            o[3] = action - current_obs[3]
            do[aid] = o

        return MDPState(actions=da, obs=do, rewards=dr)


@dataclass(frozen=True, kw_only=True)
class Quota(MechanismConfig):
    """Configuration of a :class:`QuotaMechanism`.

    Attributes
    ----------
    quota_transition_width : float
        Width of the logistic transition of the quota, in fraction of the
        carrying capacity. Default ``0.03``.
    usage_transition_width : float
        Width of the smooth excess over the allowance that is cut from the
        request, in fraction of the effort. Default ``0.005``.
    violation_transition_width : float
        Width of the smooth violation that the fine is proportional to, in
        fraction of the effort. Default ``0.03``.

    Notes
    -----
    The other fields (``action_space``, ``id``, ``acts_on``, ``obs_map``) are
    those of :class:`~core.mechanism.config.MechanismConfig`; ``action_space``
    has no default and must be given.

    When to use: in the ``mechanisms`` of the regulator's
    :class:`~core.agents.base.AgentConfig`, to cap what the followers take.

    Examples
    --------
    >>> import numpy as np
    >>> from gymnasium import spaces
    >>> config = Quota(
    ...     id="quota",
    ...     action_space=spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
    ...     quota_transition_width=0.05,
    ... )
    >>> config.build("regulator").quota_transition_width
    0.05
    """

    mechanism_cls: ClassVar[type[Mechanism]] = QuotaMechanism
    quota_transition_width: float = 0.03
    usage_transition_width: float = 0.005
    violation_transition_width: float = 0.03
