"""Smooth quota capping the action that the targeted agents request.

The regulator searches a quota ``q`` in ``[0, 1]``. With ``b`` the resource
level normalised by the carrying capacity, the mechanism computes the fraction
of the effort that the quota allows,

    allowed = (s((b - q) / w) - s(-q / w)) / (s((1 - q) / w) - s(-q / w)),

where ``s`` is the logistic function and ``w`` the quota transition width. A
targeted agent that requests the fraction ``f = s(z / 4)`` of its effort, with
``z`` its raw policy output, is delivered ``f`` minus a smooth excess over
``allowed``, and the mechanism returns the residual on ``z`` that moves the
request to that delivered fraction. The regulator acts before the agents, so the
request it reads has not been decoded yet, which is why it works on ``z``.

The cap is a modelling choice of the fishery benchmark, a logistic smoothing
that keeps the delivered action differentiable in the quota and the stock. It is
not taken from a published model.
"""

from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from core.annotations import override
from core.mechanism.base import ActType, MDPState, Mechanism
from core.mechanism.config import MechanismConfig
from core.utils import sigmoid, smooth_positive_zero_at_origin

EPS = 1e-8


class QuotaMechanism(Mechanism):
    """Cap the effort requested by the targeted agents with a smooth quota.

    The quota is the regulator's action, read in ``[0, 1]``. It is compared with
    the normalised resource level to give the fraction of effort allowed (see the
    module docstring for the formula), and every targeted agent whose request
    exceeds it is cut back. The request is the raw policy output ``z`` of the
    agent, mapped to an effort ``f = sigmoid(z / 4)`` in ``(0, 1)``; the delivered
    effort is ``f`` minus the smooth excess over the allowance, floored at
    ``1e-6`` and capped at ``1 - 1e-6`` before it is mapped back to ``z``. When the
    stock is empty the allowance is ``0``, and when it is full the allowance
    is ``1``. The mechanism returns a residual on the first component of the
    request, so the request is not modified in place.

    Parameters
    ----------
    quota_transition_width : float
        Width ``w`` of the logistic transition of the quota, positive, in
        normalised resource units (fraction of the carrying capacity). Default
        ``0.03``.
    usage_transition_width : float
        Width of the smooth excess over the allowance, positive, in fraction of
        the effort. Default ``0.005``.
    violation_transition_width : float
        Positive, default ``0.03``. It is validated and stored, but ``apply``
        does not use it.
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
    to the state of the resource, in contrast with :class:`SubsidyMechanism`,
    which prices the action, or :class:`ThresholdPenaltyMechanism`, which
    sanctions every agent alike. The constructor of the mechanism does not check
    ``acts_on`` and ``obs_map``; ``apply`` raises a ``ValueError`` when they are
    missing.

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
    >>> z = np.array([4.0 * np.log(9.0)], dtype=np.float32)  # requests 0.9
    >>> mdp = MDPState(
    ...     params={"K": 1000.0},
    ...     state={"fish": 500.0},
    ...     actions={
    ...         "regulator": {"quota": np.array([0.5], dtype=np.float32)},
    ...         "fisherman:0": {"harvest": z},
    ...     },
    ... )
    >>> delta = mechanism(mdp, np.array([0.5]))
    >>> cut = delta.actions["fisherman:0"]["harvest"][0]
    >>> delivered = 1.0 / (1.0 + np.exp(-(z + cut) / 4.0))
    >>> round(float(delivered[0]), 3)  # half the stock: about half the effort
    0.503
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
        **kwargs: Any,
    ) -> None:
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
    def decode(self, mdp: MDPState, action: ActType) -> float:
        """Decode the regulator action into a quota parameter in [0, 1].

        The first component of ``action`` is clipped to ``[0, 1]``. The map is
        idempotent, as the decoded action is written back and decoded again at
        the next steps.

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
    def apply(self, mdp: MDPState, action: ActType) -> MDPState:
        """Return the cut that brings each targeted request under the allowance.

        Parameters
        ----------
        mdp : MDPState
            Shared state; the resource is read at ``mdp.t`` and the raw actions
            of the targeted agents are read at ``mdp.t``.
        action : float
            Quota in ``[0, 1]`` returned by :meth:`decode`.

        Returns
        -------
        MDPState
            Residual with, for each targeted agent that holds an action for
            the targeted mechanism, a ``float32`` array of the shape of its
            request, zero except for the first component, which moves the
            request to the delivered effort. It also carries the allowed fraction
            under the state entry ``allowed_frac``: the residual is the
            difference between the allowed fraction and the value that entry
            already holds at ``mdp.t`` (``0`` when absent), so that adding it to
            the state leaves the allowed fraction of the step in the entry.

        Raises
        ------
        ValueError
            If ``acts_on`` is missing, or if ``obs_map`` does not map
            ``"resource_level"``.
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

        for aid, a in (mdp.actions.data or {}).items():
            if not str(aid).startswith(f"{target_agent}:"):
                continue

            if target_mechanism not in a:
                continue

            requested = np.asarray(a[target_mechanism][mdp.t], dtype=np.float32)
            temperature = 4.0
            z = float(requested.reshape(-1)[0])
            requested_frac = sigmoid(z / temperature)

            excess = smooth_positive_zero_at_origin(
                requested_frac - allowed_frac, self.usage_transition_width
            )
            delivered_frac = float(np.clip(requested_frac - excess, 1e-6, 1.0 - 1e-6))
            delivered_z = temperature * np.log(delivered_frac / (1.0 - delivered_frac))
            delivered = requested.copy().reshape(-1)
            delivered[0] = delivered_z
            delivered = delivered.reshape(requested.shape)
            da[aid] = {target_mechanism: delivered - requested}

        # State entries are stocks: a residual is added to the value already in
        # the state. Return the difference, so that the entry ends up holding the
        # allowed fraction of this step instead of a running sum of them.
        history = mdp.state.data.get("allowed_frac")
        previous = history[min(mdp.t, len(history) - 1)] if history else 0.0

        return MDPState(actions=da, state={"allowed_frac": allowed_frac - previous})


@dataclass(frozen=True, kw_only=True)
class Quota(MechanismConfig):
    """Configuration of a :class:`QuotaMechanism`.

    Attributes
    ----------
    quota_transition_width : float
        Width of the logistic transition of the quota, in fraction of the
        carrying capacity. Default ``0.03``.
    usage_transition_width : float
        Width of the smooth excess over the allowance, in fraction of the
        effort. Default ``0.005``.
    violation_transition_width : float
        Stored by the mechanism but not used by it. Default ``0.03``.

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
