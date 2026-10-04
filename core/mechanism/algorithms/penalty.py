"""Smooth threshold penalty acting on the reward of the targeted agents.

When the normalised resource level ``b`` falls below ``threshold``, every
targeted agent loses up to ``penalty_amount`` at the current step:

    delta_r_i = -penalty_amount / (1 + exp((b - threshold) / transition_width))

The penalty is a fixed regulatory rule: its three parameters are set in the
configuration and the regulator searches nothing, so the action space is empty.
The logistic transition is a smoothing choice that keeps the reward continuous
in the stock; it is not taken from a published model.
"""

from dataclasses import dataclass, field
from typing import ClassVar

import numpy as np
from gymnasium import Space

from core.annotations import override
from core.mechanism.base import ActType, MDPState, Mechanism
from core.mechanism.config import MechanismConfig, empty_action_space

EPS = 1e-8

MAX_EXPONENT = 60.0
"""Bound on the logistic exponent; ``exp(60)`` is finite in double precision."""


class ThresholdPenaltyMechanism(Mechanism):
    """Penalise the targeted agents while the resource is below a threshold.

    Parameters
    ----------
    threshold : float
        Normalised resource level in ``[0, 1]`` at which half of the penalty
        applies.
    penalty_amount : float
        Maximal reward deduction per step, non-negative, in reward units.
    transition_width : float
        Width of the logistic transition, positive, in normalised resource
        units.
    acts_on : tuple[AgentID, MechanismID]
        Only the agent type is used: every agent whose identifier starts with
        ``"<agent>:"`` and that holds an action at the current step is
        penalised, whatever the mechanism named.
    obs_map : dict[str, str]
        Must map ``"resource_level"`` to the state key of the resource, which
        is divided by ``mdp.params["K"]``.
    **kwargs
        Forwarded to :class:`core.mechanism.base.Mechanism`.

    Raises
    ------
    ValueError
        If ``threshold`` is outside ``[0, 1]``, ``penalty_amount`` is negative
        or ``transition_width`` is not positive.

    Notes
    -----
    When to use: a collective sanction tied to the state of the resource,
    identical for all targeted agents whatever their own action. To act on the
    individual action, use :class:`QuotaMechanism`.

    Examples
    --------
    >>> import numpy as np
    >>> from core.mechanism.base import MDPState
    >>> mechanism = ThresholdPenalty(
    ...     id="penalty",
    ...     acts_on=("fisherman", "harvest"),
    ...     obs_map={"resource_level": "fish"},
    ...     threshold=0.2,
    ...     penalty_amount=0.1,
    ... ).build("regulator")
    >>> mdp = MDPState(
    ...     params={"K": 1000.0},
    ...     state={"fish": 200.0},
    ...     actions={
    ...         "regulator": {"penalty": np.empty(0)},
    ...         "fisherman:0": {"harvest": np.array([0.0])},
    ...     },
    ... )
    >>> delta = mechanism(mdp, np.empty(0))
    >>> delta.rewards["fisherman:0"]
    [-0.05]
    """

    def __init__(
        self,
        *,
        threshold: float = 0.20,
        penalty_amount: float = 0.10,
        transition_width: float = 0.03,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.threshold = threshold
        self.penalty_amount = penalty_amount
        self.transition_width = transition_width

        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1].")
        if self.penalty_amount < 0.0:
            raise ValueError("penalty_amount must be non-negative.")
        if self.transition_width <= 0.0:
            raise ValueError("transition_width must be positive.")

    def penalty(self, resource_level: float) -> float:
        """Reward deduction, non-negative, for a normalised resource level."""
        z = np.clip(
            (float(resource_level) - self.threshold) / self.transition_width,
            -MAX_EXPONENT,
            MAX_EXPONENT,
        )
        return float(self.penalty_amount / (1.0 + np.exp(z)))

    @override(Mechanism)
    def apply(self, mdp: MDPState, action: ActType) -> MDPState:
        """Return the same negative reward residual for every targeted agent.

        Parameters
        ----------
        mdp : MDPState
            Shared state; the resource is read at ``mdp.t``.
        action : numpy.ndarray
            Empty array, shape ``(0,)``: the rule has no searched parameter.

        Returns
        -------
        MDPState
            Residual holding only ``rewards``, one float per targeted agent.
        """
        if self.acts_on is None:
            raise ValueError("ThresholdPenaltyMechanism requires `acts_on`.")

        if self.obs_map is None or "resource_level" not in self.obs_map:
            raise ValueError(
                "ThresholdPenaltyMechanism requires obs_map['resource_level']."
            )

        target_agent, _ = self.acts_on
        resource_level = mdp.state[self.obs_map["resource_level"]][mdp.t] / max(
            mdp.params["K"], EPS
        )
        penalty = self.penalty(resource_level)

        dr = {}

        for aid in mdp.actions.data or {}:
            if not str(aid).startswith(f"{target_agent}:"):
                continue

            dr[aid] = -penalty

        return MDPState(rewards=dr)


@dataclass(frozen=True, kw_only=True)
class ThresholdPenalty(MechanismConfig):
    mechanism_cls: ClassVar[type[Mechanism]] = ThresholdPenaltyMechanism
    action_space: Space = field(default_factory=empty_action_space)
    threshold: float = 0.20
    penalty_amount: float = 0.10
    transition_width: float = 0.03
