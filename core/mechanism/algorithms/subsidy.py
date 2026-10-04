"""Effort subsidy acting on the reward of the targeted agents.

For a targeted agent ``i`` with effort ``e_i`` in ``(0, 1)``, subsidy rate
``sigma`` and quadratic effort cost ``c``, the mechanism adds

    delta_r_i = sigma * e_i - c * e_i**2

to the agent's reward at the current step. The rate is the regulator's action,
searched in ``[0, 1]`` and scaled by ``MAX_SUBSIDY``; the cost is a fixed
parameter of the benchmark. The ecological effect of the effort (for the
fishery, the fish that restoration adds) belongs to the targeted agent's own
mechanism, not to this one.

The functional form is a modelling choice of the fishery benchmark, a linear
payment against a convex effort cost; it is not taken from a published model.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from core.annotations import override
from core.mechanism.base import ActType, MDPState, Mechanism
from core.mechanism.config import MechanismConfig
from core.utils import sigmoid

MAX_SUBSIDY = 0.5
"""Subsidy rate, in reward units per unit of effort, of a regulator action of 1."""

ACTION_TEMPERATURE = 4.0
"""Temperature of the map from a raw policy output ``z`` to an effort in ``(0, 1)``.

The regulator acts before the targeted agents, so the effort it reads has not
been decoded yet. This value must equal the one the targeted mechanism uses in
its own ``decode`` (``Fishing`` and ``Restore`` in the fishery, and
``QuotaMechanism.apply``), otherwise the subsidised effort is not the delivered
one.
"""


class SubsidyMechanism(Mechanism):
    """Pay the targeted agents for their effort, net of a quadratic cost.

    Parameters
    ----------
    cost : float
        Quadratic effort cost ``c`` in ``[0, 1]``, in reward units per squared
        unit of effort. Fixed; it is not searched by the regulator.
    acts_on : tuple[AgentID, MechanismID]
        Agent type and mechanism whose action is the subsidised effort, for
        example ``("fisherman", "restore")``. Every agent whose identifier
        starts with ``"<agent>:"`` and that holds an action for that mechanism
        is subsidised.
    **kwargs
        Forwarded to :class:`core.mechanism.base.Mechanism`. ``action_space``
        is a ``Box`` of shape ``(1,)`` in ``[0, 1]``: the normalised rate.

    Raises
    ------
    ValueError
        If ``cost`` is outside ``[0, 1]``.

    Notes
    -----
    When to use: the regulator should make an action more attractive without
    constraining it, in contrast with :class:`QuotaMechanism`, which changes
    the action itself. The effort is the one the agent requested at the current
    step; a mechanism of the same regulator that modifies that action is not
    seen, because all mechanisms of one agent read the same input state.

    Examples
    --------
    >>> import numpy as np
    >>> from gymnasium import spaces
    >>> from core.mechanism.base import MDPState
    >>> mechanism = Subsidy(
    ...     id="subsidy",
    ...     action_space=spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
    ...     acts_on=("fisherman", "restore"),
    ...     cost=0.0,
    ... ).build("regulator")
    >>> mdp = MDPState(
    ...     actions={
    ...         "regulator": {"subsidy": np.array([1.0])},
    ...         "fisherman:0": {"restore": np.array([0.0])},
    ...     }
    ... )
    >>> delta = mechanism(mdp, np.array([1.0]))
    >>> delta.rewards["fisherman:0"]  # 0.5 * sigmoid(0)
    [0.25]
    """

    def __init__(self, *, cost: float, **kwargs) -> None:
        super().__init__(**kwargs)
        self.cost = cost

        if not 0.0 <= self.cost <= 1.0:
            raise ValueError(f"cost must be in [0, 1], got {self.cost}.")

    @override(Mechanism)
    def decode(self, mdp: MDPState, action: ActType) -> float:
        """Decode the regulator action into a normalised rate in [0, 1].

        The scaling by ``MAX_SUBSIDY`` is left to :meth:`apply`: the decoded
        action is written back into the trajectory and carried forward to the
        next step, where it is decoded again, so this map must be idempotent.
        """
        value = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])
        return float(np.clip(value, 0.0, 1.0))

    @override(Mechanism)
    def apply(self, mdp: MDPState, action: ActType) -> MDPState:
        """Return ``sigma * e_i - c * e_i**2`` as a reward residual per target.

        Parameters
        ----------
        mdp : MDPState
            Shared state; the raw actions of the targeted agents are read at
            ``mdp.t``.
        action : float
            Normalised rate in ``[0, 1]`` returned by :meth:`decode`.

        Returns
        -------
        MDPState
            Residual holding only ``rewards``, one float per targeted agent.
        """
        if self.acts_on is None:
            raise ValueError("SubsidyMechanism requires `acts_on`.")

        target_agent, target_mechanism = self.acts_on
        subsidy = action * MAX_SUBSIDY

        dr = {}

        for aid, a in (mdp.actions.data or {}).items():
            if not str(aid).startswith(f"{target_agent}:"):
                continue

            if target_mechanism not in a:
                continue

            requested = np.asarray(a[target_mechanism][mdp.t], dtype=np.float32)
            z = float(requested.reshape(-1)[0])
            effort = sigmoid(z / ACTION_TEMPERATURE)
            dr[aid] = subsidy * effort - self.cost * effort**2

        return MDPState(rewards=dr)


@dataclass(frozen=True, kw_only=True)
class Subsidy(MechanismConfig):
    mechanism_cls: ClassVar[type[Mechanism]] = SubsidyMechanism
    cost: float
