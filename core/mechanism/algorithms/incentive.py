"""Reward-based incentive mechanism for targeted agent actions.

For a targeted agent ``i`` with decoded action ``e_i`` in ``(0, 1)``
and incentive rate ``alpha``, the mechanism contributes

    delta_r_i = alpha * e_i

to the agent's reward at the current step.

The rate is selected by the regulator. Positive rates reward the targeted
behavior; the current implementation supports only non-negative rates.
The regulator's action space determines the permitted rate range.

The mechanism modifies rewards without changing the targeted action or
the environment's transition dynamics. Intrinsic action costs and physical
effects remain the responsibility of the targeted agent and its mechanisms.

For example, in the fishery benchmark, restoration costs are charged by
the fisher's Restore mechanism, while the incentive rewards restoration
effort. The linear incentive is a modelling choice rather than a
requirement of a specific published economic model.
"""

from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from core.annotations import override
from core.mechanism.base import ActType, MDPState, Mechanism
from core.mechanism.config import MechanismConfig

"""Nominal maximum incentive rate, in reward units per unit of effort.

This constant is not enforced directly by the mechanism; the configured
action space should impose the intended upper bound.
"""

# TODO (nadine) temp val / max_subsidy for migration set to 0.5 later
MAX = 0.5


class IncentiveMechanism(Mechanism):
    """Add a linear reward incentive for targeted agent actions.

    Parameters
    ----------
    acts_on : tuple[AgentID, MechanismID]
        Agent type and mechanism whose action receives the incentive,
        for example ``("fisherman", "restore")``. Every agent whose
        identifier starts with ``"<agent>:"`` and has an action for
        the specified mechanism is targeted.
    **kwargs
        Forwarded to :class:`core.mechanism.base.Mechanism`.
        The regulator's ``action_space`` is a one-dimensional ``Box``
        specifying the allowed non-negative incentive rate.

    Notes
    -----
    The mechanism returns an ``MDPState`` containing only reward
    contributions. It does not compute intrinsic action costs, constrain
    actions or modify the physical environment.

    Targeted actions are interpreted as raw policy outputs and decoded
    using ``sigmoid(z / ACTION_TEMPERATURE)``. This assumes the targeted
    actions have not already been decoded or modified.

    During reset, the mechanism can expose its incentive rate to
    targeted agents through their observations.

    Examples
    --------
    >>> import numpy as np
    >>> from gymnasium import spaces
    >>> from core.mechanism.base import MDPState
    >>> mechanism = Incentive(
    ...     id="incentive",
    ...     action_space=spaces.Box(0.0, 0.5, shape=(1,), dtype=np.float32),
    ...     acts_on=("fisherman", "restore"),
    ... ).build("regulator")
    >>> mdp = MDPState(
    ...     actions={
    ...         "regulator": {"incentive": np.array([0.5])},
    ...         "fisherman:0": {"restore": np.array([0.0])},
    ...     }
    ... )
    >>> delta = mechanism(mdp, np.array([0.5]))
    >>> delta.rewards["fisherman:0"]  # 0.5 * sigmoid(0)
    [0.25]
    """

    @override(Mechanism)
    def decode(self, mdp: MDPState, action: ActType) -> ActType:
        """Decode the regulator's action into a non-negative incentive rate.

        The first component of the regulator's action is extracted and
        clipped to ``[0, 1]``. No additional scaling is applied.

        Parameters
        ----------
        mdp : MDPState
            Shared MDP state; unused.
        action : ActType
            Regulator action, array-like with at least one element.

        Returns
        -------
        float
            Incentive rate clipped to ``[0, 1]``.
        """

        value = float(np.asarray(action, dtype=np.float32).reshape(-1)[0])

        return float(np.clip(value, 0.0, 1.0))

    @override(Mechanism)
    def on_reset(self, mdp: MDPState, action: ActType) -> MDPState:
        """Expose the initial incentive rate to targeted agents.

        The current implementation uses a temporary nine-component
        observation layout inherited from the legacy fishery benchmark.
        The incentive rate occupies the final component; all preceding
        components are initialized to zero.

        Parameters
        ----------
        mdp : MDPState
            Shared state containing the identifiers of participating
            agents in ``mdp.aids``.
        action : ActType
            Initial incentive rate, or ``None`` if none is assigned.

        Returns
        -------
        MDPState
            Observation residual for targeted agents, or an empty
            residual when no initial action is provided.

        Raises
        ------
        ValueError
            If ``acts_on`` is missing and an action is provided.
        """

        if action is None:
            return MDPState()

        if self.acts_on is None:
            raise ValueError("QuotaMechanism requires `acts_on`.")

        # TODO (nadine) temp legacy attributes
        obs = {}

        for aid in mdp.aids:
            if not str(aid).startswith(f"{self.acts_on[0]}:"):
                continue

            history = mdp.obs.data.get(aid, [])
            current = (
                np.asarray(history[-1], dtype=np.float32)
                if history
                else np.zeros(9, dtype=np.float32)
            )
            delta = np.zeros(9, dtype=np.float32)
            delta[8] = np.float32(action) - current[8]
            obs[aid] = delta

        return MDPState(obs=obs)

    @override(Mechanism)
    def apply(self, mdp: MDPState, action: ActType) -> MDPState:
        """Compute the incentive reward for each targeted agent.

        For each targeted agent ``i``, the raw action ``z_i`` of the
        specified mechanism is decoded into an effort fraction:

            e_i = sigmoid(z_i / ACTION_TEMPERATURE)

        The reward contribution is then:

            delta_r_i = alpha * e_i

        where ``alpha`` is the regulator's decoded incentive rate.

        Parameters
        ----------
        mdp : MDPState
            Shared state containing targeted agents' action trajectories.
            Actions are read at the current timestep ``mdp.t``.
        action : float
            Incentive rate returned by :meth:`decode`.

        Returns
        -------
        MDPState
            Residual containing only ``rewards``, with one additive reward
            contribution per targeted agent. Agents without the specified
            action are ignored.

        Raises
        ------
        ValueError
            If ``acts_on`` is missing.
        """

        if self.acts_on is None:
            raise ValueError("IncentiveMechanism requires `acts_on`.")

        target_agent, target_mechanism = self.acts_on
        dr = {}

        for aid, a in (mdp.actions.data or {}).items():
            if not str(aid).startswith(f"{target_agent}:"):
                continue

            if target_mechanism not in a:
                continue

            effort = float(np.asarray(a[target_mechanism][mdp.t]).item())
            dr[aid] = float(action) * effort * MAX

        return MDPState(rewards=dr)


@dataclass(frozen=True, kw_only=True)
class Incentive(MechanismConfig):
    """Configuration for :class:`IncentiveMechanism`.

    Defines a reward-based intervention through which a regulator
    assigns a non-negative incentive rate to a targeted agent action.

    Attributes
    ----------
    mechanism_cls : ClassVar[type[Mechanism]]
        Mechanism implementation, set to :class:`IncentiveMechanism`.

    Notes
    -----
    The fields ``id``, ``action_space``, ``acts_on``, ``obs_map`` and
    ``default`` are inherited from
    :class:`~core.mechanism.config.MechanismConfig`.

    ``action_space`` should be a one-dimensional ``Box`` with
    non-negative bounds. Its upper bound determines the maximum
    incentive rate available to the regulator.

    The configuration does not define an intrinsic effort cost.
    Such costs belong to the targeted agent's utility.

    Examples
    --------
    >>> import numpy as np
    >>> from gymnasium import spaces
    >>> config = Incentive(
    ...     id="restoration_subsidy",
    ...     action_space=spaces.Box(
    ...         0.0, 0.5, shape=(1,), dtype=np.float32
    ...     ),
    ...     acts_on=("fisherman", "restore"),
    ... )
    >>> type(config.build("regulator")).__name__
    'IncentiveMechanism'
    """

    mechanism_cls: ClassVar[type[Mechanism]] = IncentiveMechanism
