"""Shared state of an episode and the interface of a mechanism.

A mechanism is the regulatory intervention that the outer optimizer searches
over and that the inner environment applies to its agents. This module defines
:class:`MDPState`, the container for the state, the observations, the actions
and the rewards of an episode, and :class:`Mechanism`, the abstract class every
mechanism inherits. A mechanism does not change the state it receives: it
returns a residual :class:`MDPState`, which the owning agent adds to the shared
state, and may return a second residual for the observations through
:meth:`Mechanism.observe`. The time-indexed trees that hold the fields live in
:mod:`core.mechanism.types`, and the configurations that build the concrete
mechanisms in :mod:`core.mechanism.config`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any, Optional, SupportsFloat, TypeVar

import numpy as np
from gymnasium import spaces

from core.mechanism.types import FlowTrajectory, Trajectory
from core.types import AgentID, MechanismID
from core.utils import intersect, logical_or_dict

StateType = TypeVar("StateType")
ActType = TypeVar("ActType")
ObsType = TypeVar("ObsType")


@dataclass
class MDPState:
    """Snapshot of an episode: parameters, time and the four trajectories.

    The state, the observations, the actions and the rewards are
    :class:`~core.mechanism.types.Trajectory` trees indexed by timestep. A
    ``dict`` given for one of them is converted: the state and the actions are
    stocks (an unwritten step keeps the previous value), while the rewards and
    the observations are flows (an unwritten step starts from zero). The
    environment and the agents build residual states that hold only the fields
    they change, and :meth:`add` composes them into the shared state.

    Attributes
    ----------
    t : int
        Current timestep, starting at ``0``. Default ``0``.
    params : dict[str, Any]
        Fixed parameters of the environment, for example the carrying capacity
        ``"K"`` in fish units. Default empty.
    aids : set
        Identifiers of the agents present. Default empty.
    state : Trajectory or dict
        Environment state, for example the fish stock. Default empty.
    obs : FlowTrajectory or dict
        Observation of each agent, one array per step. Default empty.
    actions : Trajectory or dict
        Action of each agent and mechanism, as ``actions[agent][mechanism]``.
        Default empty.
    rewards : FlowTrajectory or dict
        Reward of each agent at each step. Default empty.
    state_space, action_spaces, obs_space : gymnasium.spaces.Dict or None
        Spaces of the state, the actions and the observations. ``None`` means
        that the field says nothing about the space.
    terminateds, truncateds : dict[AgentID, bool] or None
        Termination and truncation flags per agent. ``None`` means that the
        field says nothing about them.

    Notes
    -----
    When to use: as the argument and the return value of everything that takes
    part in a transition, namely the environment, the agents and the
    mechanisms. Build a partial one to describe a residual.

    Examples
    --------
    >>> mdp = MDPState(state={"fish": 100.0}, rewards={"fisherman:0": 1.0})
    >>> residual = MDPState(rewards={"fisherman:0": -0.25})
    >>> mdp.add(residual).rewards["fisherman:0"]
    [0.75]
    >>> mdp.advance(state={"fish": 90.0}).state["fish"]
    [100.0, 90.0]
    """

    t: int = 0
    params: dict[str, StateType] = field(default_factory=dict)
    aids: set[AgentID] = field(default_factory=set)

    # assumes tau additive
    state: Trajectory[StateType] | dict = field(default_factory=Trajectory)
    obs: FlowTrajectory[ObsType] | dict = field(default_factory=FlowTrajectory)
    actions: Trajectory[ActType] | dict = field(default_factory=Trajectory)
    rewards: FlowTrajectory[SupportsFloat] | dict = field(
        default_factory=FlowTrajectory
    )
    state_space: Optional[spaces.Dict] = None
    action_spaces: Optional[spaces.Dict] = None
    obs_space: Optional[spaces.Dict] = None
    terminateds: Optional[dict[AgentID, bool]] = None
    truncateds: Optional[dict[AgentID, bool]] = None

    def __post_init__(self) -> None:
        for name in ("state", "obs", "actions", "rewards"):
            value = getattr(self, name)

            if isinstance(value, dict):
                # Rewards and observations are rebuilt at every step; the
                # other fields are stocks that carry forward.
                cls = FlowTrajectory if name in ("rewards", "obs") else Trajectory
                setattr(self, name, cls(value))

    # Mechanisms may introduce dimensions, but absence means "no change".
    # Deletion is not supported.
    def add(self, ds: list["MDPState"]) -> MDPState:
        """Compose this state with residual states, without modifying it.

        The trajectories of the residuals are added to this state's at the
        current step ``t``: values are summed, and a field a residual does not
        mention is left unchanged. ``params`` are merged, a later residual
        overriding an earlier key, ``aids`` are united, the spaces are
        intersected, and the termination and truncation flags are combined with
        a logical or. The time ``t`` is kept.

        Parameters
        ----------
        ds : list of MDPState
            Residual states. A single ``MDPState`` is also accepted.

        Returns
        -------
        MDPState
            New state of the same class as ``self``.
        """
        if isinstance(ds, MDPState):
            ds = [ds]

        params = self.params.copy()

        for d in ds:
            params |= d.params

        return type(self)(
            t=self.t,
            aids=set().union(self.aids or set(), *(d.aids or set() for d in ds)),
            params=params,
            state=self.state.add(self.t, [d.state for d in ds]),
            actions=self.actions.add(self.t, [d.actions for d in ds]),
            obs=self.obs.add(self.t, [d.obs for d in ds]),
            rewards=self.rewards.add(self.t, [d.rewards for d in ds]),
            state_space=intersect(self.state_space, [d.state_space for d in ds]),
            obs_space=intersect(self.obs_space, [d.obs_space for d in ds]),
            action_spaces=intersect(self.action_spaces, [d.action_spaces for d in ds]),
            terminateds=logical_or_dict(self.terminateds, [d.terminateds for d in ds]),
            truncateds=logical_or_dict(self.truncateds, [d.truncateds for d in ds]),
        )

    def advance(
        self,
        *,
        state: Optional[dict[Any, Any]] = None,
        obs: Optional[dict[Any, Any]] = None,
        actions: Optional[dict[Any, Any]] = None,
        rewards: Optional[dict[Any, Any]] = None,
    ) -> "MDPState":
        """Return a copy one timestep later, with the given values appended.

        Each trajectory that is given receives one more step holding the
        values; a trajectory left as ``None`` is not extended. This state is
        not modified.

        Parameters
        ----------
        state, obs, actions, rewards : dict or None
            Values of the new step for the matching trajectory, as trees keyed
            by agent or state name. Default ``None``.

        Returns
        -------
        MDPState
            New state with ``t`` increased by one.
        """

        return replace(
            self,
            t=self.t + 1,
            state=(self.state.append(state) if state is not None else self.state),
            obs=(self.obs.append(obs) if obs is not None else self.obs),
            actions=(
                self.actions.append(actions) if actions is not None else self.actions
            ),
            rewards=(
                self.rewards.append(rewards) if rewards is not None else self.rewards
            ),
        )

    def update(
        self,
        t: Optional[int] = None,
        *,
        state: Optional[dict[Any, Any]] = None,
        obs: Optional[dict[Any, Any]] = None,
        actions: Optional[dict[Any, Any]] = None,
        rewards: Optional[dict[Any, Any]] = None,
    ) -> "MDPState":
        """Write values at one timestep, in place, and return the state.

        Unlike :meth:`add` and :meth:`advance`, this method modifies the state
        it is called on.

        Parameters
        ----------
        t : int or None
            Timestep to write. Default ``None``, the current step ``self.t``.
        state, obs, actions, rewards : dict or None
            Values to write in the matching trajectory, which is overwritten at
            that step. A trajectory left as ``None`` is not touched.

        Returns
        -------
        MDPState
            This state, after the update.
        """
        t = self.t if t is None else t

        if state is not None:
            self.state = self.state.update(t, state)
        if obs is not None:
            self.obs = self.obs.update(t, obs)
        if actions is not None:
            self.actions = self.actions.update(t, actions)
        if rewards is not None:
            self.rewards = self.rewards.update(t, rewards)
        return self


class Mechanism(ABC):
    """Abstract regulatory mechanism, owned by an agent.

    A mechanism turns the regulator's action into a residual
    :class:`MDPState`, which the owning agent adds to the shared state. It does
    so in two steps: :meth:`decode` maps the raw action to the coordinates of
    the mechanism, and :meth:`apply` returns the residual of a transition. The
    environment calls :meth:`observe` separately, for the mechanisms of the
    leaders, to let a mechanism contribute to the observations. Subclasses
    implement :meth:`apply`, and override :meth:`decode` and :meth:`observe`
    when needed. They are usually built from a
    :class:`~core.mechanism.config.MechanismConfig`.

    Parameters
    ----------
    aid : AgentID or None
        Identifier of the agent that owns the mechanism, typically the
        regulator. Default ``None``; the configuration passes it to
        :meth:`~core.mechanism.config.MechanismConfig.build`.
    action_space : gymnasium.spaces.Box
        Space of the mechanism's action, for example ``Box(0, 1, (1,))`` for a
        quota in ``[0, 1]``. Keyword-only.
    id : MechanismID or None
        Identifier of the mechanism, the key of its action under ``aid``.
        Default ``None``.
    acts_on : tuple[AgentID, MechanismID]
        Agent type and mechanism that the mechanism acts on. Keyword-only.
    obs_map : dict[str, str] or None
        Names under which the mechanism reads the environment state, for
        example ``{"resource_level": "fish"}``. Default ``None``.
    default : numpy.ndarray or None
        Default action, kept unchanged as the private attribute ``_u``, whatever
        its size or value. Default ``None``.

    Attributes
    ----------
    aid, id, action_space, acts_on, obs_map
        The constructor arguments of the same names.
    mechanism_id : MechanismID or None
        Same value as ``id``.

    Notes
    -----
    When to use: as the base class of a new regulatory mechanism. Use the
    ready-made ones (:class:`~core.mechanism.algorithms.quota.QuotaMechanism`,
    :class:`~core.mechanism.algorithms.subsidy.SubsidyMechanism`,
    :class:`~core.mechanism.algorithms.penalty.ThresholdPenaltyMechanism`,
    :class:`~core.mechanism.algorithms.social_influence.SocialInfluenceMechanism`)
    when they fit.

    Examples
    --------
    >>> class Tax(Mechanism):
    ...     def apply(self, mdp, action):
    ...         return MDPState(rewards={"fisherman:0": -action / 2})
    >>> mechanism = Tax(
    ...     aid="regulator",
    ...     id="tax",
    ...     action_space=spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
    ...     acts_on=("fisherman", "harvest"),
    ... )
    >>> mdp = MDPState(actions={"regulator": {"tax": 0.0}})
    >>> mechanism(mdp, 0.5).rewards["fisherman:0"]
    [-0.25]
    >>> mdp.actions["regulator"]["tax"]
    [0.5]
    """

    def __init__(
        self,
        aid: Optional[AgentID] = None,
        *,
        action_space: spaces.Box,
        id: Optional[MechanismID] = None,
        acts_on: tuple[AgentID, MechanismID],
        obs_map: Optional[dict[str, str]] = None,
        default: Optional[np.ndarray] = None,
    ) -> None:
        self.aid = aid
        self.id = id
        self.action_space = action_space
        self.mechanism_id = id
        self.acts_on = acts_on
        self.obs_map = obs_map
        self._u = default

    def __call__(self, mdp: MDPState, action: ActType) -> MDPState:
        """Decode ``action``, record it in ``mdp`` and return the residual.

        The decoded action is written, in place, at step ``mdp.t`` of
        ``mdp.actions[aid][id]``, which must already exist and be long enough.
        Then :meth:`apply` is called with it.

        Parameters
        ----------
        mdp : MDPState
            Shared state at the current step.
        action : ActType
            Raw action of the regulator for this mechanism.

        Returns
        -------
        MDPState
            Residual of the transition, as returned by :meth:`apply`.
        """
        action = self.decode(mdp, action)
        mdp.actions[self.aid][self.id][mdp.t] = action
        return self.apply(mdp, action)

    def decode(self, mdp: MDPState, action: ActType) -> ActType:
        """Map optimizer/policy coordinates to mechanism coordinates.

        The decoded value is written back into the action trajectory and is
        carried forward to the following steps, where it is decoded again, so
        an override must give the same result when it receives its own output.
        The default is the identity.

        Parameters
        ----------
        mdp : MDPState
            Shared state at the current step.
        action : ActType
            Raw action of the regulator for this mechanism.

        Returns
        -------
        ActType
            Action in the mechanism's own coordinates.
        """
        return action

    @abstractmethod
    def apply(self, mdp: MDPState, action: ActType) -> MDPState:
        """Return this mechanism's contribution to the state transition.

        Parameters
        ----------
        mdp : MDPState
            Shared state at the current step, as every mechanism of the agent
            sees it.
        action : ActType
            Decoded action, as returned by :meth:`decode`.

        Returns
        -------
        MDPState
            Residual holding only the fields the mechanism changes.

        Raises
        ------
        NotImplementedError
            Always, in the base class; the method is abstract.
        """
        raise NotImplementedError

    def observe(self, mdp: MDPState) -> MDPState:
        """Return this mechanism's contribution to the agents' observations.

        ``apply`` runs before the transition, on a step whose observation the
        policies have already consumed, so what it writes into ``obs`` is never
        seen. The environment calls ``observe`` when it builds the next
        observation instead: at reset, and after the transition of each step,
        where ``mdp.t`` is the new step and ``mdp.actions[...][mdp.t - 1]``
        holds the delivered actions of the step just finished. The returned
        ``obs`` entries are summed with the observation each agent built, so a
        mechanism fills entries of the vector that the agent leaves at zero.

        The default contributes nothing.

        Parameters
        ----------
        mdp : MDPState
            Shared state after the transition, or after the reset.

        Returns
        -------
        MDPState
            Residual holding only ``obs``; the default holds nothing.
        """
        return MDPState()
