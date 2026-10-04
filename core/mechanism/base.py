"""Mechanism interface and its generic vector implementation.

A mechanism is the regulatory intervention the outer optimizer searches over
and the inner environment applies to its agents. This module defines the
:class:`Mechanism` protocol every mechanism satisfies (a normalized vector
representation exposed to the agents, parameter names and a default instance)
and :class:`VectorMechanism`, a parameter-free wrapper around a raw array used
when no semantic mechanism class is available. The geometry of the mechanism
manifold (encoding, decoding, clipping, sampling) lives in
:mod:`core.mechanism.space`.
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
        """Advance the MDP by one timestep."""

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
        """Update trajectory values"""
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
    """Runtime mechanism controller owned by an agent"""

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
        self._u = default if default else None

    def __call__(self, mdp: MDPState, action: ActType) -> MDPState:
        action = self.decode(mdp, action)
        mdp.actions[self.aid][self.id][mdp.t] = action
        return self.apply(mdp, action)

    def decode(self, mdp: MDPState, action: ActType) -> ActType:
        """Map optimizer/policy coordinates to mechanism coordinates."""
        return action

    @abstractmethod
    def apply(self, mdp: MDPState, action: ActType) -> MDPState:
        """Return this mechanism's contribution to the state transition."""
        raise NotImplementedError
