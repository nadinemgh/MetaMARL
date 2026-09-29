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
from dataclasses import dataclass, replace
from typing import Optional, SupportsFloat, TypeVar

import numpy as np
from gymnasium import spaces

from core.utils import add, intersect, logical_or_dict

StateType = TypeVar("StateType")
ActType = TypeVar("ActType")
ObsType = TypeVar("ObsType")

from dataclasses import field
from typing import Generic, TypeVar

from core.types import AgentID, MechanismID

T = TypeVar("T")

# TODO (nadine) move to env


@dataclass
class Trajectory(Generic[T]):
    data: dict[str, list[T]] = field(default_factory=dict)

    def __post_init__(self):
        self.data = {
            key: value if isinstance(value, list) else [value]
            for key, value in self.data.items()
        }

    def __getitem__(self, key: str) -> list[T]:
        return self.data[key]

    def __contains__(self, key: str) -> bool:
        return key in self.data

    def copy(self) -> "Trajectory[T]":
        return type(self)({key: values.copy() for key, values in self.data.items()})

    def add(self, t: int, deltas: list["Trajectory[T]"]) -> "Trajectory[T]":
        result = self.copy()

        for delta in deltas:
            for key, values in delta.data.items():
                value = values[-1]

                if key not in result.data:
                    result.data[key] = [0] * t + [value]
                else:
                    result.data[key][t] += value

        return result

    def append(self, values: dict[str, T]) -> "Trajectory[T]":
        result = self.copy()
        length = max((len(v) for v in result.data.values()), default=0)

        for history in result.data.values():
            history.append(history[-1])

        for key, value in values.items():
            if key in result.data:
                result.data[key][-1] = value
            else:
                result.data[key] = [0] * length + [value]

        return result


@dataclass
class MDPState:
    t: int = 0
    params: dict[str, StateType] = field(default_factory=dict)
    aids: set[AgentID] = field(default_factory=set)
    obs: dict[AgentID, ObsType] =field(default_factory=dict) # TODO trajectory
    # assumes tau additive
    state: Trajectory[StateType] | dict[str, StateType] = field(default_factory=Trajectory)  
    actions: Optional[dict[AgentID, ActType]] = None  # TODO trajectory
    rewards: Optional[dict[AgentID, SupportsFloat]] = None  # TODO trajectory
    state_space: Optional[spaces.Dict] = None
    action_spaces: Optional[spaces.Dict] = None
    obs_space: Optional[spaces.Dict] = None
    terminateds: Optional[dict[AgentID, bool]] = None
    truncateds: Optional[dict[AgentID, bool]] = None

    def __post_init__(self):
        if isinstance(self.state, dict):
            self.state = Trajectory(self.state)

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
            actions=add(self.actions, [d.actions for d in ds]),
            obs=add(self.obs, [d.obs for d in ds]),
            rewards=add(self.rewards, [d.rewards for d in ds]),
            state_space=intersect(self.state_space, [d.state_space for d in ds]),
            obs_space=intersect(self.obs_space, [d.obs_space for d in ds]),
            action_spaces=intersect(self.action_spaces, [d.action_spaces for d in ds]),
            terminateds=logical_or_dict(self.terminateds, [d.terminateds for d in ds]),
            truncateds=logical_or_dict(self.truncateds, [d.truncateds for d in ds]),
        )

    def advance(self, values: dict[str, StateType]) -> "MDPState":
        return replace(self, t=self.t + 1, state=self.state.append(values))


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
        self.action_space = action_space
        self.mechanism_id = id
        self.acts_on = acts_on
        self.obs_map = obs_map
        self._u = default if default else None

    # TODO (nadine) enforce shape 1 action
    def __call__(self, state: MDPState, action: ActType) -> MDPState:
        action = self.decode(action)

        return self.apply(state, action)

    def decode(self, action: ActType) -> ActType:
        """Map optimizer/policy coordinates to mechanism coordinates."""
        return action

    @abstractmethod
    def apply(self, mdp_state: MDPState, action: ActType) -> MDPState:
        """Return this mechanism's contribution to the state transition."""
        raise NotImplementedError
