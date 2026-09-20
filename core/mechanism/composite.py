from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import numpy as np

from core.annotations import override
from core.mechanism.base import Mechanism
from core.types import MultiAgentDict

# TODO from control theory, controllers can either be independent or dependent
# TODO vector composition where one mechanism is acting on 

@dataclass(frozen=True)
class CompositeMechanism(Mechanism):
    """
    Vector composition of mechanisms.

    For children [f, h, g]:

        action(x) = g(h(f(x)))

    The same ordering is used for reward and observation transformations.
    """

    children: tuple[Mechanism, ...]

    bindings: dict[str, Callable[[Any], Any]] = field(
        default_factory=dict,
        compare=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not self.children:
            raise ValueError(
                "ChainedMechanism requires at least one child."
            )

    @property
    def dimension(self) -> int:
        return sum(child.dimension for child in self.children)

    # TODO replace with __str__
    def param_names(self) -> list[str]:
        names: list[str] = []
        for index, child in enumerate(self.children):
            prefix = (f"{index}:{type(child).__name__}")
            names.extend(f"{prefix}.{name}" for name in child.param_names())
        return names

    def encode(self) -> np.ndarray:
        if self.dimension == 0:
            return np.empty(0, dtype=np.float32)

        return np.concatenate(
            [child.encode() for child in self.children], 
            axis=0,
        ).astype(np.float32, copy=False,)

    def decode(
        self,
        x: np.ndarray,
    ) -> "ChainedMechanism":
        x = self._validate(x)
        decoded_children: list[Mechanism] = []
        start = 0
        for child in self.children:
            stop = (start + child.dimension)
            decoded_children.append(child.decode(x[start:stop]))
            start = stop
        return replace(self, children=tuple(decoded_children))

    @override(Mechanism)
    def reward(
        self,
        reward_dict: MultiAgentDict,
        *,
        env: Any,
        **kwargs,
    ) -> MultiAgentDict:
        """
        Composite mechanism reward follows the following shape:
        r'_{t} = r_t + \sum_{i=1}^{m} d_{i}^r
        r_t is the base reward from the environment if available.
        """
        dr: Counter = Counter()
        for child in self.children:
            context = child.resolve(env)
            child_kwargs = {**kwargs, **context}
            dr.update(child.reward(reward_dict, **child_kwargs))
        dr.update(reward_dict)
        return dict(dr)

    @override(Mechanism)
    def action(
        self,
        action_dict: MultiAgentDict,
        *,
        env: Any,
        **kwargs,
    ) -> MultiAgentDict:
        """
        Composite mechanism reward follows the following shape:
        a'_{t} = a_t + \sum_{i=1}^{m} d_{i}^a
        a_t is the raw action from the environment
        """
        da: Counter = Counter()
        for child in self.children:
            context = child.resolve(env)
            child_kwargs = {**kwargs, **context}
            da.update(child.action(action_dict, **child_kwargs))
        da.update(action_dict)
        return dict(da)

    @override(Mechanism)
    def observation(
        self,
        observation_dict: MultiAgentDict,
        *,
        env: Any,
        **kwargs,
    ) -> MultiAgentDict:
        """
        Composite mechanism reward follows the following shape:
        o'_{t} = o_t + \sum_{i=1}^{m} d_{i}^o
        a_t is the raw action from the environment
        """
        do : Counter = Counter()
        for child in self.children:
            context = child.resolve(env)
            child_kwargs = {**kwargs, **context}
            do.update(child.observation(observation_dict, **child_kwargs))
        do.update(observation_dict)
        return dict(do)