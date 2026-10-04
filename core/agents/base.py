from dataclasses import dataclass
from typing import ClassVar, Optional

import numpy as np
from gym.core import ActType
from gymnasium import Space

from core.mechanism.base import MDPState, Mechanism
from core.mechanism.config import MechanismConfig
from core.types import AgentID, MechanismID, PolicyID
from core.utils import sigmoid


class Agent:
    _action: ClassVar[str | None] = None
    _reward: ClassVar[str | None] = None
    _observation: ClassVar[str | None] = None
    _observation_spaces: ClassVar[str | None] = None

    def __init__(
        self,
        *,
        id: AgentID,
        policy_id: PolicyID,
        mechanisms: dict[MechanismID, Mechanism],
        observation_space: Space | None = None,
    ):
        self.id = id
        self.policy_id = policy_id
        self.mechanisms = mechanisms
        self.observation_space = observation_space

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        for name, func in tuple(cls.__dict__.items()):
            if getattr(func, "action", False):
                cls._action = name
            if getattr(func, "reward", False):
                cls._reward = name
            if getattr(func, "observation", False):
                cls._observation = name  # o_i = O_i(S_t)
            if getattr(func, "observation_spaces", False):
                cls._observation_spaces = name

    def _normalize_action(self, action: ActType) -> np.ndarray:
        z = np.asarray(action, dtype=np.float32).reshape(-1)
        temperature = 4.0
        return np.asarray(
            [sigmoid(float(value) / temperature) for value in z], dtype=np.float32
        )

    def action(self, mdp: MDPState) -> MDPState:
        """Apply this agent's mechanisms to the shared mdp.

        Each selected mechanism receives its control ``u_t`` and independently
        evaluates the same input ``mdp``. Each mechanism returns a ``mdp``
        representing its residual effect on the shared system.

        Residual mdps are stacked along the mechanism axis and composed with
        the original mdp by summing additive quantities and intersecting
        constrained spaces.

        Args:
            action: Mapping from mechanism IDs to their controls at time ``t``.
            mdp: Shared system mdp observed by every selected mechanism.
            **kwargs: Additional mechanism arguments.

        Returns:
            The mdp resulting from composing the original mdp with all
            mechanism residuals.
        """
        acts = mdp.actions.data.get(self.id)
        if not acts:
            return mdp
        mdp = mdp.add(
            [
                self.mechanisms[mid](mdp, a[mdp.t])
                for mid, a in acts.items()
                if mid in self.mechanisms
            ]
        )
        return mdp

    def observation(self, mdp: MDPState) -> MDPState:
        return MDPState()

    def reward(self, mdp: MDPState) -> MDPState:
        return MDPState()


@dataclass(frozen=True)
class AgentConfig:
    policy_id: PolicyID
    mechanisms: MechanismConfig | tuple[MechanismConfig, ...]
    id: Optional[AgentID] = None
    count: int = 1
    shared_policy: bool = True
    observation_space: Optional[Space] = None
    agent_cls: ClassVar[type[Agent]] = Agent

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError("count must be at least 1.")

        mechanisms = (
            self.mechanisms
            if isinstance(self.mechanisms, tuple)
            else (self.mechanisms,)
        )

        if not mechanisms:
            raise ValueError("mechanisms cannot be empty.")

        object.__setattr__(self, "mechanisms", mechanisms)

        if self.id is None:
            object.__setattr__(self, "id", self.agent_cls.__name__)

    def build(self) -> Agent:
        mechanisms = {cfg.id: cfg.build(self.id) for cfg in self.mechanisms}
        return self.agent_cls(
            id=self.id,
            policy_id=self.policy_id,
            mechanisms=mechanisms,
            observation_space=self.observation_space,
        )
