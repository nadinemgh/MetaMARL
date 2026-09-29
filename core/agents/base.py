from dataclasses import dataclass, fields
from typing import ClassVar, Optional
from gym.core import ActType
from gymnasium import Space
import numpy as np
from core.mechanism.base import Mechanism, MDPState
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

    def __init_subclass__(
            cls,
            **kwargs,
        ):
        super().__init_subclass__(**kwargs)

        for name, func in tuple(cls.__dict__.items()):
            # TODO (nadine) only reset, transition and state space belong to env
            if getattr(func, "action", False): cls._action = name
            if getattr(func, "reward", False): cls._reward = name
            if getattr(func, "observation", False): cls._observation = name  # o_i = O_i(S_t)
            if getattr(func, "observation_spaces", False): cls._observation_spaces = name
    
    # TODO move this to Agent
    # TODO make this configurable in future
    def _normalize_action(
        self,
        action: ActType,
    ) -> np.ndarray:
        z = np.asarray(action, dtype=np.float32).reshape(-1)
        temperature = 4.0
        return np.asarray([sigmoid(float(value) / temperature) for value in z], dtype=np.float32)


    # TODO (nadine) MDPState should be EnvState payload to replace MDPState

    
    def action(self, mdp_state: MDPState) -> MDPState:
        """Apply this agent's mechanisms to the shared mdp_state.

        Each selected mechanism receives its control ``u_t`` and independently
        evaluates the same input ``mdp_state``. Each mechanism returns a ``mdp_state``
        representing its residual effect on the shared system.

        Residual mdp_states are stacked along the mechanism axis and composed with
        the original mdp_state by summing additive quantities and intersecting
        constrained spaces.

        Args:
            action: Mapping from mechanism IDs to their controls at time ``t``.
            mdp_state: Shared system mdp_state observed by every selected mechanism.
            **kwargs: Additional mechanism arguments.

        Returns:
            The mdp_state resulting from composing the original mdp_state with all
            mechanism residuals.
    """
        # actions = {aid: self._normalize_action(a) for aid, a in action_dict.items()}
        # if self._action is not None:
        #     actions = getattr(self, self._action)(actions)
        # [self.logger.push(key=("by_agent", aid, "actions"), value=a) for aid, a in actions]
        # observs
        acts = mdp_state.actions[self.id]
        return mdp_state.add([self.mechanisms[mid](mdp_state, a) for mid, a in acts.items()])

    def observation(
            self,
            m: MDPState,
    ) -> MDPState:
        """o_i = O_i(x_t)"""
        # if self._observation is not None:
        #     observations = getattr(
        #         self,
        #         self._observation,
        #     )(state)
        # else:
        #     """default to fully observable distributed MDP"""
        #     observations = {aid: state for aid in self.agents}
        # [self.logger.push(key=("by_agent", aid, "observation"), value=o) for aid, o in observations]
        ...

    def reward(self, m: MDPState) -> MDPState:
        # if self._reward is not None:
        #     rewards = getattr(self, self._reward)(action_dict)
        # else:
        #     rewards = {agent_id: 0.0 for agent_id in self.agents}
        # [self.logger.push(key=("by_agent", aid, "base_reward"), value=r) for aid, r in rewards]
        ...

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

        mechanisms = (self.mechanisms if isinstance(self.mechanisms, tuple) else (self.mechanisms,))

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