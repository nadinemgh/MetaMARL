import logging
from abc import abstractmethod
from typing import Any, SupportsFloat

import numpy as np
from gymnasium.core import ActType, ObsType
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import AgentID, MultiAgentDict

from core.annotations import override
from core.envs.base import BaseEnv
from core.envs.regulated import RegulatedEnv
from core.types import OptimizerID
from core.world.base import World
from core.world.context import EnvStepContext, MechanismStatus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

# TODO create a reward type

# TODO remove inheritance from regulatedEnv completley
# class SingleAgentBaseEnv(gym.Env): ...
# class MultiAgentBaseEnv(MultiAgentEnv): ...

# class RegulatedEnv(SingleAgentBaseEnv): ...
# class MultiAgentRegulatedEnv(MultiAgentBaseEnv): ...


class MultiAgentRegulatedEnv(RegulatedEnv, MultiAgentEnv):
    def __init__(
        self,
        *,
        agents: list[AgentID],
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.agents = agents
        self.possible_agents = list(self.agents)
        self.action_spaces = kwargs.get("action_spaces", {})
        self.observation_spaces = kwargs.get("observation_spaces", {})

        # TODO move this to baseenv later
        self.observation_space = spaces.Dict(self.observation_spaces)
        self.action_space = spaces.Dict(self.action_spaces)

        self._infos : MultiAgentDict = {agent_id: {} for agent_id in self.agents}

    def _update_infos(self, key: str, values: MultiAgentDict | SupportsFloat):
        values = (
            values
            if isinstance(values, dict)
            else {agent_id: values for agent_id in self._infos}
        )
        for agent_id, value in values.items():
            self._infos[agent_id][key] = value

    @abstractmethod
    def _reset(self) -> MultiAgentDict:
        raise NotImplementedError

    # TODO options doesnt get consumed
    @override(MultiAgentEnv)
    def reset(
        self, *, seed=None, options=None
    ) -> tuple[MultiAgentDict, MultiAgentDict]:
        if seed is not None and self.seed is not None and seed != self.seed:
            pass # do not mutate seed after construction

        # if seed is not None and seed != self.seed: 
        #     self.seed = seed
        #     self.rng = np.random.default_rng(seed)
        self._t = 0

        effective_seed = self.seed if self.seed is not None else seed
        
        self._pre_reset(seed=effective_seed)
        obs = self._reset()
        self._infos = {agent_id: {} for agent_id in self.agents}
        return obs, self._infos

    @override(MultiAgentEnv)
    def step(
        self, action_dict: MultiAgentDict
    ) -> tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        actions = self.action(action_dict)

        if not self.published_mechanism_assigned:
            obs = {
                agent_id: self.observation(agent_id, self.S_t)
                for agent_id in self.agents
            }

            rewards = {agent_id: 0.0 for agent_id in self.agents}

            terminated = {agent_id: False for agent_id in self.agents}
            terminated["__all__"] = False

            truncated = {agent_id: False for agent_id in self.agents}
            truncated["__all__"] = False

            self._t += 1
            return obs, rewards, terminated, truncated, self._infos
        obs, rewards, terminated, truncated, self._infos = self._step(actions)


        self._publish(
            EnvStepContext(
                env_id=self.env_id,
                seed=self.seed,
                policy_seed=self.policy_seed,
                status=MechanismStatus(self.mode),
                mechanism=self.mechanism_id,
                observation=obs,
                observation_map=self.obs_map,
                reward=rewards,
                action=actions,
                info=self._infos,
            )
        )
        self._t += 1
        return obs, rewards, terminated, truncated, self._infos
    
    def _step(
        self, action_dict: dict[AgentID, ActType]
    ) -> tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        intrinsic_rewards : MultiAgentDict = self.intrinsic_utility(A_t=action_dict)
        rewards = self.reward(rewards=intrinsic_rewards, A_t=action_dict)

        self.S_t = self.transition_kernel(A_t=action_dict, S_t=self.S_t.copy())

        obs = {
            agent_id: self.observation(agent_id, self.S_t) for agent_id in self.agents
        }

        time_limit = self._is_truncated()

        terminated = {aid: False for aid in self.agents}
        terminated["__all__"] = False

        truncated = {aid: time_limit for aid in self.agents}
        truncated["__all__"] = time_limit

        self._update_infos(key="intrinsic_utility", values=intrinsic_rewards)
        return obs, rewards, terminated, truncated, self._infos

    @abstractmethod
    def transition_kernel(
        self,
        *,
        A_t: MultiAgentDict,
        S_t: dict[str, MultiAgentDict],
        **kwargs,
    ) -> dict[str, float]:
        """S_{t+1} = T(S_t, A_t)"""
        raise NotImplementedError

    @abstractmethod
    def intrinsic_utility(
        self, 
        *,
        A_t: MultiAgentDict,
        **kwargs,
    ) -> SupportsFloat:
        """u_i = U(a_i, S_t)"""
        raise NotImplementedError

    @abstractmethod
    def violation_signal(
        self, 
        u_i: SupportsFloat, 
        **kwargs) -> SupportsFloat:
        """v_i = V(a_i, S_t, M)"""
        raise NotImplementedError

    @abstractmethod
    def penalty(
        self, 
        u_i: SupportsFloat, 
        **kwargs) -> SupportsFloat:
        """λ = λ(M)"""
        raise NotImplementedError
    
    @abstractmethod
    def _observation(
        self, agent_id: AgentID, S_t: dict[str, MultiAgentDict]
    ) -> ObsType:
        """o_i = O_i(S_t)"""
        raise NotImplementedError
    
    @abstractmethod
    def _is_truncated(self) -> bool:
        raise NotImplementedError

    # TODO Restrict Any Type
    @override(BaseEnv)
    def observation(self, agent_id: AgentID, S_t: dict[str, MultiAgentDict]) -> Any:
        """o_i = O_i(S_t, theta)"""
        # TODO may wanna normalize base_obs later
        base_obs = self._observation(agent_id=agent_id, S_t=S_t)
        theta = self.mechanism.to_vector()
        return np.concatenate([base_obs, theta], axis=0)

    def _aggregate_rewards(self, rewards: MultiAgentDict) -> MultiAgentDict:
        mean_reward = float(np.mean(list(rewards.values())))
        return {agent_id: mean_reward for agent_id in self.agents}
    
    @override(RegulatedEnv)
    def reward(self, rewards: MultiAgentDict, **kwargs) -> SupportsFloat:
        return {
            aid: float(
                u_i
                - self.penalty(u_i, **kwargs)
                * self.violation_signal(u_i, aid, **kwargs)
            )
            for aid, u_i in rewards.items()
        }