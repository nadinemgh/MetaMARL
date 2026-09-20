"""Multi-agent environment regulated by a published mechanism.

``MultiAgentRegulatedEnv`` is the RLlib-facing environment of the inner
optimization level. It combines :class:`core.envs.regulated.RegulatedEnv`
(mechanism lifecycle and regulated reward) with RLlib's ``MultiAgentEnv``. A
concrete benchmark implements the abstract pieces of the step:
``transition_kernel`` (``S_{t+1} = T(S_t, A_t)``), ``intrinsic_utility``
(``u_i = U(a_i, S_t)``), ``violation_signal`` and ``penalty`` (the regulated
reward ``u_i - lambda(M) * v_i``), ``_observation`` (``o_i = O_i(S_t)``) and
``_is_truncated``. The base class owns the step lifecycle: it computes the
intrinsic utilities, shapes and aggregates the rewards, advances the state,
appends the mechanism vector ``theta`` to every observation and publishes an
``EnvStepContext`` to the ``World``.

The mechanism in force is fetched from the ``World`` at ``reset`` by
``mechanism_id``. Until one is published the environment returns zero rewards
and does not advance its dynamics, so RLlib's environment checks can run
before training starts.

Metrics are optional: with a ``schema`` the env owns a ``MetricLogger`` fed by
``_log`` (episode identity at ``reset``, rewards and ``iter`` at every step),
and with a ``reporter_cfg`` a ``Reporter`` renders the configured ``queries``
against it (see ``core.callbacks``).
"""

from dataclasses import dataclass, field
import logging
from abc import abstractmethod
from typing import Any, ClassVar, Optional, SupportsFloat, TypeAlias, TypeVar

from gym import spaces
import numpy as np

# TODO remove ray and gymnasium dependency
from gymnasium.core import ActType, ObsType
import gymnasium as gym
import ray
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from core.annotations import override
from core.mechanism.base import Mechanism
from core.metrics.logger import MetricLogger
from core.metrics.schemas import MetricSchema
from core.reporting.base import Reporter
from core.reporting.config import ReporterConfig
from core.reporting.query import Query
from core.types import AgentID, MultiAgentDict
from core.utils import sigmoid
from core.types import OptimizerID
from core.world.base import World
from core.world.context import (
    MechanismStatus,
    MechanismContext
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

# TODO create a reward type
# TODO separate reported vs type message from agent to principal
# TODO wrapper for MultiAgentEnv adaptor to work with ray
# TODO gym wrapper.

StateType = TypeVar("StateType")
@dataclass
class Information:
    states: list[StateType] = field(default_factory=list)
    actions: list[MultiAgentDict] = field(default_factory=list)
    obs: list[MultiAgentDict] = field(default_factory=list)
@dataclass
class Message:
    t : int
    horizon: Optional[int] = None
    action_spaces : Optional[dict[AgentID, gym.Space]] = None
    observation_spaces: Optional[dict[AgentID, gym.Space]] = None
    state_space: Optional[gym.Space] = None
    info : Optional[Information] = None
    rewards: Optional[MultiAgentDict] = None
    terminateds: Optional[MultiAgentDict] = None
    truncateds: Optional[MultiAgentDict] = None

# TODO future enhancement. decouple gym inheritance to support MPC, trajectory optimization etc.
class MultiAgentRegulatedEnv(MultiAgentEnv):
    """Base class for mechanism-regulated multi-agent benchmarks.

    Parameters
    ----------
    world : World
        Ray actor handle of the shared blackboard.
    opt_id : OptimizerID, optional
        Identifier of the optimizer owning this env (set on published contexts).
    env_name : str, optional
        Label prefix of the env-level reporter.
    horizon : int, optional
        Episode length in steps.
    mechanism_id : str
        Identifier of the candidate mechanism this env instance trains
        against; used to fetch its ``MechanismContext`` from the ``World``.
    seed, policy_seed : int, optional
        Environment RNG seed and seed of the associated policy.
    mode : {"train", "eval"}
        Lifecycle status stamped on published contexts.
    reporter_cfg : ReporterConfig, optional
        Builds the env-level ``Reporter``; ``None`` disables reporting.
    queries : tuple[AnyQuery, ...], optional
        Queries rendered by the env-level reporter.
    schema : type[MetricSchema], optional
        Metric schema of the env logger; ``None`` disables logging.
    agents : list[AgentID]
        Agent identifiers; ``possible_agents`` is a copy of this list.
    action_spaces, observation_spaces : dict, optional
        Per-agent gymnasium spaces read from ``kwargs`` (forwarded by the
        RLlib env creator); they also build the ``Dict`` spaces of the env.
    **kwargs
        Forwarded to :class:`MultiAgentEnv`.
    """

    _state: StateType | None = None
    _t: int = 0
    __observation_spaces: Optional[dict[AgentID, gym.Space]] = None
    __action_spaces: Optional[dict[AgentID, gym.Space]] = None
    __state_space: Optional[gym.Space] = None
    
    _reset: ClassVar[str | None] = None
    _action: ClassVar[str | None] = None
    _reward: ClassVar[str | None] = None
    _observation: ClassVar[str | None] = None
    _transition: ClassVar[str | None] = None
    _observation_spaces: ClassVar[str | None] = None
    _action_spaces: ClassVar[str | None] = None
    _state_space: ClassVar[str | None] = None

    def __init__(
        self,
        *,
        world: World,
        opt_id: Optional[OptimizerID] = None,
        env_name: Optional[str] = None,
        horizon: Optional[int] = None,
        mechanism_id: str,
        seed: Optional[int] = None,
        policy_seed: Optional[int] = None,
        mode: Optional[str] = "train",
        agents: list[AgentID],
        reporter_cfg: Optional[ReporterConfig] = None,
        queries: Optional[tuple[Query]] = None,
        schema: Optional[MetricSchema] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.world = world
        self._opt_id = opt_id

        # training
        self.horizon = horizon
        self.env_id = None

        # seeding
        self.seed = seed
        self.policy_seed = policy_seed
        self.rng = np.random.default_rng(seed)
        self.mode = MechanismStatus(mode) # TODO change name to just Status

        # Mechanism
        self.mechanism_id = mechanism_id
        self.m_ctx: MechanismContext = None
        self.m: Mechanism = None
        self._using_default_mechanism = True

        # observation map
        self.obs_map: Optional[dict[int, str]] = None
        self.observation_spaces = kwargs.get("observation_spaces", {})
        self.observation_space = spaces.Dict(self.observation_spaces)

        # Multi-agent environment
        self.agents = agents
        self.possible_agents = list(self.agents)
        self.action_spaces = kwargs.get("action_spaces", {})
        self.action_space = spaces.Dict(self.action_spaces)
        self._infos : MultiAgentDict = {agent_id: {} for agent_id in self.agents}

        # Logger
        self.logger: Optional[MetricLogger] = (
            MetricLogger.from_schema(schema) if schema else None
        )

        # reporter
        mechanism_id = getattr(self, "mechanism_id", None)
        reporting_env_id = (
            f"{env_name}"
            f"|mode={mode}"
            f"{f'|m={mechanism_id}' if mechanism_id is not None else ''}"
            f"|ps={policy_seed}"
            f"|ss={self.seed}"
        )
        self.reporter: Reporter = reporter_cfg.build(label=reporting_env_id)
        self.reporter.schema = schema

        self.reporter.add_query(*(queries or ()))


    def __init_subclass__(
            cls,
            **kwargs,
        ):
        super().__init_subclass__(**kwargs)

        for name, func in cls.__dict__.items():
            if getattr(func, "reset", False): cls._reset = name
            if getattr(func, "action", False): cls._action = name
            if getattr(func, "reward", False): cls._reward = name
            if getattr(func, "observation", False): cls._observation = name  # o_i = O_i(S_t)
            if getattr(func, "transition", False): cls._transition = name  # S_{t+1} = T(S_t, A_t)
            if getattr(func, "action_spaces", False): cls._action_spaces = name
            if getattr(func, "observation_spaces", False): cls._observation_spaces = name
            if getattr(func, "state_space", False): cls._state_space = name

    @property
    def mechanism(self) -> Mechanism:
        if self.m is not None:
            return self.m
        return self.m_space.default()
    
    @property
    def published_mechanism_assigned(self) -> bool:
        return self.m is not None and not self._using_default_mechanism

    @property
    def action_spaces(
        self,
    ) -> dict[AgentID, gym.Space]:
        return self.__action_spaces

    @action_spaces.setter
    def action_spaces(self, state: StateType) -> None:
        if self._action_spaces is not None:
            self.__action_spaces = getattr(self, self._action_spaces)(state)

    @property
    def observation_spaces(
        self,
    ) -> dict[AgentID, gym.Space]:
        return self.__observation_spaces

    @observation_spaces.setter
    def observation_spaces(self, state: StateType):
        if self._observation_spaces is not None:
            self.__observation_spaces = getattr(self, self._observation_spaces)(state)

    @property
    def state_space(self) -> gym.Space:
        return self.__state_space

    @state_space.setter
    def state_space(self, state: StateType) -> None:
        if self._state_space is not None:
            self.__state_space = getattr(
                self,
                self._state_space,
            )(state)

    # Setter
    def set_opt_id(self, opt_id: OptimizerID) -> None:
        self._opt_id = opt_id

    # TODO make this configurable in future
    def _normalize_action(
        self,
        action: ActType,
    ) -> np.ndarray:
        z = np.asarray(action, dtype=np.float32).reshape(-1)
        temperature = 4.0
        return np.asarray([sigmoid(float(value) / temperature) for value in z], dtype=np.float32)
    
    # TODO options doesnt get consumed
    @override(MultiAgentEnv)
    def reset(
        self, 
        *, 
        seed: Optional[int] = None, 
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[MultiAgentDict, MultiAgentDict]:
        """Start a new episode under the mechanism currently in force.
        The step counter restarts (the ``iter`` metric is flushed) and the
        episode identity (``env_id``, ``mechanism_id``, ``seed``,
        ``policy_seed``) is logged. If no published candidate has been assigned
        yet, one is fetched from the ``World`` by ``mechanism_id``; otherwise
        the current mechanism is kept for the whole run. The abstract
        ``_reset`` then produces the initial per-agent observations and the
        per-agent infos are cleared. The seed fixed at construction takes
        precedence over the per-call ``seed``; ``options`` is ignored.

        Returns
        -------
        tuple[MultiAgentDict, MultiAgentDict]
            Per-agent initial observations and per-agent (empty) infos.
        """
        if seed is not None and self.seed is not None and seed != self.seed:
            pass # do not mutate seed after construction
        self._t = 0

        self.logger.flush(key=("iter",))
        self.logger.push(key=("env_id",), value=self.env_id)
        self.logger.push(key=("mechanism_id",), value=self.mechanism_id)
        self.logger.push(key=("seed",), value=self.seed)
        self.logger.push(key=("policy_seed",), value=self.policy_seed)

        # Try to fetch a new mechanism if one is available (published)
        # Otherwise keep the current mechanism for subsequent episodes
        if self.mechanism_id is None:
            raise RuntimeError(
                "RegulatedEnv has no mechanism_id. "
                "mechanism_id must be injected at env creation."
            )

        if not self.published_mechanism_assigned: 
            try:
                new_ctx = ray.get(
                    self.world.get_mechanism_by_id.remote(
                        mechanism_id = self.mechanism_id, 
                        seed=self.policy_seed,
                        mode=self.mode
                    )
                )
            except Exception as e:
                self._debug_remote(
                    "pre_reset_fetch_failed",
                    {
                        "error_type": type(e).__name__,
                        "error_repr": repr(e),
                    },
                )
                raise RuntimeError(
                    f"Could not fetch mechanism_id={self.mechanism_id} from World."
                ) from e

            if new_ctx is not None:
                self.m_ctx = new_ctx
                self.m = self.m_ctx.mechanism
                self._using_default_mechanism = False

            # TODO raising error if training started and default mechanism is still on - leads to silent error

        # Optional benchmark Reset hook to initialize state and add to observation
        if self._reset is not None:
            self._state = getattr(self, self._reset)()

        # Native constraints at x_0
        self.action_spaces = self._state
        self.observation_spaces = self._state
        self.state_space = self._state

        obs = self.observation(self._state)

        message = Message(
            t=0,
            horizon=self.horizon,
            action_spaces=self.action_spaces,
            observation_spaces=self.observation_spaces,
            state_space=self.state_space,
            info=Information(
                states=[self._state],
                obs=[obs],
            ),
        )
        message : Message = self.mechanism.apply(message)

        self._state = message.info.states[-1]
        self.__action_spaces = message.action_spaces
        self.__observation_spaces = message.observation_spaces
        self.__state_space = message.state_space

        obs = message.info.obs[-1]

        return obs, {}


    def transition(
            self, 
            state: StateType,
            action_dict: MultiAgentDict,
    ) -> StateType:
        if self._transition is not None:
            return getattr(self, self._transition)(A_t=action_dict, state=state)
        return state

    def termination(
            self,
    ) -> tuple[MultiAgentDict, MultiAgentDict]:
        time_limit = self.horizon is not None and (self._t + 1) >= self.horizon
        terminateds = {aid: False for aid in self.agents}
        terminateds["__all__"] = False
        truncateds = {aid: time_limit for aid in self.agents}
        truncateds["__all__"] = time_limit
        return terminateds, truncateds

    @override(MultiAgentEnv)
    def step(
        self, action_dict: MultiAgentDict
    ) -> tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        """Run one regulated step of the benchmark.

        Actions go through :meth:`action`; ``_step`` then computes the
        intrinsic utilities with :meth:`intrinsic_utility`, shapes and
        aggregates them with :meth:`reward`, advances ``S_t`` with
        :meth:`transition_kernel` and builds the next observations with
        :meth:`observation`. Rewards are logged per agent and as
        ``reward_mean``, the intrinsic utilities are recorded in the infos as
        ``intrinsic_utility``, and an ``EnvStepContext`` is published to the
        ``World``. Episodes never terminate; ``truncated`` is raised for every
        agent and for ``"__all__"`` when ``_is_truncated`` reports the horizon.

        Before any candidate mechanism has been published (RLlib's environment
        checks), the step is inert: observations of the unchanged state, zero
        rewards, nothing published.

        Returns
        -------
        tuple
            ``(obs, rewards, terminated, truncated, infos)``, each keyed by
            agent ID; ``terminated`` and ``truncated`` also carry ``"__all__"``.
        """
        # Policy outputs -> normalized semantic actions -> mechanisms.
        state = self._state
        actions = self.action(action_dict)
        obs = self.observation(self._state)

        # pre-transition intervention
        message = Message(
            t=self._t,
            horizon=self.horizon,
            action_spaces=self.action_spaces,
            observation_spaces=self.observation_spaces,
            state_space=self.state_space,
            information=Information(
                state=[state],
                actions=[actions],
                obs=[obs],
            ),
        )
        message : Message = self.mechanism.apply(message)
        actions = message.info.actions[0]

        state_next = self.transition(state, actions)
        obs_next = self.observation(state_next)
        rewards = self.reward(actions)
        terminateds, truncateds = self.termination()
        self.action_spaces = state_next
        self.observation_spaces = state_next
        self.state_space = state_next


        message.info.states.append(state_next)
        message.info.obs.append(obs_next)
        message.rewards = rewards
        message.terminateds = terminateds
        message.truncateds = truncateds
        message.action_spaces = self.action_spaces
        message.observation_spaces = self.observation_spaces
        message.state_space = self.state_space

        message = self.mechanism.apply(message)

        self._state = message.info.states[-1]
        obs_next = message.info.obs[-1]
        rewards = message.rewards
        terminateds = message.terminateds
        truncateds = message.truncateds
        self.__action_spaces = message.action_spaces
        self.__observation_spaces = message.observation_spaces
        self.__state_space = message.state_space

        # TODO remove this block
        # if not self.published_mechanism_assigned:
        #     obs = self.observation(self._state)
        #     rewards = {agent_id: 0.0 for agent_id in self.agents}
        #     terminated = {agent_id: False for agent_id in self.agents}
        #     terminated["__all__"] = False
        #     truncated = {agent_id: False for agent_id in self.agents}
        #     truncated["__all__"] = False
        #     self._t += 1
        #     self.logger.push(key=("iter",), value=self._t)
        #     return obs, rewards, terminated, truncated, self._infos

        self._t += 1

        self.logger.push(key=("iter",), value=self._t)

        return obs_next, rewards, terminateds, truncateds, self._infos

    @override(MultiAgentEnv)
    def action(
        self,
        action_dict: MultiAgentDict,
    ) -> MultiAgentDict:
        # apply normalization to action components
        # TODO provide wrapper hooks for benchmark envs
        actions = {aid: self._normalize_action(a) for aid, a in action_dict.items()}
        if self._action is not None:
            actions = getattr(self, self._action)(actions)
        [self.logger.push(key=("by_agent", aid, "actions"), value=a) for aid, a in actions]
        return actions


    @override(MultiAgentEnv)
    def reward(
        self,
        action_dict: MultiAgentDict,
    ) -> MultiAgentDict:
        if self._reward is not None:
            rewards = getattr(self, self._reward)(action_dict)
        else:
            rewards = {agent_id: 0.0 for agent_id in self.agents}
        [self.logger.push(key=("by_agent", aid, "base_reward"), value=r) for aid, r in rewards]
        return rewards

    @override(MultiAgentEnv)
    def observation(
        self,
        state: StateType,
    ) -> MultiAgentDict:
        """o_i = O_i(x_t)"""
        if self._observation is not None:
            observations = getattr(
                self,
                self._observation,
            )(state)
        else:
            """default to fully observable distributed MDP"""
            observations = {aid: state for aid in self.agents}
        [self.logger.push(key=("by_agent", aid, "observation"), value=o) for aid, o in observations]
        return observations
