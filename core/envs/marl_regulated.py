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
from functools import reduce
import logging
from abc import ABC
from typing import ClassVar, Optional
import ray

import numpy as np
import gymnasium as gym
from core.agents.base import Agent, AgentConfig
from core.annotations import override
from core.mechanism.base import MDPState, Mechanism, StateType
from core.metrics.logger import MetricLogger
from core.metrics.schemas import MetricSchema
from core.reporting.base import Reporter
from core.reporting.config import ReporterConfig
from core.reporting.query import Query
from core.types import AgentID, OptimizerID
from core.world.base import World
from core.world.context import (
    MechanismStatus,
    MechanismContext
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(mdp)s",
)

logger = logging.getLogger(__name__)

# TODO create a reward type
# TODO separate reported vs type mdp from agent to principal
# TODO wrapper for MultiAgentEnv adaptor to work with ray

# TODO future enhancement. decouple gym inheritance to support MPC, trajectory optimization etc.
class MultiAgentEnv(ABC):
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
    _reset: ClassVar[str | None] = None
    _transition: ClassVar[str | None] = None
    _state_space: ClassVar[str | None] = None

    def __init__(
        self,
        *,
        world: World,
        opt_id: Optional[OptimizerID] = None,
        env_name: Optional[str] = None,
        horizon: Optional[int] = None,
        agents_cfg_dict: dict[AgentID, AgentConfig],
        leaders_cfg_dict: Optional[dict[AgentID, AgentConfig]] = None,
        # TODO (nadine) later replace with planner_id
        mechanism_id: str,
        seed: Optional[int] = None,
        policy_seed: Optional[int] = None,
        mode: Optional[str] = "train",
        reporter_cfg: Optional[ReporterConfig] = None,
        queries: Optional[tuple[Query]] = None,
        schema: Optional[MetricSchema] = None,
        **kwargs,
    ):
        self._t = 0
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

        # TODO (nadine) later replace with planner id
        # Mechanism
        self.mechanism_id = mechanism_id
        self.m_ctx: MechanismContext = None
        self.m: Mechanism = None
        self._using_default_mechanism = True

        # Bilevel Multi-agent environment
        self.followers = {aid: cfg.build() for aid, cfg in agents_cfg_dict.items()}
        self.leaders = {aid: cfg.build() for aid, cfg in leaders_cfg_dict.items()}
        self.lids = set(leaders_cfg_dict.keys())
        self.agents = {**self.followers, **self.leaders,}

        # Logger
        self.logger: Optional[MetricLogger] = (
            MetricLogger.from_schema(schema) if schema else None
        )

        # reporter
        reporting_env_id = (
            f"{env_name}"
            f"|mode={mode}"
            f"{f'|m={self.mechanism_id}' if self.mechanism_id is not None else ''}"
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
        for name, func in tuple(cls.__dict__.items()):
            if getattr(func, "reset", False): cls._reset = name
            if getattr(func, "transition", False): cls._transition = name  # S_{t+1} = T(S_t, A_t)
            if getattr(func, "state_space", False): cls._state_space = name

    @property
    def mechanism(self) -> Mechanism:
        if self.m is not None:
            return self.m
    
    @property
    def published_mechanism_assigned(self) -> bool:
        return self.m is not None and not self._using_default_mechanism

    @property
    def opt_id(self) -> OptimizerID:
        return self.opt_id

    @opt_id.setter
    def opt_id(self, opt_id: OptimizerID) -> None:
        """Set the optimizer identifier stamped on every context this env publishes."""
        self._opt_id = opt_id

    @override(gym.Env)
    def reset(self, mdp : MDPState) -> None:
        self.logger.flush(key=("iter",))
        self.logger.push(key=("env_id",), value=self.env_id)
        self.logger.push(key=("mechanism_id",), value=self.mechanism_id)
        self.logger.push(key=("seed",), value=self.seed)
        self.logger.push(key=("policy_seed",), value=self.policy_seed)

        # Get regulator Agent
        # Try to fetch a new mechanism if one is available (published)
        # Otherwise keep the current mechanism for subsequent episodes
        if self.mechanism_id is None:
            raise RuntimeError(
                "RegulatedEnv has no mechanism_id. "
                "mechanism_id must be injected at env creation."
            )

        if not self.published_mechanism_assigned: 
            try:
                new_ctx: MechanismContext = ray.get(
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
                # persist action to leaders
                mdp = mdp.add(MDPState(actions={lid: new_ctx.mechanism for lid in self.lids}))
            # TODO raising error if training started and default mechanism is still on - leads to silent error

        # Optional benchmark Reset hook to initialize state and add to observation
        if self._reset is not None:
            mdp: MDPState = mdp.add(getattr(self, self._reset)(mdp))
        return mdp.add([agent.observation(mdp) for agent in self.followers.values()])

    def transition(
            self, 
            mdp: MDPState,
    ) -> MDPState:
        if self._transition is not None:
            return getattr(self, self._transition)(mdp=mdp)
        return mdp

    def termination(self, mdp: MDPState) -> MDPState:
        time_limit = ( self.horizon is not None and mdp.t >= self.horizon)
        terminateds = {aid: False for aid in self.agents}
        terminateds["__all__"] = False
        truncateds = {aid: time_limit for aid in self.agents}
        truncateds["__all__"] = time_limit
        mdp.terminateds = terminateds
        mdp.truncateds = truncateds
        return mdp

    def step(self, mdp: MDPState) -> MDPState:
        for agent in self.agents.values():
            mdp = agent.action(mdp)
            mdp = mdp.add(agent.reward(mdp))
        mdp = self.transition(mdp)
        mdp = self.termination(mdp)
        self.logger.push(key=("iter",), value=mdp.t)
        mdp = mdp.add([agent.observation(mdp) for agent in self.followers.values()])
        self._t += 1
        return mdp