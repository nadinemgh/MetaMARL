"""Outer-level environment driving the inner optimizer.

``RegulatorEnv`` is the environment seen by the outer optimizer (Evolution
Strategies). One outer step evaluates a population of candidate mechanisms:

1. ``action(population)`` decodes each optimizer vector into a ``Mechanism``
   through the env's mechanism space (or wraps it in a ``VectorMechanism``
   when no space is configured);
2. ``_step(mechanisms)`` publishes one ``MechanismContext`` per (candidate,
   seed) to the ``World``, resets the inner policy, trains it for
   ``train_iters`` inner iterations, optionally evaluates it on ``eval_seeds``,
   renders the inner reports and aggregates the inner optimizer's accumulated
   metrics into one fitness per candidate with :meth:`aggregate_rewards`;
3. consumed contexts are flushed from the World.

Without an inner optimizer the env runs in *analytic* mode: subclasses override
``_step`` with a closed-form fitness, which is how the ES is unit-tested.
"""

from typing import Any, Optional, SupportsFloat

import gymnasium as gym
import numpy as np
import ray
from gymnasium.core import ActType, ObsType, WrapperObsType

from core.agents.base import Agent, AgentConfig
from core.annotations import override
from core.metrics.logger import MetricLogger
from core.metrics.schemas import MetricSchema
from core.optimizers.base import Optimizer
from core.reporting.base import Reporter
from core.reporting.config import ReporterConfig
from core.reporting.query import Query
from core.types import AgentID, OptimizerID
from core.world.base import World
from core.world.context import Context, MechanismContext, MechanismStatus


class RegulatorEnv(gym.Env):
    """Outer-loop environment: candidate mechanisms in, one fitness per candidate out.

    Parameters
    ----------
    optimizer : Optimizer
        Agents policy optimization algorithm (e.g. ``RayOptimizer`` wrapping APPO).
        ``None`` selects the analytic mode.
    horizon : int
        Rollout length for the regulator

    seeds : list[int], optional
        Policy seeds; one ``MechanismContext`` is published per (candidate, seed).
    eval_seeds : list[int], optional
        If given, ``inner.evaluate()`` runs after training.
    **kwargs
        Forwarded to :class:`BaseEnv` (``world``, ``mechanism_space``,
        ``horizon``, ...).
    """

    def __init__(
        self,
        *,
        world: World,
        optimizer: Optimizer,
        horizon: int,
        agents_cfgs: dict[AgentID, AgentConfig],
        seeds: list[int],  # agent policy seeds required for mechanism publishing
        reporter_cfg: Optional[ReporterConfig] = None,
        queries: Optional[tuple[Query]] = None,
        schema: Optional[MetricSchema] = None,
        opt_id: Optional[OptimizerID] = None,
        mode: Optional[str] = "train",
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.world = world
        self._opt_id = opt_id
        self.horizon = horizon
        self._t = 0
        self.env_id = None
        self.agents: dict[AgentID, Agent] = {
            aid: cfg.build() for aid, cfg in agents_cfgs.items()
        }
        self.inner: Optimizer = optimizer
        self.seeds: list[int] = seeds or []

        # logging
        self.logger: Optional[MetricLogger] = (
            MetricLogger.from_schema(schema) if schema else None
        )

        # reporting
        reporting_env_id = self.__class__.__name__
        self.reporter: Reporter = reporter_cfg.build(label=reporting_env_id)
        self.reporter.schema = schema
        self.reporter.add_query(*(queries or ()))

    @property
    def opt_id(self) -> OptimizerID:
        return self.opt_id

    @opt_id.setter
    def opt_id(self, opt_id: OptimizerID) -> None:
        """Set the optimizer identifier stamped on every context this env publishes."""
        self._opt_id = opt_id

    @override(gym.Env)
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = {}
    ) -> tuple[ObsType, dict[str, Any]]:
        if seed is not None and self.seed is not None and seed != self.seed:
            self.seed = seed
            self.rng = np.random.default_rng(seed)
            pass  # do not mutate seed after construction
        self._t = 0

        if not options.get("persist_agents_policy", False):
            self.inner.reset()

        return None, {}

    @override(gym.Env)
    def step(
        self, actions: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Advance the environment by one step and publish the transition.

        The template method applies :meth:`action` to ``action``, runs the
        abstract ``_step``, then applies :meth:`observation` and :meth:`reward`
        to the raw outputs. The transformed observation and reward, together
        with the original ``action`` and ``info``, are published to the
        ``World`` as an ``EnvStepContext`` before the step counter is advanced
        and logged under ``("iter",)``.

        Returns
        -------
        tuple
            ``(obs, reward, terminated, truncated, info)`` in the gymnasium
            convention.
        """

        # Setup agents' env after regulator action sampling
        self._t_agents = 0

        for idx, a in enumerate(actions):
            for seed in self.seeds:
                ctx = Context(
                    id=None,
                    opt_id=self._opt_id,
                    step=self._t,
                    env=self.__class__.__name__,
                    payload=MechanismContext(
                        index=idx,
                        seed=seed,
                        status=MechanismStatus.published,
                        env_id=None,
                        mechanism=a,
                        metrics=None,
                    ),
                )
            ray.get(self.world.append_context.remote(ctx))

        results = self.inner.train()

        reward = self.reward(results)
        obs = None
        info = {"metrics": self.inner.reduce_metrics()}

        self._t += 1

        if self.logger is not None:
            self.logger.push(key=("iter",), value=self._t)

        truncated = False
        terminated = self.horizon is not None and (self._t + 1) >= self.horizon

        return obs, reward, terminated, truncated, info

    def action(self, action: ActType) -> ActType: ...

    def observation(self, observation: WrapperObsType): ...

    def reward(self, reward: Optional[SupportsFloat] = None, **kwargs): ...
