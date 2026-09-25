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

from abc import abstractmethod
from typing import Any, Optional, SupportsFloat

import numpy as np
import ray
from gymnasium.core import ActType, ObsType, WrapperActType, WrapperObsType
import gymnasium as gym

from core.annotations import override
from core.mechanism.base import Mechanism
from core.metrics.logger import MetricLogger
from core.metrics.schemas import MetricSchema
from core.optimizers.base import Optimizer
from core.reporting.base import Reporter
from core.reporting.config import ReporterConfig
from core.reporting.query import Query
from core.types import OptimizerID
from core.world.base import World
from core.world.context import (
    Context,
    ContextSchema,
    EnvStepContext,
    MechanismContext,
    MechanismStatus,
)


class RegulatorEnv(
    gym.Env, gym.ActionWrapper, gym.ObservationWrapper, gym.RewardWrapper
):
    """Outer-loop environment: candidate mechanisms in, one fitness per candidate out.

    Parameters
    ----------
    optimizer : Optimizer, optional
        Inner optimizer (e.g. ``RayOptimizer`` wrapping APPO). ``None`` selects
        the analytic mode.
    train_iters : int
        Inner training iterations per outer step (``>= 1`` with an optimizer).
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
        opt_id: Optional[OptimizerID] = None,
        env_name: Optional[str] = None,
        horizon: Optional[int] = None,
        seed: Optional[int] = None,
        policy_seed: Optional[int] = None,
        mode: Optional[str] = "train",
        reporter_cfg: Optional[ReporterConfig] = None,
        queries: Optional[tuple[Query]] = None,
        schema: Optional[MetricSchema] = None,
        mechanism: type[Mechanism],
        optimizer: Optional[Optimizer] = None,
        train_iters: int = 5,
        seeds: Optional[list[int]] = None,
        eval_seeds: Optional[list[int]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.world = world
        self._opt_id = opt_id
        self.horizon = horizon
        self.seed = seed
        self.policy_seed = policy_seed
        self.rng = np.random.default_rng(seed)
        self._t = 0
        self.env_id = None
        self.mechanism: type[Mechanism] = mechanism
        self.inner: Optimizer = optimizer

        # logger
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

        self.train_iters: int = train_iters
        self.seeds: list[int] = seeds or []
        self.eval_seeds: Optional[list[int]] = eval_seeds or None

        self._validate()

    @property
    def opt_id(self) -> OptimizerID:
        return self.opt_id

    @opt_id.setter
    def opt_id(self, opt_id: OptimizerID) -> None:
        """Set the optimizer identifier stamped on every context this env publishes."""

        self._opt_id = opt_id

    # TODO remove legacy method.
    def set_opt_id(self, opt_id: OptimizerID) -> None:
        self._opt_id = opt_id

    # private methods
    # TODO get rid of context publishing
    def _publish(self, payload: ContextSchema):
        ctx = Context(
            id=None,
            opt_id=self._opt_id,
            step=self._t,
            env=self.__class__.__name__,
            payload=payload,
        )

        ray.get(self.world.append_context.remote(ctx))

    def _validate(self):
        if self.inner is None:
            return  # analytic override mode allowed

        if self.train_iters <= 0:
            raise ValueError("train_iters must be >= 1 when optimizer is provided")

    @override(gym.Env)
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Start a new episode and publish its initial observation.

        The per-call ``seed`` is ignored: the environment seed is fixed at
        construction and ``_pre_reset`` always receives that one. The step
        counter is reset to zero and logged, the abstract ``_reset`` produces
        the first observation, and an ``EnvStepContext`` with zero reward and
        no action is published to the ``World``. ``options`` is accepted for
        gymnasium compatibility and unused.

        Returns
        -------
        tuple[ObsType, dict]
            The initial observation and an empty info dictionary.
        """

        # Option to pass seed directly to env --> sequential
        # TODO what are the options used for ?

        if seed is not None and self.seed is not None and seed != self.seed:
            pass  # do not mutate seed after construction

        # if seed is not None and and seed != self.seed;
        #     self.seed = seed
        #     self.rng = np.random.default_rng(seed)
        self._t = 0

        self.logger.push(key=("iter",), value=self._t)

        obs = (
            0.0
            if self.observation_space is None
            else np.zeros(self.observation_space.shape, dtype=np.float32)
        )

        self._publish(
            EnvStepContext(
                env_id=self.env_id,
                seed=self.seed,
                policy_seed=self.policy_seed,
                status=MechanismStatus(self.mode),
                mechanism=getattr(self, "mechanism_id", None),
                observation=obs,
                observation_map=self.obs_map,
                reward=0.0,
                action=None,
                info={},
            )
        )

        return obs, {}

    # TODO input should only be one action. regulator env parallelized instead
    @override(gym.Env)
    def step(
        self, actions: list[ActType] = None
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

        if self.inner is None:
            raise NotImplementedError(
                f"{self.__class__.__name__} has no inner optimizer. "
                f"Override `_step()` for analytic reward computation."
            )

        mechanisms = [self.action(a) for a in actions]

        # Reset policy weights for fresh equilibrium search each ES iteration
        if hasattr(self.inner, "reset"):
            self.inner.reset()

        # TODO PARALLELIZE Vectorize environment across θ candidates and train one ppo policy over mehcanism candidates
        # TODO other techiniques can also speed this up
        # for theta in thetas:
        #     self._publish(MechanismContext(theta=theta))
        for idx, m in enumerate(mechanisms):
            for seed in self.seeds:
                self._publish(
                    MechanismContext(
                        index=idx,
                        seed=seed,
                        status=MechanismStatus.published,
                        env_id=None,
                        mechanism=m,
                        metrics=None,
                    )
                )

        # TODO : why eval gets repeated ?
        # Train policy for train_iters iterations
        for _ in range(self.train_iters):
            ctx_registry = ray.get(self.world.get_ctx_registry.remote())

            # TODO remove env step contexsts
            ray.get(self.world.flush_ctx.remote(ctx_registry.keys()))
            ray.get(self.world.flush.remote(status=MechanismStatus.eval))
            self.inner.run()

        # TODO check if eval mechanisms published. If parallel and sequential eval both turned on
        # will be a problem
        if self.eval_seeds:
            # TODO flush all remote mechanisms and env_step ctx.
            # TODO initializing the envs with the seeds from eval_seeds
            # TODO flush all remote mechanisms and env_step ctx.
            # TODO eval results accumulate in eval and maintain training data !
            self.inner.evaluate()

        # plot results
        self.inner.report_metrics()

        metrics = self.inner.logger.peek()

        # TODO self.reward func should not be taking metrics
        reward = self.reward(metrics)
        obs = self.observation()

        # flush consumed contexts
        ray.get(self.world.flush_ctx.remote(ctx_registry.keys()))
        ray.get(self.world.flush.remote(status=MechanismStatus.eval))

        # return reduced data to outer optimizer
        # TODO route this through world in future
        # TODO applying nested reducers or in sequence
        info = {"metrics": self.inner.reduce_metrics()}

        # Publish env context to World
        self._publish(
            EnvStepContext(
                env_id=self.env_id,
                seed=self.seed,
                policy_seed=self.policy_seed,
                status=MechanismStatus(self.mode),
                mechanism=getattr(self, "mechanism_id", None),
                observation=None,
                observation_map=self.obs_map,
                reward=reward,
                action=mechanisms,
                info=info,
            )
        )

        self._t += 1

        if self.logger is not None:
            self.logger.push(key=("iter",), value=self._t)

        truncated = self.horizon is not None and (self._t + 1) >= self.horizon
        terminated = False

        # TODO return obs, terminated and truncated.
        return obs, reward, terminated, truncated, info

    @override(gym.ActionWrapper)
    def action(self, action: WrapperActType) -> Mechanism:
        if self.inner is None:
            return action

        # TODO move constraint function to optimizer.
        if not self.action_space.contains(action):
            raise ValueError(
                f"Action {action} is outside regulator action space {self.action_space}"
            )

        return self.mechanism(action)

    @override(gym.ObservationWrapper)
    def observation(self, observation: WrapperObsType):
        return observation

    @override(gym.RewardWrapper)
    def reward(self, reward: Optional[SupportsFloat] = None, **kwargs):
        raise NotImplementedError
