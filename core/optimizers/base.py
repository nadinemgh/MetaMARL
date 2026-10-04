"""Abstract optimizer node of the bilevel optimisation graph.

An ``Optimizer`` owns a frozen ``OptimizerConfig``, an optional environment,
handles to the shared ``World`` and reporter actors, and upstream/downstream
links to other optimizers. Concrete implementations (``RayOptimizer`` for the
inner RL level, the ES optimizer for the outer regulator) implement ``run`` and
optionally ``evaluate``, ``reset``, ``save`` and ``stop``.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import gymnasium as gym
from ray.actor import ActorHandle

from core.metrics.logger import MetricLogger
from core.metrics.schemas import MetricSchema
from core.optimizers.config import OptimizerConfig
from core.reporting.base import Reporter
from core.types import OptimizerID
from core.world.base import World


class Optimizer(ABC):
    """
    Abstract optimizer node.

    Represents a single logical optimizer in a possibly hierarchical
    (bilevel / multilevel) optimization graph.
    """

    # data owned by the optimizer
    config: OptimizerConfig
    opt_id: OptimizerID
    logger: MetricLogger
    reporting: Reporter

    # this is the default configuration as soon as an optimizer is created

    def __init__(
        self,
        config: Optional[OptimizerConfig] = None,
        world: Optional[ActorHandle[World]] = None,
        reporting: Optional[Reporter] = None,
        **kwargs,
    ):
        from core.optimizers.config import OptimizerConfig

        self.episodes: int = config.episodes
        self.config: OptimizerConfig = config
        self.world = world
        self.reporting: Optional[Reporter] = reporting
        self.logger: Optional[MetricLogger] = None

        # Assigned by World or Orchestrator
        self.opt_id: OptimizerID | None = None

        # Optional environment (may be None for meta-optimizers)
        self._env: gym.Env | None = config.env if config else None

        # Optimizer Graph connectivity
        self._downstream: set["Optimizer"] = set()
        self._upstream: set["Optimizer"] = set()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}(id={self.opt_id})"

    @property
    def env(self) -> gym.Env | None:
        """Environment attached to this optimizer.

        Initialised from ``config.env`` (which may be an env *class* rather
        than an instance) and replaced by ``OptimizerConfig.build_optimizer``
        with the instantiated environment. ``None`` for optimizers that do
        not step an environment themselves.
        """

        return self._env

    @env.setter
    def env(self, value: gym.Env | None) -> None:
        """Attach an environment and fire ``_on_env_init`` if it is not None."""

        self._env = value

        if value is not None:
            self._on_env_init(value)

    def _on_env_init(self, env: gym.Env) -> None:
        """Hook called after a runtime env is attached"""

        pass

    @property
    def id(self) -> OptimizerID:
        """Identifier assigned by the World, raising if not yet set.

        Raises
        ------
        RuntimeError
            If ``set_id`` has not been called.
        """

        if self.opt_id is None:
            raise RuntimeError("Optimizer ID not set")

        return self.opt_id

    @property
    def batch_capacity(self) -> int:
        """Number of candidates this optimizer can process per iteration.

        The base class returns ``self._batch_capacity`` but never sets it, so
        subclasses must either override the property (``RayOptimizer``) or
        assign the attribute (the ES optimizer); otherwise ``AttributeError``
        is raised. The bilevel driver copies the inner optimizer's capacity
        onto the outer one to size the ES population.
        """

        return self._batch_capacity

    @id.setter
    def id(self, id: OptimizerID) -> None:
        """Assign the optimizer ID once.

        Raises
        ------
        RuntimeError
            If an ID is already set.
        """

        if self.opt_id is not None:
            raise RuntimeError("Optimizer ID already set")

        self.opt_id = id

    def set_downstream(self, opt: "Optimizer") -> None:
        """Register ``opt`` as a downstream (inner) optimizer of this one."""

        self._downstream.add(opt)

    def set_upstream(self, opt: "Optimizer") -> None:
        """Register ``opt`` as an upstream (outer) optimizer of this one.

        Kept for graph consistency checks and to let an inner optimizer reach
        its parent; the core loop does not traverse ``_upstream``.
        """

        # this is only for checking!
        # and also to call the upstream optimizer
        self._upstream.add(opt)

    @classmethod
    def from_config(cls, config: OptimizerConfig) -> "Optimizer":
        """Instantiate optimizer from config."""

        return cls(config=config)

    @classmethod
    def get_default_config(cls) -> OptimizerConfig:
        """Return a default config; not provided by the base class."""

        raise NotImplementedError("Optimizers must define a default config explicitly")

    @classmethod
    def from_checkpoint(cls, file_path: Path) -> "Optimizer":
        """Restore optimizer from config."""

        raise NotImplementedError

    def reduce_metrics(self) -> MetricSchema:
        """
        Destructively reduce and return all accumulated optimizer metrics.
        """

        if self.logger is None:
            raise RuntimeError(f"{type(self).__name__} has no MetricLogger.")

        return self.logger.reduce()

    def flush_metrics(self) -> None:
        """
        Flush all accumulated optimizer metrics.
        """

        if self.logger is None:
            return

        self.logger.reset()

    def report_metrics(self) -> None:
        """Render every configured query on the accumulated metrics.

        Non-destructive: the metrics are peeked, not reduced.
        """

        if self.logger is None or self.reporting is None:
            return

        self.reporting.report(self.logger.peek())

    @abstractmethod
    def train(self) -> None:
        """
        Implementations may publish Context objects to the World, invoke downstream
        optimizers via `self._downstream`, and retrieve or aggregate contexts from
        the World in any order. The framework does not impose a fixed execution flow.

        Example
        -------
        >>> def run(self, world: World) -> None:
        >>>     # Publish contexts
        >>>     ctx = Context(
        >>>         id=None,
        >>>         opt_id=self.id,
        >>>         payload=SignalContext(value=1.0),
        >>>     )
        >>>     world.set_new_context(ctx)
        >>>
        >>>     # Execute downstream optimizers
        >>>     for opt in self._downstream:
        >>>         opt.train(world)
        >>>
        >>>     # Retrieve downstream contexts ---
        >>>     for opt in self._downstream:
        >>>         ctx_ids = world.get_opt_ctx_ids(opt.id)
        >>>         for ctx_id in ctx_ids:
        >>>             ctx = world.get_context(ctx_id)
        >>>             # aggregate or process ctx.payload here
        >>>
        >>>     # Optional: update or overwrite own context ---
        >>>     ctx.payload.value += 1.0
        >>>     world.update_context(ctx)

        """

        raise NotImplementedError

    def evaluate(self) -> None:
        """Evaluate Optimizer Performance"""

        pass

    def save(self) -> None:
        """Persist Optimizer State"""

        pass

    def reset(self) -> None:
        """Reset optimizer state (e.g., policy weights)."""

        pass

    def stop(self) -> None:
        """Release resources held by the optimizer (no-op by default)."""

        pass
