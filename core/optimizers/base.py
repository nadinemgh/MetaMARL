"""Abstract optimizer node of the bilevel optimisation graph.

This module holds the ``Optimizer`` contract that every level of the framework
implements. An ``Optimizer`` owns the ``OptimizerConfig`` it was built from, an
optional environment, a handle to the shared ``World`` actor, an optional
reporter and upstream/downstream links to other optimizers. Concrete
implementations (``RayOptimizer`` for the inner RL level, ``ESOptimizer`` for
the outer regulator, ``BilevelOptimizer`` for the composition of both)
implement ``train`` and optionally ``evaluate``, ``reset``, ``save`` and
``stop``.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import gymnasium as gym
from ray.actor import ActorHandle

from core.metrics.logger import MetricLogger
from core.metrics.schemas import MetricSchema
from core.optimizers.config import OptimizerConfig
from core.reporting.base import Reporter
from core.types import OptimizerID
from core.world.base import World


class Optimizer(ABC):
    """Abstract optimizer node of a possibly hierarchical optimisation graph.

    An ``Optimizer`` is one logical level of a bilevel (or multilevel) run. It
    keeps the configuration it was built from, an optional environment that it
    steps itself, the ``World`` actor handle and the reporter it was given, and
    two sets of links: the optimizers below it (``set_downstream``) and above it
    (``set_upstream``). Subclasses must implement :meth:`train`; the lifecycle
    hooks :meth:`evaluate`, :meth:`save`, :meth:`reset` and :meth:`stop` do
    nothing by default. Metrics accumulate in ``logger`` (a ``MetricLogger``
    that the subclass creates) and are rendered by :meth:`report_metrics`.

    Parameters
    ----------
    config : OptimizerConfig, optional
        Configuration the optimizer was built from. ``episodes`` and the
        environment are read from it; when ``None`` both stay ``None``.
    world : ActorHandle[World], optional
        Handle to the shared ``World`` actor (default ``None``).
    reporting : Reporter, optional
        Reporter that renders the accumulated metrics (default ``None``: no
        reporting).
    **kwargs : Any
        Accepted and ignored, so that builders can pass the same keywords to
        every optimizer class.

    Attributes
    ----------
    config : OptimizerConfig or None
        The configuration given at construction.
    episodes : int or None
        Number of outer iterations (``config.episodes``).
    world : ActorHandle[World] or None
        Shared ``World`` actor handle.
    reporting : Reporter or None
        Reporter used by :meth:`report_metrics`.
    logger : MetricLogger or None
        Metric accumulator; ``None`` until a subclass creates one.
    opt_id : OptimizerID or None
        Identifier assigned once by the ``World`` (see :attr:`id`).

    When to use: as the base class of a new level of the optimisation graph,
    for instance a different outer search method or another inner learner. A
    new level only has to implement :meth:`train`; the graph links, the
    identifier, the environment hook and the metric helpers come from here.

    Examples
    --------
    A leaf optimizer and a root that trains whatever is linked below it:

    >>> class Leaf(Optimizer):
    ...     def train(self):
    ...         return {"fitness": 1.0}
    >>> class Root(Optimizer):
    ...     def train(self):
    ...         return {"below": [opt.train() for opt in self._downstream]}
    >>> root, leaf = Root(), Leaf()
    >>> root.set_downstream(leaf)
    >>> leaf.set_upstream(root)
    >>> root.train()
    {'below': [{'fitness': 1.0}]}
    >>> leaf.id = "leaf-0"
    >>> str(leaf)
    'Leaf(id=leaf-0)'
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
        **kwargs: Any,
    ):
        from core.optimizers.config import OptimizerConfig

        self.episodes: Optional[int] = config.episodes if config else None
        self.config: OptimizerConfig = config
        self.world = world
        self.reporting: Optional[Reporter] = reporting
        self.logger: Optional[MetricLogger] = None

        # Assigned by World or Orchestrator
        self.opt_id: OptimizerID | None = None

        # Set by the subclass or by the bilevel driver (see ``batch_capacity``)
        self._batch_capacity: Optional[int] = None

        # Optional environment (may be None for meta-optimizers)
        self._env: gym.Env | None = config.env if config else None

        # Optimizer Graph connectivity
        self._downstream: set["Optimizer"] = set()
        self._upstream: set["Optimizer"] = set()

    def __str__(self) -> str:
        """Return ``ClassName(id=<opt_id>)``; the id is ``None`` while unset."""

        return f"{self.__class__.__name__}(id={self.opt_id})"

    @property
    def env(self) -> gym.Env | None:
        """Environment attached to this optimizer, or ``None``.

        Initialised from ``config.env`` (which may be an env *class* rather
        than an instance) and replaced by ``OptimizerConfig.build_optimizer``
        with the instantiated environment. ``None`` for optimizers that do
        not step an environment themselves.
        """

        return self._env

    @env.setter
    def env(self, value: gym.Env | None) -> None:
        """Attach an environment and fire ``_on_env_init`` if it is not None.

        Parameters
        ----------
        value : gymnasium.Env or None
            The environment instance; ``None`` detaches it without firing the
            hook.
        """

        self._env = value

        if value is not None:
            self._on_env_init(value)

    def _on_env_init(self, env: gym.Env) -> None:
        """Hook called after a runtime env is attached"""

        pass

    @property
    def id(self) -> OptimizerID:
        """Identifier assigned by the World, raising if not yet set.

        Returns
        -------
        OptimizerID
            The identifier stored in ``opt_id``.

        Raises
        ------
        RuntimeError
            If no identifier has been assigned yet.
        """

        if self.opt_id is None:
            raise RuntimeError("Optimizer ID not set")

        return self.opt_id

    @property
    def batch_capacity(self) -> int:
        """Number of candidates this optimizer can process per iteration.

        The base class returns ``self._batch_capacity``, which starts unset:
        subclasses either override the property (``RayOptimizer``) or assign
        the attribute (the ES optimizer). The bilevel driver copies the inner
        optimizer's capacity onto the outer one to size the ES population.

        Returns
        -------
        int
            Number of candidates (regulated environments per seed).

        Raises
        ------
        RuntimeError
            If the capacity has not been set yet.
        """

        if self._batch_capacity is None:
            raise RuntimeError(f"{type(self).__name__}.batch_capacity not set")

        return self._batch_capacity

    @id.setter
    def id(self, id: OptimizerID) -> None:
        """Assign the optimizer ID once.

        Parameters
        ----------
        id : OptimizerID
            Identifier handed out by the ``World`` registry.

        Raises
        ------
        RuntimeError
            If an ID is already set.
        """

        if self.opt_id is not None:
            raise RuntimeError("Optimizer ID already set")

        self.opt_id = id

    def set_downstream(self, opt: "Optimizer") -> None:
        """Register ``opt`` as a downstream (inner) optimizer of this one.

        Registering the same optimizer twice keeps one entry.

        Parameters
        ----------
        opt : Optimizer
            The optimizer placed below this one in the graph.
        """

        self._downstream.add(opt)

    def set_upstream(self, opt: "Optimizer") -> None:
        """Register ``opt`` as an upstream (outer) optimizer of this one.

        Kept for graph consistency checks and to let an inner optimizer reach
        its parent; the core loop does not traverse ``_upstream``.

        Parameters
        ----------
        opt : Optimizer
            The optimizer placed above this one in the graph.
        """

        # this is only for checking!
        # and also to call the upstream optimizer
        self._upstream.add(opt)

    @classmethod
    def from_config(cls, config: OptimizerConfig) -> "Optimizer":
        """Instantiate the optimizer from a config, without a World or reporter.

        Parameters
        ----------
        config : OptimizerConfig
            Configuration passed as ``config=`` to the constructor.

        Returns
        -------
        Optimizer
            An instance of ``cls``. Unlike ``OptimizerConfig.build_optimizer``
            it registers nothing with a ``World`` and attaches no environment.
        """

        return cls(config=config)

    @classmethod
    def get_default_config(cls) -> OptimizerConfig:
        """Return a default config; not provided by the base class.

        Raises
        ------
        NotImplementedError
            Always; subclasses that offer a default configuration override it.
        """

        raise NotImplementedError("Optimizers must define a default config explicitly")

    @classmethod
    def from_checkpoint(cls, file_path: Path) -> "Optimizer":
        """Restore an optimizer from a checkpoint file; not implemented here.

        Parameters
        ----------
        file_path : pathlib.Path
            Location of the checkpoint.

        Raises
        ------
        NotImplementedError
            Always; the base class defines no checkpoint format.
        """

        raise NotImplementedError

    def reduce_metrics(self) -> MetricSchema:
        """Reduce the accumulated metrics, clearing them, and return the result.

        Returns
        -------
        MetricSchema
            The reduced metrics held by ``logger``.

        Raises
        ------
        RuntimeError
            If the optimizer has no ``MetricLogger``.
        """

        if self.logger is None:
            raise RuntimeError(f"{type(self).__name__} has no MetricLogger.")

        return self.logger.reduce()

    def flush_metrics(self) -> None:
        """Discard the accumulated metrics; do nothing when there is no logger."""

        if self.logger is None:
            return

        self.logger.reset()

    def report_metrics(self) -> None:
        """Render every configured query on the accumulated metrics.

        Non-destructive: the metrics are peeked, not reduced. Nothing happens
        when the optimizer has no logger or no reporter.
        """

        if self.logger is None or self.reporting is None:
            return

        self.reporting.report(self.logger.peek())

    @abstractmethod
    def train(self) -> None:
        """Run the optimizer; every concrete level implements this method.

        The base class prescribes no execution flow and no return value: an
        implementation may drive the optimizers registered with
        :meth:`set_downstream`, step its environment, or call into the
        ``World`` actor, in any order. ``ESOptimizer`` and
        ``BilevelOptimizer`` return a summary dictionary; the class docstring
        has a small runnable example.

        Raises
        ------
        NotImplementedError
            If an implementation calls ``super().train()``.
        """

        raise NotImplementedError

    def evaluate(self) -> None:
        """Evaluate the current result of the optimizer; a no-op by default."""

        pass

    def save(self) -> None:
        """Persist the optimizer state; a no-op by default."""

        pass

    def reset(self) -> None:
        """Reset the optimizer state (e.g. policy weights); a no-op by default."""

        pass

    def stop(self) -> None:
        """Release resources held by the optimizer (no-op by default)."""

        pass
