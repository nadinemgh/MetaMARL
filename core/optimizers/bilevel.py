"""Bilevel optimizer: an outer mechanism search wrapping an inner policy optimizer.

This module holds the composition root of the framework. The outer level
(``ESConfig`` / ``ESOptimizer``) searches the mechanism parameters; the inner
level (``APPOptimizerConfig`` / ``RayOptimizer``) trains the agents' policies
against each candidate mechanism. Both share one ``World`` Ray actor, through
which candidates and step records are exchanged.

``BilevelConfig`` is the fluent configuration of the whole run. Its
``build_optimizer`` starts Ray, creates the reporting and ``World`` actors,
copies the inner seeds into the outer environment configuration, hands the
regulator's agent configurations to the inner environment, builds both levels
and ties the ES population size to the number of inner environments.
``BilevelOptimizer.train`` is the outer loop: it runs the ES, then stops both
levels and closes the reporter.

Example
-------
>>> from core.optimizers.appo.config import APPOptimizerConfig
>>> from core.optimizers.es.config import ESConfig
>>> from core.reporting.csv import CSVConfig
>>> cfg = (
...     BilevelConfig()
...     .world(world_name="fishery")
...     .reporter(CSVConfig(project="fishery"))
...     .society(APPOptimizerConfig())
...     .regulator(ESConfig().training(episodes=100, sigma=0.15))
... )
>>> cfg.outer_cfg.episodes, cfg.world_name.startswith("fishery_")
(100, True)
>>> optimizer = cfg.build_optimizer()  # doctest: +SKIP
>>> summary = optimizer.train()  # doctest: +SKIP

The two last lines start Ray and train the inner learner, so the example does
not run them. See ``examples/bilevel_fishery/debug.py`` for a complete
configuration, including the environments and the regulator agent.
"""

import logging
import uuid
from typing import Any, Optional, Self

from core.adaptors.ray.runtime import DeviceType, RayRuntime, RayRuntimeConfig
from core.annotations import override
from core.optimizers.base import Optimizer
from core.optimizers.config import OptimizerConfig
from core.reporting.base import Reporter
from core.reporting.config import ReporterConfig
from core.world.base import World

logger = logging.getLogger(__name__)


class BilevelConfig(OptimizerConfig):
    """Fluent configuration of a bilevel run (see module docstring).

    Builder methods of this class: :meth:`world`, :meth:`ray`,
    :meth:`reporter`, :meth:`society` (inner level) and :meth:`regulator`
    (outer level). Every one returns ``self``. The inherited builders
    (``training``, ``environment``, ``debugging``, ``reporting``, ``agents``)
    configure the bilevel config itself; the numbers of generations and of inner
    iterations are set on the levels, through ``regulator`` and ``society``.
    ``training(episodes=...)`` on the bilevel config is a shortcut for the
    number of generations: ``build_optimizer`` copies it into the regulator
    config when that one has none, and refuses a different value.

    Parameters
    ----------
    opt_class : type[Optimizer], optional
        Optimizer class built by ``build_optimizer`` (default
        ``BilevelOptimizer``).

    Attributes
    ----------
    outer_cfg : OptimizerConfig or None
        Outer (mechanism search) configuration, set by :meth:`regulator`.
    inner_cfg : OptimizerConfig or None
        Inner (policy learning) configuration, set by :meth:`society`.
    world_name : str or None
        Name of the ``World`` actor, with a random suffix, set by :meth:`world`.
    ray_cfg : RayRuntimeConfig or None
        Ray runtime configuration, set by :meth:`ray`; ``None`` means the
        defaults of ``RayRuntimeConfig``.

    When to use: to describe a full bilevel run (outer search plus inner
    learner) before building it with ``build_optimizer``; the levels themselves
    are configured through their own configs.

    Examples
    --------
    >>> from core.optimizers.es.config import ESConfig
    >>> from core.reporting.csv import CSVConfig
    >>> cfg = BilevelConfig().regulator(ESConfig()).reporter(CSVConfig(project="p"))
    >>> cfg.opt_class.__name__, cfg.inner_cfg is None
    ('BilevelOptimizer', True)
    """

    def __init__(self, opt_class: Optional[type[Optimizer]] = None):
        super().__init__(opt_class=opt_class or BilevelOptimizer)

        self.outer_cfg = None
        self.inner_cfg = None
        self.world_name: Optional[str] = None
        self.ray_cfg = None

    def society(self, cfg: Optional[OptimizerConfig] = None) -> Self:
        """Set the inner (policy learning) config, typically an ``APPOptimizerConfig``.

        The inner optimizer trains the agents' policies against each mechanism
        candidate. ``build_optimizer`` merges the regulator's agent
        configurations into its ``env_config`` as ``leaders_cfg_dict``, and
        reads its seeds and its batch capacity.

        Parameters
        ----------
        cfg : OptimizerConfig, optional
            The inner configuration; ``None`` keeps the current one.

        Returns
        -------
        BilevelConfig
            ``self`` for chaining.
        """

        if cfg is not None:
            self.inner_cfg = cfg

        return self

    def regulator(self, cfg: Optional[OptimizerConfig] = None) -> Self:
        """Set the outer (mechanism search) config, typically an ``ESConfig``.

        ``build_optimizer`` copies the inner seeds into its ``env_config`` and
        sizes its population from the inner batch capacity. The number of
        generations is the ``episodes`` of this config, or the ``episodes`` of
        the bilevel config when this one has none.

        Parameters
        ----------
        cfg : OptimizerConfig, optional
            The outer configuration; ``None`` keeps the current one.

        Returns
        -------
        BilevelConfig
            ``self`` for chaining.
        """

        if cfg is not None:
            self.outer_cfg = cfg

        return self

    def world(self, *, world_name: str, **kwargs: Any) -> Self:
        """Name the shared ``World`` actor (a random suffix keeps runs distinct).

        Parameters
        ----------
        world_name : str
            Base name; the stored ``world_name`` is ``"<world_name>_<8 hex
            characters>"``. ``None`` keeps the previous name.
        **kwargs : Any
            Accepted and ignored.

        Returns
        -------
        BilevelConfig
            ``self`` for chaining.
        """

        if world_name is not None:
            self.world_name = f"{world_name}_{uuid.uuid4().hex[:8]}"

        return self

    def ray(
        self,
        *,
        device: DeviceType = "cpu",
        num_cpus: Optional[int] = None,
        num_gpus: Optional[int] = None,
        omp_threads: int = 1,
        logging_level: str = "ERROR",
        runtime_env: Optional[dict] = None,
        **kwargs: Any,
    ) -> Self:
        """Configure the local Ray runtime (device, CPU/GPU counts, runtime env).

        Parameters
        ----------
        device : {"cpu", "cuda", "mps"}, optional
            Device type of the runtime (default ``"cpu"``).
        num_cpus, num_gpus : int, optional
            Resources given to ``ray.init``; ``None`` leaves Ray to decide.
        omp_threads : int, optional
            Number of OpenMP threads per process (default 1).
        logging_level : str, optional
            Ray logging level (default ``"ERROR"``).
        runtime_env : dict, optional
            Ray runtime environment, for example files to exclude from the
            upload.
        **kwargs : Any
            Extra keyword arguments forwarded to ``ray.init``.

        Returns
        -------
        BilevelConfig
            ``self`` for chaining.
        """

        self.ray_cfg = RayRuntimeConfig(
            device=device,
            num_cpus=num_cpus,
            num_gpus=num_gpus,
            omp_threads=omp_threads,
            logging_level=logging_level,
            runtime_env=runtime_env,
            init_kwargs=kwargs,
        )

        return self

    def reporter(self, config: ReporterConfig) -> Self:
        """Select the reporting backend through a ``ReporterConfig``.

        The config is stamped with the run identity at build time, built once
        into the primary (bilevel-level) reporter, and copied to the inner and
        outer levels so each builds its own reporter. A reporter config is
        required: ``build_optimizer`` reads it unconditionally.

        Parameters
        ----------
        config : ReporterConfig
            Backend configuration, for example ``CSVConfig`` or ``WandbConfig``.

        Returns
        -------
        BilevelConfig
            ``self`` for chaining.
        """

        self.reporter_cfg = config

        return self

    @override(OptimizerConfig)
    def build_optimizer(self) -> "BilevelOptimizer":
        """Start Ray, create the actors and build both levels.

        The inner and outer configurations are copied first, so the originals
        are not modified. The ES population size is set to the inner optimizer's
        ``batch_capacity`` (number of regulated environments divided by the
        number of seeds), so every candidate is evaluated by exactly one
        environment per seed.

        ``training(episodes=...)`` of the bilevel config is the number of ES
        generations. It is copied into the outer configuration when that one
        has no ``episodes``; when both are set they must be equal. The check
        runs before Ray starts.

        Returns
        -------
        BilevelOptimizer
            The built optimizer (an instance of ``opt_class``) over the built
            outer and inner levels and the primary reporter.

        Raises
        ------
        ValueError
            If ``training(episodes=...)`` of the bilevel config and the
            ``episodes`` of the regulator config are both set and differ.
        AttributeError
            If ``society``, ``regulator`` or ``reporter`` was not called; this
            surfaces after Ray has started.
        """

        # A conflicting generation count is a configuration error: fail before
        # starting Ray rather than silently keeping one of the two values.
        if (
            self.episodes is not None
            and self.outer_cfg is not None
            and self.outer_cfg.episodes is not None
            and self.outer_cfg.episodes != self.episodes
        ):
            raise ValueError(
                f"BilevelConfig.training(episodes={self.episodes}) conflicts with "
                + f"the regulator's episodes={self.outer_cfg.episodes}. Set the "
                + "number of generations once, or give both the same value."
            )

        RayRuntime.ensure_initialized(self.ray_cfg or RayRuntimeConfig())

        world = World.options(name=self.world_name).remote()
        inner_cfg = self.inner_cfg.copy()
        outer_cfg = self.outer_cfg.copy()

        # The bilevel ``episodes`` is the number of ES generations.
        if outer_cfg.episodes is None:
            outer_cfg.episodes = self.episodes

        # Setup reporting
        self.reporter_cfg.world = self.world_name
        primary_reporter = self.reporter_cfg.build(label="bilevel")
        inner_cfg.reporter_cfg = self.reporter_cfg.copy()
        outer_cfg.reporter_cfg = self.reporter_cfg.copy()

        # Assign see to outer cfg for looping
        if inner_cfg.seeds is not None:
            outer_cfg._merge_env_config({"seeds": inner_cfg.seeds})
        inner_cfg._merge_env_config({"leaders_cfg_dict": outer_cfg.agents_cfgs})
        inner_opt = inner_cfg.build_optimizer(world=world, world_name=self.world_name)
        outer_opt = outer_cfg.build_optimizer(world=world, inner_opt=inner_opt)

        # what if outer_opt does not have that property ??
        # override outer_opt population size with inner_opt batch_size
        # set inner batch capacity to be the same as inner for batch sampling
        outer_opt.batch_capacity = inner_opt.batch_capacity

        return self.opt_class(
            config=self, outer=outer_opt, inner=inner_opt, reporter=primary_reporter
        )


class BilevelOptimizer(Optimizer):
    """Outer loop of a bilevel run: train the outer optimizer, then shut down.

    ``train`` runs the outer optimizer, which trains the inner one through its
    regulator environment, and returns the outer summary. Both levels are
    stopped and the primary reporter is closed afterwards, even when training
    fails.

    Parameters
    ----------
    config : BilevelConfig
        Configuration of the run; ``world_name`` is read from it.
    outer : Optimizer
        Optimizer whose ``train()`` returns a dict with ``episodes``,
        ``converged``, ``best_mechanism`` and ``best_fitness`` (the ES). Its
        ``episodes`` attribute is the number of generations.
    inner : Optimizer or None
        Inner optimizer, driven by the outer env; kept for lifecycle access.
        ``None`` is accepted: :meth:`train` then stops only the outer level.
    reporter : Reporter or None
        Primary (bilevel-level) reporter built from ``config.reporter_cfg``;
        closed at the end of :meth:`train`, after both levels are stopped.
        ``None`` means there is nothing to close.

    Attributes
    ----------
    world_name : str or None
        Copied from the config.
    converged : bool
        ``False`` until :meth:`train` returns; then the ``converged`` flag of
        the outer summary (``True`` when the ES stopped on its convergence
        rule).

    When to use: it is what ``BilevelConfig.build_optimizer`` returns, so call
    :meth:`train` on it to run the experiment. Build one directly only to test
    the shutdown order with stand-in levels.

    Examples
    --------
    With stand-in levels, ``train`` returns the outer summary and stops both
    levels:

    >>> from types import SimpleNamespace
    >>> class Level:
    ...     episodes = 1
    ...     stopped = False
    ...     def train(self):
    ...         return {"episodes": 1, "converged": False, "best_fitness": 2.0,
    ...                 "best_mechanism": [0.5]}
    ...     def stop(self):
    ...         self.stopped = True
    >>> config = SimpleNamespace(episodes=None, env=None, world_name="w")
    >>> outer, inner = Level(), Level()
    >>> bilevel = BilevelOptimizer(config, outer=outer, inner=inner, reporter=None)
    >>> bilevel.train()["best_fitness"]
    2.0
    >>> outer.stopped, inner.stopped
    (True, True)
    """

    def __init__(
        self,
        config: BilevelConfig,
        outer: Optimizer,
        inner: Optional[Optimizer],
        reporter: Optional[Reporter],
    ):
        super().__init__(config)

        self.world_name = config.world_name
        self.outer = outer
        self.inner = inner
        self.converged = False

        # reporter
        self.reporting = reporter

    def train(self) -> dict:
        """Run the outer optimizer and return its summary dict unchanged.

        Both levels are stopped and the reporter is closed in a ``finally``
        block, then ``converged`` is taken from the summary and the run is
        logged.

        Returns
        -------
        dict
            The summary of the outer optimizer. For ``ESOptimizer``:
            ``episodes`` (generations run), ``converged``, ``best_fitness``,
            ``best_mechanism`` (normalized vector in ``[0, 1]``) and
            ``population_history``.

        Raises
        ------
        Exception
            Whatever the outer optimizer raises, after the shutdown.
        """

        logger.info(
            "[Bilevel] Starting run | max_outer_iters=%d | world=%s",
            self.outer.episodes,
            self.world_name,
        )

        # Stop both levels even when training fails, so the RLlib algorithm
        # and its env runners are released before the reporter closes.
        try:
            result = self.outer.train()
        finally:
            for level in (self.outer, self.inner):
                if level is not None:
                    level.stop()
            if self.reporting is not None:
                self.reporting.close()

        self.converged = bool(result["converged"])

        logger.info(
            "[Bilevel] Run finished | iters=%d | converged=%s | mechanism=%s"
            + " | best_fitness=%.4f",
            result["episodes"],
            result["converged"],
            result["best_mechanism"],
            result["best_fitness"],
        )
        return result
