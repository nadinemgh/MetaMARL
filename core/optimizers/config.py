"""Base configuration objects for optimizers.

This module holds ``OptimizerConfig``, the base of every configuration in the
framework. It follows the builder pattern of RLlib's ``AlgorithmConfig``: a
mutable object configured through chained method calls (``environment``,
``debugging``, ``training`` ...), then copied into a frozen snapshot and turned
into an ``Optimizer`` by ``build_optimizer``. Concrete configs
(``RayOptimizerConfig`` for the inner learner, ``ESConfig`` for the outer
search, ``BilevelConfig`` for the composition of both) extend it with
backend-specific builders.
"""

from __future__ import annotations

import copy
from abc import ABC
from typing import TYPE_CHECKING, Any, Optional, Self, Type, Union

import gymnasium as gym
import numpy as np
import ray
from ray.actor import ActorHandle
from ray.rllib.utils.metrics.metrics_logger import DEFAULT_STATS_CLS_LOOKUP

from core.agents.base import AgentConfig
from core.metrics.schemas import MetricSchema
from core.reporting.config import ReporterConfig
from core.reporting.query import Query
from core.types import AgentID, EnvConfigDict, EnvType
from core.utils import generate_uuid
from core.world.base import World

if TYPE_CHECKING:
    from core.optimizers.base import Optimizer


class _Config(ABC):
    """Minimal interface shared by all configuration objects."""

    def to_dict(self) -> dict:
        """Convert this configuration to dict format; not implemented here."""

        raise NotImplementedError


class OptimizerConfig(_Config, ABC):
    """Fluent, freezable configuration from which an ``Optimizer`` is built.

    Builder methods (``environment``, ``training``, ``debugging``,
    ``reporting``, ``agents``) set attributes and chain. ``build_optimizer``
    then copies the config into a frozen snapshot, registers the optimizer with
    the ``World`` actor and instantiates its environment.

    Contract
    --------
    - *Fluent*: builder methods mutate ``self`` and return it, so calls
      chain. The base ``training`` only stores ``episodes``; subclasses
      override it with their own hyperparameters.
    - *Freeze*: ``freeze()`` flips ``_is_frozen``; afterwards any attribute
      assignment raises ``AttributeError`` (enforced in ``__setattr__``).
      Freezing is shallow: nested objects such as ``env_config`` stay
      mutable. ``RayOptimizerConfig`` overrides ``freeze`` with a deferred
      RLlib mutator, so its copies are not frozen by ``copy(copy_frozen=True)``.
    - *Copy*: ``copy(copy_frozen=...)`` deep-copies the config and sets the
      frozen flag of the copy. ``build_optimizer`` builds from
      ``self.copy(copy_frozen=True)``, so the original stays editable.

    Parameters
    ----------
    opt_class : type[Optimizer], optional
        Optimizer class instantiated by ``build_optimizer`` (default ``None``;
        building without one raises ``ValueError``).

    Attributes
    ----------
    env : str or EnvType or None
        Environment class (or registered name) set by ``environment``.
    env_config : dict
        Keyword arguments passed to the environment constructor.
    horizon : int or None
        Episode length in steps. ``None`` until a subclass or the caller sets
        it; ``environment(horizon=...)`` stores the value in
        ``env_config["horizon"]`` and does not change this attribute.
    agents_cfgs : dict[AgentID, AgentConfig] or None
        Agent configurations keyed by agent id, set by ``agents``.
    episodes : int or None
        Number of iterations to run, set by ``training``. For ``ESConfig`` it
        is the number of ES generations.
    base_seed : int or None
        Root seed given to ``debugging``.
    seeds : list of int
        Training seeds derived from ``base_seed`` (empty when unseeded).
    eval_seeds : list of int or None
        Evaluation seeds; set by subclasses (see
        ``RayOptimizerConfig.evaluation``).
    evaluation_config : OptimizerConfig or None
        Optional nested evaluation config; ``copy(copy_frozen=False)`` also
        unfreezes it.
    stats_cls_lookup : dict
        RLlib ``Stats`` class lookup handed to the optimizer's
        ``MetricsLogger``.

    When to use: subclass it to give a new optimizer class its own fluent
    configuration; use an existing subclass (``ESConfig``,
    ``APPOptimizerConfig``, ``BilevelConfig``) to configure a run.

    Examples
    --------
    Builders chain, and a frozen copy refuses further assignment while the
    original stays editable:

    >>> from core.optimizers.es.config import ESConfig
    >>> cfg = ESConfig().debugging(seed=0).training(episodes=5)
    >>> (cfg.base_seed, cfg.episodes)
    (0, 5)
    >>> snapshot = cfg.copy(copy_frozen=True)
    >>> snapshot.episodes = 9  # doctest: +ELLIPSIS
    Traceback (most recent call last):
        ...
    AttributeError: Cannot set attribute (episodes) of an already frozen ...
    >>> cfg.episodes = 9
    >>> (cfg.episodes, snapshot.episodes)
    (9, 5)
    """

    def __init__(self, opt_class: Optional[Type[Optimizer]] = None):
        """Initialize an OptimizerConfig with every attribute at its default.

        Parameters
        ----------
        opt_class : type[Optimizer], optional
            Optimizer class that ``build_optimizer`` instantiates.
        """

        self.opt_class = opt_class

        # lifecycle
        self._is_frozen = False

        # environment
        self.env: Optional[Union[str, EnvType]] = None
        self.env_config: dict = {}
        self.horizon: Optional[int] = None
        self.disable_env_checking: Optional[bool] = None
        self.agents_cfgs: Optional[dict[AgentID, AgentConfig]] = None

        # training
        self.episodes: Optional[int] = None

        # debugging
        self.base_seed: Optional[int] = None
        self.seeds: list[int] = []

        # eval
        self.evaluation_config: Optional["OptimizerConfig"] = None
        self.eval_seeds: Optional[list[int]] = None

        # reporting
        self.stats_cls_lookup = DEFAULT_STATS_CLS_LOOKUP
        self._reporter_cfg: Optional[ReporterConfig] = None
        self._reporting_queries: Optional[tuple[Query, ...]] = None
        self._reporting_schema_env: Optional[type[MetricSchema]] = None
        self._reporting_queries_env: Optional[tuple[Query, ...]] = None

    @property
    def reporter_cfg(self) -> Optional[ReporterConfig]:
        """Reporter configuration used by ``_build_reporter``.

        ``None`` disables reporting.
        """

        return self._reporter_cfg

    @reporter_cfg.setter
    def reporter_cfg(self, reporter_cfg: Optional[ReporterConfig]) -> None:
        """Attach a reporter configuration.

        ``BilevelConfig`` copies one to each level.

        Parameters
        ----------
        reporter_cfg : ReporterConfig or None
            Configuration from which ``build_optimizer`` builds the
            optimizer-level reporter; ``None`` disables reporting.
        """

        self._reporter_cfg = reporter_cfg

    def __setattr__(self, name: str, value: Any) -> None:
        if hasattr(self, "_is_frozen") and self._is_frozen:
            if name not in ["_is_frozen"]:
                raise AttributeError(
                    f"Cannot set attribute ({name}) of an already frozen "
                    + "OptimizerConfig!"
                )

        super().__setattr__(name, value)

    def _merge_env_config(self, extra: dict) -> Self:
        """Shallow-merge ``extra`` into ``env_config`` (``extra`` wins).

        Note that ``environment`` resets ``env_config`` to ``{}`` before
        filling it, so merges done before that call are lost.

        Returns
        -------
        OptimizerConfig
            ``self`` for chaining.
        """

        self.env_config = {**(self.env_config or {}), **extra}

        return self

    def freeze(self) -> None:
        """Freeze this config object, such that no attributes can be set anymore.

        Optimizers should use this method to make sure their config objects
        remain read-only after this. Freezing twice is harmless.
        """

        if self._is_frozen:
            return

        self._is_frozen = True

    def copy(self, copy_frozen: Optional[bool] = None) -> Self:
        """Create a deep copy of this config and (un)freeze it if requested.

        Parameters
        ----------
        copy_frozen : bool, optional
            ``True`` freezes the copy, ``False`` unfreezes it (and its
            ``evaluation_config``), ``None`` keeps the frozen status that
            ``self`` has.

        Returns
        -------
        OptimizerConfig
            A deep copy of ``self`` of the same class.
        """

        cp = copy.deepcopy(self)

        if copy_frozen is True:
            cp.freeze()
        elif copy_frozen is False:
            cp._is_frozen = False

            if isinstance(cp.evaluation_config, OptimizerConfig):
                cp.evaluation_config._is_frozen = False

        return cp

    @classmethod
    def from_dict(cls, data: dict) -> Self:
        """Build a config from a dictionary of attribute values.

        Keys that are not attributes of a default instance are ignored; the
        values are assigned as they are, without validation.

        Parameters
        ----------
        data : dict
            Attribute names mapped to values.

        Returns
        -------
        OptimizerConfig
            A new instance of ``cls`` (which must be constructible without
            arguments).
        """

        cfg = cls()

        for k, v in data.items():
            if hasattr(cfg, k):
                setattr(cfg, k, v)

        return cfg

    @classmethod
    def from_yaml(cls, path: str) -> Self:
        """Build a config from a YAML file holding a mapping of attributes.

        Parameters
        ----------
        path : str
            Path of the YAML file, read with ``yaml.safe_load``.

        Returns
        -------
        OptimizerConfig
            The result of :meth:`from_dict` on the parsed mapping.
        """

        import yaml

        with open(path, "r") as f:
            data = yaml.safe_load(f)

        return cls.from_dict(data)

    def _env_creator(self, **env_ctx) -> gym.Env:
        """Instantiate ``self.env`` with the given keyword arguments.

        Parameters
        ----------
        **env_ctx
            Constructor arguments: ``build_optimizer`` passes ``world``,
            ``opt_id``, ``optimizer`` and the ``env_config`` entries;
            ``RayOptimizerConfig`` adds ``agents``, ``mechanism_id``, ``seed``,
            ``policy_seed`` and ``mode``.

        Returns
        -------
        gymnasium.Env
            A new environment instance. ``self.env`` must be a callable class;
            string specifiers are not resolved here.
        """

        return self.env(**env_ctx)

    def build_optimizer(
        self,
        *,
        world: Optional[ActorHandle[World]] = None,
        inner_opt: Optional[Optimizer] = None,
        **kwargs: Any,
    ) -> Optimizer:
        """Build an ``Optimizer`` from a frozen deep copy of this config.

        The optimizer is registered with ``world`` to obtain its ``opt_id``,
        receives the optimizer-level reporter (``None`` when no reporter
        configuration was set), and its environment is instantiated once
        through ``_env_creator`` with ``world``, ``opt_id``, ``inner_opt``, the
        agent configurations, the reporting settings and the ``env_config``
        entries. The environment is attached with ``opt.env``.

        With ``world=None`` no identifier is requested: ``opt.opt_id`` stays
        ``None`` and the environment receives ``opt_id=None``.

        Parameters
        ----------
        world : ActorHandle[World], optional
            Handle to the shared ``World`` actor that hands out the optimizer
            identifier.
        inner_opt : Optimizer, optional
            Inner optimizer given to the environment as ``optimizer`` (the
            outer level passes the inner one here).
        **kwargs : Any
            Accepted for subclass compatibility and ignored.

        Returns
        -------
        Optimizer
            An instance of ``opt_class`` with its id and environment set.

        Raises
        ------
        ValueError
            If the config has no ``opt_class``.
        """

        cfg = self.copy(copy_frozen=True)

        if cfg.opt_class is None:
            raise ValueError("OptimizerConfig has no opt_class")

        # Build reporter; reporting is optional
        reporter = None

        if self._reporter_cfg is not None:
            reporter = self._reporter_cfg.build(label=self.opt_class.__name__)
            reporter.add_query(*(self._reporting_queries or ()))

        opt: Optimizer = cfg.opt_class(world=world, reporting=reporter, config=cfg)

        if world is not None:
            registry = ray.get(world.get_opt_registry.remote())
            opt.id = ray.get(
                world._set_new_opt_id.remote(opt_id=generate_uuid(registry))
            )

        env = cfg._env_creator(
            world=world,
            opt_id=opt.opt_id,
            optimizer=inner_opt,
            agents_cfgs=self.agents_cfgs,
            reporter_cfg=cfg.reporter_cfg.copy()
            if cfg.reporter_cfg is not None
            else None,
            queries=cfg._reporting_queries_env,
            schema=cfg._reporting_schema_env,
            **self.env_config,
        )
        opt.env = env

        return opt

    def environment(
        self,
        env: Optional[Union[str, EnvType]] = None,
        horizon: Optional[int] = None,
        queries: Optional[tuple[Query, ...]] = None,
        schema: Optional[type[MetricSchema]] = None,
        *,
        env_config: Optional[EnvConfigDict] = None,
        observation_space: Optional[gym.Space] = None,
        action_space: Optional[gym.Space] = None,
        disable_env_checking: Optional[bool] = None,
    ) -> Self:
        """Set the environment class and the keyword arguments it is built with.

        Calling this resets ``env_config`` to an empty dict before filling it,
        so ``_merge_env_config`` calls made earlier are lost.

        Parameters
        ----------
        env : str or EnvType, optional
            Environment class instantiated by ``_env_creator`` (string
            specifiers are stored but not resolved by the base class).
        horizon : int, optional
            Episode length in steps, stored in ``env_config["horizon"]``.
        queries : tuple of Query, optional
            Env-level queries rendered by the environment's own reporter.
        schema : type[MetricSchema], optional
            Metric schema the environment logs into.
        env_config : dict, optional
            Extra constructor arguments merged into ``env_config`` (for
            example ``{"train_iters": 50}`` for a regulator environment that
            takes it).
        observation_space, action_space : gymnasium.Space, optional
            Stored in ``env_config`` for environments that take them.
        disable_env_checking : bool, optional
            Stored as an attribute; RLlib-backed subclasses forward it.

        Returns
        -------
        OptimizerConfig
            ``self`` for chaining.
        """

        self.env_config: dict = {}

        if env is not None:
            self.env = env

        if observation_space is not None:
            self.env_config.update({"observation_space": observation_space})

        if action_space is not None:
            self.env_config.update({"action_space": action_space})

        if horizon is not None:
            self.env_config.update({"horizon": horizon})

        if env_config is not None:
            self.env_config.update(env_config)

        if disable_env_checking is not None:
            self.disable_env_checking = disable_env_checking

        if queries is not None:
            self._reporting_queries_env = tuple(queries)

        if schema is not None:
            self._reporting_schema_env = schema

        return self

    def training(self, *, episodes: Optional[int] = None) -> Self:
        """Set the optimizer's training hyperparameters (backend specific).

        The base implementation stores ``episodes`` when given. Subclasses
        define further keyword arguments.

        Parameters
        ----------
        episodes : int, optional
            Number of iterations to run; ``None`` keeps the current value.

        Returns
        -------
        OptimizerConfig
            ``self`` for chaining.
        """

        if episodes is not None:
            self.episodes = episodes

        return self

    def debugging(
        self,
        *,
        seed: Optional[int] = None,  # base seed
        num_seeds: int = 3,
    ) -> Self:
        """Derive the training seeds from a base seed.

        Parameters
        ----------
        seed : int or None, optional
            Base seed stored in ``base_seed``. A
            ``numpy.random.SeedSequence(seed)`` is created and
            ``generate_state(num_seeds)`` yields ``num_seeds`` independent
            32-bit integers, stored as Python ints in ``seeds``. The same base
            seed always yields the same list, and different base seeds give
            well-separated streams (the SeedSequence hashing spreads them).
            ``None`` clears ``seeds`` to ``[]``.
        num_seeds : int, optional
            Number of training seeds to derive (default 3).

        Returns
        -------
        OptimizerConfig
            ``self`` for chaining.
        """

        if seed is not None:
            self.base_seed = seed
            ss = np.random.SeedSequence(seed)
            self.seeds = ss.generate_state(num_seeds).tolist()
        else:
            self.seeds = []

        return self

    def reporting(self, queries: Optional[tuple[Query, ...]]) -> Self:
        """Declare the queries the optimizer-level reporter renders.

        The paths of the queries refer to the metric schema the optimizer logs
        into, which the optimizer class fixes (``ESSchema`` for the evolution
        strategy, ``RaySchema`` for the RLlib society).

        Parameters
        ----------
        queries : tuple of Query or None
            Queries the optimizer's reporter renders; ``None`` keeps the
            current queries.

        Returns
        -------
        OptimizerConfig
            ``self`` for chaining.
        """

        if queries is not None:
            self._reporting_queries = tuple(queries)

        return self

    def agents(self, agents: AgentConfig | tuple[AgentConfig, ...]) -> Self:
        """Set the agent configurations, keyed by their ``id``.

        Parameters
        ----------
        agents : AgentConfig or tuple of AgentConfig
            One configuration or a non-empty tuple; replaces any earlier
            ``agents_cfgs``.

        Returns
        -------
        OptimizerConfig
            ``self`` for chaining.

        Raises
        ------
        ValueError
            If an empty tuple is given.
        """

        if isinstance(agents, tuple):
            if len(agents) < 1:
                raise ValueError("agents cannot be empty")
        else:
            agents = (agents,)
        self.agents_cfgs = {agent.id: agent for agent in agents}
        return self
