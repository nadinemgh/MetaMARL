"""Outer-level environment driving the inner optimizer.

``RegulatorEnv`` is the environment the outer optimizer (evolution strategies)
steps. One outer step receives a population of candidate mechanisms, publishes
each of them to the shared ``World`` once per policy seed, asks the inner
optimizer to train the follower policies against those candidates and returns
the value of ``reward`` computed from the training results as the fitness of
the population:

1. for every candidate and every seed, a ``Context`` carrying a
   ``MechanismContext`` in the ``published`` state is appended to the
   ``World``, where the inner environments fetch it by candidate index and
   policy seed;
2. ``inner.train()`` runs the inner optimization and returns its results;
3. ``reward(results)`` turns those results into the step's reward, which the
   subclass of a benchmark computes (see
   ``examples/bilevel_fishery/regulator_env.py``), and the inner metrics are
   reduced into the ``info`` dictionary.

The class itself defines no fitness: its ``action``, ``observation`` and
``reward`` hooks return ``None``, and a benchmark overrides the ones it needs.
The environment does not evaluate the candidates itself, does not flush
contexts from the ``World`` and does not decode optimizer vectors; the
optimizer hands it candidates that are already mechanisms.
"""

from typing import Any, Optional, SupportsFloat

import gymnasium as gym
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
    """Outer-loop environment: candidate mechanisms in, one fitness out.

    A step of this environment is one generation of the outer optimizer. It
    publishes the candidate mechanisms of the generation to the ``World``,
    trains the inner optimizer against them and returns the reward computed
    by ``reward`` from the training results. The environment keeps no
    observation (``step`` and ``reset`` return ``None`` for it) and never sets
    ``truncated``.

    Parameters
    ----------
    world : World
        Handle of the ``World`` Ray actor through which candidates are
        published to the inner environments.
    optimizer : Optimizer
        Inner optimizer (for example the Ray optimizer wrapping an RL
        algorithm). It must provide ``reset()``, ``train()`` and
        ``reduce_metrics()``; the environment stores it as ``inner``.
    horizon : int or None
        Number of outer steps of an episode. ``None`` means ``terminated`` is
        never set.
    agents_cfgs : dict[AgentID, AgentConfig]
        Configurations of the regulator's agents; each is built once and kept
        in ``agents``.
    seeds : list[int] or None
        Policy seeds of the inner training. One ``MechanismContext`` is
        published per (candidate, seed) pair; ``None`` becomes an empty list,
        in which case nothing is published.
    reporter_cfg : ReporterConfig, optional
        Builds the env-level ``Reporter``, labelled with the class name;
        ``None`` disables reporting (default ``None``).
    queries : tuple of Query, optional
        Queries added to the reporter when there is one (default ``None``).
    schema : type[MetricSchema], optional
        Metric schema from which the env logger is built, and handed to the
        reporter; ``None`` disables the logger (default ``None``).
    opt_id : OptimizerID, optional
        Identifier of the optimizer that owns this environment, stamped on
        every published context. It can be set later through the ``opt_id``
        property (default ``None``).
    mode : str, optional
        Accepted for symmetry with the inner environments; not stored and not
        used (default ``"train"``).
    **kwargs
        Forwarded to ``gymnasium.Env.__init__``, which accepts no argument:
        any extra keyword raises ``TypeError``.

    Attributes
    ----------
    world : World
        The ``World`` handle.
    horizon : int or None
        Number of outer steps of an episode.
    env_id : None
        Placeholder, always ``None``.
    agents : dict[AgentID, Agent]
        The regulator's agents built from ``agents_cfgs``.
    inner : Optimizer
        The inner optimizer.
    seeds : list[int]
        Policy seeds, one context per candidate and seed.
    logger : MetricLogger or None
        Env-level metric logger; ``step`` pushes the outer step counter to its
        ``iter`` entry.
    reporter : Reporter or None
        Env-level reporter.

    When to use: as the base class of the environment the ES optimizer steps,
    in any bilevel benchmark where each outer step must evaluate a population
    of mechanisms by training the inner agents against them. Override
    ``reward`` to turn the inner results into a fitness.

    Examples
    --------
    A step with stand-ins for the world and the inner optimizer (no Ray
    runtime is needed; ``ray.get`` is replaced by the identity for the
    duration of the call):

    >>> from types import SimpleNamespace
    >>> from unittest.mock import patch
    >>> class Fitness(RegulatorEnv):
    ...     def reward(self, results=None, **kwargs):
    ...         return [results["fitness"]]
    >>> class Inner:
    ...     def reset(self):
    ...         pass
    ...     def train(self):
    ...         return {"fitness": 1.5}
    ...     def reduce_metrics(self):
    ...         return {}
    >>> contexts = []
    >>> world = SimpleNamespace(append_context=SimpleNamespace(remote=contexts.append))
    >>> env = Fitness(
    ...     world=world, optimizer=Inner(), horizon=3, agents_cfgs={}, seeds=[7, 8]
    ... )
    >>> with patch.object(ray, "get", lambda ref: ref):
    ...     obs, reward, terminated, truncated, info = env.step([{"fee": 0.1}])
    >>> reward, terminated, truncated, info
    ([1.5], False, False, {'metrics': {}})
    >>> [(c.payload.index, c.payload.seed, c.payload.status.value) for c in contexts]
    [(0, 7, 'published'), (0, 8, 'published')]
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
        **kwargs: Any,
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
        self.reporter: Optional[Reporter] = None

        if reporter_cfg is not None:
            self.reporter = reporter_cfg.build(label=reporting_env_id)
            self.reporter.schema = schema
            self.reporter.add_query(*(queries or ()))

    @property
    def opt_id(self) -> OptimizerID:
        """Identifier of the optimizer that owns this environment.

        ``None`` until an identifier is given at construction or assigned to the
        property.
        """
        return self._opt_id

    @opt_id.setter
    def opt_id(self, opt_id: OptimizerID) -> None:
        """Set the optimizer identifier stamped on every context this env publishes."""
        self._opt_id = opt_id

    @override(gym.Env)
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[ObsType, dict[str, Any]]:
        """Start a new episode and reset the inner policy.

        The outer step counter returns to 0. Unless ``options`` holds a true
        ``persist_agents_policy``, the inner optimizer is reset as well, which
        restarts its policy from scratch.

        Parameters
        ----------
        seed : int, optional
            Accepted for the Gymnasium signature and ignored. The environment
            draws no random number: the policy seeds of the training are the
            ``seeds`` fixed at construction, and the optimizer owns the random
            generator of the search.
        options : dict, optional
            ``{"persist_agents_policy": True}`` keeps the inner policy
            instead of resetting it (default ``None``, which resets it).

        Returns
        -------
        tuple
            ``(None, {})``: the environment has no observation and no info.
        """
        # The seed is fixed at construction (``seeds``); a per-call seed is ignored.
        self._t = 0

        if not (options or {}).get("persist_agents_policy", False):
            self.inner.reset()

        return None, {}

    @override(gym.Env)
    def step(
        self, actions: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Evaluate one population of candidate mechanisms.

        Every candidate is published to the ``World`` for every policy seed as
        a ``Context`` whose payload is a ``MechanismContext`` in the
        ``published`` state, with the candidate's position in ``actions`` as
        its index, and stamped with this env's class name, the optimizer
        identifier and the current outer step. The publication is blocking
        (``ray.get``). The inner optimizer then trains once against the
        published candidates and its results are passed to :meth:`reward`.
        Afterwards the outer step counter is incremented and, when there is a
        logger, pushed to its ``iter`` entry.

        Parameters
        ----------
        actions : iterable of dict[MechanismID, array_like]
            The candidate mechanisms of the generation, each already decoded by
            the optimizer into a dictionary from mechanism identifier to the
            action array of that mechanism.

        Returns
        -------
        obs : None
            The environment has no observation.
        reward : object
            Whatever :meth:`reward` returns for the inner training results; the
            base implementation returns ``None``. Benchmarks return one fitness
            per candidate.
        terminated : bool
            ``True`` when ``horizon`` is not ``None`` and the step counter,
            already incremented, has reached ``horizon``: an episode of
            ``horizon`` steps ends on the ``horizon``-th step. With a
            ``horizon`` of 5, the fifth step is the first to report ``True``.
        truncated : bool
            Always ``False``.
        info : dict
            ``{"metrics": ...}`` holding the inner optimizer's accumulated
            metrics, reduced by ``inner.reduce_metrics()`` (which also clears
            them).
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
        terminated = self.horizon is not None and self._t >= self.horizon

        return obs, reward, terminated, truncated, info

    def action(self, action: ActType) -> ActType:
        """Hook for transforming the optimizer's action; does nothing here.

        The base implementation returns ``None``. ``step`` does not call it,
        so overriding it has no effect on the step.

        Parameters
        ----------
        action : ActType
            The action to transform.

        Returns
        -------
        ActType
            ``None`` in the base class.
        """

    def observation(self, observation: WrapperObsType) -> Any:
        """Hook for transforming the observation; does nothing here.

        The base implementation returns ``None``. ``step`` does not call it
        and always returns ``None`` as the observation.

        Parameters
        ----------
        observation : WrapperObsType
            The observation to transform.

        Returns
        -------
        object
            ``None`` in the base class; a benchmark may return a constant.
        """

    def reward(self, reward: Optional[SupportsFloat] = None, **kwargs: Any) -> Any:
        """Compute the step's reward from the inner training results.

        ``step`` calls this method with the value returned by
        ``inner.train()`` as the first positional argument. The base
        implementation returns ``None``; a benchmark overrides it to return
        the fitness of each candidate of the population.

        Parameters
        ----------
        reward : object, optional
            The inner training results (despite the parameter's name).
        **kwargs
            Unused extra keywords.

        Returns
        -------
        object
            ``None`` in the base class; the fitness (one value per candidate,
            in the benchmarks of this repository) in a subclass.
        """
