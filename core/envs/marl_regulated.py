"""Multi-agent environment regulated by a published mechanism.

``MultiAgentEnv`` is the inner-level environment of the bilevel framework. It
is a plain class, not an RLlib environment: the adapter
``core.adaptors.ray.marl_env.RLlibMultiAgentEnvAdapter`` wraps it for RLlib and
passes it the shared ``MDPState`` at every ``reset`` and ``step``. The
environment holds two groups of agents built from configurations: the
followers, whose policies RLlib trains, and the leaders, the regulator whose
mechanisms the outer optimizer searches. A concrete benchmark subclasses it and
marks two methods with the decorators of :mod:`core.envs.hooks`: a ``reset``
hook that draws the initial state of an episode and a ``transition`` hook that
advances the state by one step.

The mechanism in force is fetched from the ``World`` at every ``reset`` by
``mechanism_id``, and given to the leaders as their action. Until one is
fetched, each leader plays the ``default`` action of its mechanisms; a mechanism
without a default holds no action and contributes nothing.
Each ``step`` lets every agent apply its mechanisms and earn its reward,
applies the transition, flags the time limit, logs the step and rebuilds the
observations, to which the leaders' mechanisms add their own contribution.

Metrics are optional: with a ``schema`` the env owns a ``MetricLogger``
(episode identity at ``reset``; ``iter`` and the reward series at every step),
and with a ``reporter_cfg`` a ``Reporter`` renders the configured ``queries``
against it (see ``core.callbacks``). Without a ``schema`` there is no logger and
``reset`` and ``step`` log nothing.
"""

import logging
from abc import ABC
from typing import Any, ClassVar, Optional

import gymnasium as gym
import numpy as np
import ray
from ray.actor import ActorHandle

from core.agents.base import AgentConfig
from core.annotations import override
from core.mechanism.base import MDPState, StateType
from core.metrics.logger import MetricLogger
from core.metrics.schemas import MetricSchema
from core.reporting.base import Reporter
from core.reporting.config import ReporterConfig
from core.reporting.query import Query
from core.types import AgentID, MechanismID, OptimizerID
from core.world.base import World
from core.world.context import MechanismContext, MechanismStatus

logger = logging.getLogger(__name__)


class MultiAgentEnv(ABC):
    """Base class of mechanism-regulated multi-agent benchmarks.

    The environment owns the leaders and the followers, the episode's random
    generator, the mechanism candidate in force and the optional logger and
    reporter. It does not own the MDP state: the caller (the RLlib adapter)
    creates an ``MDPState``, hands it to ``reset`` and passes the returned state
    back to ``step`` after writing the followers' actions into it. Subclasses
    supply the dynamics through two hooks, marked with
    :func:`core.envs.hooks.reset` and :func:`core.envs.hooks.transition`; the
    class itself has no abstract method, and without a hook it starts from the
    state it is given and leaves it unchanged at every transition.

    Parameters
    ----------
    world : ActorHandle[World]
        Handle of the ``World`` Ray actor holding the published mechanism
        candidates.
    opt_id : OptimizerID, optional
        Identifier of the optimizer that owns this env (default ``None``).
    env_name : str, optional
        Prefix of the label of the env-level reporter (default ``None``).
    horizon : int, optional
        Episode length in steps; the episode is flagged as truncated once the
        state reaches it. ``None`` never truncates (default ``None``).
    agents_cfg_dict : dict[AgentID, AgentConfig]
        Configurations of the followers, keyed by agent identifier. Required.
    leaders_cfg_dict : dict[AgentID, AgentConfig], optional
        Configurations of the leaders. The default ``None`` builds an
        environment without leaders, like an empty dictionary (default
        ``None``).
    mechanism_id : int or None
        Index of the candidate mechanism this env instance trains against, in
        the generation (the Ray optimizer configuration passes it). Used to
        fetch the candidate from the ``World``. Required, and ``reset`` raises
        when it is ``None``.
    seed : int, optional
        Seed of the environment's random generator ``rng``, created once at
        construction (default ``None``, which draws fresh entropy).
    policy_seed : int, optional
        Seed of the policy trained in this env; also the seed under which the
        candidate is fetched from the ``World`` (default ``None``).
    mode : {"train", "eval"}, optional
        Lifecycle status of the candidate to fetch, converted to a
        ``MechanismStatus`` (default ``"train"``).
    reporter_cfg : ReporterConfig, optional
        Builds the env-level ``Reporter``; ``None`` disables reporting
        (default ``None``).
    queries : tuple[Query, ...], optional
        Queries added to the reporter when there is one (default ``None``).
    schema : type[MetricSchema], optional
        Metric schema from which the logger is built and which is handed to the
        reporter. With ``None`` there is no logger, and ``reset`` and ``step``
        log nothing (default ``None``).
    **kwargs
        Accepted and ignored. RLlib's environment context (worker index,
        observation and action spaces, and so on) reaches the constructor
        through it.

    Attributes
    ----------
    world : ActorHandle[World]
        The ``World`` handle.
    horizon : int or None
        Episode length in steps.
    seed, policy_seed : int or None
        Environment and policy seeds.
    rng : numpy.random.Generator
        Random generator seeded with ``seed``; never reseeded by ``reset``.
    mode : MechanismStatus
        Status of the candidate fetched at reset.
    mechanism_id : int or None
        Index of the candidate this env trains against.
    m_ctx : MechanismContext or None
        Context of the last candidate fetched from the ``World``.
    m : dict[MechanismID, numpy.ndarray] or None
        Mechanism of that candidate (a dictionary from mechanism identifier to
        action array), or ``None`` until one was fetched.
    followers, leaders : dict[AgentID, Agent]
        The agents built from ``agents_cfg_dict`` and ``leaders_cfg_dict``.
    lids : set[AgentID]
        Identifiers of the leaders.
    agents : dict[AgentID, Agent]
        Leaders first, then followers: the order in which ``step`` lets them
        act.
    logger : MetricLogger or None
        Episode logger built from ``schema``.
    reporter : Reporter or None
        Env-level reporter, labelled
        ``"<env_name>|mode=<mode>|m=<mechanism_id>|ps=<policy_seed>|ss=<seed>"``
        (the ``m=`` part is omitted when ``mechanism_id`` is ``None``).

    When to use: as the base class of the inner environment of a benchmark. A
    fishery, for instance, subclasses it, declares a ``reset`` hook that draws
    the initial fish stock and a ``transition`` hook that applies the stock
    dynamics, and lists the fishers as followers and the regulator as leader.

    Examples
    --------
    A one-fisher environment played for one step. The ``World`` is replaced by
    a stand-in that has published nothing, and ``ray.get`` by the identity for
    the duration of the reset, so no Ray runtime is needed.

    >>> from types import SimpleNamespace
    >>> from typing import ClassVar
    >>> from unittest.mock import patch
    >>> from gymnasium import spaces
    >>> from core.agents.base import Agent, AgentConfig
    >>> from core.envs.hooks import reset, transition
    >>> from core.envs.schema import EpisodeRolloutSchema
    >>> from core.mechanism.base import Mechanism
    >>> from core.mechanism.config import MechanismConfig
    >>> class Harvest(Mechanism):
    ...     def apply(self, mdp, action):
    ...         return MDPState(state={"stock": -float(np.asarray(action)[0])})
    >>> class HarvestConfig(MechanismConfig):
    ...     mechanism_cls: ClassVar[type[Mechanism]] = Harvest
    >>> class Fisher(Agent):
    ...     def reward(self, mdp):
    ...         harvest = mdp.actions[self.id]["harvest"][mdp.t]
    ...         return MDPState(rewards={self.id: float(harvest[0])})
    ...     def observation(self, mdp):
    ...         stock = mdp.state["stock"][mdp.t]
    ...         return MDPState(obs={self.id: np.asarray([stock], dtype=np.float32)})
    >>> class FisherConfig(AgentConfig):
    ...     agent_cls: ClassVar[type[Agent]] = Fisher
    >>> class Fishery(MultiAgentEnv):
    ...     @reset
    ...     def start(self, mdp):
    ...         return MDPState(state={"stock": 1.0})
    ...     @transition
    ...     def carry_stock(self, mdp):
    ...         return mdp.advance(state={"stock": mdp.state["stock"][mdp.t]})
    >>> box = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
    >>> fisher = FisherConfig(
    ...     id="f0", policy_id="fisher", mechanisms=HarvestConfig(box, id="harvest")
    ... )
    >>> nothing_published = SimpleNamespace(
    ...     get_mechanism_by_id=SimpleNamespace(remote=lambda **kwargs: None)
    ... )
    >>> env = Fishery(
    ...     world=nothing_published,
    ...     mechanism_id=0,
    ...     horizon=2,
    ...     agents_cfg_dict={"f0": fisher},
    ...     leaders_cfg_dict={},
    ...     schema=EpisodeRolloutSchema,
    ... )
    >>> with patch.object(ray, "get", lambda ref: ref):
    ...     mdp = env.reset(MDPState(aids={"f0"}))
    >>> mdp.obs["f0"][0].tolist()
    [1.0]
    >>> mdp = mdp.update(actions={"f0": {"harvest": np.asarray([0.25])}})
    >>> mdp = env.step(mdp)
    >>> mdp.t, float(mdp.rewards["f0"][0]), mdp.state["stock"]
    (1, 0.25, [0.75, 0.75])
    >>> env.logger.peek().reward_mean, mdp.truncateds["__all__"]
    ([0.25], False)
    """

    _state: StateType | None = None
    _reset: ClassVar[str | None] = None
    _transition: ClassVar[str | None] = None

    def __init__(
        self,
        *,
        world: ActorHandle[World],
        opt_id: Optional[OptimizerID] = None,
        env_name: Optional[str] = None,
        horizon: Optional[int] = None,
        agents_cfg_dict: dict[AgentID, AgentConfig],
        leaders_cfg_dict: Optional[dict[AgentID, AgentConfig]] = None,
        mechanism_id: Optional[int],
        seed: Optional[int] = None,
        policy_seed: Optional[int] = None,
        mode: str = "train",
        reporter_cfg: Optional[ReporterConfig] = None,
        queries: Optional[tuple[Query, ...]] = None,
        schema: Optional[type[MetricSchema]] = None,
        **kwargs: Any,
    ):
        self.world = world
        self._opt_id = opt_id

        # training
        self.horizon = horizon

        # seeding
        self.seed = seed
        self.policy_seed = policy_seed
        self.rng = np.random.default_rng(seed)
        self.mode = MechanismStatus(mode)

        # Mechanism
        self.mechanism_id = mechanism_id
        self.m_ctx: Optional[MechanismContext] = None
        self.m: Optional[dict[MechanismID, np.ndarray]] = None
        self._using_default_mechanism = True

        # Bilevel Multi-agent environment
        self.followers = {aid: cfg.build() for aid, cfg in agents_cfg_dict.items()}
        leaders_cfg_dict = leaders_cfg_dict or {}
        self.leaders = {aid: cfg.build() for aid, cfg in leaders_cfg_dict.items()}
        self.lids = set(self.leaders)
        self.agents = {**self.leaders, **self.followers}

        # Logger
        self.logger: Optional[MetricLogger] = (
            MetricLogger.from_schema(schema) if schema else None
        )

        # reporter
        reporting_env_id = (
            f"{env_name}"
            + f"|mode={mode}"
            + f"{f'|m={self.mechanism_id}' if self.mechanism_id is not None else ''}"
            + f"|ps={policy_seed}"
            + f"|ss={self.seed}"
        )
        self.reporter: Optional[Reporter] = None

        if reporter_cfg is not None:
            self.reporter = reporter_cfg.build(label=reporting_env_id)
            self.reporter.schema = schema
            self.reporter.add_query(*(queries or ()))

    def __init_subclass__(cls, **kwargs):
        """Record the names of the methods marked with a hook decorator.

        Each attribute of the new class that carries the ``reset`` or
        ``transition`` mark is recorded by name in ``_reset`` or
        ``_transition``, which :meth:`reset` and :meth:`transition` read to call
        the hook (the transition is S_{t+1} = T(S_t, A_t)). Hooks are
        inherited, and a subclass may mark another method to replace its
        parent's hook; a class body marks at most one method per hook.

        Raises
        ------
        TypeError
            If two methods of the class body carry the same mark. The message
            names the class, the hook and both methods.
        """
        super().__init_subclass__(**kwargs)

        marked: dict[str, str] = {}

        for name, func in tuple(cls.__dict__.items()):
            for hook in ("reset", "transition"):
                if not getattr(func, hook, False):
                    continue

                if hook in marked:
                    raise TypeError(
                        f"{cls.__name__} marks both {marked[hook]!r} and {name!r} "
                        + f"as the {hook!r} hook; a class can have only one."
                    )

                marked[hook] = name
                setattr(cls, f"_{hook}", name)

    @property
    def mechanism(self) -> Optional[dict[MechanismID, np.ndarray]]:
        """Mechanism of the candidate in force, or ``None`` before the first fetch.

        This is the dictionary from mechanism identifier to action array that
        the leaders hold as their action; it is the same object as ``m``.
        """
        if self.m is not None:
            return self.m

    @property
    def published_mechanism_assigned(self) -> bool:
        """Whether a candidate published to the ``World`` has been fetched.

        ``False`` until a ``reset`` receives a candidate; then ``True`` for the
        rest of the env's life, even when later fetches return nothing.
        """
        return self.m is not None and not self._using_default_mechanism

    @property
    def opt_id(self) -> Optional[OptimizerID]:
        """Identifier of the optimizer that owns this environment.

        ``None`` until an identifier is given at construction or assigned to the
        property.
        """
        return self._opt_id

    @opt_id.setter
    def opt_id(self, opt_id: Optional[OptimizerID]) -> None:
        """Set the optimizer identifier stamped on every context this env publishes."""
        self._opt_id = opt_id

    @override(gym.Env)
    def reset(self, mdp: MDPState) -> MDPState:
        """Start an episode: log its identity, fetch the candidate, observe.

        When the environment has a logger, it is emptied first and receives the
        ``mechanism_id``, ``seed`` and ``policy_seed`` of the episode, so its
        content covers one episode only. RLlib's environment check resets and
        steps every new environment once outside any episode; the empty logger
        keeps that step out of the first real episode.

        The candidate is then fetched from the ``World`` (blocking), for this
        env's ``mechanism_id``, ``policy_seed`` and ``mode``, at every reset.
        The ``World`` hands a training candidate out on its first fetch only
        and returns ``None`` afterwards. A candidate that is returned replaces
        the one kept (``m_ctx`` and ``m``); otherwise the kept one stays in
        force. When a mechanism is held, it is written as the action of every
        leader at the current step. Before any candidate has been fetched,
        each leader is given instead the ``default`` action of each of its
        mechanisms that has one (logged at INFO); a mechanism without a default
        holds no action. A default never counts as a published candidate:
        ``mechanism`` stays ``None`` and ``published_mechanism_assigned``
        ``False``. The ``reset`` hook, when the subclass
        declares one, is called with the state and its result is added. The
        initial observation is built last, from the followers' ``observation``
        and the leaders' ``mechanism_observations``.

        Parameters
        ----------
        mdp : MDPState
            Fresh state created by the caller, with ``aids`` (and the spaces)
            filled in. It is not modified.

        Returns
        -------
        MDPState
            A new state holding the hook's initial state, the leaders' action
            (the candidate in force, else their mechanism defaults) and the
            initial observations.

        Raises
        ------
        RuntimeError
            If ``mechanism_id`` is ``None``, or if fetching the candidate from
            the ``World`` fails; the message then names the ``mechanism_id`` and
            the original exception is chained as the cause.
        """
        # The logger holds one episode. RLlib's environment check resets and
        # steps every new environment once with an unseeded random action,
        # outside any episode; without this reset that step would be reduced
        # into the first real episode and reach the regulator's fitness.
        if self.logger is not None:
            self.logger.reset()
            self.logger.push(key=("mechanism_id",), value=self.mechanism_id)
            self.logger.push(key=("seed",), value=self.seed)
            self.logger.push(key=("policy_seed",), value=self.policy_seed)

        if self.mechanism_id is None:
            raise RuntimeError(
                "MultiAgentEnv has no mechanism_id. "
                + "mechanism_id must be injected at env creation."
            )

        # The World hands a training candidate out only on its first fetch and
        # returns None afterwards, while the adapter builds a fresh MDPState at
        # every reset. Fetch at every reset, so a newly published candidate
        # replaces the kept one, and give the kept one to the leaders otherwise.
        try:
            new_ctx: MechanismContext = ray.get(
                self.world.get_mechanism_by_id.remote(
                    mechanism_id=self.mechanism_id,
                    seed=self.policy_seed,
                    mode=self.mode,
                )
            )
        except Exception as e:
            raise RuntimeError(
                f"Could not fetch mechanism_id={self.mechanism_id} from World."
            ) from e

        if new_ctx is not None:
            self.m_ctx = new_ctx
            self.m = new_ctx.mechanism
            self._using_default_mechanism = False

        if self.m is not None:
            # persist action to leaders
            mdp = mdp.add(MDPState(actions={lid: self.m for lid in self.lids}))
        else:
            # No candidate fetched yet: each leader plays the default action of
            # its mechanisms, and a mechanism without a default stays inactive.
            defaults = {
                lid: {
                    mid: mech._u
                    for mid, mech in leader.mechanisms.items()
                    if mech._u is not None
                }
                for lid, leader in self.leaders.items()
            }
            defaults = {lid: acts for lid, acts in defaults.items() if acts}
            if defaults:
                logger.info(
                    "mechanism_id=%s policy_seed=%s: no candidate fetched yet, "
                    + "leaders play their mechanism defaults %s",
                    self.mechanism_id,
                    self.policy_seed,
                    defaults,
                )
                mdp = mdp.add(MDPState(actions=defaults))

        # Optional benchmark Reset hook to initialize state and add to observation
        if self._reset is not None:
            mdp: MDPState = mdp.add(getattr(self, self._reset)(mdp))
        return mdp.add(
            [agent.observation(mdp) for agent in self.followers.values()]
            + [
                residual
                for leader in self.leaders.values()
                for residual in leader.mechanism_observations(mdp)
            ]
        )

    def transition(self, mdp: MDPState) -> MDPState:
        """Advance the shared state by one step with the benchmark's hook.

        Calls the method marked with :func:`core.envs.hooks.transition`, passing
        the state by the keyword ``mdp``, and returns its result.

        Parameters
        ----------
        mdp : MDPState
            State after every agent applied its mechanisms at step ``mdp.t``.

        Returns
        -------
        MDPState
            The hook's result, which is expected to be the state advanced by
            one step (for example through ``MDPState.advance``). When the
            subclass declares no transition hook, ``mdp`` itself is returned
            unchanged and the clock does not advance.
        """
        if self._transition is not None:
            return getattr(self, self._transition)(mdp=mdp)
        return mdp

    def termination(self, mdp: MDPState) -> MDPState:
        """Set the termination and truncation flags of every agent.

        No agent is ever terminated. All agents, leaders included, are flagged
        as truncated once ``mdp.t`` has reached ``horizon``, and never when
        ``horizon`` is ``None``. The flags are RLlib's: one entry per agent
        plus ``"__all__"``.

        Parameters
        ----------
        mdp : MDPState
            State after the transition; its ``t`` is compared to ``horizon``.

        Returns
        -------
        MDPState
            The same ``mdp`` object, modified in place: its ``terminateds`` and
            ``truncateds`` are replaced by new dictionaries.
        """
        time_limit = self.horizon is not None and mdp.t >= self.horizon
        terminateds = {aid: False for aid in self.agents}
        terminateds["__all__"] = False
        truncateds = {aid: time_limit for aid in self.agents}
        truncateds["__all__"] = time_limit
        mdp.terminateds = terminateds
        mdp.truncateds = truncateds
        return mdp

    def step(self, mdp: MDPState) -> MDPState:
        """Play one step of the environment.

        The caller has written the followers' actions of the step into
        ``mdp``. In order, ``step``:

        1. lets every agent, leaders first, apply its mechanisms
           (``Agent.action``) and add its reward (``Agent.reward``);
        2. applies the transition hook and sets the termination flags;
        3. when there is a logger, pushes the new step index to ``iter`` and the
           mean over the followers of the reward the followers received at this
           step to the reward fields of the logger (``reward_total``,
           ``reward_mean``, ``reward_min``, ``reward_max`` and
           ``reward_terminal``). Leaders do not count in that mean, and rewards
           are per-step values, never running sums;
        4. rebuilds the observations on the new step from the followers'
           ``observation`` and the leaders' ``mechanism_observations``, so the
           contribution of the leaders' mechanisms reaches the policies at the
           next step.

        Nothing is published to the ``World`` by this method.

        Parameters
        ----------
        mdp : MDPState
            State returned by ``reset`` or by the previous ``step``, holding
            the followers' actions of the current step.

        Returns
        -------
        MDPState
            The state after the transition, with the rewards of the step,
            the new observations and the termination and truncation flags.
        """
        for agent in self.agents.values():
            mdp = agent.action(mdp)
            mdp = mdp.add(agent.reward(mdp))
        mdp = self.transition(mdp)
        mdp = self.termination(mdp)

        if self.logger is not None:
            self.logger.push(key=("iter",), value=mdp.t)

            # Mean over followers of the rewards RLlib receives for this step
            # (the RLlib adapter reads the same ``rewards[aid][t - 1]``). Every
            # reward field of the schema reduces this one per-step series.
            step_reward = float(
                np.mean([mdp.rewards[aid][mdp.t - 1] for aid in self.followers])
            )

            for key in (
                "reward_total",
                "reward_mean",
                "reward_min",
                "reward_max",
                "reward_terminal",
            ):
                self.logger.push(key=(key,), value=step_reward)

        # The leaders' mechanisms see the state after the transition, so their
        # contribution lands on the observation the policies receive next.
        mdp = mdp.add(
            [agent.observation(mdp) for agent in self.followers.values()]
            + [
                residual
                for leader in self.leaders.values()
                for residual in leader.mechanism_observations(mdp)
            ]
        )
        return mdp
