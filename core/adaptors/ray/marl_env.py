"""RLlib adapter of the framework's multi-agent environment.

``RLlibMultiAgentEnvAdapter`` is the only environment class RLlib sees in the
inner optimization level. It wraps a ``core.envs.marl_regulated.MultiAgentEnv``,
which works on an ``MDPState`` trajectory and owns the mechanism lifecycle, and
translates it into RLlib's ``MultiAgentEnv`` contract: ``reset`` returns
``(obs, infos)`` and ``step`` returns ``(obs, rewards, terminateds, truncateds,
infos)``, all keyed by agent ID. ``RayOptimizerConfig.build_optimizer`` creates
one adapter around each sub-environment that RLlib asks for.

The adapter does not compute rewards, advance the dynamics or talk to the
``World`` itself; it only builds the ``MDPState`` handed to the wrapped
environment and reads the answers back. Only the followers are RLlib agents:
the leaders of the wrapped environment never appear in the spaces or in the
returned dictionaries.
"""

import logging
from typing import Any, Optional

from gymnasium import spaces
from ray.rllib import MultiAgentEnv as RllibMultiAgentEnv

from core.annotations import override
from core.envs.marl_regulated import MultiAgentEnv
from core.mechanism.base import MDPState
from core.types import MultiAgentDict

logger = logging.getLogger(__name__)


class RLlibMultiAgentEnvAdapter(RllibMultiAgentEnv):
    """RLlib ``MultiAgentEnv`` that drives a framework ``MultiAgentEnv``.

    The wrapped environment keeps the episode state in an ``MDPState``; the
    adapter creates a fresh one at every ``reset``, hands it to
    ``env.reset`` and ``env.step`` and converts the returned trajectories to
    the per-agent dictionaries RLlib expects. The agent IDs are the keys of
    ``env.followers``, in declaration order. The observation and action spaces
    are ``gymnasium.spaces.Dict`` objects keyed by agent ID; the action space
    of an agent is itself a ``Dict`` keyed by mechanism ID, so an agent
    submits one action per mechanism it controls. The identity of the wrapped
    environment (mechanism, seeds, mode), its metric logger and its reporter
    are exposed as attributes so that RLlib callbacks can read them.

    Parameters
    ----------
    env : MultiAgentEnv
        Framework environment to wrap. It must provide ``followers`` (agent ID
        to agent configuration with ``observation_space`` and ``mechanisms``),
        ``mechanism_id``, ``seed``, ``policy_seed``, ``mode``, ``logger`` and
        ``reporter``, and the ``reset(mdp)`` and ``step(mdp)`` methods.
    worker_index : int, optional
        Index of the env runner that hosts the environment (0, the default,
        for the local runner). ``RayOptimizerConfig`` passes RLlib's
        ``EnvContext.worker_index``, which the exploration keys of
        ``core.adaptors.ray.common_random`` read.
    **kwargs : Any
        Forwarded to ``ray.rllib.MultiAgentEnv.__init__``.

    Attributes
    ----------
    env : MultiAgentEnv
        The wrapped environment.
    agents, possible_agents : list of str
        Follower IDs, ``"<agent_type>:<i>"`` in the layout built by
        ``RayOptimizerConfig``. Both lists are equal copies.
    observation_spaces : gymnasium.spaces.Dict
        Observation space of every follower.
    action_spaces : gymnasium.spaces.Dict
        For every follower, a ``Dict`` of the action space of each mechanism.
    mechanism_id, seed, policy_seed, mode
        Copied from the wrapped environment: the mechanism candidate the
        environment trains against, its random seed, the seed of the policy it
        trains and the lifecycle status (``"train"`` or ``"eval"``).
    worker_index : int
        Index of the hosting env runner.
    logger, reporter
        The wrapped environment's ``MetricLogger`` and ``Reporter`` (either may
        be ``None``).

    When to use: you rarely build it directly. ``RayOptimizerConfig`` wraps
    every environment it creates in this class so that RLlib can sample from a
    framework environment. Build it by hand to run an environment episode
    outside of RLlib, for instance in a test.

    Examples
    --------
    Wrap a minimal stand-in environment with one follower that controls one
    mechanism (the stand-in only needs the attributes the adapter reads at
    construction):

    >>> from types import SimpleNamespace
    >>> import numpy as np
    >>> from gymnasium import spaces
    >>> obs = spaces.Box(0.0, 1.0, shape=(2,), dtype=np.float32)
    >>> quota = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
    >>> follower = SimpleNamespace(
    ...     observation_space=obs,
    ...     mechanisms={"quota": SimpleNamespace(action_space=quota)},
    ... )
    >>> env = SimpleNamespace(
    ...     mechanism_id="0",
    ...     seed=11,
    ...     policy_seed=22,
    ...     mode="train",
    ...     followers={"fisher:0": follower},
    ...     logger=None,
    ...     reporter=None,
    ... )
    >>> adapter = RLlibMultiAgentEnvAdapter(env)
    >>> adapter.agents
    ['fisher:0']
    >>> adapter.action_spaces["fisher:0"]["quota"]
    Box(0.0, 1.0, (1,), float32)
    >>> adapter.policy_seed
    22
    >>> adapter.worker_index
    0
    """

    _mdp: MDPState

    def __init__(self, env: MultiAgentEnv, worker_index: int = 0, **kwargs: Any):
        super().__init__(**kwargs)
        self.env = env
        self.worker_index = int(worker_index)

        # env identity
        self.mechanism_id = env.mechanism_id
        self.seed = env.seed
        self.policy_seed = env.policy_seed
        self.mode = env.mode

        self.agents = list(self.env.followers.keys())
        self.possible_agents = list(self.env.followers.keys())

        self.observation_spaces = {
            aid: agent.observation_space for aid, agent in env.followers.items()
        }
        self.observation_spaces = spaces.Dict(self.observation_spaces)

        self.action_spaces = {
            aid: spaces.Dict(
                {mid: m.action_space for mid, m in agent.mechanisms.items()}
            )
            for aid, agent in self.env.followers.items()
        }
        self.action_spaces = spaces.Dict(self.action_spaces)

        # logger and reporter
        self.logger = env.logger
        self.reporter = env.reporter

    @override(RllibMultiAgentEnv)
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None
    ) -> tuple[MultiAgentDict, MultiAgentDict]:
        """Start a new episode and return the first observation of each agent.

        A fresh ``MDPState`` carrying the follower IDs and the observation and
        action spaces is passed to ``env.reset``; the previous episode is
        discarded. ``seed`` and ``options`` are accepted for compatibility with
        the Gymnasium signature and ignored: the wrapped environment is seeded
        once at construction and keeps its random generator across episodes.

        Parameters
        ----------
        seed : int or None, optional
            Ignored.
        options : dict or None, optional
            Ignored.

        Returns
        -------
        obs : dict of str to numpy.ndarray
            Observation of every follower at time 0, shaped like the agent's
            observation space.
        infos : dict
            Always empty.
        """

        # NOTE:
        # `seed` and `options` are environment-bound configuration.
        # The environment is seeded once at construction and owns a persistent RNG
        # (`self.rng`) for its lifetime. Episode resets do not reseed the environment.
        # `reset(seed=..., options=...)` keeps the Gymnasium/RLlib-compatible signature,
        # but these arguments are not used to mutate the environment's persistent
        # configuration after construction.
        self._mdp = MDPState(
            aids=set(self.possible_agents),
            obs_space=self.observation_spaces,
            action_spaces=self.action_spaces,
        )
        self._mdp = self.env.reset(self._mdp)
        obs = {aid: self._mdp.obs[aid][self._mdp.t] for aid in self.env.followers}
        return obs, {}

    @override(RllibMultiAgentEnv)
    def step(
        self, action_dict: MultiAgentDict
    ) -> tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        """Apply one action per agent and advance the environment by one step.

        The actions are written into the current ``MDPState`` and the wrapped
        environment steps once. The returned rewards are those earned during
        the step that just finished (the entry at time ``t - 1`` of the new
        state), while the observations belong to the new time step ``t``.
        ``reset`` must have been called first.

        Parameters
        ----------
        action_dict : dict of str to dict
            For every follower, a dict mapping mechanism ID to the action of
            that mechanism, shaped like the matching entry of
            ``action_spaces``.

        Returns
        -------
        obs : dict of str to numpy.ndarray
            Observation of every follower at the new time step.
        rewards : dict of str to float
            Reward of every follower for the step just taken (reward units of
            the benchmark).
        terminateds : dict of str to bool
            Per-agent termination flags plus the ``"__all__"`` key.
        truncateds : dict of str to bool
            Per-agent truncation flags plus the ``"__all__"`` key.
        infos : dict
            Always empty.
        """

        self._mdp.update(actions=action_dict)
        m: MDPState = self.env.step(self._mdp)
        aids = self.possible_agents
        obs = {aid: m.obs[aid][m.t] for aid in aids}
        rewards = {aid: m.rewards[aid][m.t - 1] for aid in aids}
        terminateds = {aid: m.terminateds[aid] for aid in aids}
        terminateds["__all__"] = m.terminateds["__all__"]
        truncateds = {aid: m.truncateds[aid] for aid in aids}
        truncateds["__all__"] = m.truncateds["__all__"]
        self._mdp = m
        return obs, rewards, terminateds, truncateds, {}
