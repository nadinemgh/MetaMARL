"""Agents of the bilevel framework and their configurations.

An ``Agent`` owns the mechanisms through which it acts on the shared
``MDPState`` and decides what it observes and earns at each step. Both levels
use it: the followers (for example the fishers trained by RLlib) and the
leaders (the regulator whose mechanisms the outer optimizer searches).
``AgentConfig`` is the immutable recipe that builds one agent; the optimizer
configurations store these recipes and the environments build the agents from
them.
"""

from dataclasses import dataclass
from typing import ClassVar, Optional

import numpy as np
from gymnasium import Space

from core.mechanism.base import ActType, MDPState, Mechanism
from core.mechanism.config import MechanismConfig
from core.types import AgentID, MechanismID, PolicyID
from core.utils import sigmoid


class Agent:
    """Participant of an environment, acting through the mechanisms it owns.

    An agent holds a dictionary of ``Mechanism`` objects keyed by mechanism
    identifier. Its ``action`` method applies those mechanisms to the shared
    ``MDPState`` for every mechanism the state holds an action for, and its
    ``observation`` and ``reward`` methods return what the agent sees and
    earns at the current step. The base implementations of ``observation`` and
    ``reward`` contribute nothing; a benchmark overrides them in a subclass
    (optionally marked with the decorators of :mod:`core.envs.hooks`).
    Environments build agents from an ``AgentConfig`` rather than directly.

    Parameters
    ----------
    id : AgentID
        Identifier of the agent, the key under which the state stores its
        actions, observations and rewards (for example ``"fisherman:0"``).
    policy_id : PolicyID
        Identifier of the policy that controls the agent.
    mechanisms : dict[MechanismID, Mechanism]
        Mechanisms the agent acts through, keyed by mechanism identifier.
    observation_space : gymnasium.Space, optional
        Space of the agent's observation, or ``None`` when it is not declared
        (default ``None``).

    Attributes
    ----------
    id, policy_id, mechanisms, observation_space
        The constructor arguments, stored unchanged.

    When to use: subclass it for each kind of participant of a benchmark, such
    as a fisher or a regulator, and override ``observation`` and ``reward``.
    Use the class itself for a participant that only applies mechanisms, which
    is how a leader whose mechanisms carry all of its behaviour is built.

    Examples
    --------
    >>> import numpy as np
    >>> from gymnasium import spaces
    >>> from core.mechanism.base import MDPState, Mechanism
    >>> class Harvest(Mechanism):
    ...     def apply(self, mdp, action):
    ...         return MDPState(state={"stock": -float(np.asarray(action)[0])})
    >>> box = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
    >>> harvest = Harvest(aid="f0", id="harvest", action_space=box, acts_on=None)
    >>> agent = Agent(id="f0", policy_id="fisher", mechanisms={"harvest": harvest})
    >>> mdp = MDPState(state={"stock": 1.0}, actions={"f0": {"harvest": [[0.25]]}})
    >>> agent.action(mdp).state["stock"]
    [0.75]
    """

    _action: ClassVar[str | None] = None
    _reward: ClassVar[str | None] = None
    _observation: ClassVar[str | None] = None
    _observation_spaces: ClassVar[str | None] = None

    def __init__(
        self,
        *,
        id: AgentID,
        policy_id: PolicyID,
        mechanisms: dict[MechanismID, Mechanism],
        observation_space: Space | None = None,
    ):
        """Store the identity, the mechanisms and the observation space."""
        self.id = id
        self.policy_id = policy_id
        self.mechanisms = mechanisms
        self.observation_space = observation_space

    def __init_subclass__(cls, **kwargs):
        """Record the names of the methods marked with a hook decorator.

        Each attribute of the new class that carries the ``action``,
        ``reward``, ``observation`` or ``observation_spaces`` mark is recorded
        by name in ``_action``, ``_reward``, ``_observation`` or
        ``_observation_spaces``. A class body marks at most one attribute per
        hook. Nothing in the framework reads these class variables back.

        Raises
        ------
        TypeError
            If two attributes of the class body carry the same mark. The
            message names the class, the hook and both attributes.
        """
        super().__init_subclass__(**kwargs)

        marked: dict[str, str] = {}

        for name, func in tuple(cls.__dict__.items()):
            for hook in ("action", "reward", "observation", "observation_spaces"):
                if not getattr(func, hook, False):
                    continue

                if hook in marked:
                    raise TypeError(
                        f"{cls.__name__} marks both {marked[hook]!r} and {name!r} "
                        + f"as the {hook!r} hook; a class can have only one."
                    )

                marked[hook] = name
                setattr(cls, f"_{hook}", name)

    def _normalize_action(self, action: ActType) -> np.ndarray:
        z = np.asarray(action, dtype=np.float32).reshape(-1)
        temperature = 4.0
        return np.asarray(
            [sigmoid(float(value) / temperature) for value in z], dtype=np.float32
        )

    def action(self, mdp: MDPState) -> MDPState:
        """Apply this agent's mechanisms to the shared MDP state.

        The state holds, under the agent's identifier, one action per
        mechanism, stored as a trajectory over time. For every mechanism the
        agent owns and holds an action for, the raw entry of the current step
        ``mdp.t`` (``mdp.raw_actions``, as it arrived and not as a previous step
        decoded it) is passed to the mechanism, which decodes it, records the
        decoded value in ``mdp.actions`` and returns a residual ``MDPState``
        describing its effect. The residuals are composed with
        the original state by ``MDPState.add``: additive quantities are summed
        and constrained spaces are intersected. Actions for identifiers the
        agent does not own are ignored.

        Parameters
        ----------
        mdp : MDPState
            Shared state observed by every mechanism of the agent. Its
            ``actions`` entry for this agent maps mechanism identifiers to the
            action trajectories.

        Returns
        -------
        MDPState
            The state composed with all residuals. When the state holds no
            action for this agent (or an empty one), ``mdp`` itself is
            returned unchanged.
        """
        acts = mdp.raw_actions.data.get(self.id)
        if not acts:
            return mdp
        mdp = mdp.add(
            [
                self.mechanisms[mid](mdp, a[mdp.t])
                for mid, a in acts.items()
                if mid in self.mechanisms
            ]
        )
        return mdp

    def mechanism_observations(self, mdp: MDPState) -> list[MDPState]:
        """Collect what this agent's mechanisms contribute to the observations.

        As in :meth:`action`, a mechanism takes part only while this agent
        holds an action for it, so nothing is contributed before the
        regulator's candidate reaches the environment.

        Parameters
        ----------
        mdp : MDPState
            Shared state at the step whose observation is being built.

        Returns
        -------
        list[MDPState]
            One residual per active mechanism, in the order of the agent's
            actions; empty when the agent holds no action.
        """
        acts = mdp.actions.data.get(self.id)
        if not acts:
            return []
        return [
            self.mechanisms[mid].observe(mdp) for mid in acts if mid in self.mechanisms
        ]

    def observation(self, mdp: MDPState) -> MDPState:
        """Return this agent's observation at the current step.

        The base implementation contributes nothing. A subclass overrides it
        to return an ``MDPState`` whose ``obs`` holds the observation vector
        under the agent's identifier.

        Parameters
        ----------
        mdp : MDPState
            Shared state; read the entries of step ``mdp.t``.

        Returns
        -------
        MDPState
            An empty state in the base class.
        """
        return MDPState()

    def reward(self, mdp: MDPState) -> MDPState:
        """Return this agent's reward at the current step.

        The base implementation contributes nothing. A subclass overrides it
        to return an ``MDPState`` whose ``rewards`` holds the reward earned at
        the step just played under the agent's identifier.

        Parameters
        ----------
        mdp : MDPState
            Shared state, after the agents' mechanisms were applied.

        Returns
        -------
        MDPState
            An empty state in the base class.
        """
        return MDPState()


@dataclass(frozen=True)
class AgentConfig:
    """Immutable recipe that builds one ``Agent`` with its mechanisms.

    The configuration stores the policy the agent follows, the configurations
    of its mechanisms and the agent class to instantiate. After construction
    ``mechanisms`` is always a tuple (a single configuration is wrapped) and
    ``id`` is never ``None`` (it falls back to the name of ``agent_cls``).
    ``build`` creates a new, independent agent on every call.

    Attributes
    ----------
    policy_id : PolicyID
        Identifier of the policy that controls the agent.
    mechanisms : MechanismConfig or tuple of MechanismConfig
        Configurations of the mechanisms the agent owns. Must not be empty.
    id : AgentID, optional
        Agent identifier; defaults to the name of ``agent_cls``.
    count : int
        Number of agents this configuration stands for, at least 1 (default
        1). ``build`` ignores it and always returns one agent; the Ray
        optimizer configuration reads it to expand the entry into agents named
        ``"<id>:<i>"``.
    shared_policy : bool
        Whether the ``count`` agents share one policy (default ``True``).
        Stored only; no code of the framework reads it.
    observation_space : gymnasium.Space, optional
        Observation space handed to the agent (default ``None``).
    agent_cls : type[Agent]
        Class variable, not a field: the agent class ``build`` instantiates.
        Subclasses of ``AgentConfig`` override it (default ``Agent``).

    Raises
    ------
    ValueError
        If ``count`` is below 1 or ``mechanisms`` is an empty tuple.

    When to use: to declare, in a benchmark's configuration, the agents of each
    level and the mechanisms they own, before any environment exists. Subclass
    it and set ``agent_cls`` to build a custom ``Agent`` subclass.

    Examples
    --------
    >>> import numpy as np
    >>> from gymnasium import spaces
    >>> from core.mechanism.base import Mechanism
    >>> from core.mechanism.config import MechanismConfig
    >>> class Noop(Mechanism):
    ...     def apply(self, mdp, action):
    ...         return mdp
    >>> class NoopConfig(MechanismConfig):
    ...     mechanism_cls = Noop
    >>> box = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
    >>> config = AgentConfig(policy_id="fisher", mechanisms=NoopConfig(box, id="m"))
    >>> config.id, len(config.mechanisms)
    ('Agent', 1)
    >>> agent = config.build()
    >>> type(agent).__name__, list(agent.mechanisms), agent.mechanisms["m"].aid
    ('Agent', ['m'], 'Agent')
    >>> AgentConfig(policy_id="p", mechanisms=())
    Traceback (most recent call last):
        ...
    ValueError: mechanisms cannot be empty.
    """

    policy_id: PolicyID
    mechanisms: MechanismConfig | tuple[MechanismConfig, ...]
    id: Optional[AgentID] = None
    count: int = 1
    shared_policy: bool = True
    observation_space: Optional[Space] = None
    agent_cls: ClassVar[type[Agent]] = Agent

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError("count must be at least 1.")

        mechanisms = (
            self.mechanisms
            if isinstance(self.mechanisms, tuple)
            else (self.mechanisms,)
        )

        if not mechanisms:
            raise ValueError("mechanisms cannot be empty.")

        object.__setattr__(self, "mechanisms", mechanisms)

        if self.id is None:
            object.__setattr__(self, "id", self.agent_cls.__name__)

    def build(self) -> Agent:
        """Build a new agent from this configuration.

        Each mechanism configuration is built with the agent's identifier, and
        the resulting mechanisms are keyed by their own identifier. Every call
        returns an independent agent with independent mechanisms.

        Returns
        -------
        Agent
            An instance of ``agent_cls`` carrying ``id``, ``policy_id``, the
            built mechanisms and ``observation_space``.
        """
        mechanisms = {cfg.id: cfg.build(self.id) for cfg in self.mechanisms}
        return self.agent_cls(
            id=self.id,
            policy_id=self.policy_id,
            mechanisms=mechanisms,
            observation_space=self.observation_space,
        )
