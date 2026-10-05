"""Cart-pole benchmark: one balancing agent, an inert regulator dial.

The environment wraps the ``CartPole-v1`` task of Gymnasium (Barto, Sutton &
Anderson, 1983): a pole is hinged on a cart that moves along a frictionless
track, and the agent pushes the cart to the left or to the right at every step
to keep the pole upright. The state is ``[x, x_dot, theta, theta_dot]`` (cart
position in metres, cart velocity in metres per second, pole angle in radians,
pole angular velocity in radians per second). The reward is ``1`` for every
step played, so an episode return is the number of steps the pole stayed up. An
episode terminates when the pole angle exceeds 12 degrees or the cart leaves
the track (``|x| > 2.4``) and is truncated by Gymnasium's own 500-step limit,
or by the ``horizon`` of the environment when that is shorter.

The benchmark has no mechanism design problem of its own. It exists to check a
whole bilevel run on a task with a known solution. The outer level therefore
searches a single parameter of an inert mechanism, the ``dial`` of
``DialConfig``, that has no effect on the dynamics or the reward; the fitness
of a candidate is the mean per-step reward of the balancing agent
(see :mod:`examples.cartpole.regulator_env`). The dynamics, the reward and the
termination rules are those of Gymnasium and are not re-implemented here.

The agent holds one mechanism, ``push``, whose action is the discrete choice
``0`` (push left) or ``1`` (push right). The mechanism applies no residual: the
transition hook reads the decoded push from the state and steps the Gymnasium
environment, because the reward and the termination are only known after that
call. The reward is therefore written by the transition, at the step that was
just played, and not by ``Agent.reward``.

References
----------
Barto, A. G., Sutton, R. S., & Anderson, C. W. (1983). Neuronlike adaptive
elements that can solve difficult learning control problems. IEEE Transactions
on Systems, Man, and Cybernetics, SMC-13(5), 834-846.
https://doi.org/10.1109/TSMC.1983.6313077
"""

import logging
from typing import Any, ClassVar

import gymnasium
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType

from core.agents.base import Agent, AgentConfig
from core.envs.hooks import reset, transition
from core.envs.marl_regulated import MultiAgentEnv
from core.mechanism.base import MDPState, Mechanism
from core.mechanism.config import MechanismConfig

logger = logging.getLogger(__name__)

GYM_ENV_ID = "CartPole-v1"
"""Identifier of the Gymnasium task the environment wraps."""

PUSH_ID = "push"
"""Mechanism identifier of the discrete push of the balancing agent."""

DIAL_ID = "dial"
"""Mechanism identifier of the inert dial the regulator searches."""

OBSERVATION_SIZE = 4
"""Length of the observation ``[x, x_dot, theta, theta_dot]``."""

PUSH_SPACE = spaces.Discrete(2)
"""Action space of the push: ``0`` pushes the cart left, ``1`` pushes it right."""

DIAL_SPACE = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
"""Search space of the dial: one coordinate in ``[0, 1]``."""


class CartpoleAgent(Agent):
    """Balancing agent: sees the cart-pole state and is paid by the environment.

    The agent holds the ``Push`` mechanism (built from ``CartpoleAgentConfig``).
    Its ``observation`` returns the physical state of the cart-pole. It does
    not override ``reward``: the reward of a step comes from the Gymnasium call
    inside the transition hook of :class:`CartpoleRegulatedEnv`.

    When to use: build it through ``CartpoleAgentConfig``, which the environment
    does for you; instantiate it directly only to test the observation in
    isolation.

    Examples
    --------
    >>> import numpy as np
    >>> from core.mechanism.base import MDPState
    >>> agent = CartpoleAgent(id="agent_0:0", policy_id="cartpole", mechanisms={})
    >>> state = np.asarray([0.1, 0.0, -0.02, 0.5], dtype=np.float32)
    >>> mdp = MDPState(state={"cartpole": state})
    >>> agent.observation(mdp).obs["agent_0:0"][0].tolist()
    [0.10000000149011612, 0.0, -0.019999999552965164, 0.5]
    """

    def observation(self, mdp: MDPState) -> MDPState:
        """Return the cart-pole state at the current step as the observation.

        Parameters
        ----------
        mdp : MDPState
            Shared state whose ``state["cartpole"]`` trajectory holds the
            ``float32`` vector ``[x, x_dot, theta, theta_dot]`` (metres,
            metres per second, radians, radians per second) of every step.

        Returns
        -------
        MDPState
            An MDP whose ``obs`` holds, for this agent, the ``float32`` vector
            of shape ``(4,)`` read at ``mdp.t``.
        """

        state = np.asarray(mdp.state["cartpole"][mdp.t], dtype=np.float32)
        return MDPState(obs={self.id: state})


class CartpoleAgentConfig(AgentConfig):
    """Configuration that builds ``CartpoleAgent`` agents.

    A frozen dataclass inherited from :class:`core.agents.base.AgentConfig`;
    only ``agent_cls`` changes.

    Attributes
    ----------
    policy_id : str
        Identifier of the RL policy that controls the agent.
    mechanisms : MechanismConfig or tuple[MechanismConfig, ...]
        Mechanisms the agent holds, normally one ``PushConfig`` (id
        ``"push"``).
    id : str or None
        Agent type; instances are named ``"<id>:<index>"``.
    count : int
        Number of agents built from this configuration. The environment
        wraps a single cart-pole, so it must be ``1``.
    shared_policy : bool
        Whether the agents share one policy (default ``True``).
    observation_space : gymnasium.Space or None
        Observation space of the agent, ``Box(-inf, inf, (4,), float32)`` in
        the benchmark.

    When to use: in the ``agents`` of the inner optimizer configuration, once,
    with ``count=1``.

    Examples
    --------
    >>> config = CartpoleAgentConfig(
    ...     id="agent_0",
    ...     policy_id="cartpole_policy",
    ...     mechanisms=(PushConfig(action_space=PUSH_SPACE, id="push"),),
    ... )
    >>> sorted(config.build().mechanisms)
    ['push']
    """

    agent_cls: ClassVar[type[Agent]] = CartpoleAgent


class Push(Mechanism):
    """Discrete push of the cart; the Gymnasium step applies it.

    The mechanism only normalizes the raw policy output into an ``int`` in
    ``{0, 1}``. It contributes no residual to the state, because the effect of
    the push is computed by the Gymnasium environment inside the transition
    hook of :class:`CartpoleRegulatedEnv`, which reads the decoded value back
    from ``mdp.actions``.

    When to use: as the ``push`` mechanism of ``CartpoleAgentConfig``, through
    ``PushConfig``.

    Examples
    --------
    >>> from core.mechanism.base import MDPState
    >>> push = PushConfig(action_space=PUSH_SPACE, id="push").build("agent_0:0")
    >>> push.decode(None, 1)
    1
    >>> push.apply(MDPState(), 1).state.data
    {}
    """

    def decode(self, mdp: MDPState, action: ActType) -> ActType:
        """Turn the sampled action into a plain ``int``.

        The framework always passes the raw policy output, and the result is
        written back into the action trajectory for the transition to read.

        Parameters
        ----------
        mdp : MDPState
            Unused.
        action : ActType
            Sampled action: an ``int``, a ``numpy`` integer or an array holding
            one of them; only its first element is read.

        Returns
        -------
        ActType
            ``0`` (push left) or ``1`` (push right), as a Python ``int``.
        """

        return int(np.asarray(action).reshape(-1)[0])

    def apply(self, mdp: MDPState, action: ActType) -> MDPState:
        """Return an empty residual: the transition performs the push.

        Parameters
        ----------
        mdp : MDPState
            Unused.
        action : ActType
            Decoded push, unused here.

        Returns
        -------
        MDPState
            A state with no change.
        """

        return MDPState()


class PushConfig(MechanismConfig):
    """Configuration that builds the ``Push`` mechanism.

    A frozen dataclass inherited from
    :class:`core.mechanism.config.MechanismConfig`; only ``mechanism_cls``
    changes.

    Attributes
    ----------
    action_space : gymnasium.Space
        ``Discrete(2)`` in the benchmark.
    id : str or None
        Mechanism identifier, ``"push"``; the environment reads the action
        under that key.
    acts_on : tuple[str, str] or None
        Unused (default ``None``).
    obs_map : dict[str, str] or None
        Unused (default ``None``).
    default : numpy.ndarray or None
        Unused (default ``None``).

    When to use: in the ``mechanisms`` of ``CartpoleAgentConfig``.

    Examples
    --------
    >>> type(PushConfig(action_space=PUSH_SPACE, id="push").build("a:0")).__name__
    'Push'
    """

    mechanism_cls: ClassVar[type[Mechanism]] = Push


class Dial(Mechanism):
    """Inert one-parameter mechanism the outer level searches.

    The dial takes a value in ``[0, 1]`` and does nothing with it: ``apply``
    returns no residual and ``observe`` adds nothing to the observations. It
    keeps the bilevel loop well formed (the ES optimizer needs a mechanism with
    at least one searched coordinate) while the cart-pole has no regulation
    problem. It replaces the dimension-1 inert mechanism space of the earlier
    version of this example.

    When to use: as the only mechanism of the regulator agent of the
    cart-pole experiment, through ``DialConfig``.

    Examples
    --------
    >>> import numpy as np
    >>> from core.mechanism.base import MDPState
    >>> dial = DialConfig(action_space=DIAL_SPACE, id="dial").build("regulator")
    >>> dial.apply(MDPState(), np.asarray([0.3], dtype=np.float32)).state.data
    {}
    """

    def apply(self, mdp: MDPState, action: ActType) -> MDPState:
        """Return an empty residual: the dial changes nothing.

        Parameters
        ----------
        mdp : MDPState
            Unused.
        action : ActType
            Candidate value in ``[0, 1]``, unused.

        Returns
        -------
        MDPState
            A state with no change.
        """

        return MDPState()


class DialConfig(MechanismConfig):
    """Configuration that builds the inert ``Dial`` mechanism.

    A frozen dataclass inherited from
    :class:`core.mechanism.config.MechanismConfig`; only ``mechanism_cls``
    changes.

    Attributes
    ----------
    action_space : gymnasium.spaces.Box
        ``Box(0, 1, (1,), float32)``, the search space of the ES optimizer.
    id : str or None
        Mechanism identifier, ``"dial"``.
    acts_on : tuple[str, str] or None
        Unused (default ``None``).
    obs_map : dict[str, str] or None
        Unused (default ``None``).
    default : numpy.ndarray or None
        Unused (default ``None``).

    When to use: in the ``mechanisms`` of the regulator ``AgentConfig`` of the
    ES optimizer.

    Examples
    --------
    >>> type(DialConfig(action_space=DIAL_SPACE, id="dial").build("r")).__name__
    'Dial'
    """

    mechanism_cls: ClassVar[type[Mechanism]] = Dial


class CartpoleRegulatedEnv(MultiAgentEnv):
    """Cart-pole that the inner level of the bilevel loop trains on.

    The environment owns one ``CartPole-v1`` instance of Gymnasium. Its reset
    hook, ``reset_cartpole``, resets that instance and its transition hook,
    ``step_cartpole``, plays the push the agent chose, pays the Gymnasium
    reward and records whether the episode terminated or was truncated. The
    ``termination`` method of the base class is extended so that those two
    flags reach RLlib.

    Parameters
    ----------
    **kwargs : Any
        Forwarded to :class:`core.envs.marl_regulated.MultiAgentEnv`: ``world``,
        ``mechanism_id``, ``horizon``, ``agents_cfg_dict`` (exactly one
        follower), ``leaders_cfg_dict``, ``seed``, ``schema``
        (:class:`examples.cartpole.metric_schema.CartpoleMetricSchema`; the
        environment logs through it and cannot reset without one),
        ``reporter_cfg`` and the other options of the base class.

    Attributes
    ----------
    gym_env : gymnasium.Env
        The wrapped ``CartPole-v1`` instance.
    aid : str
        Identifier of the single follower.

    Raises
    ------
    ValueError
        If ``agents_cfg_dict`` does not hold exactly one agent.

    When to use: as the ``env`` of the inner (society) optimizer of the
    cart-pole experiment, with one ``CartpoleAgentConfig`` of ``count=1``.

    Examples
    --------
    The first reset seeds Gymnasium, so two environments with the same seed
    start from the same state:

    >>> from examples.cartpole.metric_schema import CartpoleMetricSchema
    >>> from core.mechanism.base import MDPState
    >>> def build(seed):
    ...     cfg = CartpoleAgentConfig(
    ...         id="agent_0:0",
    ...         policy_id="cartpole_policy",
    ...         mechanisms=(PushConfig(action_space=PUSH_SPACE, id="push"),),
    ...     )
    ...     return CartpoleRegulatedEnv(
    ...         world=None,
    ...         mechanism_id=0,
    ...         seed=seed,
    ...         agents_cfg_dict={"agent_0:0": cfg},
    ...         schema=CartpoleMetricSchema,
    ...     )
    >>> a = build(3).reset_cartpole(MDPState()).state["cartpole"]
    >>> b = build(3).reset_cartpole(MDPState()).state["cartpole"]
    >>> bool((np.asarray(a) == np.asarray(b)).all()), np.asarray(a).shape
    (True, (1, 4))
    """

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

        if len(self.followers) != 1:
            raise ValueError(
                "CartpoleRegulatedEnv wraps one cart-pole and needs exactly one "
                + f"agent, got {len(self.followers)}."
            )
        (self.aid,) = self.followers
        self.gym_env = gymnasium.make(GYM_ENV_ID)
        self._seeded = False

    @reset
    def reset_cartpole(self, mdp: MDPState) -> MDPState:
        """Reset the Gymnasium task and expose its state to the agent.

        The first reset seeds Gymnasium with the ``seed`` of the environment;
        later resets continue its random stream, so an environment seeded once
        replays the same sequence of episodes.

        Parameters
        ----------
        mdp : MDPState
            State being reset; only used to satisfy the hook signature.

        Returns
        -------
        MDPState
            A residual with ``state["cartpole"]`` (the ``float32`` vector
            ``[x, x_dot, theta, theta_dot]``) and the ``terminated`` and
            ``truncated`` flags of the step, both ``0.0``.

        Examples
        --------
        >>> from core.mechanism.base import MDPState
        >>> from examples.cartpole.metric_schema import CartpoleMetricSchema
        >>> cfg = CartpoleAgentConfig(
        ...     id="agent_0:0",
        ...     policy_id="cartpole_policy",
        ...     mechanisms=(PushConfig(action_space=PUSH_SPACE, id="push"),),
        ... )
        >>> env = CartpoleRegulatedEnv(
        ...     world=None,
        ...     mechanism_id=0,
        ...     seed=0,
        ...     agents_cfg_dict={"agent_0:0": cfg},
        ...     schema=CartpoleMetricSchema,
        ... )
        >>> residual = env.reset_cartpole(MDPState())
        >>> residual.state["terminated"], residual.state["truncated"]
        ([0.0], [0.0])
        """

        seed = None if self._seeded else self.seed
        observation, _ = self.gym_env.reset(seed=seed)
        self._seeded = True
        return MDPState(
            state={
                "cartpole": np.asarray(observation, dtype=np.float32),
                "terminated": 0.0,
                "truncated": 0.0,
            }
        )

    @transition
    def step_cartpole(self, mdp: MDPState) -> MDPState:
        """Play the agent's push and pay the Gymnasium reward.

        The decoded push stored at ``mdp.t`` is given to ``CartPole-v1``. The
        reward Gymnasium returns (``1.0`` while the episode goes on, including
        on the terminating step) is added to the agent's reward at the step
        that was just played, and the new state and the two end-of-episode
        flags are appended to the trajectories. The method pushes the cart
        position, the pole angle and the absolute pole angle to the metric
        logger.

        Parameters
        ----------
        mdp : MDPState
            Shared state holding the agent's ``push`` action at ``mdp.t``.

        Returns
        -------
        MDPState
            The state advanced by one time index, with the reward of the step
            added at the old index.
        """

        push = int(np.asarray(mdp.actions[self.aid][PUSH_ID][mdp.t]).reshape(-1)[0])
        observation, reward, terminated, truncated, _ = self.gym_env.step(push)
        observation = np.asarray(observation, dtype=np.float32)

        if self.logger is not None:
            self.logger.push(key=("cart_position",), value=float(observation[0]))
            self.logger.push(key=("pole_angle",), value=float(observation[2]))
            self.logger.push(
                key=("pole_angle_abs_max",), value=abs(float(observation[2]))
            )

        mdp = mdp.add(MDPState(rewards={self.aid: float(reward)}))
        return mdp.advance(
            state={
                "cartpole": observation,
                "terminated": float(terminated),
                "truncated": float(truncated),
            }
        )

    def termination(self, mdp: MDPState) -> MDPState:
        """Combine the horizon with the end-of-episode flags of Gymnasium.

        The base class truncates every agent once ``mdp.t`` reaches
        ``horizon``. This method keeps that and adds the flags the transition
        recorded: a terminated Gymnasium episode terminates every agent and
        ``"__all__"``, and a truncated one (Gymnasium's 500-step limit)
        truncates them.

        Parameters
        ----------
        mdp : MDPState
            State after the transition.

        Returns
        -------
        MDPState
            The same ``mdp`` object, with its ``terminateds`` and
            ``truncateds`` replaced.
        """

        mdp = super().termination(mdp)
        terminated = bool(mdp.state["terminated"][mdp.t])
        truncated = bool(mdp.state["truncated"][mdp.t])

        mdp.terminateds = {aid: terminated for aid in self.agents}
        mdp.terminateds["__all__"] = terminated
        mdp.truncateds = {
            aid: bool(flag) or truncated for aid, flag in mdp.truncateds.items()
        }
        return mdp
