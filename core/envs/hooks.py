"""Decorators that mark the hook methods of agents and environments.

A benchmark customises the framework by subclassing ``Agent`` or
``MultiAgentEnv`` and decorating the methods that play a given role. Each
decorator in this module only sets a boolean attribute named after the hook on
the function and returns the same function, so a decorated method behaves like
any other method. The subclass hooks (``Agent.__init_subclass__`` and
``MultiAgentEnv.__init_subclass__``) then scan the class body for those
attributes and record the name of the marked method in a class variable.
"""

from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def _hook(func: F, name: str) -> F:
    setattr(func, name, True)
    return func


def action(func: F) -> F:
    """Mark a method as the action hook of an agent.

    The decorator sets ``func.action = True`` and returns ``func`` itself.
    ``Agent.__init_subclass__`` records the name of the marked method in the
    class variable ``_action``. The environments call the agent method named
    ``action`` directly, so the recorded name is informational: a marked method
    with another name is recorded but not called.

    Parameters
    ----------
    func : Callable
        Method to mark, typically ``def action(self, mdp) -> MDPState``.

    Returns
    -------
    Callable
        The same function, with its ``action`` attribute set to ``True``.

    When to use: on the method of an ``Agent`` subclass that applies the agent's
    mechanisms to the shared MDP state, when you want the role to be explicit
    in the class body.

    Examples
    --------
    >>> from core.agents.base import Agent
    >>> class Picker(Agent):
    ...     @action
    ...     def pick(self, mdp):
    ...         return mdp
    >>> Picker._action
    'pick'
    >>> Picker.pick.action
    True
    """
    return _hook(func, "action")


def observation(func: F) -> F:
    """Mark a method as the observation hook of an agent.

    The decorator sets ``func.observation = True`` and returns ``func`` itself.
    ``Agent.__init_subclass__`` records the name of the marked method in the
    class variable ``_observation``. The environments call the agent method
    named ``observation`` directly, so the recorded name is informational: a
    marked method with another name is recorded but not called.

    Parameters
    ----------
    func : Callable
        Method to mark, typically ``def observation(self, mdp) -> MDPState``.

    Returns
    -------
    Callable
        The same function, with its ``observation`` attribute set to ``True``.

    When to use: on the method of an ``Agent`` subclass that builds the
    agent's observation vector from the MDP state, when you want the role to
    be explicit in the class body.

    Examples
    --------
    >>> from core.agents.base import Agent
    >>> class Watcher(Agent):
    ...     @observation
    ...     def look(self, mdp):
    ...         return mdp
    >>> Watcher._observation
    'look'
    """
    return _hook(func, "observation")


def reset(func: F) -> F:
    """Mark a method as the reset hook of an environment.

    The decorator sets ``func.reset = True`` and returns ``func`` itself.
    ``MultiAgentEnv.__init_subclass__`` records the name of the marked method
    in the class variable ``_reset``. At each episode start,
    ``MultiAgentEnv.reset`` calls that method with the current ``MDPState`` and
    adds the ``MDPState`` it returns (the initial state, parameters and any
    observation entries) to the episode's MDP.

    Parameters
    ----------
    func : Callable
        Method to mark, ``def reset_hook(self, mdp: MDPState) -> MDPState``.

    Returns
    -------
    Callable
        The same function, with its ``reset`` attribute set to ``True``.

    When to use: on the one method of a ``MultiAgentEnv`` subclass that draws
    the initial state of an episode, such as the starting fish stock. When two
    methods of the same class carry the mark, the last one in the class body
    is recorded.

    Examples
    --------
    >>> from core.envs.marl_regulated import MultiAgentEnv
    >>> class Env(MultiAgentEnv):
    ...     @reset
    ...     def start(self, mdp):
    ...         return mdp
    >>> Env._reset
    'start'
    >>> MultiAgentEnv._reset is None
    True
    """
    return _hook(func, "reset")


def reward(func: F) -> F:
    """Mark a method as the reward hook of an agent.

    The decorator sets ``func.reward = True`` and returns ``func`` itself.
    ``Agent.__init_subclass__`` records the name of the marked method in the
    class variable ``_reward``. The environments call the agent method named
    ``reward`` directly, so the recorded name is informational: a marked method
    with another name is recorded but not called.

    Parameters
    ----------
    func : Callable
        Method to mark, typically ``def reward(self, mdp) -> MDPState``.

    Returns
    -------
    Callable
        The same function, with its ``reward`` attribute set to ``True``.

    When to use: on the method of an ``Agent`` subclass that returns the
    agent's per-step reward, when you want the role to be explicit in the
    class body.

    Examples
    --------
    >>> from core.agents.base import Agent
    >>> class Earner(Agent):
    ...     @reward
    ...     def pay(self, mdp):
    ...         return mdp
    >>> Earner._reward
    'pay'
    """
    return _hook(func, "reward")


def transition(func: F) -> F:
    """Mark a method as the transition hook of an environment.

    The decorator sets ``func.transition = True`` and returns ``func`` itself.
    ``MultiAgentEnv.__init_subclass__`` records the name of the marked method
    in the class variable ``_transition``. ``MultiAgentEnv.transition`` calls
    that method with the keyword argument ``mdp`` once per step, after every
    agent has acted, and uses the value it returns as the new MDP state.

    Parameters
    ----------
    func : Callable
        Method to mark, ``def transition_hook(self, mdp: MDPState) -> MDPState``;
        the parameter must be accepted by keyword.

    Returns
    -------
    Callable
        The same function, with its ``transition`` attribute set to ``True``.

    When to use: on the one method of a ``MultiAgentEnv`` subclass that
    advances the shared state by one step, such as the stock dynamics of a
    population model. When two methods of the same class carry the mark, the
    last one in the class body is recorded.

    Examples
    --------
    >>> from core.envs.marl_regulated import MultiAgentEnv
    >>> class Env(MultiAgentEnv):
    ...     @transition
    ...     def advance_stock(self, mdp):
    ...         return mdp
    >>> Env._transition
    'advance_stock'
    """
    return _hook(func, "transition")
