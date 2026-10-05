"""Decorators that mark the hook methods of environments.

A benchmark customises the framework by subclassing ``MultiAgentEnv`` and
decorating the methods that draw the initial state (``reset``) and advance it
(``transition``). Each decorator only sets a boolean attribute named after the
hook on the function and returns the same function, so a decorated method
behaves like any other method. ``MultiAgentEnv.__init_subclass__`` then scans
the class body for those attributes and records the name of the marked method
in a class variable. Agents need no decorator: an ``Agent`` subclass overrides
its ``action``, ``reward`` and ``observation`` methods by name.
"""

from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def _hook(func: F, name: str) -> F:
    setattr(func, name, True)
    return func


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
    the initial state of an episode, such as the starting fish stock. A class
    body that marks two methods raises ``TypeError``.

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
    population model. A class body that marks two methods raises
    ``TypeError``.

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
