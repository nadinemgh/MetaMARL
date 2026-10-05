"""Hook decorators and their discovery by ``MultiAgentEnv`` subclasses.

A hook decorator only marks a function with an attribute named after the hook.
``MultiAgentEnv.__init_subclass__`` then records, as class variables, the name
of the method that carries the ``reset`` or ``transition`` mark. The
``action``, ``observation`` and ``reward`` marks are read by
``Agent.__init_subclass__`` instead (see ``tests/agents/test_agent_base.py``).
"""

import pytest

from core.envs import hooks
from core.envs.marl_regulated import MultiAgentEnv
from core.mechanism.base import MDPState

ALL_HOOKS = ["reset", "action", "reward", "observation", "transition"]
ENV_HOOKS = ["reset", "transition"]


@pytest.mark.unit
@pytest.mark.parametrize("hook", ALL_HOOKS)
def test_decorator_marks_the_function_and_returns_it(hook):
    def function():
        return None

    decorated = getattr(hooks, hook)(function)

    assert decorated is function
    assert getattr(function, hook) is True
    assert [h for h in ALL_HOOKS if getattr(function, h, False)] == [hook]


@pytest.mark.unit
@pytest.mark.parametrize("hook", ENV_HOOKS)
def test_subclass_records_the_name_of_the_marked_method(hook):
    class Env(MultiAgentEnv):
        @getattr(hooks, hook)
        def my_hook(self, mdp):
            return MDPState()

    assert getattr(Env, f"_{hook}") == "my_hook"
    other = next(h for h in ENV_HOOKS if h != hook)
    assert getattr(Env, f"_{other}") is None


@pytest.mark.unit
@pytest.mark.parametrize("hook", ["action", "reward", "observation"])
def test_agent_hooks_are_not_recorded_by_the_environment(hook):
    class Env(MultiAgentEnv):
        @getattr(hooks, hook)
        def my_hook(self, mdp):
            return MDPState()

    assert (Env._reset, Env._transition) == (None, None)


@pytest.mark.unit
def test_inherited_hooks_are_kept_and_can_be_redeclared():
    class Parent(MultiAgentEnv):
        @hooks.reset
        def parent_reset(self, mdp):
            return MDPState()

    class Child(Parent):
        @hooks.transition
        def child_transition(self, mdp):
            return mdp

    class Override(Parent):
        @hooks.reset
        def child_reset(self, mdp):
            return MDPState()

    assert (Child._reset, Child._transition) == ("parent_reset", "child_transition")
    assert Override._reset == "child_reset"
    # A subclass never changes what its parents or the base class record.
    assert (Parent._transition, MultiAgentEnv._reset, MultiAgentEnv._transition) == (
        None,
        None,
        None,
    )


@pytest.mark.unit
def test_transition_calls_the_hook_with_the_mdp_keyword(toy):
    seen = {}

    class Env(toy.ToyEnv):
        @hooks.transition
        def spy(self, *, mdp):
            seen["mdp"] = mdp
            return mdp.advance(state={"stock": 42.0})

    env = toy.make_env(toy.ScriptedWorld(), env_cls=Env)
    mdp = MDPState(state={"stock": 1.0})

    advanced = env.transition(mdp)

    assert seen["mdp"] is mdp
    assert advanced.t == 1
    assert advanced.state["stock"] == [1.0, 42.0]


@pytest.mark.unit
def test_transition_without_a_hook_returns_the_mdp_unchanged(toy):
    class Bare(MultiAgentEnv):
        pass

    env = toy.make_env(toy.ScriptedWorld(), env_cls=Bare)
    mdp = MDPState(t=2)

    assert env.transition(mdp) is mdp
