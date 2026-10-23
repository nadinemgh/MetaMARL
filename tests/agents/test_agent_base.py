"""``core.agents.base``: ``Agent`` behaviour and ``AgentConfig`` validation.

``Agent.action`` is where an agent's mechanisms meet the shared ``MDPState``:
each mechanism the agent holds an action for is called with that (already
decoded) action at the current timestep, and the residuals it returns are
combined into one residual ``MDPState`` that the caller composes with the shared
state. The agent neither decodes the actions nor writes them back.

``AgentConfig.count`` and ``AgentConfig.shared_policy`` are stored but not read
by ``build``, which always returns one agent.
"""

import dataclasses
import subprocess
import sys
from typing import ClassVar

import numpy as np
import pytest
from gymnasium import spaces

from core.agents.base import Agent, AgentConfig
from core.mechanism.base import MDPState, Mechanism
from core.mechanism.config import MechanismConfig

BOX = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)


class Adder(Mechanism):
    """Add the (already decoded) action to the ``total`` state entry."""

    def apply(self, mdp: MDPState, action: float) -> MDPState:
        return MDPState(state={"total": action})


class AdderConfig(MechanismConfig):
    mechanism_cls: ClassVar[type[Mechanism]] = Adder


def make_agent(*mechanism_ids: str) -> Agent:
    mechanisms = {
        mid: Adder(aid="a0", id=mid, action_space=BOX, acts_on=None)
        for mid in mechanism_ids
    }

    return Agent(id="a0", policy_id="p", mechanisms=mechanisms)


def mdp_holding(actions: dict, *, t: int = 0) -> MDPState:
    """MDP at step ``t`` whose ``total`` entry exists and starts at zero."""

    mdp = MDPState(state={"total": 0.0}, actions={"a0": actions})

    for _ in range(t):
        mdp = mdp.advance(state={"total": 0.0})

    return mdp


@pytest.mark.unit
def test_constructor_stores_its_arguments():
    mechanisms = {"m": object()}
    agent = Agent(
        id="a0", policy_id="policy", mechanisms=mechanisms, observation_space=BOX
    )

    assert (agent.id, agent.policy_id) == ("a0", "policy")
    assert agent.mechanisms is mechanisms
    assert agent.observation_space is BOX


@pytest.mark.unit
def test_observation_space_defaults_to_none():
    assert Agent(id="a", policy_id="p", mechanisms={}).observation_space is None


@pytest.mark.unit
def test_default_observation_and_reward_contribute_nothing():
    agent = make_agent()
    mdp = MDPState(aids={"a0"})

    for contribution in (agent.observation(mdp), agent.reward(mdp)):
        assert contribution.obs.data == {} and contribution.rewards.data == {}


@pytest.mark.unit
def test_action_without_an_entry_for_the_agent_returns_an_empty_residual():
    mdp = MDPState(actions={"someone_else": {"m": np.array([1.0])}})
    residual = make_agent("m").action(mdp)

    assert residual is not mdp
    assert residual.state.data == {} and residual.actions.data == {}


@pytest.mark.unit
def test_action_with_an_empty_entry_returns_an_empty_residual():
    mdp = MDPState(actions={"a0": {}})
    residual = make_agent("m").action(mdp)

    assert residual is not mdp
    assert residual.state.data == {} and residual.actions.data == {}


@pytest.mark.unit
def test_action_applies_every_mechanism_it_holds_an_action_for():
    agent = make_agent("m1", "m2")
    mdp = mdp_holding({"m1": [1.0], "m2": [2.0]})
    result = agent.action(mdp)

    # Both actions are summed into the residual.
    assert result.state["total"] == pytest.approx([3.0])


@pytest.mark.unit
def test_action_ignores_actions_for_mechanisms_the_agent_does_not_own():
    agent = make_agent("m1")
    mdp = mdp_holding({"m1": [1.0], "stranger": [5.0]})
    result = agent.action(mdp)

    assert result.state["total"] == pytest.approx([1.0])


@pytest.mark.unit
def test_action_uses_the_entry_of_the_current_timestep():
    agent = make_agent("m")
    mdp = mdp_holding({"m": [1.0, 3.0]}, t=1)
    result = agent.action(mdp)

    # The residual starts at step 0 whatever the step of ``mdp``, and holds the
    # action of the current step of ``mdp`` (3.0), not that of step 0 (1.0).
    assert result.state["total"] == pytest.approx([3.0])


@pytest.mark.unit
def test_action_does_not_write_the_action_back():
    agent = make_agent("m")
    mdp = mdp_holding({"m": [1.0]})
    result = agent.action(mdp)

    # Decoding happens in the adapter and in ``env.reset``: the mechanism uses
    # the action as stored and leaves the action trajectories alone.
    assert result.actions.data == {}
    assert mdp.actions["a0"]["m"] == pytest.approx([1.0])


@pytest.mark.unit
def test_config_wraps_a_single_mechanism_config_in_a_tuple():
    mechanism = AdderConfig(action_space=BOX, id="m")
    config = AgentConfig(policy_id="p", mechanisms=mechanism)

    assert config.mechanisms == (mechanism,)


@pytest.mark.unit
def test_config_keeps_a_tuple_of_mechanism_configs():
    mechanisms = (AdderConfig(action_space=BOX, id="m1"), AdderConfig(BOX, id="m2"))
    assert AgentConfig(policy_id="p", mechanisms=mechanisms).mechanisms == mechanisms


@pytest.mark.unit
def test_config_id_defaults_to_the_agent_class_name():
    class Fisherman(Agent):
        pass

    class FishermanConfig(AgentConfig):
        agent_cls: ClassVar[type[Agent]] = Fisherman

    mechanism = AdderConfig(action_space=BOX, id="m")
    assert AgentConfig(policy_id="p", mechanisms=mechanism).id == "Agent"
    assert FishermanConfig(policy_id="p", mechanisms=mechanism).id == "Fisherman"
    assert FishermanConfig(policy_id="p", mechanisms=mechanism, id="x").id == "x"


@pytest.mark.unit
@pytest.mark.parametrize("count", [0, -2])
def test_config_rejects_a_count_below_one(count):
    mechanism = AdderConfig(action_space=BOX, id="m")

    with pytest.raises(ValueError, match="count must be at least 1"):
        AgentConfig(policy_id="p", mechanisms=mechanism, count=count)


@pytest.mark.unit
def test_config_rejects_an_empty_tuple_of_mechanisms():
    with pytest.raises(ValueError, match="mechanisms cannot be empty"):
        AgentConfig(policy_id="p", mechanisms=())


@pytest.mark.unit
def test_config_is_frozen():
    config = AgentConfig(policy_id="p", mechanisms=AdderConfig(BOX, id="m"))

    with pytest.raises(dataclasses.FrozenInstanceError):
        config.policy_id = "other"


@pytest.mark.unit
def test_build_returns_an_agent_holding_mechanisms_keyed_by_their_id():
    class Fisherman(Agent):
        pass

    class FishermanConfig(AgentConfig):
        agent_cls: ClassVar[type[Agent]] = Fisherman

    config = FishermanConfig(
        id="f0",
        policy_id="fisher_policy",
        observation_space=BOX,
        mechanisms=(
            AdderConfig(action_space=BOX, id="m1"),
            AdderConfig(action_space=BOX, id="m2", acts_on=("fisherman", "harvest")),
        ),
    )
    agent = config.build()

    assert type(agent) is Fisherman
    assert (agent.id, agent.policy_id) == ("f0", "fisher_policy")
    assert agent.observation_space is BOX
    assert list(agent.mechanisms) == ["m1", "m2"]
    assert all(
        isinstance(m, Adder) and m.aid == "f0" for m in agent.mechanisms.values()
    )
    assert agent.mechanisms["m2"].acts_on == ("fisherman", "harvest")
    assert agent.mechanisms["m1"].action_space is BOX


@pytest.mark.unit
def test_every_build_creates_independent_agents():
    config = AgentConfig(id="a", policy_id="p", mechanisms=AdderConfig(BOX, id="m"))
    first, second = config.build(), config.build()

    assert first is not second
    assert first.mechanisms["m"] is not second.mechanisms["m"]


@pytest.mark.unit
def test_a_mechanism_can_be_built_with_a_default_of_several_entries():
    default = np.asarray([0.5, 0.2], dtype=np.float32)
    config = AgentConfig(
        id="a",
        policy_id="p",
        mechanisms=AdderConfig(action_space=BOX, id="m", default=default),
    )
    agent = config.build()

    assert isinstance(agent.mechanisms["m"], Adder)
    np.testing.assert_array_equal(agent.mechanisms["m"]._u, default)


@pytest.mark.unit
def test_importing_the_agents_does_not_import_the_unmaintained_gym_package():
    # ``gym`` prints a deprecation notice when it is imported; the framework
    # uses ``gymnasium``. A fresh interpreter is needed: this process may have
    # imported ``gym`` through another module.
    code = "import sys, core.agents.base; print('gym' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )

    assert result.stdout.strip() == "False"
    assert "unmaintained" not in result.stderr
