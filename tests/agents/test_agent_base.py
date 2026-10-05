"""``core.agents.base``: ``Agent`` behaviour and ``AgentConfig`` validation.

``Agent.action`` is where an agent's mechanisms meet the shared ``MDPState``:
each mechanism the agent holds an action for is called with that action at the
current timestep and the residuals it returns are added to the state. The
observation side of the agent (``mechanism_observations``) is tested in
``test_mechanism_observations.py``.

``AgentConfig.count`` and ``AgentConfig.shared_policy`` are stored but not read
by ``build``, which always returns one agent.
"""

import dataclasses
from typing import ClassVar

import numpy as np
import pytest
from gymnasium import spaces

from core.agents.base import Agent, AgentConfig
from core.envs import hooks
from core.mechanism.base import MDPState, Mechanism
from core.mechanism.config import MechanismConfig
from core.utils import sigmoid

BOX = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)


class Adder(Mechanism):
    """Add the (decoded) action to the ``total`` state entry."""

    def decode(self, mdp: MDPState, action):
        return float(np.asarray(action).reshape(-1)[0]) * 10.0

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
def test_subclass_hooks_are_recorded_by_name():
    def spaces_hook(self):
        return None

    spaces_hook.observation_spaces = True

    class Hooked(Agent):
        @hooks.action
        def pick(self, mdp):
            return mdp

        @hooks.reward
        def pay(self, mdp):
            return mdp

        @hooks.observation
        def look(self, mdp):
            return mdp

        declare_spaces = spaces_hook

    assert Hooked._action == "pick"
    assert Hooked._reward == "pay"
    assert Hooked._observation == "look"
    assert Hooked._observation_spaces == "declare_spaces"
    assert (Agent._action, Agent._reward, Agent._observation) == (None, None, None)
    assert Agent._observation_spaces is None


@pytest.mark.unit
def test_normalize_action_squashes_each_component_with_a_temperature_of_four():
    agent = make_agent()

    normalized = agent._normalize_action([[0.0, 8.0], [-8.0, 4.0]])

    assert normalized.dtype == np.float32 and normalized.shape == (4,)
    expected = [sigmoid(0.0), sigmoid(2.0), sigmoid(-2.0), sigmoid(1.0)]
    np.testing.assert_allclose(normalized, expected, rtol=1e-6)


@pytest.mark.unit
def test_action_without_an_entry_for_the_agent_returns_the_same_mdp():
    mdp = MDPState(actions={"someone_else": {"m": np.array([1.0])}})

    assert make_agent("m").action(mdp) is mdp


@pytest.mark.unit
def test_action_with_an_empty_entry_returns_the_same_mdp():
    mdp = MDPState(actions={"a0": {}})

    assert make_agent("m").action(mdp) is mdp


@pytest.mark.unit
def test_action_applies_every_mechanism_it_holds_an_action_for():
    agent = make_agent("m1", "m2")
    mdp = mdp_holding({"m1": [0.1], "m2": [0.2]})

    result = agent.action(mdp)

    # Both decoded actions (x10) are summed into the state.
    assert result.state["total"] == pytest.approx([3.0])


@pytest.mark.unit
def test_action_ignores_actions_for_mechanisms_the_agent_does_not_own():
    agent = make_agent("m1")
    mdp = mdp_holding({"m1": [0.1], "stranger": [5.0]})

    result = agent.action(mdp)

    assert result.state["total"] == pytest.approx([1.0])


@pytest.mark.unit
def test_action_uses_the_entry_of_the_current_timestep():
    agent = make_agent("m")
    mdp = mdp_holding({"m": [0.1, 0.3]}, t=1)

    result = agent.action(mdp)

    assert result.state["total"] == pytest.approx([0.0, 3.0])


@pytest.mark.unit
def test_action_records_the_decoded_action_in_the_mdp():
    agent = make_agent("m")
    mdp = mdp_holding({"m": [0.1]})

    result = agent.action(mdp)

    assert result.actions["a0"]["m"] == pytest.approx([1.0])


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
