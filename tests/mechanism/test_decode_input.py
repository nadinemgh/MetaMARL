"""``decode`` runs once per raw action, outside the mechanism call.

The action of a leader is written once, when the episode starts, and is carried
forward from step to step like any stock. A mechanism whose ``decode`` is not
idempotent would receive its own output again if the call decoded the stored
action at every step. Decoding therefore lives in the callers: the Ray adaptor
decodes a follower's action as it arrives, and ``MultiAgentEnv.reset`` decodes
a leader's candidate or default once. The stored action is the decoded one,
``Mechanism.__call__`` uses it as it is and neither decodes it nor writes it
back, and the state holds no separate ``raw_actions`` trajectory.

The first class shows that a ``decode`` that is not idempotent delivers a
constant action when the caller decodes once. The second class checks that the
state holds the decoded actions only. The third class shows that the four
mechanisms of the package, whose ``decode`` is idempotent, give bit-identical
results to the earlier behaviour, in which the call decoded the stored action
at every step and wrote the result back.
"""

import copy

import numpy as np
import pytest
from gymnasium import spaces

from core.agents.base import Agent
from core.mechanism.algorithms.incentive import Incentive
from core.mechanism.algorithms.penalty import ThresholdPenalty
from core.mechanism.algorithms.quota import Quota
from core.mechanism.algorithms.social_influence import SocialInfluence
from core.mechanism.base import MDPState, Mechanism

REGULATOR = "regulator"
K = 1_000.0
UNIT = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
NOTHING = np.empty(0, dtype=np.float32)
NO_PARAMETER = spaces.Box(low=0.0, high=1.0, shape=(0,), dtype=np.float32)


class Halver(Mechanism):
    """Mechanism whose ``decode`` halves the action, so it is not idempotent."""

    def decode(self, mdp, action):
        return 0.5 * float(np.asarray(action).reshape(-1)[0])

    def apply(self, mdp, action):
        return MDPState(rewards={"fisherman:0": action})


def decode_actions(agent, mdp, raw=None):
    """Decode ``raw`` (or the stored actions of the current step) once and store it.

    This is what the callers of ``Mechanism.__call__`` do: the Ray adaptor for
    the followers at each step and ``MultiAgentEnv.reset`` for the leaders.
    """

    if raw is None:
        raw = {
            mid: acts[mdp.t]
            for mid, acts in mdp.actions.data[agent.id].items()
            if mid in agent.mechanisms
        }

    decoded = {
        agent.id: {
            mid: agent.mechanisms[mid].decode(mdp, value) for mid, value in raw.items()
        }
    }

    return mdp.update(actions=decoded)


def run_as_an_environment(agent, mdp, steps, followers=None):
    """Play ``steps`` steps the way an environment does, and record each one.

    The raw action of the first step is decoded once, as ``reset`` does. Each
    step then lets the agent act, composes its residual with the state,
    advances the state, and stores the new action of the step after decoding
    it, as the adaptor does. Returns the list of the states reached after each
    ``Agent.action``.
    """

    reached = []
    mdp = decode_actions(agent, mdp)

    for step in range(steps):
        mdp = mdp.add(agent.action(mdp))

        reached.append(mdp)

        mdp = mdp.advance(state={"fish": mdp.state["fish"][mdp.t] * 0.9})

        if followers is not None:
            # Writing at the new step carries the other actions forward, as the
            # adaptor does when it stores the followers' actions.
            new = followers(step + 1)

            mdp = (
                decode_actions(agent, mdp, new[agent.id])
                if agent.id in new
                else mdp.update(actions={})
            )

    return reached


@pytest.mark.unit
class TestNonIdempotentDecode:
    def make_agent(self):
        halver = Halver(
            aid=REGULATOR, id="halve", action_space=UNIT, acts_on=("fisherman", "x")
        )

        return Agent(id=REGULATOR, policy_id="p", mechanisms={"halve": halver})

    def test_the_delivered_action_does_not_compound_over_the_steps(self):
        mdp = MDPState(
            state={"fish": 100.0}, actions={REGULATOR: {"halve": np.array([0.3])}}
        )
        reached = run_as_an_environment(
            self.make_agent(), mdp, steps=4, followers=lambda t: {}
        )
        delivered = [state.actions[REGULATOR]["halve"][state.t] for state in reached]

        assert delivered == pytest.approx([0.15, 0.15, 0.15, 0.15])

    def test_the_effect_of_the_mechanism_does_not_decay_over_the_steps(self):
        mdp = MDPState(
            state={"fish": 100.0}, actions={REGULATOR: {"halve": np.array([0.3])}}
        )
        reached = run_as_an_environment(
            self.make_agent(), mdp, steps=4, followers=lambda t: {}
        )
        rewards = [state.rewards["fisherman:0"][state.t] for state in reached]

        assert rewards == pytest.approx([0.15, 0.15, 0.15, 0.15])

    def test_the_stored_action_is_the_decoded_one_and_stays_so(self):
        mdp = MDPState(
            state={"fish": 100.0}, actions={REGULATOR: {"halve": np.array([0.3])}}
        )
        reached = run_as_an_environment(
            self.make_agent(), mdp, steps=3, followers=lambda t: {}
        )
        stored = [state.actions[REGULATOR]["halve"][state.t] for state in reached]

        # Decoded once (0.3 / 2) and carried forward, never decoded again.
        assert stored == pytest.approx([0.15, 0.15, 0.15])

    def test_a_new_raw_action_of_a_follower_is_decoded_once_per_step(self):
        halver = Halver(
            aid="fisherman:0", id="halve", action_space=UNIT, acts_on=("a", "b")
        )
        agent = Agent(id="fisherman:0", policy_id="p", mechanisms={"halve": halver})
        raw = [0.8, 0.4, 0.2]
        mdp = MDPState(
            state={"fish": 100.0}, actions={"fisherman:0": {"halve": raw[0]}}
        )

        def next_raw(step):
            return {"fisherman:0": {"halve": raw[step]}} if step < len(raw) else {}

        reached = run_as_an_environment(agent, mdp, steps=3, followers=next_raw)
        delivered = [s.actions["fisherman:0"]["halve"][s.t] for s in reached]

        assert delivered == pytest.approx([0.4, 0.2, 0.1])


@pytest.mark.unit
class TestActionsTrajectory:
    def test_the_state_holds_the_decoded_actions_only(self):
        mdp = MDPState(actions={"a": {"m": 1.0}})

        assert not hasattr(mdp, "raw_actions")

        with pytest.raises(TypeError, match="raw_actions"):
            MDPState(actions={"a": {"m": 1.0}}, raw_actions={"a": {"m": 3.0}})

    def test_calling_a_mechanism_leaves_the_stored_action_alone(self):
        halver = Halver(aid="a", id="m", action_space=UNIT, acts_on=("f", "h"))
        mdp = MDPState(actions={"a": {"m": 0.4}})

        residual = halver(mdp, 0.4)

        # The call uses the action as given: it does not decode it again and
        # does not write anything back.
        assert residual.rewards["fisherman:0"] == [0.4]
        assert mdp.actions["a"]["m"] == [0.4]

    def test_update_advance_and_add_write_the_actions(self):
        mdp = MDPState(actions={"a": {"m": 1.0}})
        mdp = mdp.update(actions={"a": {"m": 1.5}})

        assert mdp.actions["a"]["m"] == [1.5]

        mdp = mdp.advance(actions={"a": {"m": 2.0}})

        assert mdp.actions["a"]["m"] == [1.5, 2.0]

        mdp = mdp.add(MDPState(actions={"a": {"m": 0.25}}))

        assert mdp.actions["a"]["m"] == [1.5, 2.25]


def regulator_with_the_four_mechanisms() -> Agent:
    mechanisms = {
        "quota": Quota(
            id="quota",
            action_space=UNIT,
            acts_on=("fisherman", "harvest"),
            obs_map={"resource_level": "fish"},
        ).build(REGULATOR),
        "subsidy": Incentive(
            id="subsidy", action_space=UNIT, acts_on=("fisherman", "restore")
        ).build(REGULATOR),
        "penalty": ThresholdPenalty(
            id="penalty",
            action_space=NO_PARAMETER,
            acts_on=("fisherman", "harvest"),
            obs_map={"resource_level": "fish"},
            threshold=0.6,
        ).build(REGULATOR),
        "social": SocialInfluence(
            id="social",
            action_space=NO_PARAMETER,
            obs_offset=2,
            acts_on=("fisherman", "harvest"),
        ).build(REGULATOR),
    }

    return Agent(id=REGULATOR, policy_id="p", mechanisms=mechanisms)


def followers_actions(step: int) -> dict:
    """Raw policy outputs of two fishers at ``step``, different at each step."""

    return {
        f"fisherman:{i}": {
            "harvest": np.array([0.7 * (i + 1) - 0.4 * step], dtype=np.float32),
            "restore": np.array([0.3 * step - 0.5 * i], dtype=np.float32),
        }
        for i in range(2)
    }


def initial_state() -> MDPState:
    return MDPState(
        aids={"fisherman:0", "fisherman:1"},
        params={"K": K},
        state={"fish": 0.8 * K},
        actions={
            REGULATOR: {
                "quota": np.array([0.55], dtype=np.float32),
                "subsidy": np.array([0.8], dtype=np.float32),
                "penalty": NOTHING,
                "social": NOTHING,
            },
            **followers_actions(0),
        },
        obs={f"fisherman:{i}": np.zeros(6, dtype=np.float32) for i in range(2)},
    )


def decoding_inside_the_call(agent: Agent, mdp: MDPState) -> MDPState:
    """``Agent.action`` as it was: decode in the call and write the result back."""

    acts = mdp.actions.data.get(agent.id)

    if not acts:
        return mdp

    decoded = {
        mid: agent.mechanisms[mid].decode(mdp, a[mdp.t])
        for mid, a in acts.items()
        if mid in agent.mechanisms
    }
    mdp = mdp.add([agent.mechanisms[mid](mdp, value) for mid, value in decoded.items()])

    return mdp.update(actions={agent.id: decoded})


def decoding_once_in_the_caller(agent: Agent, mdp: MDPState) -> MDPState:
    """The current behaviour: the leader's action is decoded once, at reset."""

    if mdp.t == 0:
        mdp = decode_actions(agent, mdp)

    return mdp.add(agent.action(mdp))


def snapshot(mdp: MDPState) -> dict:
    return {
        name: copy.deepcopy(getattr(mdp, name).data)
        for name in ("state", "actions", "rewards", "obs")
    }


def assert_identical(left, right, path="") -> None:
    """Exact comparison of two trees whose leaves are numbers or arrays."""

    if isinstance(left, dict):
        assert isinstance(right, dict) and left.keys() == right.keys(), path

        for key in left:
            assert_identical(left[key], right[key], f"{path}/{key}")
    elif isinstance(left, list):
        assert isinstance(right, list) and len(left) == len(right), path

        for index, (a, b) in enumerate(zip(left, right)):
            assert_identical(a, b, f"{path}[{index}]")
    else:
        assert np.array_equal(left, right), f"{path}: {left!r} != {right!r}"
        assert np.asarray(left).dtype == np.asarray(right).dtype, path


def play(act, steps: int) -> list[dict]:
    """Four steps of an environment: act, advance, new follower actions, observe."""

    agent = regulator_with_the_four_mechanisms()
    mdp = initial_state()
    recorded = []

    for step in range(steps):
        mdp = act(agent, mdp)

        recorded.append(snapshot(mdp))

        mdp = mdp.advance(
            state={"fish": mdp.state["fish"][mdp.t] * (0.9 - 0.1 * step)},
            obs={f"fisherman:{i}": np.zeros(6, dtype=np.float32) for i in range(2)},
        )
        mdp = mdp.update(actions=followers_actions(step + 1))

        recorded.append(snapshot(mdp))

    return recorded


@pytest.mark.unit
class TestIdempotentMechanismsAreUnchanged:
    @pytest.mark.parametrize("name", ["quota", "subsidy", "penalty", "social"])
    def test_decode_gives_the_same_value_when_it_receives_its_own_output(self, name):
        mechanism = regulator_with_the_four_mechanisms().mechanisms[name]
        mdp = initial_state()

        for raw in (0.0, 0.3, 0.55, 1.0, 1.7, -0.4):
            once = mechanism.decode(mdp, np.array([raw], dtype=np.float32))
            twice = mechanism.decode(mdp, once)

            np.testing.assert_array_equal(np.asarray(twice), np.asarray(once))

    def test_five_steps_are_bit_identical_to_decoding_inside_the_call(self):
        now = play(decoding_once_in_the_caller, steps=5)
        before = play(decoding_inside_the_call, steps=5)

        assert len(now) == len(before) == 10

        for new, old in zip(now, before):
            assert_identical(new, old)
