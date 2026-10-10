"""``SubsidyMechanism``: a reward residual on the targeted agents' effort.

The regulator acts before the fishermen, so the effort it reads is still the
raw policy output ``z``; the mechanism maps it to ``e = sigmoid(z / 4)`` as the
followers' own mechanisms do. The tests build ``z`` from the effort they want.
"""

import dataclasses

import numpy as np
import pytest
from gymnasium import spaces

from core.agents.base import Agent
from core.mechanism.algorithms.subsidy import MAX_SUBSIDY, Subsidy, SubsidyMechanism
from core.mechanism.base import MDPState
from core.utils import ACTION_TEMPERATURE

REGULATOR = "regulator"


def make(cost: float = 0.5, acts_on=("fisherman", "restore")) -> SubsidyMechanism:
    return Subsidy(
        id="subsidy",
        action_space=spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        acts_on=acts_on,
        cost=cost,
    ).build(REGULATOR)


def raw(effort: float) -> np.ndarray:
    """Policy output whose decoded effort is ``effort``."""
    z = ACTION_TEMPERATURE * np.log(effort / (1.0 - effort))
    return np.array([z], dtype=np.float32)


def follower_actions(efforts: dict[str, float]) -> dict:
    return {
        aid: {"harvest": raw(0.8), "restore": raw(effort)}
        for aid, effort in efforts.items()
    }


def mdp_with(efforts: dict[str, float], rate: float = 0.6) -> MDPState:
    """State at ``t = 0`` where the regulator holds the normalised ``rate``."""
    return MDPState(
        aids=set(efforts),
        actions={
            REGULATOR: {"subsidy": np.array([rate], dtype=np.float32)},
            **follower_actions(efforts),
        },
    )


@pytest.mark.unit
class TestValidation:
    def test_cost_must_be_in_unit_interval(self):
        with pytest.raises(ValueError, match="cost"):
            make(cost=1.5)
        with pytest.raises(ValueError, match="cost"):
            make(cost=-0.1)

    def test_apply_requires_a_target(self):
        mechanism = make(acts_on=None)
        with pytest.raises(ValueError, match="acts_on"):
            mechanism(mdp_with({"fisherman:0": 0.4}), np.array([0.6]))


@pytest.mark.unit
class TestDecode:
    def test_decode_clips_the_rate_to_the_unit_interval(self):
        mechanism = make()
        mdp = MDPState()
        assert mechanism.decode(mdp, np.array([1.7])) == 1.0
        assert mechanism.decode(mdp, np.array([-0.2])) == 0.0
        assert mechanism.decode(mdp, np.array([0.25])) == pytest.approx(0.25)

    def test_decode_is_idempotent(self):
        # A rate already in [0, 1] is returned unchanged. The mechanism no
        # longer relies on it (see ``test_decode_input.py``), but the clip
        # keeps this property.
        mechanism = make()
        once = mechanism.decode(MDPState(), np.array([0.6]))
        assert mechanism.decode(MDPState(), once) == once


@pytest.mark.unit
class TestRewardResidual:
    def test_analytical_value(self):
        rate, cost, effort = 0.6, 0.5, 0.4
        mdp = mdp_with({"fisherman:0": effort}, rate=rate)

        delta = make(cost=cost)(mdp, mdp.actions[REGULATOR]["subsidy"][0])

        sigma = rate * MAX_SUBSIDY
        assert delta.rewards["fisherman:0"][0] == pytest.approx(
            sigma * effort - cost * effort**2, abs=1e-6
        )

    def test_no_effort_leaves_the_reward_unchanged(self):
        mdp = mdp_with({"fisherman:0": 0.5})
        mdp.actions["fisherman:0"]["restore"][0] = np.array([-400.0], dtype=np.float32)

        delta = make()(mdp, np.array([1.0]))

        assert delta.rewards["fisherman:0"][0] == pytest.approx(0.0, abs=1e-12)

    def test_effort_is_read_from_the_targeted_mechanism(self):
        mdp = mdp_with({"fisherman:0": 0.2})

        delta = make(cost=0.0, acts_on=("fisherman", "harvest"))(mdp, np.array([1.0]))

        # ``follower_actions`` sets the harvest effort to 0.8.
        assert delta.rewards["fisherman:0"][0] == pytest.approx(
            MAX_SUBSIDY * 0.8, abs=1e-6
        )

    def test_each_targeted_agent_gets_its_own_residual(self):
        mdp = mdp_with({"fisherman:0": 0.2, "fisherman:1": 0.9})

        delta = make(cost=0.0)(mdp, np.array([1.0]))

        assert delta.rewards["fisherman:0"][0] == pytest.approx(0.1, abs=1e-6)
        assert delta.rewards["fisherman:1"][0] == pytest.approx(0.45, abs=1e-6)

    def test_other_agents_are_not_subsidised(self):
        mdp = mdp_with({"fisherman:0": 0.4, "farmer:0": 0.4})

        delta = make()(mdp, np.array([1.0]))

        assert set(delta.rewards.data) == {"fisherman:0"}

    def test_only_rewards_are_touched(self):
        mdp = mdp_with({"fisherman:0": 0.4})

        delta = make()(mdp, np.array([1.0]))

        assert delta.actions.data == {}
        assert delta.state.data == {}
        assert delta.obs.data == {}


@pytest.mark.unit
class TestComposition:
    def test_residual_adds_to_the_agents_own_reward(self):
        rate, cost, effort = 1.0, 0.25, 0.4
        regulator = Agent(
            id=REGULATOR, policy_id="p", mechanisms={"subsidy": make(cost=cost)}
        )
        mdp = regulator.action(mdp_with({"fisherman:0": effort}, rate=rate))
        mdp = mdp.add(MDPState(rewards={"fisherman:0": 2.0}))

        assert mdp.rewards["fisherman:0"][0] == pytest.approx(
            2.0 + MAX_SUBSIDY * effort - cost * effort**2, abs=1e-6
        )

    def test_rate_does_not_drift_over_the_steps_of_an_episode(self):
        # The regulator's action is set once at reset and carried forward; the
        # mechanism decodes the raw action again at every step.
        efforts = {"fisherman:0": 0.4}
        regulator = Agent(
            id=REGULATOR, policy_id="p", mechanisms={"subsidy": make(cost=0.0)}
        )
        mdp = MDPState(
            aids=set(efforts),
            actions={REGULATOR: {"subsidy": np.array([0.6], dtype=np.float32)}},
        )

        residuals = []
        for _ in range(4):
            mdp.update(actions=follower_actions(efforts))
            mdp = regulator.action(mdp)
            residuals.append(mdp.rewards["fisherman:0"][mdp.t])
            mdp = dataclasses.replace(mdp, t=mdp.t + 1)

        assert residuals == pytest.approx([0.6 * MAX_SUBSIDY * 0.4] * 4, abs=1e-6)
