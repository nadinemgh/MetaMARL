"""``IncentiveMechanism``: a reward residual on the targeted agents' effort.

The adapter decodes the followers' raw policy outputs before the regulator's
mechanisms run, so the effort ``e`` the mechanism reads from the state is already
decoded, and the rate it receives is the regulator's decoded action. The tests
therefore store the effort and the rate directly.
"""

import dataclasses

import numpy as np
import pytest
from gymnasium import spaces

from core.agents.base import Agent
from core.mechanism.algorithms.incentive import MAX, Incentive, IncentiveMechanism
from core.mechanism.base import MDPState

REGULATOR = "regulator"


def make(acts_on=("fisherman", "restore")) -> IncentiveMechanism:
    return Incentive(
        id="subsidy",
        action_space=spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        acts_on=acts_on,
    ).build(REGULATOR)


def decoded(effort: float) -> np.float32:
    """Decoded effort as the adapter stores it in the state."""

    return np.float32(effort)


def follower_actions(efforts: dict[str, float]) -> dict:
    return {
        aid: {"harvest": decoded(0.8), "restore": decoded(effort)}
        for aid, effort in efforts.items()
    }


def mdp_with(efforts: dict[str, float], rate: float = 0.6) -> MDPState:
    """State at ``t = 0`` where the regulator holds the normalised ``rate``."""

    return MDPState(
        aids=set(efforts),
        actions={REGULATOR: {"subsidy": np.float32(rate)}, **follower_actions(efforts)},
    )


@pytest.mark.unit
class TestValidation:
    def test_apply_requires_a_target(self):
        mechanism = make(acts_on=None)

        with pytest.raises(ValueError, match="acts_on"):
            mechanism(mdp_with({"fisherman:0": 0.4}), 0.6)


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
        rate, effort = 0.6, 0.4
        mdp = mdp_with({"fisherman:0": effort}, rate=rate)
        delta = make()(mdp, mdp.actions[REGULATOR]["subsidy"][0])

        assert delta.rewards["fisherman:0"][0] == pytest.approx(
            rate * MAX * effort, abs=1e-6
        )

    def test_no_effort_leaves_the_reward_unchanged(self):
        mdp = mdp_with({"fisherman:0": 0.5})
        mdp.actions["fisherman:0"]["restore"][0] = np.float32(0.0)
        delta = make()(mdp, 1.0)

        assert delta.rewards["fisherman:0"][0] == pytest.approx(0.0, abs=1e-12)

    def test_effort_is_read_from_the_targeted_mechanism(self):
        mdp = mdp_with({"fisherman:0": 0.2})
        delta = make(acts_on=("fisherman", "harvest"))(mdp, 1.0)

        # ``follower_actions`` sets the harvest effort to 0.8.
        assert delta.rewards["fisherman:0"][0] == pytest.approx(MAX * 0.8, abs=1e-6)

    def test_each_targeted_agent_gets_its_own_residual(self):
        mdp = mdp_with({"fisherman:0": 0.2, "fisherman:1": 0.9})
        delta = make()(mdp, 1.0)

        assert delta.rewards["fisherman:0"][0] == pytest.approx(0.1, abs=1e-6)
        assert delta.rewards["fisherman:1"][0] == pytest.approx(0.45, abs=1e-6)

    def test_other_agents_are_not_subsidised(self):
        mdp = mdp_with({"fisherman:0": 0.4, "farmer:0": 0.4})
        delta = make()(mdp, 1.0)

        assert set(delta.rewards.data) == {"fisherman:0"}

    def test_only_rewards_are_touched(self):
        mdp = mdp_with({"fisherman:0": 0.4})
        delta = make()(mdp, 1.0)

        assert delta.actions.data == {}
        assert delta.state.data == {}
        assert delta.obs.data == {}


@pytest.mark.unit
class TestComposition:
    def test_residual_adds_to_the_agents_own_reward(self):
        rate, effort = 1.0, 0.4
        regulator = Agent(id=REGULATOR, policy_id="p", mechanisms={"subsidy": make()})
        mdp = regulator.action(mdp_with({"fisherman:0": effort}, rate=rate))
        mdp = mdp.add(MDPState(rewards={"fisherman:0": 2.0}))

        assert mdp.rewards["fisherman:0"][0] == pytest.approx(
            2.0 + MAX * effort, abs=1e-6
        )

    def test_rate_does_not_drift_over_the_steps_of_an_episode(self):
        # The regulator's decoded action is set once at reset and carried
        # forward; the mechanism only reads it, so the residual is the same at
        # every step and the stored rate is never rewritten.
        efforts = {"fisherman:0": 0.4}
        regulator = Agent(id=REGULATOR, policy_id="p", mechanisms={"subsidy": make()})
        mdp = MDPState(
            aids=set(efforts), actions={REGULATOR: {"subsidy": np.float32(0.6)}}
        )
        residuals = []

        for t in range(4):
            mdp.update(t=t, actions=follower_actions(efforts))

            mdp = dataclasses.replace(mdp, t=t)

            delta = regulator.action(mdp)

            residuals.append(delta.rewards["fisherman:0"][delta.t])

            assert mdp.actions[REGULATOR]["subsidy"][t] == np.float32(0.6)

        assert residuals == pytest.approx([0.6 * MAX * 0.4] * 4, abs=1e-6)


@pytest.mark.unit
class TestOnReset:
    def test_without_a_rate_nothing_is_written(self):
        assert make().on_reset(mdp_with({"fisherman:0": 0.4}), None).obs.data == {}

    def test_the_rate_is_written_in_the_last_slot_only(self):
        residual = make().on_reset(mdp_with({"fisherman:0": 0.4}), 0.3)
        expected = np.zeros(9, dtype=np.float32)
        expected[8] = np.float32(0.3)

        np.testing.assert_array_equal(residual.obs["fisherman:0"][0], expected)

    def test_the_delta_is_taken_from_the_current_step(self):
        mdp = mdp_with({"fisherman:0": 0.4}).add(
            MDPState(obs={"fisherman:0": np.full(9, 2.0, dtype=np.float32)})
        )
        residual = make().on_reset(mdp, 0.3).obs["fisherman:0"][0]

        assert residual[8] == pytest.approx(0.3 - 2.0, abs=1e-6)

    def test_a_step_without_an_observation_yet_receives_the_rate(self):
        # Observations are per-step flows: the previous step's value must not
        # enter the difference, or the slot alternates between the rate and
        # zero from one step to the next.
        mdp = mdp_with({"fisherman:0": 0.4}).add(
            MDPState(obs={"fisherman:0": np.full(9, 2.0, dtype=np.float32)})
        )
        mdp = mdp.advance(state={})
        residual = make().on_reset(mdp, 0.3).obs["fisherman:0"][0]

        assert residual[8] == pytest.approx(0.3, abs=1e-6)
