"""``ThresholdPenaltyMechanism``: a reward residual below a resource threshold."""

import dataclasses

import numpy as np
import pytest

from core.agents.base import Agent
from core.mechanism.algorithms.penalty import (
    ThresholdPenalty,
    ThresholdPenaltyMechanism,
)
from core.mechanism.base import MDPState

REGULATOR = "regulator"
K = 5_000.0
NO_ACTION = np.empty(0, dtype=np.float32)


def make(**kwargs) -> ThresholdPenaltyMechanism:
    kwargs.setdefault("acts_on", ("fisherman", "harvest"))
    kwargs.setdefault("obs_map", {"resource_level": "fish"})
    return ThresholdPenalty(id="penalty", **kwargs).build(REGULATOR)


def mdp_at(fish_norm: float, aids=("fisherman:0", "fisherman:1")) -> MDPState:
    """State at ``t = 0`` with the stock at ``fish_norm`` of the capacity."""
    return MDPState(
        aids=set(aids),
        params={"K": K},
        state={"fish": fish_norm * K},
        actions={
            REGULATOR: {"penalty": NO_ACTION},
            **{aid: {"harvest": np.array([0.0], dtype=np.float32)} for aid in aids},
        },
    )


@pytest.mark.unit
class TestValidation:
    def test_parameter_ranges(self):
        with pytest.raises(ValueError, match="threshold"):
            make(threshold=1.5)
        with pytest.raises(ValueError, match="penalty_amount"):
            make(penalty_amount=-1.0)
        with pytest.raises(ValueError, match="transition_width"):
            make(transition_width=0.0)

    def test_apply_requires_a_target(self):
        with pytest.raises(ValueError, match="acts_on"):
            make(acts_on=None)(mdp_at(0.5), NO_ACTION)

    def test_apply_requires_the_resource_level(self):
        with pytest.raises(ValueError, match="resource_level"):
            make(obs_map=None)(mdp_at(0.5), NO_ACTION)
        with pytest.raises(ValueError, match="resource_level"):
            make(obs_map={"stock": "fish"})(mdp_at(0.5), NO_ACTION)


@pytest.mark.unit
class TestFixedRule:
    def test_default_action_space_is_empty(self):
        config = ThresholdPenalty(id="penalty")

        assert config.action_space.shape == (0,)
        assert config.build(REGULATOR).action_space.shape == (0,)


@pytest.mark.unit
class TestPenaltyCurve:
    def test_half_penalty_at_the_threshold(self):
        mechanism = make(threshold=0.2, penalty_amount=0.1)
        assert mechanism.penalty(0.2) == pytest.approx(0.05)

    def test_extreme_inputs_do_not_overflow(self):
        mechanism = make(threshold=0.5, penalty_amount=1.0, transition_width=1e-4)
        assert mechanism.penalty(0.0) == pytest.approx(1.0)
        assert mechanism.penalty(1.0) == pytest.approx(0.0, abs=1e-12)


@pytest.mark.unit
class TestRewardResidual:
    def test_far_above_the_threshold_there_is_no_penalty(self):
        mechanism = make(threshold=0.2, penalty_amount=0.1, transition_width=0.03)

        delta = mechanism(mdp_at(0.9), NO_ACTION)

        assert delta.rewards["fisherman:0"][0] == pytest.approx(0.0, abs=1e-6)
        assert delta.rewards["fisherman:1"][0] == pytest.approx(0.0, abs=1e-6)

    def test_far_below_the_threshold_the_penalty_is_full(self):
        mechanism = make(threshold=0.2, penalty_amount=0.1, transition_width=0.03)

        delta = mechanism(mdp_at(0.0), NO_ACTION)

        assert delta.rewards["fisherman:0"][0] == pytest.approx(-0.1, abs=1e-3)

    def test_resource_level_is_normalised_by_the_capacity(self):
        mechanism = make(threshold=0.2, penalty_amount=0.1)

        delta = mechanism(mdp_at(0.2), NO_ACTION)

        assert delta.rewards["fisherman:0"][0] == pytest.approx(-0.05)

    def test_every_targeted_agent_gets_the_same_penalty(self):
        mdp = mdp_at(0.1, aids=("fisherman:0", "fisherman:1", "farmer:0"))

        delta = make()(mdp, NO_ACTION)

        assert set(delta.rewards.data) == {"fisherman:0", "fisherman:1"}
        assert delta.rewards["fisherman:0"] == delta.rewards["fisherman:1"]

    def test_only_rewards_are_touched(self):
        delta = make()(mdp_at(0.1), NO_ACTION)

        assert delta.actions.data == {}
        assert delta.state.data == {}
        assert delta.obs.data == {}


@pytest.mark.unit
class TestComposition:
    def test_penalty_follows_the_stock_step_by_step(self):
        mechanism = make(threshold=0.2, penalty_amount=0.1, transition_width=0.03)
        regulator = Agent(
            id=REGULATOR, policy_id="p", mechanisms={"penalty": mechanism}
        )
        mdp = mdp_at(0.9, aids=("fisherman:0",))

        rewards = []
        for fish_norm in (0.9, 0.2, 0.0):
            mdp.update(state={"fish": fish_norm * K})
            mdp.update(actions={"fisherman:0": {"harvest": np.array([0.0])}})
            mdp = regulator.action(mdp)
            mdp = mdp.add(MDPState(rewards={"fisherman:0": 1.0}))
            rewards.append(mdp.rewards["fisherman:0"][mdp.t])
            mdp = dataclasses.replace(mdp, t=mdp.t + 1)

        assert rewards == pytest.approx([1.0, 0.95, 0.9], abs=1e-3)
