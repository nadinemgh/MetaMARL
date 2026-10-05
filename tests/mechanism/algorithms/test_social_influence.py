"""``SocialInfluenceMechanism``: peers' last actions in each observation.

The mechanism contributes at observation time. After the transition of step
``t - 1`` the state is at ``t`` and ``actions[...][t - 1]`` holds what each
agent delivered; the tests build that situation directly.
"""

import dataclasses

import numpy as np
import pytest

from core.agents.base import Agent
from core.mechanism.algorithms.social_influence import (
    SocialInfluence,
    SocialInfluenceMechanism,
)
from core.mechanism.base import MDPState

REGULATOR = "regulator"
NO_ACTION = np.empty(0, dtype=np.float32)


def make(obs_offset: int = 2, **kwargs) -> SocialInfluenceMechanism:
    kwargs.setdefault("acts_on", ("fisherman", "harvest"))
    return SocialInfluence(id="social", obs_offset=obs_offset, **kwargs).build(
        REGULATOR
    )


def mdp_after_a_step(delivered: dict[str, list[float]], obs_size: int = 6) -> MDPState:
    """State at ``t = 1`` whose step 0 delivered the given harvest actions."""
    mdp = MDPState(
        aids=set(delivered),
        actions={
            REGULATOR: {"social": NO_ACTION},
            **{aid: {"harvest": np.array(a)} for aid, a in delivered.items()},
        },
        obs={aid: np.zeros(obs_size, dtype=np.float32) for aid in delivered},
    )
    return dataclasses.replace(mdp, t=1)


THREE = {
    "fisherman:0": [1.0, 10.0],
    "fisherman:1": [2.0, 20.0],
    "fisherman:2": [3.0, 30.0],
}


@pytest.mark.unit
class TestValidation:
    def test_parameter_ranges(self):
        with pytest.raises(ValueError, match="obs_offset"):
            make(obs_offset=-1)
        with pytest.raises(ValueError, match="influence_weight"):
            make(influence_weight=-0.5)

    def test_observe_requires_a_target(self):
        with pytest.raises(ValueError, match="acts_on"):
            make(acts_on=None).observe(mdp_after_a_step(THREE))

    def test_reserved_entries_must_fit_in_the_observation(self):
        # Two peers with two action components need entries 2 to 5.
        with pytest.raises(ValueError, match="fisherman:0"):
            make(obs_offset=2).observe(mdp_after_a_step(THREE, obs_size=5))


@pytest.mark.unit
class TestFixedRule:
    def test_default_action_space_is_empty(self):
        assert SocialInfluence(id="social", obs_offset=0).action_space.shape == (0,)

    def test_influence_weight_is_reserved(self):
        assert make().influence_weight == 0.0

    def test_apply_contributes_nothing(self):
        mdp = dataclasses.replace(mdp_after_a_step(THREE), t=0)

        delta = make()(mdp, NO_ACTION)

        assert delta.obs.data == {}
        assert delta.rewards.data == {}
        assert delta.actions.data == {}
        assert delta.state.data == {}


@pytest.mark.unit
class TestObservationResidual:
    def test_peer_ordering_and_self_exclusion(self):
        delta = make(obs_offset=2).observe(mdp_after_a_step(THREE))

        # (N - 1) * d = 2 * 2 = 4 entries, after the two the agent fills itself.
        np.testing.assert_allclose(delta.obs["fisherman:0"][0], [0, 0, 2, 20, 3, 30])
        np.testing.assert_allclose(delta.obs["fisherman:1"][0], [0, 0, 1, 10, 3, 30])
        np.testing.assert_allclose(delta.obs["fisherman:2"][0], [0, 0, 1, 10, 2, 20])
        assert all(v[0].dtype == np.float32 for v in delta.obs.data.values())

    def test_peer_order_does_not_depend_on_the_order_of_the_actions(self):
        shuffled = {aid: THREE[aid] for aid in reversed(THREE)}

        delta = make(obs_offset=2).observe(mdp_after_a_step(shuffled))

        np.testing.assert_allclose(delta.obs["fisherman:1"][0], [0, 0, 1, 10, 3, 30])

    def test_peers_follow_the_numeric_order_of_their_index(self):
        # String order would put "fisherman:10" before "fisherman:2".
        delivered = {"fisherman:2": [2.0], "fisherman:10": [10.0], "fisherman:1": [1.0]}

        delta = make(obs_offset=0).observe(mdp_after_a_step(delivered, obs_size=2))

        np.testing.assert_allclose(delta.obs["fisherman:1"][0], [2, 10])
        np.testing.assert_allclose(delta.obs["fisherman:10"][0], [1, 2])

    def test_non_numeric_indices_follow_the_numeric_ones_in_string_order(self):
        delivered = {
            "fisherman:b": [-2.0],
            "fisherman:10": [10.0],
            "fisherman:a": [-1.0],
            "fisherman:9": [9.0],
        }

        delta = make(obs_offset=0).observe(mdp_after_a_step(delivered, obs_size=3))

        np.testing.assert_allclose(delta.obs["fisherman:b"][0], [9, 10, -1])

    def test_scalar_actions_are_accepted(self):
        # A mechanism such as ``Fishing`` decodes its action to a plain float.
        mdp = mdp_after_a_step({"fisherman:0": 0.25, "fisherman:1": 0.75}, obs_size=3)

        delta = make(obs_offset=2).observe(mdp)

        np.testing.assert_allclose(delta.obs["fisherman:0"][0], [0, 0, 0.75])
        np.testing.assert_allclose(delta.obs["fisherman:1"][0], [0, 0, 0.25])

    def test_other_agents_are_neither_peers_nor_recipients(self):
        mdp = mdp_after_a_step({**THREE, "farmer:0": [9.0, 90.0]})

        delta = make(obs_offset=2).observe(mdp)

        assert set(delta.obs.data) == set(THREE)
        np.testing.assert_allclose(delta.obs["fisherman:0"][0], [0, 0, 2, 20, 3, 30])

    def test_nothing_is_written_at_reset(self):
        mdp = dataclasses.replace(mdp_after_a_step(THREE), t=0)

        assert make().observe(mdp).obs.data == {}

    def test_a_single_agent_has_no_peer(self):
        delta = make().observe(mdp_after_a_step({"fisherman:0": [1.0, 10.0]}))

        assert delta.obs.data == {}

    def test_only_the_last_step_is_exposed(self):
        mdp = mdp_after_a_step({"fisherman:0": 0.25, "fisherman:1": 0.75}, obs_size=3)
        mdp.update(
            actions={"fisherman:0": {"harvest": 0.5}, "fisherman:1": {"harvest": 0.9}}
        )
        mdp.update(obs={aid: np.zeros(3, dtype=np.float32) for aid in mdp.aids})
        mdp = dataclasses.replace(mdp, t=2)

        delta = make(obs_offset=2).observe(mdp)

        np.testing.assert_allclose(delta.obs["fisherman:0"][0], [0, 0, 0.9])


@pytest.mark.unit
class TestComposition:
    def test_contribution_fills_the_entries_the_agent_left_at_zero(self):
        mdp = mdp_after_a_step({"fisherman:0": 0.25, "fisherman:1": 0.75}, obs_size=3)
        regulator = Agent(
            id=REGULATOR, policy_id="p", mechanisms={"social": make(obs_offset=2)}
        )
        own = MDPState(
            obs={aid: np.asarray([0.8, 0.1, 0.0], dtype=np.float32) for aid in mdp.aids}
        )

        mdp = mdp.add([own, *regulator.mechanism_observations(mdp)])

        np.testing.assert_allclose(mdp.obs["fisherman:0"][1], [0.8, 0.1, 0.75])
        np.testing.assert_allclose(mdp.obs["fisherman:1"][1], [0.8, 0.1, 0.25])
