"""``Trajectory`` and ``MDPState``: edge cases of the tree and time operations.

``test_trajectory.py`` covers how unwritten timesteps are filled. This file
covers the structural rules around it: a leaf and a branch never share a key,
a leaf that lags behind the rest of the tree is brought up to date before it
is written, time cannot go backwards or be skipped, and ``MDPState.update`` and
``MDPState.add`` write the right fields.
"""

import numpy as np
import pytest

from core.mechanism.base import MDPState
from core.mechanism.types import FlowTrajectory, Trajectory


@pytest.mark.unit
class TestTreeMismatch:
    def test_update_rejects_a_branch_where_a_leaf_exists(self):
        with pytest.raises(TypeError, match="destination is a leaf"):
            Trajectory({"a": 1.0}).update(0, {"a": {"x": 1.0}})

    def test_update_rejects_a_leaf_where_a_branch_exists(self):
        with pytest.raises(TypeError, match="destination is a branch"):
            Trajectory({"a": {"x": 1.0}}).update(0, {"a": 1.0})

    def test_add_rejects_a_branch_where_a_leaf_exists(self):
        with pytest.raises(TypeError, match="destination is a leaf"):
            Trajectory({"a": 1.0}).add(0, [Trajectory({"a": {"x": 1.0}})])

    def test_add_rejects_a_leaf_where_a_branch_exists(self):
        with pytest.raises(TypeError, match="destination is a branch"):
            Trajectory({"a": {"x": 1.0}}).add(0, [Trajectory({"a": 1.0})])


@pytest.mark.unit
class TestLaggingLeaf:
    def test_stock_leaf_behind_the_tree_is_filled_with_its_last_value(self):
        trajectory = Trajectory({"a": [1.0, 2.0, 3.0], "b": [7.0]})

        updated = trajectory.update(2, {"b": 9.0})

        assert updated["b"] == [7.0, 7.0, 9.0]
        assert updated["a"] == [1.0, 2.0, 3.0]

    def test_flow_leaf_behind_the_tree_is_filled_with_zero(self):
        trajectory = FlowTrajectory({"a": [1.0, 2.0, 3.0], "b": [7.0]})

        updated = trajectory.update(2, {"b": 9.0})

        assert updated["b"] == [7.0, 0, 9.0]

    def test_leaf_one_step_behind_is_appended_to(self):
        trajectory = Trajectory({"a": [1.0, 2.0], "b": [7.0]})

        updated = trajectory.update(1, {"b": 9.0})

        assert updated["b"] == [7.0, 9.0]

    def test_new_key_written_late_is_zero_padded(self):
        updated = Trajectory({"a": [1.0, 2.0]}).update(1, {"b": 5.0})

        assert updated["b"] == [0, 5.0]
        assert updated["a"] == [1.0, 2.0]

    def test_add_creates_a_missing_branch(self):
        base = Trajectory({"a": {"x": [1.0]}})

        added = base.add(0, [Trajectory({"b": {"y": 2.0}})])

        assert added["a"]["x"] == [1.0]
        assert added["b"]["y"] == [2.0]

    def test_add_to_a_lagging_leaf_fills_then_sums(self):
        base = Trajectory({"a": [1.0, 2.0, 3.0], "b": [7.0]})

        added = base.add(2, [Trajectory({"b": 1.0})])

        assert added["b"] == [7.0, 7.0, 8.0]


@pytest.mark.unit
class TestTime:
    def test_update_rejects_a_negative_step(self):
        with pytest.raises(ValueError, match="non-negative"):
            Trajectory({"a": 1.0}).update(-1, {"a": 2.0})

    def test_update_rejects_skipping_a_step(self):
        with pytest.raises(ValueError, match="Cannot skip from trajectory length 1"):
            Trajectory({"a": 1.0}).update(3, {"a": 2.0})

    def test_update_at_the_last_step_overwrites_it(self):
        updated = Trajectory({"a": [1.0, 2.0]}).update(1, {"a": 5.0})

        assert updated["a"] == [1.0, 5.0]

    def test_update_at_the_length_opens_a_new_step(self):
        updated = Trajectory({"a": [1.0, 2.0]}).update(2, {"a": 5.0})

        assert updated["a"] == [1.0, 2.0, 5.0]

    def test_update_does_not_touch_the_original(self):
        original = Trajectory({"a": [1.0, 2.0]})

        original.update(1, {"a": 5.0})
        original.append({"a": 9.0})

        assert original["a"] == [1.0, 2.0]

    def test_append_adds_one_step_to_every_leaf(self):
        appended = Trajectory({"a": [1.0], "b": {"c": [2.0]}}).append({"a": 3.0})

        assert appended["a"] == [1.0, 3.0]
        assert appended["b"]["c"] == [2.0, 2.0]
        assert appended.length == 2

    def test_length_of_an_empty_trajectory_is_zero(self):
        assert Trajectory().length == 0


@pytest.mark.unit
class TestContainer:
    def test_membership_and_lookup_use_the_top_level_keys(self):
        trajectory = Trajectory({"a": 1.0, "b": {"c": 2.0}})

        assert "a" in trajectory
        assert "b" in trajectory
        assert "c" not in trajectory
        assert trajectory["b"] == {"c": [2.0]}

    def test_copy_is_independent(self):
        original = Trajectory({"a": [1.0], "b": {"c": [2.0]}})
        copy = original.copy()

        copy.data["a"].append(5.0)
        copy.data["b"]["c"].append(5.0)

        assert original["a"] == [1.0]
        assert original["b"]["c"] == [2.0]

    def test_a_list_leaf_is_taken_as_a_history(self):
        assert Trajectory({"a": [1.0, 2.0]})["a"] == [1.0, 2.0]

    def test_an_array_leaf_is_one_timestep(self):
        trajectory = Trajectory({"a": np.array([1.0, 2.0])})

        assert len(trajectory["a"]) == 1
        np.testing.assert_array_equal(trajectory["a"][0], [1.0, 2.0])

    def test_add_does_not_mutate_array_leaves_shared_with_the_original(self):
        shared = np.array([1.0, 2.0])
        base = Trajectory({"a": [shared]})

        added = base.add(0, [Trajectory({"a": np.array([10.0, 10.0])})])

        np.testing.assert_array_equal(added["a"][0], [11.0, 12.0])
        np.testing.assert_array_equal(shared, [1.0, 2.0])


@pytest.mark.unit
class TestMDPStateUpdate:
    def test_update_writes_the_given_fields_at_the_current_step(self):
        mdp = MDPState(
            state={"fish": 1.0},
            obs={"a": 0.5},
            actions={"a": {"harvest": 0.1}},
            rewards={"a": 2.0},
        )

        returned = mdp.update(
            state={"fish": 3.0},
            obs={"a": 0.7},
            actions={"a": {"harvest": 0.2}},
            rewards={"a": 4.0},
        )

        assert returned is mdp
        assert mdp.state["fish"] == [3.0]
        assert mdp.obs["a"] == [0.7]
        assert mdp.actions["a"]["harvest"] == [0.2]
        assert mdp.rewards["a"] == [4.0]

    def test_update_leaves_omitted_fields_alone(self):
        mdp = MDPState(state={"fish": 1.0}, rewards={"a": 2.0})

        mdp.update(rewards={"a": 5.0})

        assert mdp.state["fish"] == [1.0]
        assert mdp.rewards["a"] == [5.0]

    def test_update_at_an_explicit_step(self):
        mdp = MDPState(state={"fish": [1.0, 2.0]})

        mdp.update(0, state={"fish": 9.0})

        assert mdp.state["fish"] == [9.0, 2.0]

    def test_advance_opens_the_next_step_and_keeps_the_original(self):
        mdp = MDPState(state={"fish": 1.0}, rewards={"a": 2.0})

        advanced = mdp.advance(state={"fish": 0.5}, rewards={"a": 3.0})

        assert advanced.t == 1
        assert advanced.state["fish"] == [1.0, 0.5]
        assert advanced.rewards["a"] == [2.0, 3.0]
        assert mdp.t == 0
        assert mdp.state["fish"] == [1.0]


@pytest.mark.unit
class TestMDPStateAdd:
    def test_a_single_residual_is_accepted_without_a_list(self):
        mdp = MDPState(state={"fish": 10.0}).add(MDPState(state={"fish": -2.0}))

        assert mdp.state["fish"] == [8.0]

    def test_params_and_aids_are_merged(self):
        mdp = MDPState(params={"K": 1.0, "r": 0.3}, aids={"a"})

        merged = mdp.add(
            [MDPState(params={"r": 0.5, "p": 1.0}, aids={"b"}), MDPState(aids={"c"})]
        )

        assert merged.params == {"K": 1.0, "r": 0.5, "p": 1.0}
        assert merged.aids == {"a", "b", "c"}
        assert mdp.params == {"K": 1.0, "r": 0.3}

    def test_time_is_kept(self):
        mdp = MDPState(t=2, state={"fish": [1.0, 1.0, 1.0]})

        assert mdp.add(MDPState(state={"fish": 1.0})).t == 2

    def test_termination_flags_are_or_ed(self):
        mdp = MDPState(terminateds={"a": False, "b": False})

        merged = mdp.add(MDPState(terminateds={"b": True, "c": False}))

        assert merged.terminateds == {"a": False, "b": True, "c": False}
