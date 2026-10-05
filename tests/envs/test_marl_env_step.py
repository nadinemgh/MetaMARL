"""``MultiAgentEnv.step``: rewards, observations, termination and metrics.

The step runs the leaders' mechanisms, then each follower's, over the shared
``MDPState``; the transition hook advances the state; the environment logs the
step and rebuilds the observations. Rewards and observations are per-step
values (``FlowTrajectory``): each step's entry holds what was earned or
observed at that step only, never a running sum.

The tests drive the environment the way the RLlib adapter does: the actions of
the step are written with ``mdp.update(actions=...)`` before ``step`` is called.
"""

import numpy as np
import pytest

from core.mechanism.base import MDPState

FOLLOWERS = ["f0", "f1"]
AIDS = set(FOLLOWERS)


def run_episode(env, toy, harvests):
    """Reset, then step once per entry of ``harvests``; return every MDP state."""
    mdp = env.reset(MDPState(aids=AIDS))
    states = [mdp]
    for amount in harvests:
        mdp.update(actions=toy.harvest_actions(FOLLOWERS, amount))
        mdp = env.step(mdp)
        states.append(mdp)
    return states


@pytest.fixture
def unregulated(toy, identity_ray_get):
    """Environment whose World has published nothing yet."""
    return toy.make_env(toy.ScriptedWorld(), horizon=3)


@pytest.fixture
def regulated(toy, identity_ray_get):
    """Environment whose leader holds a fee of 0.1 per step."""
    return toy.make_env(toy.ScriptedWorld([toy.candidate(0.1)]), horizon=3)


@pytest.mark.unit
def test_rewards_are_per_step_values_not_running_sums(unregulated, toy):
    states = run_episode(unregulated, toy, [0.2, 0.2, 0.2])

    final = states[-1]
    for aid in FOLLOWERS:
        assert final.rewards[aid] == pytest.approx([0.2, 0.2, 0.2])


@pytest.mark.unit
def test_rewards_follow_the_harvest_of_each_step(unregulated, toy):
    states = run_episode(unregulated, toy, [0.1, 0.4, 0.3])

    assert states[-1].rewards["f0"] == pytest.approx([0.1, 0.4, 0.3])


@pytest.mark.unit
def test_leader_mechanism_shapes_every_followers_reward(regulated, toy):
    states = run_episode(regulated, toy, [0.3, 0.3])

    final = states[-1]
    for aid in FOLLOWERS:
        assert final.rewards[aid] == pytest.approx([0.3 - 0.1, 0.3 - 0.1])


@pytest.mark.unit
def test_leader_fee_is_charged_each_step_from_the_kept_candidate(regulated, toy):
    # The World returns the candidate on the first fetch only; the second
    # episode must be charged the same fee.
    run_episode(regulated, toy, [0.3])
    second = run_episode(regulated, toy, [0.3, 0.5])

    assert second[-1].rewards["f1"] == pytest.approx([0.2, 0.4])


@pytest.mark.unit
def test_stock_is_depleted_by_the_harvests_and_carried_over(unregulated, toy):
    states = run_episode(unregulated, toy, [0.1, 0.2])

    # Two followers harvest at each step; the transition copies the stock.
    stock = states[-1].state["stock"]
    assert stock == pytest.approx([0.8, 0.4, 0.4])


@pytest.mark.unit
def test_observations_are_the_current_stock_plus_the_leader_signal(regulated, toy):
    states = run_episode(regulated, toy, [0.1, 0.1])

    final = states[-1]
    # The observation of step t is built after the transition of step t - 1.
    np.testing.assert_allclose(final.obs["f0"][0], [1.0, 0.1])
    np.testing.assert_allclose(final.obs["f0"][1], [0.8, 0.1])
    np.testing.assert_allclose(final.obs["f0"][2], [0.6, 0.1])


@pytest.mark.unit
def test_step_advances_the_clock(unregulated, toy):
    states = run_episode(unregulated, toy, [0.0, 0.0])

    assert [s.t for s in states] == [0, 1, 2]


@pytest.mark.unit
def test_step_logs_iter_and_the_mean_follower_reward(regulated, toy):
    run_episode(regulated, toy, [0.2, 0.4, 0.6])

    peeked = regulated.logger.peek()
    assert peeked.iter == [1, 2, 3]
    assert peeked.reward_mean == pytest.approx([0.1, 0.3, 0.5])
    assert peeked.reward_total == pytest.approx([0.1, 0.3, 0.5])
    assert peeked.reward_min == pytest.approx([0.1, 0.3, 0.5])
    assert peeked.reward_max == pytest.approx([0.1, 0.3, 0.5])
    assert peeked.reward_terminal == pytest.approx([0.1, 0.3, 0.5])


@pytest.mark.unit
def test_leaders_do_not_count_in_the_logged_reward(toy, identity_ray_get):
    # Only followers earn rewards; a leader-less run gives the same series.
    with_leader = toy.make_env(toy.ScriptedWorld(), leader=True)
    without = toy.make_env(toy.ScriptedWorld(), leader=False)

    run_episode(with_leader, toy, [0.2, 0.2])
    run_episode(without, toy, [0.2, 0.2])

    assert with_leader.logger.peek().reward_mean == pytest.approx(
        without.logger.peek().reward_mean
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    "horizon, truncated_after_each_step",
    [(None, [False, False, False]), (2, [False, True, True]), (1, [True, True, True])],
)
def test_truncation_follows_the_horizon(
    toy, identity_ray_get, horizon, truncated_after_each_step
):
    env = toy.make_env(toy.ScriptedWorld(), horizon=horizon)

    states = run_episode(env, toy, [0.0, 0.0, 0.0])[1:]

    assert [s.truncateds["__all__"] for s in states] == truncated_after_each_step
    assert [s.terminateds["__all__"] for s in states] == [False, False, False]


@pytest.mark.unit
def test_termination_flags_cover_every_agent_leaders_included(unregulated, toy):
    states = run_episode(unregulated, toy, [0.0, 0.0, 0.0])

    final = states[-1]
    assert set(final.truncateds) == {"regulator", "f0", "f1", "__all__"}
    assert set(final.terminateds) == {"regulator", "f0", "f1", "__all__"}
    assert all(final.truncateds.values())
    assert not any(final.terminateds.values())


@pytest.mark.unit
def test_a_new_episode_starts_from_a_clean_logger(regulated, toy):
    run_episode(regulated, toy, [0.2, 0.2])
    run_episode(regulated, toy, [0.4])

    peeked = regulated.logger.peek()
    assert peeked.iter == [1]
    assert peeked.reward_mean == pytest.approx([0.3])


@pytest.mark.unit
def test_environment_without_a_schema_steps_without_logging(toy, identity_ray_get):
    env = toy.make_env(toy.ScriptedWorld(), schema=None, horizon=2)

    states = run_episode(env, toy, [0.2, 0.2])

    assert env.logger is None
    assert states[-1].t == 2
    assert states[-1].rewards["f0"] == pytest.approx([0.2, 0.2])
