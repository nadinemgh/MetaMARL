"""``MultiAgentEnv.reset``: episode identity, candidate fetch and first observation.

The candidate-keeping behaviour (a later reset still gives the leaders the last
candidate fetched, and a newly published one replaces it) is pinned by
``test_marl_regulated_mechanism.py``; here the surrounding contract is tested:
what the environment asks the World, what it logs, what the initial MDP holds
and how a failing fetch is reported.
"""

import numpy as np
import pytest

from core.mechanism.base import MDPState
from core.world.context import MechanismStatus

AIDS = {"f0", "f1"}


@pytest.mark.unit
def test_reset_asks_the_world_for_this_instances_candidate(toy, identity_ray_get):
    world = toy.ScriptedWorld([toy.candidate(0.2)])
    env = toy.make_env(world, mechanism_id=2, policy_seed=5, mode="eval")

    env.reset(MDPState(aids=AIDS))

    assert world.fetches == [
        {"mechanism_id": 2, "seed": 5, "mode": MechanismStatus.eval}
    ]


@pytest.mark.unit
def test_reset_fetches_at_every_episode(toy, identity_ray_get):
    world = toy.ScriptedWorld([toy.candidate(0.2), None, None])
    env = toy.make_env(world)

    for _ in range(3):
        env.reset(MDPState(aids=AIDS))

    assert len(world.fetches) == 3
    assert env.published_mechanism_assigned


@pytest.mark.unit
def test_reset_logs_the_episode_identity_and_no_step(toy, identity_ray_get):
    env = toy.make_env(toy.ScriptedWorld(), mechanism_id=1, seed=3, policy_seed=4)

    env.reset(MDPState(aids=AIDS))

    peeked = env.logger.peek()
    assert (peeked.mechanism_id, peeked.seed, peeked.policy_seed) == ([1], [3], [4])
    assert peeked.iter == [] and peeked.reward_mean == []


@pytest.mark.unit
def test_reset_does_not_reseed_the_environment(toy, identity_ray_get):
    env = toy.make_env(toy.ScriptedWorld(), seed=3)
    state_before = env.rng.bit_generator.state

    env.reset(MDPState(aids=AIDS))

    assert env.seed == 3
    assert env.rng.bit_generator.state == state_before


@pytest.mark.unit
def test_reset_requires_a_mechanism_id(toy, identity_ray_get):
    env = toy.make_env(toy.ScriptedWorld(), mechanism_id=None)

    with pytest.raises(RuntimeError, match="MultiAgentEnv has no mechanism_id"):
        env.reset(MDPState(aids=AIDS))


@pytest.mark.unit
def test_a_failing_fetch_is_reported_with_the_mechanism_id(toy, identity_ray_get):
    world = toy.ScriptedWorld()

    def fail(**kwargs):
        raise ConnectionError("actor died")

    world.get_mechanism_by_id.remote = fail
    env = toy.make_env(world, mechanism_id=4)

    with pytest.raises(RuntimeError, match="Could not fetch mechanism_id=4") as caught:
        env.reset(MDPState(aids=AIDS))

    assert isinstance(caught.value.__cause__, ConnectionError)


@pytest.mark.unit
def test_initial_mdp_holds_the_hook_state_and_every_followers_observation(
    toy, identity_ray_get
):
    env = toy.make_env(toy.ScriptedWorld(), leader=False)

    mdp = env.reset(MDPState(aids=AIDS))

    assert mdp.t == 0
    assert mdp.state["stock"] == [toy.initial_stock]
    assert sorted(mdp.obs.data) == ["f0", "f1"]
    for aid in AIDS:
        np.testing.assert_allclose(mdp.obs[aid][0], [toy.initial_stock, 0.0])
    assert mdp.actions.data == {}


@pytest.mark.unit
def test_leaders_hold_the_candidate_as_their_first_action(toy, identity_ray_get):
    env = toy.make_env(toy.ScriptedWorld([toy.candidate(0.25)]))

    mdp = env.reset(MDPState(aids=AIDS))

    assert list(mdp.actions.data) == ["regulator"]
    np.testing.assert_allclose(mdp.actions["regulator"]["fee"][0], [0.25])
    # The Fee mechanism writes the fee it holds into the second entry.
    for aid in AIDS:
        np.testing.assert_allclose(mdp.obs[aid][0], [toy.initial_stock, 0.25])


@pytest.mark.unit
def test_leaders_without_a_default_hold_no_action_before_a_candidate(
    toy, identity_ray_get
):
    env = toy.make_env(toy.ScriptedWorld())

    mdp = env.reset(MDPState(aids=AIDS))

    assert mdp.actions.data == {}
    for aid in AIDS:
        np.testing.assert_allclose(mdp.obs[aid][0], [toy.initial_stock, 0.0])


@pytest.mark.unit
def test_leaders_hold_their_mechanism_defaults_before_a_candidate(
    toy, identity_ray_get
):
    env = toy.make_env(
        toy.ScriptedWorld(), leaders_cfg_dict=toy.regulator_configs(fee_default=0.4)
    )

    mdp = env.reset(MDPState(aids=AIDS))

    assert list(mdp.actions.data) == ["regulator"]
    np.testing.assert_allclose(mdp.actions["regulator"]["fee"][0], [0.4])
    for aid in AIDS:
        np.testing.assert_allclose(mdp.obs[aid][0], [toy.initial_stock, 0.4])
    # A default is not a published candidate.
    assert not env.published_mechanism_assigned
    assert env.mechanism is None


@pytest.mark.unit
def test_a_fetched_candidate_replaces_the_mechanism_defaults(toy, identity_ray_get):
    env = toy.make_env(
        toy.ScriptedWorld([None, toy.candidate(0.25), None]),
        leaders_cfg_dict=toy.regulator_configs(fee_default=0.4),
    )

    fees = [
        float(env.reset(MDPState(aids=AIDS)).actions["regulator"]["fee"][0][0])
        for _ in range(3)
    ]

    np.testing.assert_allclose(fees, [0.4, 0.25, 0.25])
    assert env.published_mechanism_assigned


@pytest.mark.unit
def test_reset_without_a_reset_hook_only_builds_the_observations(toy, identity_ray_get):
    class Bare(toy.ToyEnv):
        _reset = None

    env = toy.make_env(
        toy.ScriptedWorld(),
        env_cls=Bare,
        agents_cfg_dict={
            "o0": toy.ObserverConfig(
                id="o0",
                policy_id="p",
                mechanisms=toy.HarvestConfig(action_space=None, id="harvest"),
            )
        },
        leaders_cfg_dict={},
    )

    mdp = env.reset(MDPState(aids={"o0"}))

    assert mdp.state.data == {}
    np.testing.assert_allclose(mdp.obs["o0"][0], [7.0])


@pytest.mark.unit
def test_reset_returns_a_new_mdp_and_leaves_the_argument_alone(toy, identity_ray_get):
    env = toy.make_env(toy.ScriptedWorld([toy.candidate(0.25)]))
    start = MDPState(aids=AIDS)

    mdp = env.reset(start)

    assert mdp is not start
    assert start.obs.data == {} and start.actions.data == {} and start.state.data == {}
