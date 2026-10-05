"""``FisheryRegulatorEnv.reward``: the fitness is computed on the tail of each episode.

The inner environment logs, for every evaluation episode, the step series of
the reward, the normalized biomass and the realized harvest. The regulator
keeps the last ``fitness_tail_steps`` steps of each series (all of them when
the episode is shorter), summarizes them, and averages the summaries over the
episodes and the seeds of a candidate. The tests feed hand-built metrics to
``reward`` and compare the objective, ``harvest_score + weight * mean_fish``,
with values computed by hand.
"""

from types import SimpleNamespace

import pytest
import ray

from examples.bilevel_fishery.contexts import FitnessContext
from examples.bilevel_fishery.regulator_env import FisheryRegulatorEnv

K = 1000.0
MSY = 100.0
WEIGHT = 2.0


@pytest.fixture(autouse=True)
def ray_get_is_identity(monkeypatch):
    monkeypatch.setattr(ray, "get", lambda ref, *args, **kwargs: ref)


def make_env(**ecology) -> FisheryRegulatorEnv:
    world = SimpleNamespace(
        append_context=SimpleNamespace(remote=lambda context: context)
    )
    return FisheryRegulatorEnv(
        world=world,
        optimizer=None,
        horizon=1,
        agents_cfgs={},
        seeds=[0],
        ecology_cfg={
            "K": K,
            "sustainability_weight": WEIGHT,
            "sustainability_threshold": 0.1,
            **ecology,
        },
    )


def episode(fish, harvest, reward=None):
    """One logged episode: each leaf holds one list of steps per logged episode."""
    return SimpleNamespace(
        reward_series=[reward if reward is not None else [0.0] * len(fish)],
        fish_norm_next_series=[fish],
        H_realized_series=[harvest],
        MSY=[MSY],
    )


def metrics_of(mechanisms, split="eval"):
    """``mechanisms`` maps a mechanism id to ``{seed id: [episode, ...]}``."""
    rollout = SimpleNamespace(
        by_mechanism={
            str(mechanism): SimpleNamespace(
                by_seed={
                    str(seed): SimpleNamespace(
                        by_episode={str(i): ep for i, ep in enumerate(episodes)}
                    )
                    for seed, episodes in seeds.items()
                }
            )
            for mechanism, seeds in mechanisms.items()
        }
    )
    return SimpleNamespace(**{split: SimpleNamespace(rollout=rollout)})


@pytest.mark.unit
def test_the_objective_uses_only_the_last_steps_of_the_episode():
    env = make_env(fitness_tail_steps=3)
    # The first three steps are a healthy stock with no catch; the tail is a
    # depleted stock (0.3) with a catch of 50 = MSY / 2 at every step.
    fish = [0.9, 0.9, 0.9, 0.3, 0.3, 0.3]
    harvest = [0.0, 0.0, 0.0, 50.0, 50.0, 50.0]

    fitness = env.reward(metrics_of({0: {0: [episode(fish, harvest)]}}))

    assert fitness == pytest.approx([0.5 + WEIGHT * 0.3])
    assert env.last_metrics[0]["mean_fish"] == pytest.approx(0.3)
    assert env.last_metrics[0]["min_fish"] == pytest.approx(0.3)


@pytest.mark.unit
def test_an_episode_shorter_than_the_tail_contributes_all_its_steps():
    env = make_env(fitness_tail_steps=50)
    fish = [0.2, 0.4]
    harvest = [100.0, 0.0]

    fitness = env.reward(metrics_of({0: {0: [episode(fish, harvest)]}}))

    assert fitness == pytest.approx([0.5 + WEIGHT * 0.3])


@pytest.mark.unit
def test_the_tail_statistics_are_averaged_over_episodes_and_seeds():
    env = make_env(fitness_tail_steps=2)
    low = episode([0.9, 0.2, 0.2], [0.0, 0.0, 0.0])
    high = episode([0.1, 0.6, 0.6], [0.0, 100.0, 100.0])
    two_in_one_leaf = SimpleNamespace(
        reward_series=[[0.0] * 3, [0.0] * 3],
        fish_norm_next_series=[[0.9, 0.2, 0.2], [0.1, 0.6, 0.6]],
        H_realized_series=[[0.0] * 3, [0.0, 100.0, 100.0]],
        MSY=[MSY, MSY],
    )

    by_seed = env.reward(metrics_of({0: {0: [low], 1: [high]}}))
    by_leaf = env.reward(metrics_of({0: {0: [two_in_one_leaf]}}))

    # Mean biomass 0.2 and 0.6 give 0.4; harvest score 0 and 1 give 0.5.
    assert by_seed == pytest.approx([0.5 + WEIGHT * 0.4])
    assert by_leaf == pytest.approx(by_seed)


@pytest.mark.unit
def test_the_gaps_of_the_metric_logger_are_skipped():
    env = make_env(fitness_tail_steps=2)
    # The key logged an episode in the second inner iteration only; the first
    # entry of every leaf is the logger's gap.
    gapped = SimpleNamespace(
        reward_series=[None, [0.0] * 3],
        fish_norm_next_series=[None, [0.9, 0.2, 0.2]],
        H_realized_series=[None, [0.0] * 3],
        MSY=[None, MSY],
    )

    fitness = env.reward(metrics_of({0: {0: [gapped]}}))

    assert fitness == pytest.approx([0.0 + WEIGHT * 0.2])


@pytest.mark.unit
def test_collapse_rate_counts_the_tail_steps_below_the_threshold():
    env = make_env(fitness_tail_steps=4)
    fish = [0.05, 0.5, 0.05, 0.05, 0.5, 0.5]

    env.reward(metrics_of({0: {0: [episode(fish, [0.0] * 6)]}}))

    # The tail holds steps 2 to 5: two of its four values are below 0.1.
    assert env.last_metrics[0]["collapse_rate"] == pytest.approx(0.5)


@pytest.mark.unit
def test_the_split_named_by_aggregation_status_is_read():
    env = make_env(aggregation_status="train", fitness_tail_steps=2)
    metrics = metrics_of({0: {0: [episode([0.5, 0.5], [0.0, 0.0])]}}, split="train")

    assert env.reward(metrics) == pytest.approx([WEIGHT * 0.5])


@pytest.mark.unit
def test_an_index_with_no_rollout_holds_minus_infinity():
    env = make_env(fitness_tail_steps=2)

    fitness = env.reward(metrics_of({1: {0: [episode([0.5, 0.5], [0.0, 0.0])]}}))

    assert fitness[0] == float("-inf")
    assert fitness[1] == pytest.approx(WEIGHT * 0.5)


@pytest.mark.unit
def test_no_fines_are_reported_because_no_fine_is_computed():
    env = make_env(fitness_tail_steps=2)
    env.reward(metrics_of({0: {0: [episode([0.5, 0.5], [0.0, 0.0])]}}))

    assert "total_fines" not in env.last_metrics[0]
    assert "total_fines" not in FitnessContext.model_fields
    with pytest.raises(TypeError):
        FitnessContext.from_metrics(
            mean_reward=0.0,
            collapse_rate=0.0,
            sustainability_penalty=0.0,
            sustainability_weight=1.0,
            total_fines=0.5,
        )


@pytest.mark.unit
def test_a_missing_capacity_is_reported_by_name():
    world = SimpleNamespace(
        append_context=SimpleNamespace(remote=lambda context: context)
    )

    with pytest.raises(ValueError, match="'K'"):
        FisheryRegulatorEnv(
            world=world,
            optimizer=None,
            horizon=1,
            agents_cfgs={},
            seeds=[0],
            ecology_cfg={"sustainability_weight": WEIGHT},
        )


@pytest.mark.unit
@pytest.mark.parametrize("split", ["train", "eval"])
def test_a_split_without_rollouts_is_reported_by_name(split):
    env = make_env(aggregation_status=split)

    with pytest.raises(ValueError, match=f"No rollout found for the '{split}' split"):
        env.reward(metrics_of({}, split=split))
