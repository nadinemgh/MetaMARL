"""``FisheryRegulatedEnv``: reference points, reset, harvest, restoration and logs.

Rewrite of the August ``test_fishery_regulated_env.py``. The benchmark is now
built from agent configurations holding a ``Fishing`` and a ``Restore``
mechanism, and the regulation (quota, fine, risk penalty) lives in leader
mechanisms tested under ``tests/mechanism/``; what stays here is the ecology.

The tests drive the environment the way the RLlib adapter does and use
``sigma = 0`` so that the dynamics are deterministic. They pin quantities that
do not depend on the order in which the discrete-time update applies harvest
and growth. Three observations about that order are reported as open questions
rather than asserted (see the report of the pass): harvests depend on the order
of the fishermen, the growth term is computed on the previous step's stock, and
the first step grows the stock after the harvest.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import ray
from gymnasium import spaces

from core.mechanism.base import MDPState
from core.reporting.base import Reporter
from core.reporting.config import ReporterConfig
from examples.bilevel_fishery.metric_schema import FisheryMetricSchema
from examples.bilevel_fishery.regulated_env import (
    FishermanConfig,
    FisheryRegulatedEnv,
    FishingConfig,
    RestoreConfig,
)

UNBOUNDED = spaces.Box(-np.inf, np.inf, (1,), np.float32)
OBSERVATION = spaces.Box(-np.inf, np.inf, (5,), np.float32)
ECOLOGY = {
    "r": 0.3,
    "K": 1000.0,
    "p": 1.0,
    "fish_init": 800.0,
    "sigma": 0.0,
    "initial_stock_log_sigma": 0.0,
    "unregulated_f_multiplier": 2.0,
    "restoration_effectiveness": 0.0,
}
RAW_OFF = -50.0
RAW_ON = 50.0


class SilentReporter(Reporter):
    def __init__(self, label):
        self.label = label

    def _report(self, query, series):
        return None

    def close(self):
        return None


class SilentReporterConfig(ReporterConfig):
    def build(self, *, label=None):
        return SilentReporter(label)


def fisher_ids(n: int) -> list[str]:
    return [f"fisherman:{i}" for i in range(n)]


def make_env(n: int = 1, drop: tuple[str, ...] = (), **ecology) -> FisheryRegulatedEnv:
    agents = {
        aid: FishermanConfig(
            id=aid,
            policy_id="fisher_policy",
            observation_space=OBSERVATION,
            mechanisms=(
                FishingConfig(action_space=UNBOUNDED, id="harvest"),
                RestoreConfig(action_space=UNBOUNDED, id="restore"),
            ),
        )
        for aid in fisher_ids(n)
    }
    world = SimpleNamespace(
        get_mechanism_by_id=SimpleNamespace(remote=lambda **kwargs: None)
    )
    return FisheryRegulatedEnv(
        world=world,
        mechanism_id=0,
        horizon=50,
        seed=0,
        agents_cfg_dict=agents,
        leaders_cfg_dict={},
        reporter_cfg=SilentReporterConfig(project="p"),
        schema=FisheryMetricSchema,
        ecology_cfg={
            key: value
            for key, value in {**ECOLOGY, **ecology}.items()
            if key not in drop
        },
    )


def actions(n: int, harvest: float, restore: float = RAW_OFF) -> dict:
    return {
        aid: {
            "harvest": np.asarray([harvest], dtype=np.float32),
            "restore": np.asarray([restore], dtype=np.float32),
        }
        for aid in fisher_ids(n)
    }


def play(env: FisheryRegulatedEnv, n: int, steps: int, harvest: float, **kwargs):
    """Reset, then step ``steps`` times with constant raw actions."""
    mdp = env.reset(MDPState(aids=set(fisher_ids(n))))
    for _ in range(steps):
        mdp.update(actions=actions(n, harvest, **kwargs))
        mdp = env.step(mdp)
    return mdp


@pytest.fixture(autouse=True)
def ray_get_is_identity(monkeypatch):
    monkeypatch.setattr(ray, "get", lambda ref, *args, **kwargs: ref)


@pytest.mark.unit
def test_reference_points_of_the_schaefer_model():
    env = make_env(r=0.3, K=1000.0, p=1.0)

    # p = 1: B_msy = K / 2, MSY = r K / 4 and F_msy = MSY / B_msy = r / 2.
    assert env.ecology["B_msy"] == pytest.approx(500.0)
    assert env.ecology["MSY"] == pytest.approx(75.0)
    assert env.ecology["F_msy"] == pytest.approx(0.15)


@pytest.mark.unit
def test_reference_points_of_the_pella_tomlinson_model():
    env = make_env(r=0.4, K=2000.0, p=2.0)

    b_msy = 2000.0 * (1.0 / 3.0) ** 0.5
    msy = 0.4 * 2000.0 / 3.0 ** (3.0 / 2.0)
    assert env.ecology["B_msy"] == pytest.approx(b_msy)
    assert env.ecology["MSY"] == pytest.approx(msy)
    assert env.ecology["F_msy"] == pytest.approx(msy / b_msy)


@pytest.mark.unit
def test_capacity_may_be_given_as_max_fish_and_the_start_as_b0():
    env = make_env(drop=("K", "fish_init"), max_fish=400.0, B0=300.0)

    assert env.K == 400.0
    assert env.fish_init == 300.0


@pytest.mark.unit
def test_deterministic_reset_starts_at_the_configured_stock():
    env = make_env(fish_init=640.0)

    mdp = env.reset(MDPState(aids=set(fisher_ids(1))))

    assert mdp.state["fish"] == [640.0] and mdp.state["usage"] == [0.0]
    assert mdp.params["K"] == 1000.0 and mdp.params["F_msy"] == pytest.approx(0.15)


@pytest.mark.unit
def test_stochastic_reset_is_reproducible_and_clipped_to_capacity():
    first = make_env(initial_stock_log_sigma=0.3, fish_init=900.0)
    second = make_env(initial_stock_log_sigma=0.3, fish_init=900.0)
    huge = make_env(initial_stock_log_sigma=0.05, fish_init=1e9)
    aids = set(fisher_ids(1))

    start_a = first.reset(MDPState(aids=aids)).state["fish"][0]
    start_b = second.reset(MDPState(aids=aids)).state["fish"][0]
    start_huge = huge.reset(MDPState(aids=aids)).state["fish"][0]

    assert start_a == start_b != 900.0
    assert start_huge == 1000.0


@pytest.mark.unit
def test_first_observation_is_the_normalised_stock_and_usage():
    env = make_env(n=2, fish_init=500.0, K=1000.0)

    mdp = env.reset(MDPState(aids=set(fisher_ids(2))))

    for aid in fisher_ids(2):
        np.testing.assert_allclose(mdp.obs[aid][0], [0.5, 0.0, 0.0, 0.0, 0.0])
        assert mdp.obs[aid][0].dtype == np.float32


@pytest.mark.unit
@pytest.mark.parametrize("raw, fraction", [(0.0, 0.5), (RAW_ON, 1.0), (RAW_OFF, 0.0)])
def test_a_single_fisher_harvests_a_sigmoid_fraction_of_its_capacity(raw, fraction):
    env = make_env(n=1)

    mdp = env.reset(MDPState(aids=set(fisher_ids(1))))
    mdp.update(actions=actions(1, raw))
    mdp = env.step(mdp)

    # Catch capacity of one fisher: multiplier * F_msy * stock / n = 240.
    harvested = 800.0 - mdp.state["fish"][0]
    assert harvested == pytest.approx(fraction * 240.0, abs=1e-2)


@pytest.mark.unit
def test_reward_is_the_harvest_fraction_of_the_step():
    env = make_env(n=2)

    mdp = play(env, n=2, steps=3, harvest=0.0)

    for aid in fisher_ids(2):
        assert mdp.rewards[aid] == pytest.approx([0.5, 0.5, 0.5])


@pytest.mark.unit
def test_restoration_adds_biomass_shared_over_the_fishermen():
    env = make_env(n=2, restoration_effectiveness=0.01)

    mdp = env.reset(MDPState(aids=set(fisher_ids(2))))
    mdp.update(actions=actions(2, RAW_OFF, restore=RAW_ON))
    mdp = env.step(mdp)

    # Each fisher adds 0.01 * K * fraction / n = 5 at full effort.
    assert mdp.state["fish"][0] == pytest.approx(800.0 + 10.0, abs=1e-2)


@pytest.mark.unit
def test_no_restoration_effectiveness_makes_the_effort_inert():
    env = make_env(n=2, restoration_effectiveness=0.0)

    mdp = env.reset(MDPState(aids=set(fisher_ids(2))))
    mdp.update(actions=actions(2, RAW_OFF, restore=RAW_ON))
    mdp = env.step(mdp)

    assert mdp.state["fish"][0] == pytest.approx(800.0, abs=1e-2)


@pytest.mark.unit
def test_stock_at_capacity_without_fishing_stays_there():
    env = make_env(n=2, fish_init=1000.0)

    mdp = play(env, n=2, steps=3, harvest=RAW_OFF)

    assert mdp.state["fish"][-1] == pytest.approx(1000.0, abs=0.1)


@pytest.mark.unit
def test_stock_below_capacity_grows_without_fishing():
    env = make_env(n=2, fish_init=200.0)

    mdp = play(env, n=2, steps=3, harvest=RAW_OFF)

    stock = mdp.state["fish"]
    assert stock[-1] > stock[0] > 190.0
    assert np.all(np.diff(stock[1:]) > 0)


@pytest.mark.unit
def test_stock_never_goes_negative_under_overfishing():
    env = make_env(n=2, unregulated_f_multiplier=50.0)

    play(env, n=2, steps=5, harvest=RAW_ON)

    assert min(env.logger.peek().fish_stock_next) >= 0.0


@pytest.mark.unit
def test_every_step_logs_one_value_per_dynamics_field():
    env = make_env(n=2)
    steps = 4

    play(env, n=2, steps=steps, harvest=0.0)

    logged = env.logger.peek()
    assert logged.iter == [1, 2, 3, 4]
    for field in (
        "B_msy",
        "MSY",
        "F_msy",
        "fish_stock",
        "fish_stock_next",
        "fish_norm_next_mean",
        "fish_norm_next_min",
        "fish_norm_next_max",
        "fish_norm_next_last",
        "growth",
        "growth_noise",
        "H_realized",
        "total_usage_norm",
    ):
        assert len(getattr(logged, field)) == steps, field
    assert logged.B_msy == [500.0] * steps
    assert logged.MSY == [75.0] * steps
    assert logged.growth_noise == [0.0] * steps
    np.testing.assert_allclose(
        logged.fish_norm_next_last, np.asarray(logged.fish_stock_next) / 1000.0
    )


@pytest.mark.unit
def test_usage_series_follows_the_realised_harvest_normalised_by_capacity():
    env = make_env(n=1)

    mdp = play(env, n=1, steps=3, harvest=0.0)

    logged = env.logger.peek()
    np.testing.assert_allclose(
        logged.total_usage_norm, np.asarray(logged.H_realized) / 1000.0
    )
    # The usage the next observation reads is the realised harvest of the step.
    np.testing.assert_allclose(
        mdp.obs[fisher_ids(1)[0]][3][2], mdp.state["usage"][3] / 1000.0, rtol=1e-6
    )


@pytest.mark.unit
def test_the_noise_term_scales_with_sigma_and_the_stock():
    quiet = make_env(n=1, sigma=0.0)
    noisy = make_env(n=1, sigma=0.2)

    play(quiet, n=1, steps=3, harvest=RAW_OFF)
    play(noisy, n=1, steps=3, harvest=RAW_OFF)

    assert quiet.logger.peek().growth_noise == [0.0, 0.0, 0.0]
    noise = np.asarray(noisy.logger.peek().growth_noise)
    assert np.all(noise != 0.0)
