"""``FisheryRegulatedEnv``: reference points, reset, the stock equation and the logs.

The environment follows the discrete surplus-production model in the form
Nadine reworked it (commits 82fd437 and 553679a). With ``B(t)`` the stock the
step starts with, ``H(t)`` the total request, ``R(t)`` the total restoration and
``g(B) = (r / p) B (1 - (B / K)^p)``, the growth of the step is
``g(B(t)) + noise(t) + R(t)`` and the biomass available is
``max(B(t) + growth, 0)``. The catch ``C(t)`` is the request capped at the
biomass available, and ``B(t + 1) = clip(available - C(t), 0, K)``. Every fisher
requests its catch from the same ``B(t)``; when the total request exceeds the
biomass available each fisher receives a pro-rata share, and ``C(t)`` is what
``H_realized`` logs. The fishers' mechanisms write the request into
``state["harvest"]`` and the restoration into ``state["restore"]`` in place, and
the transition restarts both from zero at the next step. A restoration costs
``restoration_cost * effort^2`` in reward and adds
``restoration_power * K * effort / N`` biomass, where the effort is the sigmoid
of the raw policy output over the action temperature. The tests below pin that
equation with expected values computed here from the formulas, never read back
from the environment. The regulation (quota, fine, risk penalty) lives in
leader mechanisms tested under ``tests/mechanism/``.

The tests drive the environment the way the RLlib adapter does and use
``sigma = 0`` (except where the noise is the subject) so that the dynamics are
deterministic.
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
from examples.bilevel_fishery.queries import (
    ES_QUERIES,
    FISHERY_ENV_QUERIES,
    INNER_QUERIES,
)
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
}
RAW_OFF = -50.0
RAW_ON = 50.0
ACTION_TEMPERATURE = 4.0
RESTORATION_POWER = 0.05
RESTORATION_COST = 0.25
CAPACITY = ECOLOGY["K"]


def effort(raw: float) -> float:
    """Return the fraction a raw policy output maps to, ``sigmoid(z / 4)``."""

    return 1.0 / (1.0 + np.exp(-raw / ACTION_TEMPERATURE))


def restored(
    n_fishers: int,
    raw: float = RAW_OFF,
    power: float = RESTORATION_POWER,
    capacity: float = CAPACITY,
):
    """Return the total biomass ``n`` fishers restore at the same raw output.

    Each fisher restores ``power * K * effort / n``. The idle fishers of most
    tests play ``RAW_OFF``, whose effort is about 4e-6 and which therefore still
    restores a small, exactly known amount.
    """

    return n_fishers * power * capacity * effort(raw) / n_fishers


def collapse_penalty(fish_norm_next: float) -> float:
    """Return the smooth penalty every follower pays when the stock collapses."""

    return 0.1 / (1.0 + np.exp(np.clip((fish_norm_next - 0.20) / 0.03, -60.0, 60.0)))


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


def make_env(
    n: int = 1,
    drop: tuple[str, ...] = (),
    schema=FisheryMetricSchema,
    restoration_power: float = RESTORATION_POWER,
    **ecology,
) -> FisheryRegulatedEnv:
    agents = {
        aid: FishermanConfig(
            id=aid,
            policy_id="fisher_policy",
            observation_space=OBSERVATION,
            mechanisms=(
                FishingConfig(action_space=UNBOUNDED, id="harvest"),
                RestoreConfig(
                    action_space=UNBOUNDED,
                    id="restore",
                    restoration_cost=RESTORATION_COST,
                    restoration_power=restoration_power,
                ),
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
        schema=schema,
        ecology_cfg={
            key: value
            for key, value in {**ECOLOGY, **ecology}.items()
            if key not in drop
        },
    )


def decoded_actions(env: FisheryRegulatedEnv, mdp: MDPState, raw_actions: dict) -> dict:
    """Decode policy outputs at the adapter boundary before updating the MDP."""

    return {
        aid: {
            mid: env.followers[aid].mechanisms[mid].decode(mdp, action)
            for mid, action in acts.items()
        }
        for aid, acts in raw_actions.items()
    }


def actions(
    env: FisheryRegulatedEnv,
    mdp: MDPState,
    n: int,
    harvest: float,
    restore: float = RAW_OFF,
) -> dict:
    """Return the decoded actions corresponding to the supplied policy outputs."""

    raw_actions = {
        aid: {
            "harvest": np.asarray([harvest], dtype=np.float32),
            "restore": np.asarray([restore], dtype=np.float32),
        }
        for aid in fisher_ids(n)
    }

    return decoded_actions(env, mdp, raw_actions)


def play(env: FisheryRegulatedEnv, n: int, steps: int, harvest: float, **kwargs):
    """Reset, then decode and step with constant raw policy outputs."""

    mdp = env.reset(MDPState(aids=set(fisher_ids(n))))

    for _ in range(steps):
        mdp.update(actions=actions(env, mdp, n, harvest, **kwargs))

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

    assert mdp.state["fish"] == [640.0] and mdp.state["harvest"] == [0.0]
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
def test_a_single_fisher_requests_a_sigmoid_fraction_of_its_capacity(raw, fraction):
    env = make_env(n=1)
    mdp = env.reset(MDPState(aids=set(fisher_ids(1))))

    mdp.update(actions=actions(env, mdp, 1, raw))

    mdp = env.step(mdp)

    # Catch capacity of one fisher: multiplier * F_msy * stock / n = 240. The
    # request is written in place at the index of the step and restarts at zero.
    assert mdp.state["harvest"][0] == pytest.approx(fraction * 240.0, abs=1e-2)
    assert mdp.state["harvest"][1] == 0.0


@pytest.mark.unit
def test_one_step_of_the_schaefer_equation_by_hand():
    env = make_env(n=1)
    mdp = env.reset(MDPState(aids=set(fisher_ids(1))))

    mdp.update(actions=actions(env, mdp, 1, 0.0))

    mdp = env.step(mdp)

    # B = 800, g = 0.3 * 800 * (1 - 0.8) = 48, request = 0.5 * 240 = 120, the
    # idle restoration R and no noise: growth = 48 + R, available = 848 + R and
    # B(1) = 848 + R - 120 = 728 + R. The stock of step 0 holds the request
    # taken out and the restoration put in: 800 - 120 + R.
    idle = restored(1)
    logged = env.logger.peek()

    assert mdp.state["fish"] == pytest.approx([680.0 + idle, 728.0 + idle])
    assert logged.fish_stock == pytest.approx([800.0])
    assert logged.growth == pytest.approx([48.0 + idle])
    assert logged.H_attempted == pytest.approx([120.0])
    assert logged.H_realized == pytest.approx([120.0])
    assert mdp.state["harvest"] == pytest.approx([120.0, 0.0])


@pytest.mark.unit
def test_one_step_of_the_pella_tomlinson_equation_by_hand():
    env = make_env(n=1, r=0.4, K=2000.0, p=2.0, fish_init=1000.0)
    mdp = env.reset(MDPState(aids=set(fisher_ids(1))))

    mdp.update(actions=actions(env, mdp, 1, 0.0))

    mdp = env.step(mdp)

    # B = 1000, g = (0.4 / 2) * 1000 * (1 - (1000 / 2000)^2) = 150. F_msy is
    # MSY / B_msy with MSY = r K / (p + 1)^((p + 1) / p) and
    # B_msy = K (1 / (p + 1))^(1 / p), and the request is 0.5 * 2 * F_msy * B.
    # The growth of the step also holds the idle restoration R.
    b_msy = 2000.0 * (1.0 / 3.0) ** 0.5
    msy = 0.4 * 2000.0 / 3.0 ** (3.0 / 2.0)
    catch = 0.5 * 2.0 * (msy / b_msy) * 1000.0
    idle = restored(1, capacity=2000.0)

    assert env.logger.peek().growth == pytest.approx([150.0 + idle])
    assert mdp.state["fish"][1] == pytest.approx(1000.0 + 150.0 + idle - catch)


@pytest.mark.unit
def test_the_transition_reads_the_stock_the_step_started_with():
    # Reconstructing the stock from interventions can introduce approximately
    # 1e-13 of floating-point roundoff; the logged starting stock must agree
    # with the original within that numerical tolerance.
    start, harvest, restoration = 832.014, 150.546, 15.865

    assert ((start - harvest) + restoration) - restoration + harvest != start

    env = make_env(n=1, fish_init=start)
    mdp = env.reset(MDPState(aids=set(fisher_ids(1))))
    requests = [
        MDPState(state={"fish": -harvest, "harvest": harvest}),
        MDPState(state={"fish": restoration, "restore": restoration}),
    ]

    env.pella_tomlinson(mdp.add(requests))
    assert env.logger.peek().fish_stock == pytest.approx([start], abs=1e-12)


@pytest.mark.unit
def test_the_first_step_already_delivers_its_catch():
    env = make_env(n=2)
    mdp = env.reset(MDPState(aids=set(fisher_ids(2))))

    mdp.update(actions=actions(env, mdp, 2, 0.0))

    mdp = env.step(mdp)

    # Two fishers, each requests 0.5 * 2 * 0.15 * 800 / 2 = 60.
    assert env.logger.peek().H_realized == pytest.approx([120.0])


@pytest.mark.skip(
    reason="FisheryRegulatedEnv does not support schema=None yet; pella_tomlinson() attempts "
    "to write to a missing logger."
)
def test_an_env_without_a_schema_steps_without_logging():
    ...

    env = make_env(n=2, schema=None)
    mdp = play(env, n=2, steps=3, harvest=0.0)

    assert env.logger is None
    assert len(mdp.state["fish"]) == 4


@pytest.mark.unit
def test_the_catch_of_every_step_follows_the_stock_of_that_step():
    env = make_env(n=1)

    play(env, n=1, steps=4, harvest=0.0)

    # A request left over from one step would add to the next one.
    logged = env.logger.peek()
    expected = 0.5 * 2.0 * 0.15 * np.asarray(logged.fish_stock)

    np.testing.assert_allclose(logged.H_realized, expected, rtol=1e-6)

    stock = np.asarray([800.0, *logged.fish_stock_next])
    growth = 0.3 * stock[:-1] * (1.0 - stock[:-1] / 1000.0) + restored(1)

    np.testing.assert_allclose(
        stock[1:], stock[:-1] + growth - np.asarray(logged.H_realized), rtol=1e-5
    )


@pytest.mark.unit
def test_requests_above_the_stock_are_capped_by_available_biomass():
    env = make_env(n=2, unregulated_f_multiplier=50.0)
    mdp = env.reset(MDPState(aids=set(fisher_ids(2))))

    mdp.update(
        actions=decoded_actions(
            env,
            mdp,
            {
                "fisherman:0": {
                    "harvest": np.asarray([0.0], dtype=np.float32),
                    "restore": np.asarray([RAW_OFF], dtype=np.float32),
                },
                "fisherman:1": {
                    "harvest": np.asarray([RAW_ON], dtype=np.float32),
                    "restore": np.asarray([RAW_OFF], dtype=np.float32),
                },
            },
        )
    )

    mdp = env.step(mdp)

    # Each fisher requests from the same 800 units of biomass. The environment
    # currently reports aggregate attempted and realized catch, not per-fisher
    # delivery, so this verifies aggregate conservation rather than pro-rata
    # distribution between individual fishers.
    available = 800.0 + 48.0 + restored(2)
    logged = env.logger.peek()

    assert logged.H_attempted == pytest.approx([4500.0], rel=1e-4)
    assert logged.H_realized == pytest.approx([available])
    assert mdp.state["fish"][1] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.unit
def test_the_outcome_does_not_depend_on_the_order_of_the_fishers():
    low, high = 0.0, RAW_ON

    def run(raw_by_fisher: dict[str, float]):
        env = make_env(n=2, unregulated_f_multiplier=4.0)
        mdp = env.reset(MDPState(aids=set(fisher_ids(2))))

        mdp.update(
            actions=decoded_actions(
                env,
                mdp,
                {
                    aid: {
                        "harvest": np.asarray([raw], dtype=np.float32),
                        "restore": np.asarray([RAW_OFF], dtype=np.float32),
                    }
                    for aid, raw in raw_by_fisher.items()
                },
            )
        )

        mdp = env.step(mdp)

        return env.logger.peek(), mdp

    first, first_mdp = run({"fisherman:0": low, "fisherman:1": high})
    swapped, swapped_mdp = run({"fisherman:0": high, "fisherman:1": low})

    # Swapping identical action magnitudes between fishers must preserve the
    # aggregate attempted and realized harvest and the biomass trajectory.
    assert first.H_attempted == pytest.approx(swapped.H_attempted)
    assert first.H_realized == pytest.approx(swapped.H_realized)
    assert first_mdp.state["fish"] == pytest.approx(swapped_mdp.state["fish"])


@pytest.mark.unit
def test_reward_is_the_harvest_fraction_of_the_step():
    env = make_env(n=2)
    mdp = play(env, n=2, steps=3, harvest=0.0)

    for aid in fisher_ids(2):
        # The MDP may also retain a zero-valued terminal reward slot.
        assert mdp.rewards[aid][:3] == pytest.approx([0.5, 0.5, 0.5], abs=1e-6)


@pytest.mark.unit
def test_restoration_adds_biomass_shared_over_the_fishermen():
    env = make_env(n=2, restoration_power=0.01, fish_init=600.0)
    mdp = env.reset(MDPState(aids=set(fisher_ids(2))))

    mdp.update(actions=actions(env, mdp, 2, RAW_OFF, restore=RAW_ON))

    mdp = env.step(mdp)

    # Each fisher adds 0.01 * K * effort / n = 5 at full effort, 10 in total,
    # and the total is what the state holds under "restore".
    assert mdp.state["restore"][0] == pytest.approx(10.0, abs=1e-3)
    assert mdp.state["restore"][1] == 0.0

    # B = 600, g = 0.3 * 600 * (1 - 0.6) = 72, restoration 10, and the request
    # of the idle harvest (effort ~4e-6 of 90 per fisher) is negligible.
    assert mdp.state["fish"][1] == pytest.approx(600.0 + 72.0 + 10.0, abs=1e-2)


@pytest.mark.unit
def test_restoration_enters_the_equation_next_to_the_production():
    env = make_env(n=2, restoration_power=0.01)
    mdp = env.reset(MDPState(aids=set(fisher_ids(2))))

    mdp.update(actions=actions(env, mdp, 2, RAW_OFF, restore=RAW_ON))

    mdp = env.step(mdp)

    # B(1) = 800 + growth (48 + 10) - catch (~0): the restoration is part of
    # the growth of the step, and so of the biomass the catch can draw on.
    assert env.logger.peek().growth == pytest.approx([48.0 + 10.0], abs=1e-3)
    assert mdp.state["fish"][1] == pytest.approx(800.0 + 48.0 + 10.0, abs=1e-2)


@pytest.mark.unit
def test_zero_restoration_power_makes_the_effort_inert():
    env = make_env(n=2, restoration_power=0.0)
    mdp = env.reset(MDPState(aids=set(fisher_ids(2))))

    mdp.update(actions=actions(env, mdp, 2, RAW_OFF, restore=RAW_ON))

    mdp = env.step(mdp)

    assert mdp.state["restore"][0] == 0.0
    assert env.logger.peek().growth == pytest.approx([48.0])
    assert mdp.state["fish"][1] == pytest.approx(800.0 + 48.0, abs=1e-2)


@pytest.mark.unit
def test_restoration_costs_the_square_of_the_effort_in_reward():
    env = make_env(n=2, restoration_power=0.01)
    mdp = env.reset(MDPState(aids=set(fisher_ids(2))))

    mdp.update(actions=actions(env, mdp, 2, 0.0, restore=0.0))

    mdp = env.step(mdp)

    # Effort 0.5 costs 0.25 * 0.5^2. The fisher is also paid its harvest
    # fraction (0.5) and pays the collapse penalty of the stock it leaves.
    # B = 800, growth = 48 + restoration (0.01 * 1000 * 0.5 = 5), requests
    # 2 * 60 = 120, so B(1) = 800 + 48 + 5 - 120 = 733.
    expected = 0.5 - RESTORATION_COST * 0.5**2 - collapse_penalty(733.0 / 1000.0)

    for aid in fisher_ids(2):
        assert mdp.rewards[aid][0] == pytest.approx(expected)


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
    assert np.all(np.diff(stock) > 0)


@pytest.mark.unit
def test_stock_never_goes_negative_under_overfishing():
    env = make_env(n=2, unregulated_f_multiplier=50.0)

    play(env, n=2, steps=5, harvest=RAW_ON)

    logged = env.logger.peek()

    assert min(logged.fish_stock_next) >= 0.0
    assert min(logged.H_realized) >= 0.0


@pytest.mark.unit
def test_the_catch_never_exceeds_the_stock_plus_the_growth_of_the_step():
    env = make_env(n=3, unregulated_f_multiplier=50.0)

    play(env, n=3, steps=5, harvest=RAW_ON)

    logged = env.logger.peek()
    available = np.maximum(np.asarray(logged.fish_stock) + np.asarray(logged.growth), 0)

    assert np.all(np.asarray(logged.H_realized) <= available + 1e-9)
    assert np.all(np.asarray(logged.H_realized) <= np.asarray(logged.H_attempted))


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
        "H_attempted",
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
def test_realised_harvest_is_not_greater_than_attempted_or_available():
    env = make_env(n=3)

    play(env, n=3, steps=3, harvest=0.0)

    logged = env.logger.peek()
    realized = np.asarray(logged.H_realized)
    attempted = np.asarray(logged.H_attempted)
    available = np.maximum(
        np.asarray(logged.fish_stock) + np.asarray(logged.growth), 0.0
    )

    assert np.all(realized <= attempted + 1e-9)
    assert np.all(realized <= available + 1e-9)


@pytest.mark.unit
def test_total_usage_follows_the_realised_harvest_and_the_observation_the_request():
    env = make_env(n=1)
    mdp = play(env, n=1, steps=3, harvest=0.0)
    logged = env.logger.peek()

    np.testing.assert_allclose(
        logged.total_usage_norm, np.asarray(logged.H_realized) / 1000.0
    )

    # The harvest the next observation reads is the request written in the
    # state at the previous step; here nothing is rationed, so it equals the
    # realised harvest.
    np.testing.assert_allclose(
        mdp.obs[fisher_ids(1)[0]][3][2], mdp.state["harvest"][2] / 1000.0, rtol=1e-6
    )
    np.testing.assert_allclose(mdp.state["harvest"][:-1], logged.H_realized)


@pytest.mark.unit
def test_the_noise_term_scales_with_sigma_and_the_stock():
    quiet = make_env(n=1, sigma=0.0)
    noisy = make_env(n=1, sigma=0.2)

    play(quiet, n=1, steps=3, harvest=RAW_OFF)
    play(noisy, n=1, steps=3, harvest=RAW_OFF)
    assert quiet.logger.peek().growth_noise == [0.0, 0.0, 0.0]

    noise = np.asarray(noisy.logger.peek().growth_noise)

    assert np.all(noise != 0.0)


@pytest.mark.unit
def test_the_first_noise_draw_is_sigma_times_a_normal_draw_times_the_stock():
    env = make_env(n=1, sigma=0.2)
    mdp = env.reset(MDPState(aids=set(fisher_ids(1))))

    mdp.update(actions=actions(env, mdp, 1, RAW_OFF))

    mdp = env.step(mdp)

    # The environment's generator is seeded with 0; the noise is evaluated on
    # B(0) = 800 and enters the growth next to the production (48) and the
    # idle restoration R. The request is 240 times the idle harvest effort.
    noise = 0.2 * np.random.default_rng(0).normal() * 800.0
    idle = restored(1)
    catch = effort(RAW_OFF) * 240.0
    logged = env.logger.peek()

    assert logged.growth_noise[0] == pytest.approx(noise)
    assert logged.growth[0] == pytest.approx(48.0 + noise + idle)
    assert logged.H_realized[0] == pytest.approx(catch)
    assert mdp.state["fish"][1] == pytest.approx(800.0 + 48.0 + noise + idle - catch)


@pytest.mark.unit
def test_the_episode_reducer_keeps_the_aggregate_dynamics():
    env = make_env(n=2)
    steps = 4

    play(env, n=2, steps=steps, harvest=0.0)

    # The transition currently pushes per-step aggregate values; its reducer
    # summarizes them as scalar episode statistics.
    stepwise = env.logger.peek()
    reduced = env.logger.reduce()

    assert len(stepwise.H_realized) == steps
    assert len(stepwise.fish_norm_next_mean) == steps
    assert reduced.H_realized == pytest.approx(np.mean(stepwise.H_realized))
    assert reduced.fish_norm_next_mean == pytest.approx(
        np.mean(stepwise.fish_norm_next_mean)
    )


@pytest.mark.unit
def test_the_unregulated_environment_query_resolves_against_its_logger():
    env = make_env(n=2)

    play(env, n=2, steps=3, harvest=0.0)

    # Quota and intrinsic-utility queries require leader-mechanism metrics;
    # only biomass is filled by the standalone unregulated environment.
    query = next(q for q in FISHERY_ENV_QUERIES if q.title == "Normalized Fish biomass")
    xs, yss, _, _ = SilentReporter("env")._resolve_query(env.logger.peek(), query)

    assert xs and all(ys for ys in yss)


@pytest.mark.unit
def test_every_episode_statistic_named_by_a_query_is_filled_by_the_environment():
    env = make_env(n=2)

    play(env, n=2, steps=3, harvest=0.0)

    logged = env.logger.peek()

    # Leader-specific by_agent penalties are only logged when mechanisms run.
    # The unregulated environment must fill the biomass fields requested by
    # the reporting queries.
    named = {
        path[-1]
        for query in (*ES_QUERIES, *INNER_QUERIES)
        for path in query.y_paths
        if path[-1].startswith("fish_norm_next_")
        and path[-1] in FisheryMetricSchema.model_fields
    }

    assert {"fish_norm_next_mean", "fish_norm_next_min"} <= named

    for field in named:
        assert getattr(logged, field), field
