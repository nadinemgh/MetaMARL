"""Fresh-water example: crop model, lake, water policy, composition, end-to-end run.

The expected values are written by hand from the formulas of the example, not
taken from its code.

Crop model (Blaney-Criddle form, July is 23 degrees and has a daylight share of
0.34): on July 15, 40 days after planting, the development stage has a crop
coefficient of 0.80, so

    ETo   = 0.34 * (0.46 * 23 + 8) = 6.3172 mm/day
    ETc   = 0.80 * 6.3172          = 5.05376 mm/day
    deficit with 1 mm of rain      = 4.05376 mm/day

and a farm of one million square metres needs 5053.76 cubic metres a day, of
which 1000 come from the rain and 4053.76 must be irrigated.

Water policy with the default rules (quota 0.85, minimum demand 0.05, maximum
demand 1.0, fine 0.05, risk scale 0.5, risk power 2) at a filled fraction of 0.9
and a release pressure of 0.5, for a request of 80 % of a deficit of 1000 cubic
metres:

    stress  = (0.9 - 0.85) / (1 - 0.85)    = 1/3
    allowed = 0.05 + 1/3 * (1.0 - 0.05)    = 0.366667  -> 366.667 m3
    quota   = 0.05 * (800 - 366.667) / 1000 = 0.021667
    flow    = 0.5 * 0.5 * 0.8 ** 2          = 0.16
    penalty = 0.181667

Surrogate lake (area one million square metres, depth 10 m, full at 420 m, no
noise, no rain, inflow 10 m3/s, release 60 % of the inflow): from the level 0.5
(415 m, 5e6 m3) with a withdrawal of 2 m3/s the volume changes by
``(10 - 2 - 6) * 86400 = 172800`` m3, which is 415.1728 m; the baseline with no
withdrawal changes by ``(10 - 6) * 86400``, which is 415.3456 m.

The composition tests step the environment behind the real RLlib adapter with a
scripted stand-in for the ``World`` actor, as
``tests/integration/test_fishery_regulator_composition.py`` does, so they run
without Ray. The end-to-end tests launch ``examples.fresh_water.debug`` as child
processes with a time limit. The Raven model is not part of the repository, so
the Raven path is tested against a stand-in executable that writes the output
files Raven would write.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from gymnasium import spaces

from core.adaptors.ray.marl_env import RLlibMultiAgentEnvAdapter
from core.agents.base import AgentConfig
from core.envs import marl_regulated
from core.reporting.csv import CSVConfig
from core.world.context import MechanismContext, MechanismStatus
from examples.fresh_water.contexts import FitnessContext
from examples.fresh_water.hydrology import (
    RavenLake,
    RavenOutputError,
    SurrogateLake,
    estimate_temp_c,
    write_stand_in_raven,
)
from examples.fresh_water.mechanism import (
    DEFAULT_RULES,
    IRRIGATE_ID,
    OBSERVATION_SIZE,
    RULE_NAMES,
    RULES_SPACE,
    WATER_POLICY_ID,
    WaterPolicyConfig,
    allowed_fraction,
    decode_rules,
    encode_rules,
    irrigation_fraction,
    quota_stress,
    violation_signal,
)
from examples.fresh_water.metric_schema import WaterMetricSchema
from examples.fresh_water.regulated_env import (
    IRRIGATE_SPACE,
    IrrigateConfig,
    UtilizerConfig,
    WaterRegulatedEnv,
    crop_demand,
    crop_stage,
)
from examples.fresh_water.regulator_env import WaterRegulatorEnv, streamflow_deviation

REPO_ROOT = Path(__file__).resolve().parents[2]
REGULATOR = "water_regulator"
HORIZON = 12
SECONDS_PER_DAY = 86400.0

# A candidate under which the quota binds: protected level 0.67, minimum demand
# 0.035, maximum demand 0.48, a fine of 0.06, a risk scale of 0.7 and a risk
# power of 1 + 4 * 0.4 = 2.6.
TIGHT = np.asarray([0.2, 0.1, 0.2, 0.6, 0.7, 0.4, 0.5, 0.5], dtype=np.float32)
# Every request is allowed whatever the level.
LOOSE = np.asarray([0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.5, 0.5], dtype=np.float32)


def farm_ids(count: int) -> list[str]:
    return [f"utilizer:{index}" for index in range(count)]


class ScriptedWorld:
    """Stand-in for the ``World`` actor: hands out the scripted candidates once."""

    def __init__(self, answers: list[MechanismContext | None]) -> None:
        self.answers = list(answers)
        self.get_mechanism_by_id = SimpleNamespace(remote=self._fetch)

    def _fetch(self, **kwargs: Any) -> MechanismContext | None:
        return self.answers.pop(0) if self.answers else None


def candidate(rules: np.ndarray) -> MechanismContext:
    """Candidate the regulator publishes: the normalized rules."""
    return MechanismContext(
        index=0,
        env_id=None,
        seed=0,
        status=MechanismStatus.train,
        mechanism={WATER_POLICY_ID: np.asarray(rules, dtype=np.float32)},
        metrics=None,
    )


@pytest.fixture
def build_env(tmp_path, monkeypatch):
    """Factory of a fresh-water environment, with or without a regulator."""
    monkeypatch.setattr(marl_regulated.ray, "get", lambda ref, *a, **k: ref)

    def build(
        *,
        rules: np.ndarray | None = None,
        farms: int = 3,
        horizon: int = HORIZON,
        seed: int = 0,
        **env_kwargs: Any,
    ) -> WaterRegulatedEnv:
        followers = {
            aid: UtilizerConfig(
                id=aid,
                policy_id="utilizer_policy",
                mechanisms=(
                    IrrigateConfig(id=IRRIGATE_ID, action_space=IRRIGATE_SPACE),
                ),
                observation_space=spaces.Box(
                    -np.inf, np.inf, (OBSERVATION_SIZE,), np.float32
                ),
            )
            for aid in farm_ids(farms)
        }
        leaders = {}
        if rules is not None:
            leaders[REGULATOR] = AgentConfig(
                id=REGULATOR,
                policy_id="water_policy_net",
                mechanisms=(
                    WaterPolicyConfig(
                        id=WATER_POLICY_ID,
                        action_space=RULES_SPACE,
                        acts_on=("utilizer", IRRIGATE_ID),
                    ),
                ),
            )
        return WaterRegulatedEnv(
            world=ScriptedWorld([candidate(rules)] if rules is not None else []),
            opt_id="inner",
            env_name="fresh_water_test",
            horizon=horizon,
            agents_cfg_dict=followers,
            leaders_cfg_dict=leaders,
            mechanism_id=0,
            seed=seed,
            policy_seed=seed,
            mode="train",
            reporter_cfg=CSVConfig(
                project="fresh_water_test", output_dir=str(tmp_path)
            ),
            queries=(),
            schema=WaterMetricSchema,
            **env_kwargs,
        )

    return build


# --- Crop model ------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    ("days", "stage"),
    [
        (-1, "offseason"),
        (0, "initial"),
        (30, "initial"),
        (31, "development"),
        (70, "development"),
        (71, "mid"),
        (110, "mid"),
        (111, "late"),
        (150, "late"),
        (151, "offseason"),
    ],
)
def test_crop_stage_changes_at_the_documented_day(days, stage):
    assert crop_stage(days) == stage


@pytest.mark.unit
def test_monthly_temperature_of_july_is_23_degrees():
    assert estimate_temp_c(datetime(1980, 7, 15)) == 23.0


@pytest.mark.unit
def test_crop_demand_matches_the_hand_computed_water_balance():
    demand = crop_demand(datetime(1980, 7, 15), 40, 1.0, 1_000_000.0)
    assert demand.stage == "development"
    assert demand.kc == 0.80
    assert demand.eto_mm_day == pytest.approx(0.34 * (0.46 * 23 + 8))
    assert demand.etcrop_mm_day == pytest.approx(5.05376)
    assert demand.deficit_mm_day == pytest.approx(4.05376)
    assert demand.full_required_m3_day == pytest.approx(4053.76)
    assert demand.crop_water_need_m3_day == pytest.approx(5053.76)
    assert demand.precip_water_m3_day == pytest.approx(1000.0)


@pytest.mark.unit
def test_rain_above_the_crop_use_leaves_no_deficit():
    demand = crop_demand(datetime(1980, 7, 15), 40, 20.0, 1_000_000.0)
    assert demand.deficit_mm_day == 0.0
    assert demand.full_required_m3_day == 0.0
    assert demand.precip_water_m3_day == pytest.approx(20_000.0)


@pytest.mark.unit
def test_a_crop_that_is_not_growing_needs_no_water():
    before = crop_demand(datetime(1980, 5, 1), -5, 0.0, 1_000_000.0)
    after = crop_demand(datetime(1980, 10, 20), 200, 0.0, 1_000_000.0)
    for demand in (before, after):
        assert demand.stage == "offseason"
        assert demand.crop_water_need_m3_day == 0.0
        assert demand.full_required_m3_day == 0.0


# --- Water policy ----------------------------------------------------------------


@pytest.mark.unit
def test_decode_rules_maps_the_unit_cube_to_the_documented_ranges():
    assert decode_rules(np.zeros(8)).tolist() == pytest.approx(
        [0.6, 0.0, 0.35, 0.0, 0.0, 1.0, 0.0, 100_000.0]
    )
    assert decode_rules(np.ones(8)).tolist() == pytest.approx(
        [0.95, 0.35, 1.0, 0.1, 1.0, 5.0, 1.0, 20_000_000.0]
    )
    # Out-of-range outputs are clipped instead of leaving the ranges.
    np.testing.assert_allclose(decode_rules(np.full(8, 7.0)), decode_rules(np.ones(8)))


@pytest.mark.unit
def test_the_maximum_demand_is_never_below_the_minimum_demand():
    rules = decode_rules([0.5, 1.0, 0.0, 0.5, 0.5, 0.5, 0.5, 0.5])
    assert rules[1] == pytest.approx(0.35)
    assert rules[2] == pytest.approx(0.35)


@pytest.mark.unit
def test_encode_rules_inverts_decode_rules_and_the_default_is_inside_the_cube():
    u = np.asarray([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    np.testing.assert_allclose(encode_rules(decode_rules(u)), u, atol=1e-6)
    encoded_default = encode_rules(DEFAULT_RULES)
    assert np.all((0.0 <= encoded_default) & (encoded_default <= 1.0))
    assert len(RULE_NAMES) == 8


@pytest.mark.unit
def test_decode_rules_refuses_a_vector_of_the_wrong_size():
    with pytest.raises(ValueError, match="8"):
        decode_rules([0.5, 0.5])


@pytest.mark.unit
def test_quota_stress_and_allowed_fraction_at_a_known_level():
    assert quota_stress(0.9, 0.85) == pytest.approx(1 / 3)
    assert quota_stress(0.85, 0.85) == 0.0
    assert quota_stress(0.4, 0.85) == 0.0
    assert quota_stress(1.0, 0.85) == 1.0
    assert allowed_fraction(0.9, DEFAULT_RULES) == pytest.approx(0.05 + 0.95 / 3)
    assert allowed_fraction(0.3, DEFAULT_RULES) == pytest.approx(0.05)
    assert allowed_fraction(1.0, DEFAULT_RULES) == pytest.approx(1.0)


@pytest.mark.unit
def test_irrigation_fraction_clips_the_policy_output():
    assert irrigation_fraction(np.asarray([-0.3], dtype=np.float32)) == 0.0
    assert irrigation_fraction(np.asarray([1.7], dtype=np.float32)) == 1.0
    assert irrigation_fraction(0.25) == 0.25


@pytest.mark.unit
def test_violation_signal_matches_the_hand_computed_penalties():
    v = violation_signal(
        0.8,
        full_required_m3_day=1000.0,
        level_norm=0.9,
        release_pressure=0.5,
        rules=DEFAULT_RULES,
    )
    assert v.requested_m3_day == pytest.approx(800.0)
    assert v.allowed_m3_day == pytest.approx(1100 / 3)
    assert v.delivered_m3_day == pytest.approx(1100 / 3)
    assert v.quota_violation_m3_day == pytest.approx(800 - 1100 / 3)
    assert v.quota_penalty == pytest.approx(0.05 * (800 - 1100 / 3) / 1000)
    assert v.flow_penalty == pytest.approx(0.5 * 0.5 * 0.8**2)
    assert v.total_penalty == pytest.approx(0.181667, abs=1e-6)


@pytest.mark.unit
def test_a_request_inside_the_quota_pays_only_the_flow_penalty():
    v = violation_signal(
        0.2,
        full_required_m3_day=1000.0,
        level_norm=1.0,
        release_pressure=1.0,
        rules=DEFAULT_RULES,
    )
    assert v.quota_violation_m3_day == 0.0
    assert v.quota_penalty == 0.0
    assert v.delivered_m3_day == pytest.approx(200.0)
    assert v.total_penalty == pytest.approx(0.5 * 1.0 * 0.2**2)


@pytest.mark.unit
def test_the_penalty_is_capped_at_one_and_a_dry_day_has_none():
    rules = decode_rules(np.ones(8))
    capped = violation_signal(
        1.0,
        full_required_m3_day=1000.0,
        level_norm=0.0,
        release_pressure=1.0,
        rules=rules,
    )
    assert capped.total_penalty == 1.0
    dry = violation_signal(
        1.0,
        full_required_m3_day=0.0,
        level_norm=0.5,
        release_pressure=1.0,
        rules=DEFAULT_RULES,
    )
    assert dry.requested_m3_day == 0.0
    assert dry.total_penalty == 0.0


# --- Lake models -----------------------------------------------------------------


def quiet_lake(**overrides: Any) -> SurrogateLake:
    parameters = dict(
        full_stage_m=420.0,
        max_depth_m=10.0,
        lake_area_m2=1e6,
        inflow_log_sigma=0.0,
        seasonal_amplitude=0.0,
        rain_probability=0.0,
        initial_level_range=(0.5, 0.5),
    )
    parameters.update(overrides)
    return SurrogateLake(**parameters)


@pytest.mark.unit
def test_surrogate_lake_mass_balance_and_baseline_match_the_hand_values():
    lake = quiet_lake()
    rng = np.random.default_rng(0)
    first = lake.start(datetime(1980, 5, 10), rng)
    assert first.stage_m == 415.0
    assert lake.level_norm(first.stage_m) == pytest.approx(0.5)

    day = lake.advance(datetime(1980, 5, 11), withdrawal_m3s=2.0, rng=rng)
    assert day.stage_m == pytest.approx(415.1728)
    assert day.baseline_stage_m == pytest.approx(415.3456)
    assert day.inflow_m3s == pytest.approx(10.0)
    assert day.outflow_m3s == pytest.approx(6.0)


@pytest.mark.unit
def test_surrogate_lake_spills_when_full_and_never_goes_below_empty():
    full = quiet_lake(initial_level_range=(1.0, 1.0), release_fraction=0.0)
    rng = np.random.default_rng(0)
    full.start(datetime(1980, 5, 10), rng)
    day = full.advance(datetime(1980, 5, 11), withdrawal_m3s=0.0, rng=rng)
    assert day.stage_m == pytest.approx(420.0)

    empty = quiet_lake(initial_level_range=(0.0, 0.0), release_fraction=0.0)
    empty.start(datetime(1980, 5, 10), rng)
    day = empty.advance(datetime(1980, 5, 11), withdrawal_m3s=500.0, rng=rng)
    assert day.stage_m == pytest.approx(410.0)


@pytest.mark.unit
def test_surrogate_lake_with_the_same_seed_repeats_its_weather():
    def run(seed: int) -> list[float]:
        lake = SurrogateLake(full_stage_m=420.0, max_depth_m=10.0, lake_area_m2=1e6)
        rng = np.random.default_rng(seed)
        lake.start(datetime(1980, 6, 1), rng)
        return [
            lake.advance(
                datetime(1980, 6, 2 + day), withdrawal_m3s=1.0, rng=rng
            ).inflow_m3s
            for day in range(5)
        ]

    assert run(3) == run(3)
    assert run(3) != run(4)


@pytest.mark.unit
def test_raven_lake_reads_the_stand_in_outputs(tmp_path):
    model, command = write_stand_in_raven(tmp_path)
    lake = RavenLake(
        full_stage_m=420.0,
        max_depth_m=10.0,
        raven_cwd=model,
        raven_cmd=command,
        key="unit",
        work_dir=tmp_path / "runs",
    )
    rng = np.random.default_rng(0)
    start = lake.start(datetime(1980, 5, 1), rng)
    assert (start.stage_m, start.baseline_stage_m) == (420.0, 420.0)

    first = lake.advance(datetime(1980, 5, 2), withdrawal_m3s=2.0, rng=rng)
    second = lake.advance(datetime(1980, 5, 3), withdrawal_m3s=0.5, rng=rng)

    # The stand-in lowers the stage by the sum of the withdrawals so far.
    assert first.stage_m == 418.0
    assert second.stage_m == 417.5
    assert second.baseline_stage_m == 420.0
    assert (second.inflow_m3s, second.outflow_m3s, second.precip_mm_day) == (
        5.0,
        3.0,
        2.0,
    )
    assert second.gauges["gauge_west_montrose_m3s"] == 9.0


@pytest.mark.unit
def test_raven_lake_names_the_missing_executable_output(tmp_path):
    model, _ = write_stand_in_raven(tmp_path)
    script = tmp_path / "does_nothing.sh"
    script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(0o755)
    lake = RavenLake(
        full_stage_m=420.0,
        max_depth_m=10.0,
        raven_cwd=model,
        raven_cmd=script,
        key="silent",
        work_dir=tmp_path / "runs",
    )
    with pytest.raises(RavenOutputError):
        lake.start(datetime(1980, 5, 1), np.random.default_rng(0))


# --- Environment construction ----------------------------------------------------


@pytest.mark.unit
def test_environment_refuses_an_unknown_lake_model(build_env):
    with pytest.raises(ValueError, match="hydrology"):
        build_env(hydrology="bathtub")


@pytest.mark.unit
def test_raven_lake_needs_the_model_directory_and_the_executable(build_env):
    with pytest.raises(ValueError, match="raven_cwd"):
        build_env(hydrology="raven")


@pytest.mark.unit
def test_ecology_options_reach_the_lake_and_the_farm_area(build_env):
    env = build_env(
        ecology_cfg={"max_farm_area_m2": 250_000.0, "rain_probability": 0.0}
    )
    assert env.max_farm_area_m2 == 250_000.0
    assert env.lake.rain_probability == 0.0


# --- Dynamics of the environment, behind the adapter -----------------------------


def level_state(adapter: RLlibMultiAgentEnvAdapter) -> dict[str, float]:
    """State of the step about to be played, read from the shared MDP."""
    mdp = adapter._mdp
    t = mdp.t
    return {
        name: float(mdp.state[name][t])
        for name in (
            "reservoir_level_norm",
            "release_pressure",
            "full_required_m3_day",
            "crop_water_need_m3_day",
            "precip_water_m3_day",
        )
    }


def expected_reward(
    request: float, day: dict[str, float], u: np.ndarray | None
) -> tuple[float, float]:
    """Satisfaction minus penalty and delivered volume, from the formulas.

    ``u`` is the normalized rules, ``None`` for the default rules without a
    regulator (so with no penalty).
    """
    if u is None:
        quota, low, high = 0.85, 0.05, 1.0
        fine, scale, power = 0.05, 0.5, 2.0
    else:
        u = np.asarray(u, dtype=np.float64)
        quota = 0.6 + 0.35 * u[0]
        low = 0.35 * u[1]
        high = max(0.35 + 0.65 * u[2], low)
        fine, scale, power = 0.1 * u[3], u[4], 1.0 + 4.0 * u[5]
    level = day["reservoir_level_norm"]
    stress = min(1.0, max(0.0, (level - quota) / (1.0 - quota)))
    full = day["full_required_m3_day"]
    allowed = (low + stress * (high - low)) * full
    requested = request * full
    delivered = min(requested, allowed)
    need = day["crop_water_need_m3_day"]
    satisfaction = (
        1.0
        if need <= 1e-8
        else min(1.0, (delivered + day["precip_water_m3_day"]) / need)
    )
    penalty = 0.0
    if u is not None:
        quota_penalty = min(1.0, fine * max(0.0, requested - allowed) / max(1e-8, full))
        flow_penalty = (
            scale * day["release_pressure"] * (requested / max(1e-8, full)) ** power
        )
        penalty = min(1.0, quota_penalty + flow_penalty)
    return satisfaction - penalty, delivered


def requests_for(step: int, farms: int) -> dict[str, dict[str, np.ndarray]]:
    """Scripted policy outputs: a different request per farm and per day."""
    values = [0.9, 0.5, 0.15]
    return {
        aid: {
            IRRIGATE_ID: np.asarray(
                [min(1.0, values[index % 3] + 0.01 * step)], dtype=np.float32
            )
        }
        for index, aid in enumerate(farm_ids(farms))
    }


@pytest.mark.unit
@pytest.mark.parametrize("rules", [TIGHT, LOOSE, None], ids=["tight", "loose", "none"])
def test_rewards_follow_crop_satisfaction_minus_the_policy_penalty(build_env, rules):
    env = build_env(rules=rules)
    adapter = RLlibMultiAgentEnvAdapter(env)
    adapter.reset()

    for step in range(HORIZON):
        day = level_state(adapter)
        actions = requests_for(step, 3)
        _, rewards, _, truncateds, _ = adapter.step(actions)
        for aid in farm_ids(3):
            request = float(actions[aid][IRRIGATE_ID][0])
            expected, _ = expected_reward(request, day, rules)
            assert rewards[aid] == pytest.approx(expected, abs=1e-6), (step, aid)
    assert truncateds["__all__"] is True


@pytest.mark.unit
def test_the_leader_action_is_the_decoded_candidate_and_stays_in_force(build_env):
    env = build_env(rules=TIGHT)
    adapter = RLlibMultiAgentEnvAdapter(env)
    adapter.reset()
    adapter.step(requests_for(0, 3))
    adapter.step(requests_for(1, 3))

    stored = adapter._mdp.actions[REGULATOR][WATER_POLICY_ID]
    assert len(stored) >= 2
    for rules in stored:
        np.testing.assert_allclose(rules, decode_rules(TIGHT), atol=1e-6)
    assert env.published_mechanism_assigned


@pytest.mark.unit
def test_observation_carries_the_level_the_allowed_fraction_and_the_rules(build_env):
    env = build_env(rules=TIGHT)
    adapter = RLlibMultiAgentEnvAdapter(env)
    first, _ = adapter.reset()
    assert first[farm_ids(3)[0]].shape == (OBSERVATION_SIZE,)

    obs, _, *_ = adapter.step(requests_for(0, 3))
    day = level_state(adapter)
    rules = decode_rules(TIGHT)
    quota = rules[0]
    stress = min(1.0, max(0.0, (day["reservoir_level_norm"] - quota) / (1 - quota)))
    allowed = rules[1] + stress * (rules[2] - rules[1])
    delivered_yesterday = float(adapter._mdp.state["usage_m3_day"][adapter._mdp.t])

    for aid in farm_ids(3):
        vector = obs[aid]
        assert vector[0] == pytest.approx(day["reservoir_level_norm"], abs=1e-5)
        assert vector[1] == pytest.approx(day["release_pressure"], abs=1e-5)
        assert vector[2] == pytest.approx(allowed, abs=1e-5)
        assert vector[3] == pytest.approx(delivered_yesterday, rel=1e-6)
        np.testing.assert_allclose(vector[4:], TIGHT, atol=1e-6)


@pytest.mark.unit
def test_without_a_regulator_the_observation_has_no_rules(build_env):
    adapter = RLlibMultiAgentEnvAdapter(build_env())
    obs, _ = adapter.reset()
    vector = obs[farm_ids(3)[0]]
    assert vector[2] == 0.0
    assert not vector[4:].any()


@pytest.mark.unit
def test_delivered_water_is_withdrawn_from_the_lake_and_leaves_the_baseline(build_env):
    env = build_env(
        rules=LOOSE,
        ecology_cfg={"rain_probability": 0.0, "initial_level_range": (0.5, 0.5)},
    )
    adapter = RLlibMultiAgentEnvAdapter(env)
    adapter.reset()
    day = level_state(adapter)

    actions = requests_for(0, 3)
    adapter.step(actions)
    usage = float(adapter._mdp.state["usage_m3_day"][adapter._mdp.t])

    expected_usage = sum(
        expected_reward(float(actions[aid][IRRIGATE_ID][0]), day, LOOSE)[1]
        for aid in farm_ids(3)
    )
    assert usage == pytest.approx(expected_usage, rel=1e-6)
    # Away from the spill and from the empty level, a volume V lowers the stage
    # by V / lake area against the world with no withdrawal.
    reading = env._reading
    lowered = reading.baseline_stage_m - reading.stage_m
    assert lowered == pytest.approx(usage / env.lake.lake_area_m2, rel=1e-6)


@pytest.mark.unit
def test_a_tight_policy_withdraws_less_and_keeps_the_lake_higher(build_env):
    def run(rules: np.ndarray) -> tuple[float, SimpleNamespace]:
        env = build_env(
            rules=rules,
            ecology_cfg={"rain_probability": 0.0, "initial_level_range": (0.6, 0.6)},
        )
        adapter = RLlibMultiAgentEnvAdapter(env)
        adapter.reset()
        usage = 0.0
        for step in range(HORIZON):
            adapter.step(requests_for(step, 3))
            usage += float(adapter._mdp.state["usage_m3_day"][adapter._mdp.t])
        return usage, env._reading

    loose_usage, loose = run(LOOSE)
    tight_usage, tight = run(TIGHT)

    assert tight_usage < loose_usage
    assert tight.stage_m > loose.stage_m
    # The no-withdrawal world sees the same weather whatever the policy does.
    assert tight.baseline_stage_m == pytest.approx(loose.baseline_stage_m)
    assert tight.baseline_inflow_m3s == pytest.approx(loose.baseline_inflow_m3s)
    assert loose.stage_m < loose.baseline_stage_m


@pytest.mark.unit
def test_the_logger_reduces_the_crop_and_lake_series_of_an_episode(build_env):
    env = build_env(rules=TIGHT, farms=2)
    adapter = RLlibMultiAgentEnvAdapter(env)
    adapter.reset()
    levels = []
    rewards = []
    for step in range(HORIZON):
        _, step_rewards, *_ = adapter.step(requests_for(step, 2))
        mdp = adapter._mdp
        levels.append(float(mdp.state["reservoir_level_norm"][mdp.t]))
        rewards.append(float(np.mean(list(step_rewards.values()))))

    reduced = env.logger.reduce()
    assert reduced.reward_total == pytest.approx(sum(rewards), abs=1e-5)
    assert reduced.reward_mean == pytest.approx(np.mean(rewards), abs=1e-5)
    assert reduced.reservoir_level_norm_min == pytest.approx(min(levels))
    assert reduced.reservoir_level_norm == pytest.approx(np.mean(levels))
    assert reduced.reservoir_level_norm_series == pytest.approx(levels)
    assert len(reduced.streamflow_m3s_series) == HORIZON
    # The inflow of the surrogate does not depend on the withdrawals.
    assert reduced.streamflow_m3s_series == pytest.approx(
        reduced.baseline_streamflow_m3s_series
    )
    assert reduced.min_demand_frac == pytest.approx(decode_rules(TIGHT)[1])


@pytest.mark.unit
def test_episodes_after_a_reset_start_on_a_new_planting_day(build_env):
    env = build_env(horizon=3)
    adapter = RLlibMultiAgentEnvAdapter(env)
    starts = []
    for _ in range(4):
        adapter.reset()
        starts.append(env._planting_date)
        for step in range(3):
            adapter.step(requests_for(step, 3))
    day_of_year = [(d - datetime(1980, 1, 1)).days for d in starts]
    assert all(121 <= day < 274 for day in day_of_year)
    assert len(set(day_of_year)) > 1


@pytest.mark.unit
def test_the_raven_lake_runs_inside_the_environment(build_env, tmp_path):
    model, command = write_stand_in_raven(tmp_path)
    env = build_env(
        rules=LOOSE,
        farms=1,
        horizon=2,
        hydrology="raven",
        raven_cwd=model,
        raven_cmd=command,
        raven_work_dir=tmp_path / "runs",
    )
    adapter = RLlibMultiAgentEnvAdapter(env)
    adapter.reset()
    day = level_state(adapter)
    actions = requests_for(0, 1)
    adapter.step(actions)

    _, delivered = expected_reward(
        float(actions[farm_ids(1)[0]][IRRIGATE_ID][0]), day, LOOSE
    )
    # The stand-in stage is 420 minus the withdrawal in m3/s.
    assert env._reading.stage_m == pytest.approx(420.0 - delivered / SECONDS_PER_DAY)
    reduced_gauge = env.logger.reduce().gauge_west_montrose_m3s
    assert reduced_gauge == 9.0


# --- Regulator environment -------------------------------------------------------


@pytest.mark.unit
def test_streamflow_deviation_is_the_relative_absolute_change():
    assert streamflow_deviation([9.0, 12.0], [10.0, 10.0]) == pytest.approx(0.15)
    assert streamflow_deviation([5.0, 5.0], [5.0, 5.0]) == 0.0
    assert streamflow_deviation([1.0], [0.0]) == pytest.approx(1.0 / 1e-8)
    with pytest.raises(ValueError, match="shape"):
        streamflow_deviation([1.0, 2.0], [1.0])


@pytest.mark.unit
def test_fitness_context_combines_the_scores_with_the_weights():
    context = FitnessContext.from_scores(
        economic_score=0.6,
        streamflow_deviation=0.25,
        economic_weight=2.0,
        sustainability_weight=3.0,
    )
    assert context.sustainability_score == pytest.approx(0.8)
    assert context.objective_score == pytest.approx(2.0 * 0.6 + 3.0 * 0.8)


def regulator(published: list, monkeypatch, **ecology: Any) -> WaterRegulatorEnv:
    import ray

    world = SimpleNamespace(
        append_context=SimpleNamespace(remote=lambda ctx: published.append(ctx))
    )
    monkeypatch.setattr(ray, "get", lambda ref, *a, **k: ref)
    return WaterRegulatorEnv(
        world=world,
        optimizer=None,
        horizon=1,
        agents_cfgs={},
        seeds=[0, 1],
        ecology_cfg=ecology,
    )


def episode(reward: float, flow: list[float], baseline: list[float]) -> Any:
    return SimpleNamespace(
        reward_mean=[reward],
        streamflow_m3s_series=[flow],
        baseline_streamflow_m3s_series=[baseline],
        outflow_m3s_series=[flow],
        baseline_outflow_m3s_series=[baseline],
    )


@pytest.mark.unit
def test_regulator_reward_averages_episodes_and_seeds(monkeypatch):
    published: list = []
    env = regulator(published, monkeypatch, sustainability_weight=2.0)
    by_seed = {
        # Deviations 0.2 and 0.0 and rewards 0.5 and 0.7.
        "0": SimpleNamespace(
            by_episode={
                "0": episode(0.5, [12.0, 12.0], [10.0, 10.0]),
                "1": episode(0.7, [10.0], [10.0]),
            }
        ),
        # Deviation 0.4 and reward 0.9.
        "1": SimpleNamespace(by_episode={"0": episode(0.9, [14.0], [10.0])}),
    }
    rollout = SimpleNamespace(by_mechanism={"2": SimpleNamespace(by_seed=by_seed)})
    metrics = SimpleNamespace(train=SimpleNamespace(rollout=rollout))

    fitness = env.reward(metrics)

    mean_reward = (0.5 + 0.7 + 0.9) / 3
    mean_deviation = (0.2 + 0.0 + 0.4) / 3
    expected = mean_reward + 2.0 / (1.0 + mean_deviation)
    # Mechanisms 0 and 1 have no rollout and are scored minus infinity.
    assert fitness[:2] == [-np.inf, -np.inf]
    assert fitness[2] == pytest.approx(expected, rel=1e-6)
    (context,) = published
    assert context.payload.status is MechanismStatus.done
    assert context.payload.index == 2
    assert context.payload.metrics.streamflow_deviation == pytest.approx(mean_deviation)


@pytest.mark.unit
def test_regulator_reads_the_series_named_by_the_option(monkeypatch):
    env = regulator([], monkeypatch, deviation_series="outflow")
    branch = episode(0.5, [1.0], [1.0])
    branch.outflow_m3s_series = [[15.0]]
    branch.baseline_outflow_m3s_series = [[10.0]]
    by_seed = {"0": SimpleNamespace(by_episode={"0": branch})}
    rollout = SimpleNamespace(by_mechanism={"0": SimpleNamespace(by_seed=by_seed)})
    metrics = SimpleNamespace(train=SimpleNamespace(rollout=rollout))
    assert env.reward(metrics) == [pytest.approx(0.5 + 1.0 / 1.5)]


@pytest.mark.unit
def test_regulator_refuses_bad_options_and_empty_metrics(monkeypatch):
    with pytest.raises(ValueError, match="aggregation_status"):
        regulator([], monkeypatch, aggregation_status="done")
    with pytest.raises(ValueError, match="deviation_series"):
        regulator([], monkeypatch, deviation_series="rain")
    env = regulator([], monkeypatch)
    empty = SimpleNamespace(
        train=SimpleNamespace(rollout=SimpleNamespace(by_mechanism={}))
    )
    with pytest.raises(ValueError, match="No rollout"):
        env.reward(empty)


# --- Script options --------------------------------------------------------------


@pytest.mark.unit
def test_importing_the_script_runs_nothing_and_defaults_are_the_full_run():
    from examples.fresh_water import debug

    args = debug.parse_args([])
    assert (args.hydrology, args.outer_iters, args.train_iters) == (
        "surrogate",
        100,
        200,
    )
    assert (args.horizon, args.num_agents, args.population) == (150, 500, 1)


@pytest.mark.unit
def test_raven_options_are_required_together_and_reach_the_environment(tmp_path):
    from examples.fresh_water import debug

    with pytest.raises(SystemExit):
        debug.parse_args(["--hydrology", "raven"])
    args = debug.parse_args(
        ["--hydrology", "raven", "--raven-cwd", "model", "--raven-cmd", "run"]
    )
    config = debug.inner_env_config(args)
    assert (config["hydrology"], config["raven_cwd"], config["raven_cmd"]) == (
        "raven",
        "model",
        "run",
    )
    assert "raven_work_dir" not in config
    assert "/Users/" not in repr(config)


@pytest.mark.unit
def test_the_script_holds_no_machine_specific_path():
    for path in (REPO_ROOT / "examples" / "fresh_water").glob("*.py"):
        assert "/Users/" not in path.read_text(), path.name


# --- End to end, as child processes ----------------------------------------------

CHILD_TIMEOUT_S = 240
SMOKE_ARGS = (
    "--outer-iters",
    "2",
    "--train-iters",
    "2",
    "--horizon",
    "10",
    "--num-agents",
    "4",
    "--reporter",
    "csv",
)


def run_child(args: tuple[str, ...], workdir: Path) -> dict:
    """Run ``python -m examples.fresh_water.debug args`` with a time limit."""
    env = dict(os.environ, WANDB_MODE="offline", PYTHONPATH=str(REPO_ROOT))
    # A fixed hash seed would make the child's world identifiers repeat across runs.
    env.pop("PYTHONHASHSEED", None)
    try:
        done = subprocess.run(
            [sys.executable, "-m", "examples.fresh_water.debug", *args],
            cwd=workdir,
            env=env,
            capture_output=True,
            text=True,
            timeout=CHILD_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(
            f"debug.py exceeded {CHILD_TIMEOUT_S} s; stdout: {exc.stdout!r}"[-800:]
        )
    return {
        "returncode": done.returncode,
        "output": done.stdout + "\n" + done.stderr,
        "workdir": workdir,
    }


@pytest.fixture(scope="module")
def surrogate_run(tmp_path_factory):
    return run_child(SMOKE_ARGS, tmp_path_factory.mktemp("fresh_water_surrogate"))


@pytest.fixture(scope="module")
def raven_run(tmp_path_factory):
    workdir = tmp_path_factory.mktemp("fresh_water_raven")
    model, command = write_stand_in_raven(workdir / "stand_in")
    return run_child(
        (
            "--hydrology",
            "raven",
            "--raven-cwd",
            str(model),
            "--raven-cmd",
            str(command),
            "--raven-work-dir",
            str(workdir / "runs"),
            *SMOKE_ARGS[:-4],
            "--num-agents",
            "3",
            "--reporter",
            "csv",
        ),
        workdir,
    )


@pytest.mark.integration
def test_debug_script_runs_to_completion_with_the_surrogate_lake(surrogate_run):
    assert surrogate_run["returncode"] == 0, surrogate_run["output"][-2000:]
    assert "Run finished | iters=2" in surrogate_run["output"]


@pytest.mark.integration
def test_surrogate_run_writes_csv_files_and_every_query_renders(surrogate_run):
    csv_files = sorted((surrogate_run["workdir"] / "results").rglob("*.csv"))
    assert len(csv_files) == 7, [path.name for path in csv_files]
    assert "could not render query" not in surrogate_run["output"]


@pytest.mark.integration
def test_surrogate_run_reports_a_finite_fitness_per_generation(surrogate_run):
    fitness_files = list(
        (surrogate_run["workdir"] / "results").rglob("Fitness_over*.csv")
    )
    assert fitness_files
    values = [
        float(line.split(",")[3])
        for line in fitness_files[0].read_text().splitlines()[1:]
        if "Candidates" in line
    ]
    # The reward of a farm is at most 1 and the sustainability score is at most 1.
    assert len(values) == 2
    assert all(np.isfinite(value) and value <= 2.0 for value in values)


@pytest.mark.integration
def test_debug_script_runs_to_completion_with_the_raven_stand_in(raven_run):
    assert raven_run["returncode"] == 0, raven_run["output"][-2000:]
    assert "Run finished | iters=2" in raven_run["output"]
