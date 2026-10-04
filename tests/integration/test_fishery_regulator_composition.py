"""A regulator that holds several mechanisms, stepped through the real fishery.

The environment is the shrunk fishery of ``examples/bilevel_fishery/debug.py``
(two fishermen, a horizon of 20 steps, the same ecology) behind the RLlib
adapter, stepped with seeded actions. Only the ``World`` actor is replaced, by
a script of candidates, so the tests run without Ray. They check what a unit
test of one mechanism cannot: that the residuals of the regulator's mechanisms
compose in the reward and in the observation the policies receive.

The expected values are written from the formulas of the mechanisms and from
the stock read before each step, not from the mechanisms:

- delivered harvest fraction: the quota cuts the request ``f = sigmoid(z / 4)``
  to ``f - softplus0(f - allowed)`` (see ``tests/mechanism/algorithms/test_quota.py``);
- subsidy residual: ``rate * MAX_SUBSIDY * e - cost * e**2`` with the restoration
  effort ``e = sigmoid(z_restore / 4)``;
- penalty residual: ``-amount / (1 + exp((b - threshold) / width))`` at the
  normalised stock ``b`` of the step.
"""

import math

import numpy as np
import pytest
from gymnasium import spaces

from core.mechanism.algorithms.penalty import ThresholdPenalty
from core.mechanism.algorithms.quota import Quota
from core.mechanism.algorithms.social_influence import SocialInfluence
from core.mechanism.algorithms.subsidy import MAX_SUBSIDY, Subsidy

CAPACITY = 5_000.0
HORIZON = 20
TEMPERATURE = 4.0
QUOTA_WIDTH = 0.03
USAGE_WIDTH = 0.005

# Under these seeded actions the stock of the shrunk fishery stays between 0.72
# and 0.81 of the capacity, so a quota inside that range cuts the larger
# requests and a penalty threshold inside it is crossed during the episode.
QUOTA = float(np.float32(0.78))
SUBSIDY_RATE = float(np.float32(0.6))
SUBSIDY_COST = 0.5
PENALTY_THRESHOLD = 0.77
PENALTY_AMOUNT = 0.1
PENALTY_WIDTH = 0.03

HARVEST_ENTRY = 3
RESTORE_ENTRY = 4

UNIT_BOX = spaces.Box(0.0, 1.0, (1,), np.float32)
Z_SCALE = 2.0
"""Spread of the seeded policy outputs; large enough for requests on both sides
of the quota."""

# Reward and observation are computed in double precision from float32 policy
# outputs and float32 residuals.
FLOAT32_ATOL = 1e-5


def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def softplus0(x: float, width: float = USAGE_WIDTH) -> float:
    return max(0.0, width * (math.log1p(math.exp(x / width)) - math.log(2.0)))


def allowed_fraction(level: float, quota: float) -> float:
    lower = logistic(-quota / QUOTA_WIDTH)
    upper = logistic((1.0 - quota) / QUOTA_WIDTH)
    return (logistic((level - quota) / QUOTA_WIDTH) - lower) / (upper - lower)


def requested_fraction(z: float) -> float:
    return logistic(z / TEMPERATURE)


def delivered_fraction(z: float, level: float, quota: float) -> float:
    requested = requested_fraction(z)
    return requested - softplus0(requested - allowed_fraction(level, quota))


def subsidy_residual(z_restore: float) -> float:
    effort = requested_fraction(z_restore)
    return SUBSIDY_RATE * MAX_SUBSIDY * effort - SUBSIDY_COST * effort**2


def penalty_residual(level: float) -> float:
    return -PENALTY_AMOUNT / (
        1.0 + math.exp((level - PENALTY_THRESHOLD) / PENALTY_WIDTH)
    )


def quota_config():
    return Quota(
        id="quota",
        action_space=UNIT_BOX,
        acts_on=("fisherman", "harvest"),
        obs_map={"resource_level": "fish"},
    )


def subsidy_config():
    return Subsidy(
        id="subsidy",
        action_space=UNIT_BOX,
        acts_on=("fisherman", "restore"),
        cost=SUBSIDY_COST,
    )


def penalty_config():
    return ThresholdPenalty(
        id="penalty",
        acts_on=("fisherman", "harvest"),
        obs_map={"resource_level": "fish"},
        threshold=PENALTY_THRESHOLD,
        penalty_amount=PENALTY_AMOUNT,
        transition_width=PENALTY_WIDTH,
    )


def social_configs():
    return (
        SocialInfluence(
            id="social_harvest",
            acts_on=("fisherman", "harvest"),
            obs_offset=HARVEST_ENTRY,
        ),
        SocialInfluence(
            id="social_restore",
            acts_on=("fisherman", "restore"),
            obs_offset=RESTORE_ENTRY,
        ),
    )


def candidate(with_social: bool) -> dict[str, np.ndarray]:
    """The regulator's action: one array per mechanism it holds."""
    action = {
        "quota": np.array([QUOTA], dtype=np.float32),
        "subsidy": np.array([SUBSIDY_RATE], dtype=np.float32),
        "penalty": np.empty(0, dtype=np.float32),
    }
    if with_social:
        action["social_harvest"] = np.empty(0, dtype=np.float32)
        action["social_restore"] = np.empty(0, dtype=np.float32)
    return action


def policy_outputs(rng: np.random.Generator, aids: list[str]) -> dict[str, dict]:
    return {
        aid: {
            "harvest": (Z_SCALE * rng.normal(size=1)).astype(np.float32),
            "restore": (Z_SCALE * rng.normal(size=1)).astype(np.float32),
        }
        for aid in aids
    }


def level_of(adapter) -> float:
    """Normalised stock of the step about to be played."""
    mdp = adapter._mdp
    return float(mdp.state["fish"][mdp.t]) / CAPACITY


def z_of(actions: dict, aid: str, mechanism: str) -> float:
    return float(actions[aid][mechanism][0])


@pytest.mark.unit
class TestRewardComposition:
    def play(self, build_fishery, candidate_context):
        fishery = build_fishery(
            quota_config(),
            subsidy_config(),
            penalty_config(),
            candidates=[candidate_context(candidate(with_social=False))],
        )
        adapter = fishery.adapter
        adapter.reset()
        rng = np.random.default_rng(0)
        records = []

        for _ in range(HORIZON):
            level = level_of(adapter)
            t = adapter._mdp.t
            actions = policy_outputs(rng, adapter.possible_agents)
            _, rewards, _, truncateds, _ = adapter.step(actions)
            for aid in adapter.possible_agents:
                records.append(
                    {
                        "aid": aid,
                        "reward": rewards[aid],
                        "level": level,
                        "z_harvest": z_of(actions, aid, "harvest"),
                        "z_restore": z_of(actions, aid, "restore"),
                        "stored_harvest": adapter._mdp.actions[aid]["harvest"][t],
                        "stored_restore": adapter._mdp.actions[aid]["restore"][t],
                    }
                )
        return records, truncateds

    def test_reward_is_harvest_plus_subsidy_plus_penalty(
        self, build_fishery, candidate_context
    ):
        records, _ = self.play(build_fishery, candidate_context)

        assert len(records) == HORIZON * 2
        for r in records:
            effort = r["stored_restore"]
            expected = (
                r["stored_harvest"]
                + (SUBSIDY_RATE * MAX_SUBSIDY * effort - SUBSIDY_COST * effort**2)
                + penalty_residual(r["level"])
            )
            assert r["reward"] == pytest.approx(expected, abs=1e-12)

    def test_reward_matches_the_formulas_of_the_three_mechanisms(
        self, build_fishery, candidate_context
    ):
        records, _ = self.play(build_fishery, candidate_context)

        for r in records:
            expected = (
                delivered_fraction(r["z_harvest"], r["level"], QUOTA)
                + subsidy_residual(r["z_restore"])
                + penalty_residual(r["level"])
            )
            assert r["reward"] == pytest.approx(expected, abs=FLOAT32_ATOL)

    def test_each_residual_is_active_in_the_episode(
        self, build_fishery, candidate_context
    ):
        # Guards against a vacuous composition: the quota must bind on some
        # requests, and the subsidy and the penalty must both move the reward.
        records, _ = self.play(build_fishery, candidate_context)

        cut = [
            requested_fraction(r["z_harvest"]) - r["stored_harvest"] for r in records
        ]
        assert max(cut) > 0.05
        assert min(cut) == pytest.approx(0.0, abs=1e-3)
        assert max(subsidy_residual(r["z_restore"]) for r in records) > 0.02
        penalties = [penalty_residual(r["level"]) for r in records]
        assert min(penalties) < -0.05
        assert max(penalties) - min(penalties) > 0.02

    def test_the_episode_is_truncated_at_the_horizon(
        self, build_fishery, candidate_context
    ):
        _, truncateds = self.play(build_fishery, candidate_context)

        assert truncateds["__all__"] is True


@pytest.mark.unit
class TestSocialObservations:
    def play(self, build_fishery, answers, steps=HORIZON):
        fishery = build_fishery(
            quota_config(),
            subsidy_config(),
            penalty_config(),
            *social_configs(),
            candidates=answers,
        )
        adapter = fishery.adapter
        obs, _ = adapter.reset()
        rng = np.random.default_rng(1)
        history = [{"obs": obs, "level": level_of(adapter), "delivered": None}]

        for _ in range(steps):
            level = level_of(adapter)
            t = adapter._mdp.t
            actions = policy_outputs(rng, adapter.possible_agents)
            obs, _, _, _, _ = adapter.step(actions)
            delivered = {
                aid: {
                    "harvest": adapter._mdp.actions[aid]["harvest"][t],
                    "restore": adapter._mdp.actions[aid]["restore"][t],
                    "z_harvest": z_of(actions, aid, "harvest"),
                    "z_restore": z_of(actions, aid, "restore"),
                }
                for aid in adapter.possible_agents
            }
            history.append(
                {
                    "obs": obs,
                    "level": level_of(adapter),
                    "delivered": delivered,
                    "previous_level": level,
                }
            )
        return fishery, history

    def test_entries_hold_the_peer_s_last_delivered_actions(
        self, build_fishery, candidate_context
    ):
        _, history = self.play(
            build_fishery, [candidate_context(candidate(with_social=True))]
        )

        for step in history[1:]:
            for aid, obs in step["obs"].items():
                (peer,) = [other for other in step["obs"] if other != aid]
                delivered = step["delivered"][peer]
                assert obs[HARVEST_ENTRY] == np.float32(delivered["harvest"])
                assert obs[RESTORE_ENTRY] == np.float32(delivered["restore"])

    def test_entries_match_the_formulas_of_the_mechanisms(
        self, build_fishery, candidate_context
    ):
        _, history = self.play(
            build_fishery, [candidate_context(candidate(with_social=True))]
        )

        for step in history[1:]:
            for aid, obs in step["obs"].items():
                (peer,) = [other for other in step["obs"] if other != aid]
                delivered = step["delivered"][peer]
                assert obs[HARVEST_ENTRY] == pytest.approx(
                    delivered_fraction(
                        delivered["z_harvest"], step["previous_level"], QUOTA
                    ),
                    abs=FLOAT32_ATOL,
                )
                assert obs[RESTORE_ENTRY] == pytest.approx(
                    requested_fraction(delivered["z_restore"]), abs=FLOAT32_ATOL
                )

    def test_the_agents_own_entries_are_untouched(
        self, build_fishery, candidate_context
    ):
        _, history = self.play(
            build_fishery, [candidate_context(candidate(with_social=True))]
        )

        for step in history:
            for obs in step["obs"].values():
                assert obs.shape == (5,)
                assert obs.dtype == np.float32
                assert obs[0] == pytest.approx(step["level"], abs=FLOAT32_ATOL)
                assert obs[1] == 0.0

    def test_peers_see_different_values_when_their_actions_differ(
        self, build_fishery, candidate_context
    ):
        _, history = self.play(
            build_fishery, [candidate_context(candidate(with_social=True))]
        )

        gaps = [
            abs(
                step["obs"]["fisherman:0"][HARVEST_ENTRY]
                - step["obs"]["fisherman:1"][HARVEST_ENTRY]
            )
            for step in history[1:]
        ]
        assert max(gaps) > 0.05

    def test_nothing_is_exposed_at_reset(self, build_fishery, candidate_context):
        _, history = self.play(
            build_fishery, [candidate_context(candidate(with_social=True))], steps=1
        )

        for obs in history[0]["obs"].values():
            assert obs[HARVEST_ENTRY] == 0.0
            assert obs[RESTORE_ENTRY] == 0.0

    def test_entries_stay_zero_while_no_candidate_is_published(self, build_fishery):
        fishery, history = self.play(build_fishery, [None])

        assert fishery.world.calls == 1
        for step in history:
            for obs in step["obs"].values():
                assert obs[HARVEST_ENTRY] == 0.0
                assert obs[RESTORE_ENTRY] == 0.0

    def test_without_a_candidate_the_reward_is_the_requested_harvest(
        self, build_fishery
    ):
        fishery = build_fishery(
            quota_config(),
            subsidy_config(),
            penalty_config(),
            *social_configs(),
            candidates=[None],
        )
        adapter = fishery.adapter
        adapter.reset()
        rng = np.random.default_rng(2)

        for _ in range(5):
            actions = policy_outputs(rng, adapter.possible_agents)
            _, rewards, _, _, _ = adapter.step(actions)
            for aid in adapter.possible_agents:
                assert rewards[aid] == pytest.approx(
                    requested_fraction(z_of(actions, aid, "harvest")), abs=FLOAT32_ATOL
                )

    def test_entries_fill_in_once_the_candidate_arrives(
        self, build_fishery, candidate_context
    ):
        # The first episode runs before the regulator publishes; the second
        # reset fetches the candidate.
        fishery = build_fishery(
            quota_config(),
            subsidy_config(),
            penalty_config(),
            *social_configs(),
            candidates=[None, candidate_context(candidate(with_social=True))],
        )
        adapter = fishery.adapter
        rng = np.random.default_rng(3)

        adapter.reset()
        first, *_ = adapter.step(policy_outputs(rng, adapter.possible_agents))
        obs, _ = adapter.reset()
        after_reset = {aid: o.copy() for aid, o in obs.items()}
        second, *_ = adapter.step(policy_outputs(rng, adapter.possible_agents))

        for aid in adapter.possible_agents:
            assert first[aid][HARVEST_ENTRY] == 0.0
            assert first[aid][RESTORE_ENTRY] == 0.0
            assert after_reset[aid][HARVEST_ENTRY] == 0.0
            assert after_reset[aid][RESTORE_ENTRY] == 0.0
            assert second[aid][HARVEST_ENTRY] > 0.0
            assert second[aid][RESTORE_ENTRY] > 0.0
        assert fishery.world.calls == 2


@pytest.mark.unit
class TestEpisodeShape:
    def test_observations_and_rewards_are_finite_and_the_stock_stays_in_range(
        self, build_fishery, candidate_context
    ):
        fishery = build_fishery(
            quota_config(),
            subsidy_config(),
            penalty_config(),
            *social_configs(),
            candidates=[candidate_context(candidate(with_social=True))],
            horizon=5,
        )
        adapter = fishery.adapter
        obs, _ = adapter.reset()
        assert set(obs) == {"fisherman:0", "fisherman:1"}
        rng = np.random.default_rng(4)

        for step in range(5):
            obs, rewards, terminateds, truncateds, _ = adapter.step(
                policy_outputs(rng, adapter.possible_agents)
            )
            assert all(o.shape == (5,) and np.isfinite(o).all() for o in obs.values())
            assert all(math.isfinite(r) for r in rewards.values())
            assert terminateds["__all__"] is False
            assert truncateds["__all__"] is (step == 4)
            assert 0.0 <= level_of(adapter) <= 1.0

    def test_the_env_logs_the_mean_follower_reward_of_each_step(
        self, build_fishery, candidate_context
    ):
        fishery = build_fishery(
            quota_config(),
            subsidy_config(),
            penalty_config(),
            candidates=[candidate_context(candidate(with_social=False))],
            horizon=3,
        )
        adapter = fishery.adapter
        adapter.reset()
        rng = np.random.default_rng(5)

        total = 0.0
        for _ in range(3):
            _, rewards, *_ = adapter.step(policy_outputs(rng, adapter.possible_agents))
            total += float(np.mean(list(rewards.values())))

        assert fishery.env.logger.reduce().reward_total == pytest.approx(total)
