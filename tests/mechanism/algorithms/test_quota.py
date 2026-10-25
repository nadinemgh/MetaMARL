"""``QuotaMechanism``: a smooth cap and a fine on the targeted agents' request.

The adapter decodes the followers' policy outputs before the regulator acts, so
the request the mechanism reads is already an effort fraction ``f`` in
``[0, 1]``. For each targeted agent the mechanism returns three residuals, and
the expected values below are written from their formulas, not read back from
the mechanism:

    allowed = (s((b - q) / w) - s(-q / w)) / (s((1 - q) / w) - s(-q / w))
    action residual  = -softplus0(f - allowed; usage width)
    reward residual  = -min(1, FINE * softplus0(f - allowed; violation width)
                               + RISK_SCALE * max(0, 1 - b) * delivered ** POWER)
    observation residual: slot 1 moves to ``allowed`` and slot 3 to ``q``

with ``b`` the normalised resource level, ``q`` the regulator's quota, ``w`` the
quota transition width, ``s`` the logistic function, ``softplus0`` the softplus
of the given width shifted to be zero at the origin and ``delivered`` the
request minus the action residual. The mechanism no longer publishes the
allowed fraction in a state entry; it exposes it through the observation of the
targeted agents instead. Two quotas on the same target compose additively:
each corrects the same request, and the cuts, the penalties and the
observation residuals are summed.
"""

import math

import numpy as np
import pytest
from gymnasium import spaces

from core.agents.base import Agent
from core.mechanism.algorithms import quota as quota_module
from core.mechanism.algorithms.quota import Quota, QuotaMechanism
from core.mechanism.base import MDPState
from tests.issues import KNOWN_ISSUES

REGULATOR = "regulator"
K = 5_000.0
QUOTA_WIDTH = 0.03
USAGE_WIDTH = 0.005
VIOLATION_WIDTH = 0.03

# Values the mechanism writes in the observation of the targeted agents. They
# are repeated here on purpose, so that a change of the constants is noticed.
FINE = 0.2
MAX_DEMAND = 1.0
OBS_SIZE = 9  # the layout of examples/bilevel_fishery/debug.py

# Indices of the "legacy attribute" slots written at reset.
LEGACY_SLOTS = [1, 3, 4, 5, 6, 7]

# The observation is float32.
FLOAT32_ATOL = 1e-5

# Observation of a targeted agent before the regulator acts. Every slot differs
# from the values the mechanism writes, so that a residual that is zero by
# accident cannot pass.
BASE_OBS = (np.arange(1, OBS_SIZE + 1, dtype=np.float32) / 10.0).astype(np.float32)


def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def allowed_fraction(level: float, quota: float, width: float = QUOTA_WIDTH) -> float:
    lower = logistic(-quota / width)
    upper = logistic((1.0 - quota) / width)

    return (logistic((level - quota) / width) - lower) / (upper - lower)


def softplus0(x: float, width: float) -> float:
    """Softplus of the given width, shifted to zero at the origin, floored at 0."""

    return max(0.0, width * (math.log1p(math.exp(x / width)) - math.log(2.0)))


def cut(requested: float, allowed: float) -> float:
    """Action residual in fraction space (non-positive)."""

    return -softplus0(requested - allowed, USAGE_WIDTH)


def delivered_fraction(requested: float, allowed: float) -> float:
    return requested + cut(requested, allowed)


def penalty(
    requested: float,
    allowed: float,
    level: float = 0.5,
    fine: float = FINE,
    risk_scale: float = 0.0,
    risk_power: float = 1.0,
) -> float:
    """Reward residual (non-positive): the fine plus the optional risk term."""

    violation = softplus0(requested - allowed, VIOLATION_WIDTH)
    delivered = min(1.0, max(0.0, delivered_fraction(requested, allowed)))
    risk = risk_scale * max(0.0, 1.0 - level) * delivered**risk_power

    return -min(1.0, fine * violation + risk)


def make(
    acts_on=("fisherman", "harvest"), obs_map=None, mechanism_id="quota", **widths
) -> QuotaMechanism:
    if obs_map is None:
        obs_map = {"resource_level": "fish"}

    return Quota(
        id=mechanism_id,
        action_space=spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        acts_on=acts_on,
        obs_map=obs_map,
        **widths,
    ).build(REGULATOR)


def mdp_with(
    requests: dict[str, float], level: float = 0.5, quota: float = 0.5
) -> MDPState:
    """State at ``t = 0``: stock at ``level`` of the capacity, quota held.

    Every agent in ``requests`` holds a decoded request and an observation of
    size :data:`OBS_SIZE`, which the mechanism reads when it builds the
    observation residual.
    """

    return MDPState(
        aids=set(requests),
        params={"K": K},
        state={"fish": level * K},
        actions={
            REGULATOR: {"quota": make().decode(MDPState(), quota)},
            **{aid: {"harvest": f} for aid, f in requests.items()},
        },
        obs={aid: BASE_OBS.copy() for aid in requests},
    )


def two_quota_regulator() -> Agent:
    """Regulator holding the quotas ``quota`` and ``strict`` on the same target."""

    return Agent(
        id=REGULATOR,
        policy_id="p",
        mechanisms={"quota": make(), "strict": make(mechanism_id="strict")},
    )


def two_quota_mdp(request: float) -> MDPState:
    """Stock at half the capacity; ``quota`` holds 0.5 and ``strict`` 0.55.

    Both quotas bind on a request of 0.9: they allow about 0.50 and 0.16.
    """

    return MDPState(
        aids={"fisherman:0"},
        params={"K": K},
        state={"fish": 0.5 * K},
        actions={
            REGULATOR: {
                "quota": make().decode(MDPState(), 0.5),
                "strict": make(mechanism_id="strict").decode(MDPState(), 0.55),
            },
            "fisherman:0": {"harvest": request},
        },
        obs={"fisherman:0": BASE_OBS.copy()},
    )


def run(
    mechanism: QuotaMechanism,
    requests: dict[str, float],
    level: float = 0.5,
    quota: float = 0.5,
) -> MDPState:
    mdp = mdp_with(requests, level=level, quota=quota)

    return mechanism(mdp, mdp.actions[REGULATOR]["quota"][0])


def cut_of(delta: MDPState, aid: str) -> float:
    return delta.actions[aid]["harvest"][0]


def delivered_of(delta: MDPState, aid: str, request: float) -> float:
    """Delivered fraction after adding the residual to the decoded request."""

    return request + cut_of(delta, aid)


def reward_of(delta: MDPState, aid: str) -> float:
    return delta.rewards[aid][0]


@pytest.mark.unit
class TestValidation:
    @pytest.mark.parametrize(
        "name",
        [
            "quota_transition_width",
            "usage_transition_width",
            "violation_transition_width",
        ],
    )
    @pytest.mark.parametrize("value", [0.0, -0.01])
    def test_transition_widths_must_be_positive(self, name, value):
        with pytest.raises(ValueError, match=f"{name} must be > 0"):
            make(**{name: value})

    def test_apply_requires_a_target(self):
        with pytest.raises(ValueError, match="acts_on"):
            run(make(acts_on=None), {"fisherman:0": 0.5})

    def test_apply_requires_the_resource_level(self):
        requests = {"fisherman:0": 0.5}
        mechanism = make()
        mechanism.obs_map = None

        with pytest.raises(ValueError, match="resource_level"):
            run(mechanism, requests)

        mechanism.obs_map = {"stock": "fish"}

        with pytest.raises(ValueError, match="resource_level"):
            run(mechanism, requests)

    def test_on_reset_requires_a_target_once_a_quota_is_given(self):
        mdp = mdp_with({"fisherman:0": 0.5})

        with pytest.raises(ValueError, match="acts_on"):
            make(acts_on=None).on_reset(mdp, 0.5)

    def test_on_reset_requires_the_resource_level_once_a_quota_is_given(self):
        mdp = mdp_with({"fisherman:0": 0.5})

        with pytest.raises(ValueError, match="resource_level"):
            make(obs_map={"stock": "fish"}).on_reset(mdp, 0.5)

    def test_config_forwards_the_widths(self):
        mechanism = make(
            quota_transition_width=0.1,
            usage_transition_width=0.02,
            violation_transition_width=0.07,
        )

        assert isinstance(mechanism, QuotaMechanism)
        assert mechanism.quota_transition_width == 0.1
        assert mechanism.usage_transition_width == 0.02
        assert mechanism.violation_transition_width == 0.07
        assert mechanism.aid == REGULATOR
        assert mechanism.acts_on == ("fisherman", "harvest")

    def test_default_widths(self):
        mechanism = make()

        assert mechanism.quota_transition_width == QUOTA_WIDTH
        assert mechanism.usage_transition_width == USAGE_WIDTH
        assert mechanism.violation_transition_width == VIOLATION_WIDTH


@pytest.mark.unit
class TestDecode:
    @pytest.mark.parametrize(
        "action, expected",
        [(1.7, 1.0), (-0.2, 0.0), (0.25, 0.25), (0.0, 0.0), (1.0, 1.0)],
    )
    def test_decode_clips_the_quota_to_the_unit_interval(self, action, expected):
        assert make().decode(MDPState(), np.array([action])) == pytest.approx(expected)

    def test_decode_reads_the_first_component(self):
        assert make().decode(MDPState(), np.array([0.3, 0.9])) == pytest.approx(0.3)

    def test_decode_returns_a_python_float(self):
        assert type(make().decode(MDPState(), np.array([0.3], dtype=np.float32))) is (
            float
        )

    def test_decode_is_idempotent(self):
        # A quota already in [0, 1] is returned unchanged. The mechanism no
        # longer relies on it (see ``test_decode_input.py``), but the clip
        # keeps this property.
        mechanism = make()
        once = mechanism.decode(MDPState(), np.array([1.4]))

        assert mechanism.decode(MDPState(), once) == once

    def test_call_clips_the_quota_before_applying_it(self):
        # A quota of 1.4 behaves as a quota of 1: nothing binds above the stock.
        mdp = mdp_with({"fisherman:0": 0.9}, level=0.5, quota=1.4)
        delta = make()(mdp, mdp.actions[REGULATOR]["quota"][0])
        expected = delivered_fraction(0.9, allowed_fraction(0.5, 1.0))

        assert delivered_of(delta, "fisherman:0", 0.9) == pytest.approx(
            expected, abs=FLOAT32_ATOL
        )

    def test_call_leaves_the_stored_quota_untouched(self):
        # The decoded quota is stored once by the environment; the call reads
        # it and does not write it back (it was ``test_call_writes_the_decoded_
        # quota_back`` when the mechanism decoded the action itself).
        mdp = mdp_with({"fisherman:0": 0.5}, quota=1.4)

        make()(mdp, mdp.actions[REGULATOR]["quota"][0])
        assert mdp.actions[REGULATOR]["quota"][0] == 1.0


@pytest.mark.unit
class TestAllowedFraction:
    def test_half_the_stock_at_the_quota(self):
        # At ``b = q = 1/2`` the logistic is symmetric about the quota, so the
        # allowed fraction is exactly 1/2 and a request of 0.9 is cut to
        # ``0.9 - softplus0(0.4)``.
        delta = run(make(), {"fisherman:0": 0.9}, level=0.5, quota=0.5)

        assert allowed_fraction(0.5, 0.5) == pytest.approx(0.5)
        assert delivered_of(delta, "fisherman:0", 0.9) == pytest.approx(
            delivered_fraction(0.9, 0.5), abs=FLOAT32_ATOL
        )

    @pytest.mark.parametrize(
        "level, quota, width",
        [
            (0.4, 0.4, 0.05),
            (0.2, 0.5, QUOTA_WIDTH),
            (0.45, 0.5, QUOTA_WIDTH),
            (0.55, 0.5, QUOTA_WIDTH),
            (0.8, 0.3, 0.1),
            (0.1, 0.9, 0.2),
        ],
    )
    def test_delivery_follows_the_allowed_fraction(self, level, quota, width):
        mechanism = make(quota_transition_width=width)
        delta = run(mechanism, {"fisherman:0": 0.95}, level=level, quota=quota)
        expected = delivered_fraction(0.95, allowed_fraction(level, quota, width))

        assert delivered_of(delta, "fisherman:0", 0.95) == pytest.approx(
            expected, abs=FLOAT32_ATOL
        )

    def test_nothing_is_allowed_when_the_stock_is_empty(self):
        # ``b = 0`` gives ``current == lower``, hence an allowed fraction of 0:
        # a large request is cut to ``usage_width * ln 2``, the offset of the
        # shifted softplus.
        delta = run(make(), {"fisherman:0": 0.9}, level=0.0, quota=0.5)

        assert delivered_of(delta, "fisherman:0", 0.9) == pytest.approx(
            USAGE_WIDTH * math.log(2.0), abs=FLOAT32_ATOL
        )

    def test_everything_is_allowed_when_the_stock_is_full(self):
        # ``b = 1`` gives ``current == upper``, hence an allowed fraction of 1.
        delta = run(make(), {"fisherman:0": 0.9}, level=1.0, quota=0.5)

        assert delivered_of(delta, "fisherman:0", 0.9) == pytest.approx(
            0.9, abs=FLOAT32_ATOL
        )

    def test_delivery_grows_with_the_resource_level(self):
        delivered = [
            delivered_of(
                run(make(), {"fisherman:0": 0.9}, level=level, quota=0.5),
                "fisherman:0",
                0.9,
            )
            for level in np.linspace(0.0, 1.0, 21)
        ]

        assert np.all(np.diff(delivered) >= -FLOAT32_ATOL)
        assert delivered[0] < 0.01
        assert delivered[-1] == pytest.approx(0.9, abs=FLOAT32_ATOL)

    def test_a_higher_quota_lets_less_through(self):
        delivered = [
            delivered_of(
                run(make(), {"fisherman:0": 0.9}, level=0.4, quota=quota),
                "fisherman:0",
                0.9,
            )
            for quota in np.linspace(0.1, 0.9, 9)
        ]

        assert np.all(np.diff(delivered) <= FLOAT32_ATOL)

    def test_resource_level_is_the_state_over_the_capacity(self):
        # The same normalised level reached with another capacity gives the
        # same delivery.
        mdp = mdp_with({"fisherman:0": 0.9}, level=0.4)
        mdp.params["K"] = 2 * K
        mdp.state.data["fish"] = [0.4 * 2 * K]
        delta = make()(mdp, mdp.actions[REGULATOR]["quota"][0])

        assert delivered_of(delta, "fisherman:0", 0.9) == pytest.approx(
            delivered_fraction(0.9, allowed_fraction(0.4, 0.5)), abs=FLOAT32_ATOL
        )


@pytest.mark.unit
class TestActionCut:
    """The action residual lives in fraction space and is a plain float."""

    @pytest.mark.parametrize("request_fraction", [0.01, 0.1, 0.4])
    def test_request_below_the_allowance_is_left_unchanged(self, request_fraction):
        # A stock at the quota allows 1/2. The shifted softplus is floored at
        # zero, so a request under the allowance is not cut at all.
        delta = run(make(), {"fisherman:0": request_fraction}, level=0.5)

        assert cut_of(delta, "fisherman:0") == 0.0

    def test_request_at_the_allowance_is_not_cut(self):
        delta = run(make(), {"fisherman:0": 0.5}, level=0.5)

        assert cut_of(delta, "fisherman:0") == pytest.approx(0.0, abs=1e-12)

    def test_request_above_the_allowance_is_capped_and_lowered(self):
        delta = run(make(), {"fisherman:0": 0.9}, level=0.5)
        delivered = delivered_of(delta, "fisherman:0", 0.9)

        assert delivered < 0.9
        assert delivered == pytest.approx(
            0.5 + USAGE_WIDTH * math.log(2.0), abs=FLOAT32_ATOL
        )
        assert cut_of(delta, "fisherman:0") < 0

    def test_cut_is_continuous_around_the_allowance(self):
        requests = np.linspace(0.45, 0.55, 201)
        delivered = [
            delivered_of(
                run(make(), {"fisherman:0": float(r)}, level=0.5),
                "fisherman:0",
                float(r),
            )
            for r in requests
        ]

        assert np.max(np.abs(np.diff(delivered))) < 1e-3
        assert np.all(np.diff(delivered) >= -FLOAT32_ATOL)

    def test_a_wider_usage_transition_cuts_less_just_above_the_allowance(self):
        # The wider the smooth hinge, the more of a small excess it lets
        # through; far above the allowance all widths cut almost the same.
        narrow = run(make(usage_transition_width=0.005), {"fisherman:0": 0.55})
        wide = run(make(usage_transition_width=0.05), {"fisherman:0": 0.55})

        assert cut_of(wide, "fisherman:0") > cut_of(narrow, "fisherman:0")
        assert cut_of(wide, "fisherman:0") == pytest.approx(
            -softplus0(0.05, 0.05), abs=FLOAT32_ATOL
        )

    def test_residual_holds_a_single_cut_under_the_targeted_mechanism(self):
        # The request is a decoded fraction, so the residual is one number per
        # agent (it used to be an array shaped like the raw policy output).
        delta = run(make(), {"fisherman:0": 0.9}, level=0.5)

        assert set(delta.actions["fisherman:0"]) == {"harvest"}
        assert np.ndim(cut_of(delta, "fisherman:0")) == 0

    def test_request_stored_as_a_one_element_array_is_accepted(self):
        delta = run(make(), {"fisherman:0": np.array([0.9])}, level=0.5)

        assert cut_of(delta, "fisherman:0") == pytest.approx(
            cut(0.9, 0.5), abs=FLOAT32_ATOL
        )

    def test_inputs_are_not_mutated_in_place(self):
        mdp = mdp_with({"fisherman:0": 0.9}, level=0.5)
        obs = mdp.obs["fisherman:0"][0].copy()

        make()(mdp, mdp.actions[REGULATOR]["quota"][0])
        assert mdp.actions["fisherman:0"]["harvest"][0] == 0.9
        assert mdp.state["fish"][0] == 0.5 * K
        np.testing.assert_array_equal(mdp.obs["fisherman:0"][0], obs)
        assert mdp.rewards.data == {}

    def test_each_agent_is_capped_on_its_own_request(self):
        requests = {"fisherman:0": 0.2, "fisherman:1": 0.9}
        delta = run(make(), requests, level=0.5)

        assert delivered_of(delta, "fisherman:0", 0.2) == pytest.approx(0.2)
        assert delivered_of(delta, "fisherman:1", 0.9) == pytest.approx(
            delivered_fraction(0.9, 0.5), abs=FLOAT32_ATOL
        )

    @pytest.mark.parametrize("request_fraction", [0.0, 1.0])
    def test_extreme_requests_stay_in_the_unit_interval(self, request_fraction):
        # This replaces the floor and cap at ``1e-6`` that protected the logit
        # of the old delivery: the cut now works on the fraction itself, so
        # the delivered effort needs no clipping to stay finite.
        for level in (0.0, 0.5, 1.0):
            delta = run(make(), {"fisherman:0": request_fraction}, level=level)
            delivered = delivered_of(delta, "fisherman:0", request_fraction)

            assert np.isfinite(delivered)
            assert -FLOAT32_ATOL <= delivered <= 1.0 + FLOAT32_ATOL


@pytest.mark.unit
class TestRewardPenalty:
    """The fine on the part of the request above the allowance."""

    @pytest.mark.parametrize("request_fraction", [0.0, 0.1, 0.4, 0.5])
    def test_no_penalty_at_or_below_the_allowance(self, request_fraction):
        delta = run(make(), {"fisherman:0": request_fraction}, level=0.5)

        assert reward_of(delta, "fisherman:0") == pytest.approx(0.0, abs=1e-12)

    @pytest.mark.parametrize("request_fraction", [0.55, 0.7, 0.9, 1.0])
    def test_penalty_is_the_fine_times_the_smooth_violation(self, request_fraction):
        delta = run(make(), {"fisherman:0": request_fraction}, level=0.5)
        expected = -FINE * softplus0(request_fraction - 0.5, VIOLATION_WIDTH)

        assert reward_of(delta, "fisherman:0") == pytest.approx(expected, abs=1e-9)
        assert reward_of(delta, "fisherman:0") < 0

    def test_penalty_grows_with_the_violation(self):
        penalties = [
            reward_of(run(make(), {"fisherman:0": float(r)}, level=0.5), "fisherman:0")
            for r in np.linspace(0.0, 1.0, 41)
        ]

        assert np.all(np.diff(penalties) <= 1e-12)
        assert np.max(np.abs(np.diff(penalties))) < 0.01

    def test_penalty_follows_the_allowed_fraction(self):
        delta = run(make(), {"fisherman:0": 0.9}, level=0.2, quota=0.5)
        allowed = allowed_fraction(0.2, 0.5)

        assert reward_of(delta, "fisherman:0") == pytest.approx(
            penalty(0.9, allowed), abs=1e-9
        )

    def test_violation_width_sets_the_smoothness_of_the_penalty(self):
        delta = run(make(violation_transition_width=0.2), {"fisherman:0": 0.6})
        expected = -FINE * softplus0(0.1, 0.2)

        assert reward_of(delta, "fisherman:0") == pytest.approx(expected, abs=1e-9)

    def test_penalty_is_capped_at_one(self, monkeypatch):
        monkeypatch.setattr(quota_module, "FINE_AMOUNT", 50.0)

        delta = run(make(), {"fisherman:0": 0.95}, level=0.5)

        assert reward_of(delta, "fisherman:0") == -1.0

    def test_risk_term_is_added_to_the_fine(self, monkeypatch):
        # The risk scale is zero by default (a placeholder of the benchmark);
        # with it on, the term scales with the missing stock and the delivered
        # effort.
        monkeypatch.setattr(quota_module, "RISK_PENALTY_SCALE", 0.5)
        monkeypatch.setattr(quota_module, "RISK_PENALTY_POWER", 2.0)

        delta = run(make(), {"fisherman:0": 0.9}, level=0.3, quota=0.5)
        allowed = allowed_fraction(0.3, 0.5)

        assert reward_of(delta, "fisherman:0") == pytest.approx(
            penalty(0.9, allowed, level=0.3, risk_scale=0.5, risk_power=2.0), abs=1e-9
        )
        assert reward_of(delta, "fisherman:0") < penalty(0.9, allowed, level=0.3)

    def test_risk_term_vanishes_on_a_full_stock(self, monkeypatch):
        monkeypatch.setattr(quota_module, "RISK_PENALTY_SCALE", 0.5)

        delta = run(make(), {"fisherman:0": 0.9}, level=1.0, quota=0.5)

        assert reward_of(delta, "fisherman:0") == pytest.approx(0.0, abs=1e-9)

    def test_every_targeted_agent_gets_its_own_penalty(self):
        requests = {"fisherman:0": 0.2, "fisherman:1": 0.9}
        delta = run(make(), requests, level=0.5)

        assert set(delta.rewards.data) == set(requests)
        assert reward_of(delta, "fisherman:0") == pytest.approx(0.0, abs=1e-12)
        assert reward_of(delta, "fisherman:1") == pytest.approx(
            penalty(0.9, 0.5), abs=1e-9
        )


@pytest.mark.unit
class TestObservationResidual:
    """The residual moves slots 1 and 3 of the observation, and only those."""

    def test_slots_one_and_three_move_to_the_allowance_and_the_quota(self):
        delta = run(make(), {"fisherman:0": 0.9}, level=0.4, quota=0.6)
        expected = np.zeros(OBS_SIZE, dtype=np.float32)
        expected[1] = allowed_fraction(0.4, 0.6) - BASE_OBS[1]
        expected[3] = 0.6 - BASE_OBS[3]

        np.testing.assert_allclose(
            delta.obs["fisherman:0"][0], expected, atol=FLOAT32_ATOL
        )

    def test_the_other_slots_are_not_touched(self):
        delta = run(make(), {"fisherman:0": 0.9}, level=0.4, quota=0.6)
        residual = delta.obs["fisherman:0"][0]
        others = [i for i in range(OBS_SIZE) if i not in (1, 3)]

        np.testing.assert_array_equal(residual[others], np.zeros(len(others)))

    def test_adding_the_residual_leaves_the_allowance_and_the_quota_in_the_obs(self):
        mdp = mdp_with({"fisherman:0": 0.9}, level=0.4, quota=0.6)
        composed = mdp.add(make()(mdp, mdp.actions[REGULATOR]["quota"][0]))
        observed = composed.obs["fisherman:0"][0]

        assert observed[1] == pytest.approx(allowed_fraction(0.4, 0.6), abs=1e-6)
        assert observed[3] == pytest.approx(0.6, abs=1e-6)
        np.testing.assert_allclose(
            np.delete(observed, [1, 3]), np.delete(BASE_OBS, [1, 3])
        )

    def test_the_residual_is_the_difference_with_the_current_observation(self):
        # The residual is added to the observation like every flow delta, so
        # it must be the difference with what the observation holds: another
        # starting observation changes the residual, not the result.
        mdp = mdp_with({"fisherman:0": 0.9}, level=0.4, quota=0.6)

        mdp.update(obs={"fisherman:0": np.full(OBS_SIZE, 3.0, dtype=np.float32)})

        delta = make()(mdp, mdp.actions[REGULATOR]["quota"][0])

        assert delta.obs["fisherman:0"][0][1] == pytest.approx(
            allowed_fraction(0.4, 0.6) - 3.0, abs=FLOAT32_ATOL
        )
        assert delta.obs["fisherman:0"][0][3] == pytest.approx(0.6 - 3.0, abs=1e-6)

    def test_the_obs_follows_the_resource_level_over_the_steps(self):
        # The observation of each step is rebuilt from the base, so the
        # allowance shown at a step must be that of the step, not a running sum.
        regulator = Agent(id=REGULATOR, policy_id="p", mechanisms={"quota": make()})
        levels = [0.8, 0.5, 0.2, 0.2]
        mdp = mdp_with({"fisherman:0": 0.9}, level=levels[0], quota=0.5)
        recorded = []

        for step, level in enumerate(levels):
            if step:
                mdp = mdp.advance(
                    state={"fish": level * K},
                    obs={"fisherman:0": BASE_OBS.copy()},
                    actions={"fisherman:0": {"harvest": 0.9}},
                    rewards={"fisherman:0": 0.0},
                )

            composed = mdp.add(regulator.action(mdp))

            recorded.append(composed.obs["fisherman:0"][mdp.t][1])

        assert recorded == pytest.approx(
            [allowed_fraction(level, 0.5) for level in levels], abs=1e-6
        )

    def test_the_obs_residual_is_built_for_each_targeted_agent(self):
        requests = {"fisherman:0": 0.2, "fisherman:1": 0.9}
        delta = run(make(), requests, level=0.5)

        assert set(delta.obs.data) == set(requests)

    def test_no_state_entry_is_published(self):
        # The allowed fraction used to be published in the state entry
        # ``allowed_frac:<id>``; it now reaches the agents through their
        # observation only.
        mdp = mdp_with({"fisherman:0": 0.9}, level=0.4, quota=0.5)
        delta = make()(mdp, mdp.actions[REGULATOR]["quota"][0])
        composed = mdp.add(delta)

        assert delta.state.data == {}
        assert set(composed.state.data) == {"fish"}


@pytest.mark.unit
class TestOnReset:
    def test_without_a_quota_nothing_is_written(self):
        mdp = mdp_with({"fisherman:0": 0.9})
        delta = make().on_reset(mdp, None)

        assert delta.obs.data == {}
        assert delta.actions.data == {}
        assert delta.rewards.data == {}

    def test_the_legacy_slots_are_set_from_an_empty_observation(self):
        # Without a previous observation the delta is the target itself.
        mdp = MDPState(aids={"fisherman:0"}, params={"K": K}, state={"fish": 0.4 * K})
        delta = make().on_reset(mdp, 0.6)
        expected = np.zeros(OBS_SIZE, dtype=np.float32)
        expected[1] = allowed_fraction(0.4, 0.6)  # effective quota
        expected[3] = 0.6  # fixed quota
        expected[4] = MAX_DEMAND
        expected[5] = FINE
        expected[6] = 0.0  # risk penalty scale
        expected[7] = 0.0  # risk penalty power

        np.testing.assert_allclose(
            delta.obs["fisherman:0"][0], expected, atol=FLOAT32_ATOL
        )

    def test_the_delta_is_taken_from_the_current_step(self):
        mdp = mdp_with({"fisherman:0": 0.9}, level=0.4)
        later = np.full(OBS_SIZE, 2.0, dtype=np.float32)
        mdp = mdp.advance(state={"fish": 0.4 * K}, obs={"fisherman:0": later})
        delta = make().on_reset(mdp, 0.6)
        target = {
            1: allowed_fraction(0.4, 0.6),
            3: 0.6,
            4: MAX_DEMAND,
            5: FINE,
            6: 0.0,
            7: 0.0,
        }
        residual = delta.obs["fisherman:0"][0]

        for slot, value in target.items():
            assert residual[slot] == pytest.approx(value - 2.0, abs=FLOAT32_ATOL)

    def test_a_step_without_an_observation_yet_receives_the_target(self):
        # Observations are per-step flows: the previous step's value must not
        # enter the difference, or the slots alternate between the target and
        # zero from one step to the next.
        mdp = mdp_with({"fisherman:0": 0.9}, level=0.4)
        mdp = mdp.advance(state={"fish": 0.4 * K})

        mdp.update(
            obs={
                aid: np.zeros_like(values[-1])
                for aid, values in mdp.obs.data.items()
                if values
            }
        )

        residual = make().on_reset(mdp, 0.6).obs["fisherman:0"][0]
        target = [allowed_fraction(0.4, 0.6), 0.6, MAX_DEMAND, FINE, 0.0, 0.0]

        np.testing.assert_allclose(residual[LEGACY_SLOTS], target, atol=FLOAT32_ATOL)

    def test_only_the_legacy_slots_are_written(self):
        mdp = mdp_with({"fisherman:0": 0.9}, level=0.4)
        residual = make().on_reset(mdp, 0.6).obs["fisherman:0"][0]
        untouched = [i for i in range(OBS_SIZE) if i not in LEGACY_SLOTS]

        assert untouched == [0, 2, 8]
        np.testing.assert_array_equal(residual[untouched], np.zeros(3))

    def test_adding_the_residual_sets_the_slots_to_the_target(self):
        mdp = mdp_with({"fisherman:0": 0.9}, level=0.4)
        observed = mdp.add(make().on_reset(mdp, 0.6)).obs["fisherman:0"][0]
        expected = BASE_OBS.copy()
        expected[LEGACY_SLOTS] = [
            allowed_fraction(0.4, 0.6),
            0.6,
            MAX_DEMAND,
            FINE,
            0.0,
            0.0,
        ]

        np.testing.assert_allclose(observed, expected, atol=FLOAT32_ATOL)

    def test_only_agents_of_the_targeted_type_are_written(self):
        mdp = MDPState(
            aids={"fisherman:0", "fisherman:1", "farmer:0", "fishermen:0", REGULATOR},
            params={"K": K},
            state={"fish": 0.4 * K},
        )
        delta = make().on_reset(mdp, 0.6)

        assert set(delta.obs.data) == {"fisherman:0", "fisherman:1"}

    def test_the_call_with_reset_runs_on_reset(self):
        mdp = mdp_with({"fisherman:0": 0.9}, level=0.4)
        via_call = make()(mdp, 0.6, reset=True)
        direct = make().on_reset(mdp, 0.6)

        np.testing.assert_array_equal(
            via_call.obs["fisherman:0"][0], direct.obs["fisherman:0"][0]
        )
        assert via_call.actions.data == {}
        assert via_call.rewards.data == {}


@pytest.mark.unit
class TestTargeting:
    def test_only_agents_of_the_targeted_type_are_capped(self):
        requests = {"fisherman:0": 0.9, "farmer:0": 0.9}
        delta = run(make(), requests, level=0.5)

        assert set(delta.actions.data) == {"fisherman:0"}
        assert set(delta.rewards.data) == {"fisherman:0"}
        assert set(delta.obs.data) == {"fisherman:0"}

    def test_identifier_must_start_with_the_type_and_a_colon(self):
        requests = {"fisherman:0": 0.9, "fishermen:0": 0.9}
        delta = run(make(), requests, level=0.5)

        assert set(delta.actions.data) == {"fisherman:0"}

    def test_the_regulator_is_not_a_target(self):
        delta = run(make(), {"fisherman:0": 0.9}, level=0.5)

        assert REGULATOR not in delta.actions.data
        assert REGULATOR not in delta.rewards.data
        assert REGULATOR not in delta.obs.data

    def test_agents_without_the_targeted_mechanism_are_skipped(self):
        # ``fisherman:1`` holds neither the targeted mechanism nor an
        # observation: it must be skipped before its observation is read.
        mdp = mdp_with({"fisherman:0": 0.9})
        mdp.actions.data["fisherman:1"] = {"restore": [0.9]}
        delta = make()(mdp, mdp.actions[REGULATOR]["quota"][0])

        assert set(delta.actions.data) == {"fisherman:0"}
        assert set(delta.rewards.data) == {"fisherman:0"}
        assert set(delta.obs.data) == {"fisherman:0"}

    def test_the_targeted_mechanism_is_the_one_named(self):
        mdp = mdp_with({"fisherman:0": 0.9})
        mdp.actions.data["fisherman:0"]["restore"] = [0.9]
        delta = make(acts_on=("fisherman", "restore"))(
            mdp, mdp.actions[REGULATOR]["quota"][0]
        )

        assert set(delta.actions["fisherman:0"].keys()) == {"restore"}

    def test_no_target_gives_no_residual(self):
        delta = run(make(), {}, level=0.5)

        assert delta.actions.data == {}
        assert delta.rewards.data == {}
        assert delta.obs.data == {}

    @pytest.mark.skip(reason=KNOWN_ISSUES["OBS-001"])
    def test_a_targeted_agent_without_an_observation_starts_from_zeros(self):
        # TODO: Re-enable once observation initialization and dimension
        # ownership are defined across environments and mechanisms.
        mdp = mdp_with({"fisherman:0": 0.9}, level=0.4)
        mdp.actions.data["fisherman:1"] = {"harvest": [0.9]}
        quota = float(mdp.actions[REGULATOR]["quota"][0])
        residual = make()(mdp, quota).obs["fisherman:1"][0]
        expected = np.zeros(OBS_SIZE, dtype=np.float32)
        expected[[1, 3]] = [allowed_fraction(0.4, quota), quota]

        np.testing.assert_allclose(residual, expected, atol=FLOAT32_ATOL)


@pytest.mark.unit
class TestComposition:
    def test_residual_carries_the_cut_the_penalty_and_the_observation(self):
        # The agent returns the residual; the caller adds it to the shared
        # state. The follower then reads the cut request, the reward carries
        # the fine and the observation shows the allowance.
        regulator = Agent(id=REGULATOR, policy_id="p", mechanisms={"quota": make()})
        mdp = mdp_with({"fisherman:0": 0.9}, level=0.5)
        composed = mdp.add(regulator.action(mdp))

        assert composed.actions["fisherman:0"]["harvest"][0] == pytest.approx(
            delivered_fraction(0.9, 0.5), abs=FLOAT32_ATOL
        )
        assert composed.rewards["fisherman:0"][0] == pytest.approx(
            penalty(0.9, 0.5), abs=1e-9
        )
        assert composed.obs["fisherman:0"][0][1] == pytest.approx(0.5, abs=1e-6)

    def test_regulator_without_a_candidate_applies_nothing(self):
        regulator = Agent(id=REGULATOR, policy_id="p", mechanisms={"quota": make()})
        mdp = MDPState(
            params={"K": K},
            state={"fish": 0.5 * K},
            actions={"fisherman:0": {"harvest": 0.9}},
            obs={"fisherman:0": BASE_OBS.copy()},
        )
        residual = regulator.action(mdp)

        assert residual.actions.data == {}
        assert residual.rewards.data == {}
        assert residual.obs.data == {}


@pytest.mark.unit
class TestTwoQuotasOnOneTarget:
    def test_the_cuts_are_summed(self):
        # Both quotas read the same request, so each returns the full cut to
        # its own allowance and the cuts add: the delivered effort is below
        # what the stricter quota alone delivers.
        residual = two_quota_regulator().action(two_quota_mdp(0.9))
        allowed_loose = allowed_fraction(0.5, 0.5)
        allowed_strict = allowed_fraction(0.5, 0.55)
        combined = cut_of(residual, "fisherman:0")

        assert combined == pytest.approx(
            cut(0.9, allowed_loose) + cut(0.9, allowed_strict), abs=FLOAT32_ATOL
        )
        assert 0.9 + combined < delivered_fraction(0.9, allowed_strict)

    def test_the_penalties_are_summed(self):
        residual = two_quota_regulator().action(two_quota_mdp(0.9))
        expected = penalty(0.9, allowed_fraction(0.5, 0.5)) + penalty(
            0.9, allowed_fraction(0.5, 0.55)
        )

        # The strict quota 0.55 is stored through a float32 decode.
        assert reward_of(residual, "fisherman:0") == pytest.approx(expected, abs=1e-6)

    def test_the_observation_residuals_are_summed_too(self):
        # Each quota writes the difference between its own allowance and the
        # same current observation, so the two differences add: once added to
        # the observation, slots 1 and 3 do not show either quota.
        mdp = two_quota_mdp(0.9)
        residual = two_quota_regulator().action(mdp)
        slot_one = allowed_fraction(0.5, 0.5) + allowed_fraction(0.5, 0.55)

        assert residual.obs["fisherman:0"][0][1] == pytest.approx(
            slot_one - 2 * BASE_OBS[1], abs=FLOAT32_ATOL
        )
        assert residual.obs["fisherman:0"][0][3] == pytest.approx(
            0.5 + 0.55 - 2 * BASE_OBS[3], abs=FLOAT32_ATOL
        )

    def test_no_state_entry_is_published_for_either_quota(self):
        mdp = two_quota_mdp(0.9)
        composed = mdp.add(two_quota_regulator().action(mdp))

        assert set(composed.state.data) == {"fish"}
