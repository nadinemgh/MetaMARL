"""``QuotaMechanism``: a smooth cap on the targeted agents' requested action.

The regulator acts before the followers, so the request it reads is still the
raw policy output ``z``; the mechanism maps it to ``f = sigmoid(z / 4)`` and
returns a residual on ``z`` that moves the request to the delivered fraction.
The expected values below are written from the formulas of the mechanism, not
read back from it:

    allowed = (s((b - q) / w) - s(-q / w)) / (s((1 - q) / w) - s(-q / w))
    delivered = f - softplus0(f - allowed)

with ``b`` the normalised resource level, ``q`` the regulator's quota, ``w``
the quota transition width, ``s`` the logistic function and ``softplus0`` the
softplus of width ``usage_transition_width`` shifted to be zero at the origin.
The allowed fraction is also published in the state entry ``allowed_frac``;
the entry must hold the fraction of the current step, not a running sum.
"""

import math

import numpy as np
import pytest
from gymnasium import spaces

from core.agents.base import Agent
from core.mechanism.algorithms.quota import Quota, QuotaMechanism
from core.mechanism.base import MDPState

REGULATOR = "regulator"
K = 5_000.0
TEMPERATURE = 4.0
QUOTA_WIDTH = 0.03
USAGE_WIDTH = 0.005
# Single-precision residuals: the mechanism stores the delivered action in a
# float32 array.
FLOAT32_ATOL = 1e-5


def logistic(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def raw(fraction: float) -> np.ndarray:
    """Policy output ``z`` whose decoded fraction is ``fraction``."""
    z = TEMPERATURE * math.log(fraction / (1.0 - fraction))
    return np.array([z], dtype=np.float32)


def allowed_fraction(level: float, quota: float, width: float = QUOTA_WIDTH) -> float:
    lower = logistic(-quota / width)
    upper = logistic((1.0 - quota) / width)
    return (logistic((level - quota) / width) - lower) / (upper - lower)


def softplus0(x: float, width: float = USAGE_WIDTH) -> float:
    """Softplus of the given width, shifted to zero at the origin, floored at 0."""
    return max(0.0, width * (math.log1p(math.exp(x / width)) - math.log(2.0)))


def delivered_fraction(requested: float, allowed: float) -> float:
    return requested - softplus0(requested - allowed)


def make(acts_on=("fisherman", "harvest"), obs_map=None, **widths) -> QuotaMechanism:
    if obs_map is None:
        obs_map = {"resource_level": "fish"}
    return Quota(
        id="quota",
        action_space=spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
        acts_on=acts_on,
        obs_map=obs_map,
        **widths,
    ).build(REGULATOR)


def mdp_with(
    requests: dict[str, np.ndarray], level: float = 0.5, quota: float = 0.5
) -> MDPState:
    """State at ``t = 0``: stock at ``level`` of the capacity, quota held."""
    return MDPState(
        aids=set(requests),
        params={"K": K},
        state={"fish": level * K},
        actions={
            REGULATOR: {"quota": np.array([quota], dtype=np.float32)},
            **{aid: {"harvest": z} for aid, z in requests.items()},
        },
    )


def run(
    mechanism: QuotaMechanism,
    requests: dict[str, np.ndarray],
    level: float = 0.5,
    quota: float = 0.5,
) -> MDPState:
    mdp = mdp_with(requests, level=level, quota=quota)
    return mechanism(mdp, mdp.actions[REGULATOR]["quota"][0])


def delivered_of(delta: MDPState, aid: str, z: np.ndarray) -> float:
    """Delivered fraction after the residual of ``delta`` is added to ``z``."""
    return logistic(float((z + delta.actions[aid]["harvest"][0]).item()) / TEMPERATURE)


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
            run(make(acts_on=None), {"fisherman:0": raw(0.5)})

    def test_apply_requires_the_resource_level(self):
        requests = {"fisherman:0": raw(0.5)}
        mechanism = make()
        mechanism.obs_map = None
        with pytest.raises(ValueError, match="resource_level"):
            run(mechanism, requests)
        mechanism.obs_map = {"stock": "fish"}
        with pytest.raises(ValueError, match="resource_level"):
            run(mechanism, requests)

    def test_config_forwards_the_widths(self):
        mechanism = make(
            quota_transition_width=0.1,
            usage_transition_width=0.02,
            violation_transition_width=0.2,
        )

        assert isinstance(mechanism, QuotaMechanism)
        assert mechanism.quota_transition_width == 0.1
        assert mechanism.usage_transition_width == 0.02
        assert mechanism.violation_transition_width == 0.2
        assert mechanism.aid == REGULATOR
        assert mechanism.acts_on == ("fisherman", "harvest")

    def test_default_widths(self):
        mechanism = make()

        assert mechanism.quota_transition_width == QUOTA_WIDTH
        assert mechanism.usage_transition_width == USAGE_WIDTH
        assert mechanism.violation_transition_width == 0.03


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
        requests = {"fisherman:0": raw(0.9)}
        mdp = mdp_with(requests, level=0.5, quota=1.4)

        delta = make()(mdp, mdp.actions[REGULATOR]["quota"][0])

        expected = delivered_fraction(0.9, allowed_fraction(0.5, 1.0))
        assert delivered_of(delta, "fisherman:0", requests["fisherman:0"]) == (
            pytest.approx(expected, abs=FLOAT32_ATOL)
        )

    def test_call_writes_the_decoded_quota_back(self):
        mdp = mdp_with({"fisherman:0": raw(0.5)}, quota=1.4)

        make()(mdp, mdp.actions[REGULATOR]["quota"][0])

        assert mdp.actions[REGULATOR]["quota"][0] == 1.0


@pytest.mark.unit
class TestAllowedFraction:
    def test_half_the_stock_at_the_quota(self):
        # At ``b = q = 1/2`` the logistic is symmetric about the quota, so the
        # allowed fraction is exactly 1/2 and a request of 0.9 is cut to
        # ``0.9 - softplus0(0.4)``.
        z = raw(0.9)
        delta = run(make(), {"fisherman:0": z}, level=0.5, quota=0.5)

        assert allowed_fraction(0.5, 0.5) == pytest.approx(0.5)
        assert delivered_of(delta, "fisherman:0", z) == pytest.approx(
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
        z = raw(0.95)
        mechanism = make(quota_transition_width=width)

        delta = run(mechanism, {"fisherman:0": z}, level=level, quota=quota)

        expected = delivered_fraction(0.95, allowed_fraction(level, quota, width))
        assert delivered_of(delta, "fisherman:0", z) == pytest.approx(
            expected, abs=FLOAT32_ATOL
        )

    def test_nothing_is_allowed_when_the_stock_is_empty(self):
        # ``b = 0`` gives ``current == lower``, hence an allowed fraction of 0:
        # a large request is cut to ``usage_width * ln 2``, the offset of the
        # shifted softplus.
        z = raw(0.9)

        delta = run(make(), {"fisherman:0": z}, level=0.0, quota=0.5)

        assert delivered_of(delta, "fisherman:0", z) == pytest.approx(
            USAGE_WIDTH * math.log(2.0), abs=FLOAT32_ATOL
        )

    def test_everything_is_allowed_when_the_stock_is_full(self):
        # ``b = 1`` gives ``current == upper``, hence an allowed fraction of 1.
        z = raw(0.9)

        delta = run(make(), {"fisherman:0": z}, level=1.0, quota=0.5)

        assert delivered_of(delta, "fisherman:0", z) == pytest.approx(
            0.9, abs=FLOAT32_ATOL
        )

    def test_delivery_grows_with_the_resource_level(self):
        z = raw(0.9)
        delivered = [
            delivered_of(
                run(make(), {"fisherman:0": z}, level=level, quota=0.5),
                "fisherman:0",
                z,
            )
            for level in np.linspace(0.0, 1.0, 21)
        ]

        assert np.all(np.diff(delivered) >= -FLOAT32_ATOL)
        assert delivered[0] < 0.01
        assert delivered[-1] == pytest.approx(0.9, abs=FLOAT32_ATOL)

    def test_a_higher_quota_lets_less_through(self):
        z = raw(0.9)
        delivered = [
            delivered_of(
                run(make(), {"fisherman:0": z}, level=0.4, quota=quota),
                "fisherman:0",
                z,
            )
            for quota in np.linspace(0.1, 0.9, 9)
        ]

        assert np.all(np.diff(delivered) <= FLOAT32_ATOL)

    def test_resource_level_is_the_state_over_the_capacity(self):
        # The same normalised level reached with another capacity gives the
        # same delivery.
        z = raw(0.9)
        mdp = mdp_with({"fisherman:0": z}, level=0.4)
        mdp.params["K"] = 2 * K
        mdp.state.data["fish"] = [0.4 * 2 * K]

        delta = make()(mdp, mdp.actions[REGULATOR]["quota"][0])

        assert delivered_of(delta, "fisherman:0", z) == pytest.approx(
            delivered_fraction(0.9, allowed_fraction(0.4, 0.5)), abs=FLOAT32_ATOL
        )


@pytest.mark.unit
class TestDelivery:
    @pytest.mark.parametrize("request_fraction", [0.01, 0.1, 0.4])
    def test_request_below_the_allowance_is_left_unchanged(self, request_fraction):
        # A stock at the quota allows 1/2, ``softplus0`` is zero far below the
        # allowance and the residual vanishes.
        delta = run(make(), {"fisherman:0": raw(request_fraction)}, level=0.5)

        assert delta.actions["fisherman:0"]["harvest"][0] == pytest.approx(
            np.zeros(1), abs=1e-3
        )

    def test_request_above_the_allowance_is_capped_and_lowered(self):
        z = raw(0.9)

        delta = run(make(), {"fisherman:0": z}, level=0.5)

        delivered = delivered_of(delta, "fisherman:0", z)
        assert delivered < 0.9
        assert delivered == pytest.approx(
            0.5 + USAGE_WIDTH * math.log(2.0), abs=FLOAT32_ATOL
        )
        assert delta.actions["fisherman:0"]["harvest"][0][0] < 0

    def test_cap_is_continuous_around_the_allowance(self):
        requests = np.linspace(0.45, 0.55, 201)
        delivered = [
            delivered_of(
                run(make(), {"fisherman:0": raw(float(r))}, level=0.5),
                "fisherman:0",
                raw(float(r)),
            )
            for r in requests
        ]

        assert np.max(np.abs(np.diff(delivered))) < 1e-3
        assert np.all(np.diff(delivered) >= -FLOAT32_ATOL)

    def test_residual_keeps_the_shape_and_leaves_other_components_alone(self):
        z = np.array([raw(0.9)[0], 1.5, -2.0], dtype=np.float32)

        delta = run(make(), {"fisherman:0": z}, level=0.5)

        residual = delta.actions["fisherman:0"]["harvest"][0]
        assert residual.shape == z.shape
        assert residual[0] < 0
        np.testing.assert_array_equal(residual[1:], np.zeros(2, dtype=np.float32))

    def test_request_is_not_mutated_in_place(self):
        z = raw(0.9)
        original = z.copy()

        run(make(), {"fisherman:0": z}, level=0.5)

        np.testing.assert_array_equal(z, original)

    def test_each_agent_is_capped_on_its_own_request(self):
        requests = {"fisherman:0": raw(0.2), "fisherman:1": raw(0.9)}

        delta = run(make(), requests, level=0.5)

        assert delivered_of(delta, "fisherman:0", requests["fisherman:0"]) == (
            pytest.approx(0.2, abs=1e-3)
        )
        assert delivered_of(delta, "fisherman:1", requests["fisherman:1"]) == (
            pytest.approx(delivered_fraction(0.9, 0.5), abs=FLOAT32_ATOL)
        )

    def test_extreme_request_is_floored_before_the_logit(self):
        # A request of effectively zero is floored at 1e-6 before it is mapped
        # back to ``z``; the delivered fraction is that floor.
        z = np.array([-400.0], dtype=np.float32)

        delta = run(make(), {"fisherman:0": z}, level=0.5)

        delivered = z + delta.actions["fisherman:0"]["harvest"][0]
        assert logistic(float(delivered.item()) / TEMPERATURE) == pytest.approx(
            1e-6, rel=1e-3
        )


@pytest.mark.unit
class TestTargeting:
    def test_only_agents_of_the_targeted_type_are_capped(self):
        requests = {"fisherman:0": raw(0.9), "farmer:0": raw(0.9)}

        delta = run(make(), requests, level=0.5)

        assert set(delta.actions.data) == {"fisherman:0"}

    def test_identifier_must_start_with_the_type_and_a_colon(self):
        requests = {"fisherman:0": raw(0.9), "fishermen:0": raw(0.9)}

        delta = run(make(), requests, level=0.5)

        assert set(delta.actions.data) == {"fisherman:0"}

    def test_the_regulator_is_not_a_target(self):
        delta = run(make(), {"fisherman:0": raw(0.9)}, level=0.5)

        assert REGULATOR not in delta.actions.data

    def test_agents_without_the_targeted_mechanism_are_skipped(self):
        mdp = mdp_with({"fisherman:0": raw(0.9)})
        mdp.actions.data["fisherman:1"] = {"restore": raw(0.9)}

        delta = make()(mdp, mdp.actions[REGULATOR]["quota"][0])

        assert set(delta.actions.data) == {"fisherman:0"}

    def test_the_targeted_mechanism_is_the_one_named(self):
        mdp = mdp_with({"fisherman:0": raw(0.9)})
        mdp.actions.data["fisherman:0"]["restore"] = raw(0.9)

        delta = make(acts_on=("fisherman", "restore"))(
            mdp, mdp.actions[REGULATOR]["quota"][0]
        )

        assert set(delta.actions["fisherman:0"].keys()) == {"restore"}

    def test_no_target_gives_no_residual(self):
        delta = run(make(), {}, level=0.5)

        assert delta.actions.data == {}

    def test_rewards_and_observations_are_untouched(self):
        delta = run(make(), {"fisherman:0": raw(0.9)}, level=0.5)

        assert delta.rewards.data == {}
        assert delta.obs.data == {}


@pytest.mark.unit
class TestComposition:
    def test_follower_decodes_the_capped_action(self):
        # The regulator's residual is added to the follower's raw output, so
        # decoding it with the follower's own temperature gives the cap.
        regulator = Agent(id=REGULATOR, policy_id="p", mechanisms={"quota": make()})
        z = raw(0.9)
        mdp = regulator.action(mdp_with({"fisherman:0": z}, level=0.5))

        capped = mdp.actions["fisherman:0"]["harvest"][0]

        assert logistic(float(capped.item()) / TEMPERATURE) == pytest.approx(
            delivered_fraction(0.9, 0.5), abs=FLOAT32_ATOL
        )

    def test_regulator_without_a_candidate_applies_nothing(self):
        regulator = Agent(id=REGULATOR, policy_id="p", mechanisms={"quota": make()})
        mdp = MDPState(
            params={"K": K},
            state={"fish": 0.5 * K},
            actions={"fisherman:0": {"harvest": raw(0.9)}},
        )

        after = regulator.action(mdp)

        np.testing.assert_array_equal(
            after.actions["fisherman:0"]["harvest"][0], raw(0.9)
        )


@pytest.mark.unit
class TestAllowedFractionEntry:
    def test_the_entry_holds_the_allowed_fraction_of_a_fresh_state(self):
        mdp = mdp_with({"fisherman:0": raw(0.9)}, level=0.4, quota=0.5)

        composed = mdp.add(make()(mdp, mdp.actions[REGULATOR]["quota"][0]))

        assert composed.state["allowed_frac"][0] == pytest.approx(
            allowed_fraction(0.4, 0.5)
        )

    def test_the_entry_replaces_the_value_already_in_the_state(self):
        mdp = mdp_with({"fisherman:0": raw(0.9)}, level=0.4, quota=0.5)
        mdp.state.data["allowed_frac"] = [0.7]

        composed = mdp.add(make()(mdp, mdp.actions[REGULATOR]["quota"][0]))

        assert composed.state["allowed_frac"][0] == pytest.approx(
            allowed_fraction(0.4, 0.5)
        )

    def test_the_entry_follows_the_resource_level_over_the_steps(self):
        # The residual is added to the entry like every state delta, so it
        # must be the difference with the entry: the values would otherwise
        # pile up from step to step.
        regulator = Agent(id=REGULATOR, policy_id="p", mechanisms={"quota": make()})
        levels = [0.8, 0.5, 0.2, 0.2]
        mdp = mdp_with({"fisherman:0": raw(0.9)}, level=levels[0], quota=0.5)

        recorded = []
        for step, level in enumerate(levels):
            if step:
                mdp = mdp.advance(state={"fish": level * K})
                mdp = mdp.update(actions={"fisherman:0": {"harvest": raw(0.9)}})
            mdp = regulator.action(mdp)
            recorded.append(mdp.state["allowed_frac"][mdp.t])

        assert recorded == pytest.approx(
            [allowed_fraction(level, 0.5) for level in levels]
        )
