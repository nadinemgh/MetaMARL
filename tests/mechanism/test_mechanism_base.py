"""``Mechanism``: the interface a concrete mechanism inherits.

``__call__`` decodes the action, writes the decoded value back into the
trajectory and applies it; ``apply`` is abstract and ``observe`` contributes
nothing by default.
"""

import numpy as np
import pytest
from gymnasium import spaces

from core.mechanism.base import MDPState, Mechanism

SPACE = spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)


class Recorder(Mechanism):
    """Mechanism that records what ``apply`` receives."""

    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("action_space", SPACE)
        kwargs.setdefault("acts_on", ("fisherman", "harvest"))
        super().__init__(**kwargs)
        self.received = []

    def decode(self, mdp, action):
        return float(action) * 2.0

    def apply(self, mdp, action):
        self.received.append(action)
        return MDPState(rewards={"fisherman:0": action})


class Abstract(Mechanism):
    def apply(self, mdp, action):
        return super().apply(mdp, action)


@pytest.mark.unit
class TestInterface:
    def test_apply_is_abstract(self):
        with pytest.raises(TypeError, match="abstract"):
            Mechanism(action_space=SPACE, acts_on=("a", "b"))

    def test_the_base_apply_is_not_implemented(self):
        mechanism = Abstract(action_space=SPACE, acts_on=("a", "b"))

        with pytest.raises(NotImplementedError):
            mechanism.apply(MDPState(), 0.0)

    def test_the_default_decode_is_the_identity(self):
        mechanism = Recorder()
        assert Mechanism.decode(mechanism, MDPState(), 0.25) == 0.25

    def test_the_default_observation_is_empty(self):
        residual = Recorder().observe(MDPState(t=3))

        assert residual.obs.data == {}
        assert residual.rewards.data == {}
        assert residual.actions.data == {}
        assert residual.state.data == {}

    def test_the_constructor_keeps_its_arguments(self):
        mechanism = Recorder(
            aid="regulator", id="quota", obs_map={"resource_level": "fish"}
        )

        assert mechanism.aid == "regulator"
        assert mechanism.id == "quota"
        assert mechanism.mechanism_id == "quota"
        assert mechanism.acts_on == ("fisherman", "harvest")
        assert mechanism.obs_map == {"resource_level": "fish"}
        assert mechanism.action_space is SPACE


@pytest.mark.unit
class TestCall:
    def test_call_applies_the_decoded_action(self):
        mechanism = Recorder(aid="regulator", id="quota")
        mdp = MDPState(actions={"regulator": {"quota": np.array([0.5])}})

        residual = mechanism(mdp, 0.25)

        assert mechanism.received == [0.5]
        assert residual.rewards["fisherman:0"] == [0.5]

    def test_call_writes_the_decoded_action_back_at_the_current_step(self):
        mechanism = Recorder(aid="regulator", id="quota")
        mdp = MDPState(actions={"regulator": {"quota": [0.0, 0.0]}}, t=1)

        mechanism(mdp, 0.25)

        assert mdp.actions["regulator"]["quota"] == [0.0, 0.5]


@pytest.mark.unit
class TestDefault:
    def test_a_scalar_default_is_kept(self):
        mechanism = Recorder(default=np.asarray(0.56224, dtype=np.float32))

        assert float(mechanism._u) == pytest.approx(0.56224)

    def test_no_default_gives_none(self):
        assert Recorder()._u is None

    @pytest.mark.xfail(
        strict=True,
        reason="`default if default else None` evaluates the truth of an array",
    )
    def test_a_vector_default_is_kept(self):
        default = np.array([0.2, 0.3], dtype=np.float32)

        mechanism = Recorder(default=default)

        np.testing.assert_array_equal(mechanism._u, default)

    @pytest.mark.xfail(
        strict=True,
        reason="`default if default else None` drops a default that is all zeros",
    )
    def test_a_default_equal_to_zero_is_kept(self):
        default = np.array([0.0], dtype=np.float32)

        mechanism = Recorder(default=default)

        np.testing.assert_array_equal(mechanism._u, default)
