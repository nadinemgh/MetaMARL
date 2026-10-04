"""Unit tests for ``core.adaptors.ray.mps_model.MPSFullyConnectedNetwork``.

The wrapper builds an old-API-stack ``FullyConnectedNetwork`` on the resolved
device and moves observations in and logits out. ``MPS_DEVICE`` is patched to
``None`` in most tests so the forward pass runs on CPU regardless of the host;
one test keeps the module-level device to check that outputs come back on CPU
either way. The test that needs a real MPS device is skipped on hosts without
one, so the file passes with and without Apple MPS. No Ray runtime is involved:
the model is a plain ``torch.nn.Module``.

Copied from the August suite; the last two tests are new.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from gymnasium.spaces import Box, Discrete
from ray.rllib.models.catalog import MODEL_DEFAULTS

import core.adaptors.ray.mps_model as mps_model_module
from core.adaptors.ray.mps_model import MPSFullyConnectedNetwork

OBS_SPACE = Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)
ACT_SPACE = Discrete(3)


def make_model(hidden=(8,)) -> MPSFullyConnectedNetwork:
    model_config = dict(MODEL_DEFAULTS)
    model_config["fcnet_hiddens"] = list(hidden)
    return MPSFullyConnectedNetwork(
        OBS_SPACE, ACT_SPACE, ACT_SPACE.n, model_config, name="test_model"
    )


@pytest.fixture
def cpu_device(monkeypatch):
    """Force the wrapper onto CPU so the assertions are host-independent."""
    monkeypatch.setattr(mps_model_module, "MPS_DEVICE", None)


@pytest.mark.unit
def test_init_falls_back_to_cpu_when_mps_unavailable(cpu_device):
    model = make_model()
    assert model.device == torch.device("cpu")
    assert model._last_value is None
    assert model.num_outputs == ACT_SPACE.n
    assert model._base_model.name == "test_model_base"


@pytest.mark.unit
def test_module_device_matches_torch_backend():
    expected = torch.device("mps") if torch.backends.mps.is_available() else None
    assert mps_model_module.MPS_DEVICE == expected


@pytest.mark.unit
def test_forward_accepts_numpy_obs_and_returns_cpu_logits(cpu_device):
    model = make_model()
    obs = np.zeros((2, 4), dtype=np.float32)

    logits, state = model({"obs": obs}, [], None)

    assert isinstance(logits, torch.Tensor)
    assert logits.device.type == "cpu"
    assert logits.shape == (2, ACT_SPACE.n)
    assert state == []
    assert model._last_value is not None


@pytest.mark.unit
def test_forward_accepts_tensor_obs_and_caches_value(cpu_device):
    model = make_model()
    obs = torch.ones((3, 4), dtype=torch.float32)

    logits, _ = model({"obs": obs, "extra": 1}, ["s"], torch.tensor([1, 1, 1]))
    value = model.value_function()

    assert logits.shape == (3, ACT_SPACE.n)
    assert value.shape == (3,)
    assert value.device.type == "cpu"
    # Same input, same output: the wrapper adds no stochasticity.
    logits2, _ = model({"obs": obs}, [], None)
    assert torch.equal(logits, logits2)


@pytest.mark.unit
def test_value_function_requires_forward_first(cpu_device):
    model = make_model()
    with pytest.raises(ValueError, match="forward\\(\\) must be called"):
        model.value_function()


@pytest.mark.unit
def test_forward_on_module_device_returns_cpu_tensors():
    # Uses whatever device the host resolved at import time (MPS on Apple
    # Silicon, CPU elsewhere); the contract is that outputs come back on CPU.
    model = make_model()
    logits, _ = model({"obs": np.ones((1, 4), dtype=np.float32)}, [], None)
    assert logits.device.type == "cpu"
    assert model.value_function().device.type == "cpu"
    assert model.device == (mps_model_module.MPS_DEVICE or torch.device("cpu"))


@pytest.mark.unit
def test_forward_hands_the_base_model_obs_and_obs_flat_on_the_device(cpu_device):
    class RecordingBase(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.seen = None

        def forward(self, input_dict, state, seq_lens):
            self.seen = (input_dict, state, seq_lens)
            return torch.zeros((2, ACT_SPACE.n)), ["new"]

        def value_function(self):
            return torch.full((2,), 7.0)

    model = make_model()
    base = RecordingBase()
    model._base_model = base
    obs = np.ones((2, 4), dtype=np.float32)

    logits, new_state = model({"obs": obs, "extra": "kept"}, ["s"], "lens")

    input_dict, state, seq_lens = base.seen
    assert input_dict["obs"] is input_dict["obs_flat"]
    assert torch.equal(input_dict["obs"], torch.from_numpy(obs))
    assert input_dict["extra"] == "kept"
    assert (state, seq_lens) == (["s"], "lens")
    assert new_state == ["new"]
    assert logits.shape == (2, ACT_SPACE.n)
    assert torch.equal(model.value_function(), torch.full((2,), 7.0))


@pytest.mark.unit
@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires an Apple MPS device"
)
def test_parameters_live_on_mps_while_outputs_return_on_cpu():
    model = make_model()

    assert model.device.type == "mps"
    assert all(p.device.type == "mps" for p in model._base_model.parameters())
    logits, _ = model({"obs": np.ones((1, 4), dtype=np.float32)}, [], None)
    assert logits.device.type == "cpu"
    assert model.value_function().device.type == "cpu"
