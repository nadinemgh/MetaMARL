"""Fully connected RLlib model that runs its forward pass on Apple's MPS device.

``MPSFullyConnectedNetwork`` wraps RLlib's old-API-stack
``FullyConnectedNetwork`` so that the observations are moved to the Metal
Performance Shaders (MPS) device for the forward pass and the logits come back
to the CPU. ``MPS_DEVICE`` is the device resolved at import time (``None`` when
MPS is unavailable, in which case the wrapper runs on the CPU). The inner
optimizer of the bilevel framework uses RLModules on the new API stack and does
not use this model; it is kept for the old-stack examples (``examples/cartpole``).
"""

import torch
from gymnasium.spaces import Space
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.utils.annotations import override
from ray.rllib.utils.typing import ModelConfigDict, TensorType

MPS_DEVICE = torch.device("mps") if torch.backends.mps.is_available() else None


class MPSFullyConnectedNetwork(TorchModelV2, torch.nn.Module):
    """Fully connected network that evaluates its layers on the MPS device.

    The base network is built by RLlib's ``FullyConnectedNetwork`` and moved to
    ``MPS_DEVICE`` (or to the CPU when MPS is unavailable). ``forward`` copies
    the observations to that device, runs the base network and returns the
    logits on the CPU, since the rest of the RLlib pipeline runs there; the
    value-branch output is cached for ``value_function``.

    Parameters
    ----------
    obs_space : gymnasium.spaces.Space
        Observation space of the policy.
    action_space : gymnasium.spaces.Space
        Action space of the policy.
    num_outputs : int
        Number of logits the network produces (for example the number of
        discrete actions, or twice the action dimension for a Gaussian head).
    model_config : ModelConfigDict
        RLlib model configuration (``fcnet_hiddens``, ``fcnet_activation``...),
        passed to the base network unchanged.
    name : str
        Model name; the base network is named ``name + "_base"``.

    Attributes
    ----------
    device : torch.device
        ``MPS_DEVICE`` when MPS is available, else the CPU.

    When to use: with old-API-stack RLlib on an Apple Silicon machine, when the
    network is large enough for the transfer to the GPU to pay off. For the
    small networks of the fishery example the CPU is usually enough.

    Examples
    --------
    >>> import numpy as np
    >>> from gymnasium.spaces import Box, Discrete
    >>> from ray.rllib.models.catalog import MODEL_DEFAULTS
    >>> model_config = {**MODEL_DEFAULTS, "fcnet_hiddens": [8]}
    >>> model = MPSFullyConnectedNetwork(
    ...     Box(-1.0, 1.0, shape=(4,), dtype=np.float32),
    ...     Discrete(3),
    ...     3,
    ...     model_config,
    ...     name="fc",
    ... )
    >>> logits, state = model({"obs": np.zeros((2, 4), dtype=np.float32)}, [], None)
    >>> tuple(logits.shape), logits.device.type, state
    ((2, 3), 'cpu', [])
    >>> tuple(model.value_function().shape)
    (2,)
    """

    def __init__(
        self,
        obs_space: Space,
        action_space: Space,
        num_outputs: int,
        model_config: ModelConfigDict,
        name: str,
    ):
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        torch.nn.Module.__init__(self)

        self.device = MPS_DEVICE or torch.device("cpu")
        self._base_model = FullyConnectedNetwork(
            obs_space, action_space, num_outputs, model_config, name + "_base"
        )

        self._base_model.to(self.device)

        self._last_value = None

    @override(TorchModelV2)
    def forward(
        self,
        input_dict: dict[str, TensorType],
        state: list[TensorType],
        seq_lens: TensorType,
    ) -> tuple[TensorType, list[TensorType]]:
        """Run the wrapped fully connected network on the MPS device.

        Parameters
        ----------
        input_dict : dict[str, TensorType]
            RLlib input dict; ``"obs"`` is converted to a tensor if needed and
            copied to ``self.device``, and ``"obs_flat"`` is set to the same
            tensor because ``FullyConnectedNetwork`` reads that key.
        state : list[TensorType]
            RNN state (unused by the FC net, passed through).
        seq_lens : TensorType
            Sequence lengths (passed through).

        Returns
        -------
        tuple[TensorType, list[TensorType]]
            ``(logits, new_state)`` with logits moved back to CPU, since the
            rest of the RLlib pipeline runs there. The value branch output is
            cached for ``value_function``.
        """

        obs = input_dict["obs"]

        if not isinstance(obs, torch.Tensor):
            obs = torch.as_tensor(obs)

        obs_mps = obs.to(self.device)
        input_dict_mps = {**input_dict, "obs": obs_mps, "obs_flat": obs_mps}
        output, new_state = self._base_model(input_dict_mps, state, seq_lens)

        # Cache value for value_function() call
        self._last_value = self._base_model.value_function()

        # Return to CPU for RLlib
        return output.cpu(), new_state

    @override(TorchModelV2)
    def value_function(self) -> TensorType:
        """Return the value estimate cached by the last ``forward`` call.

        Returns
        -------
        TensorType
            Value tensor of shape ``[B]`` (reward units) moved to CPU.

        Raises
        ------
        ValueError
            If ``forward`` has not been called yet.
        """

        if self._last_value is None:
            raise ValueError("forward() must be called before value_function()")

        return self._last_value.cpu()
