"""Common random numbers across the mechanism slots of one policy seed.

The inner optimizer trains one RLModule per ``(mechanism slot, policy seed)``
pair, and the outer ES compares the fitness of the slots of one generation.
That comparison measures the effect of the candidate mechanism only if the
slots of one policy seed share every other source of randomness, which is the
method of common random numbers (Glasserman & Yao, 1992,
https://doi.org/10.1287/mnsc.38.6.884). The environments already share it,
since every training environment of a seed is seeded with that seed. Two
sources did not, as measured on the fishery on 2026-10-05: while the quota did
not bind, the fitness of a slot depended only on its index, with a spread of
about 0.036 in the full configuration, the order of the mechanism's own
effect.

The first source is the initial weights of the output layers. RLlib's
``PPOCatalog`` (also used by APPO) builds the pi and vf heads without the
``head_fcnet_*_initializer`` of the model config, so those layers took
torch's default initialisation from the global stream, one module after the
other. :class:`SeededHeadsPPOCatalog` passes the configured initializers to
the hidden and output layers of both heads, and :class:`SeededXavierUniform`
gives the heads a seed stream of their own and restarts at every build, so
that the learner's and the env runners' copies of a module agree.

The second source is the exploration draws. RLlib's ``GetActions`` connector
samples every module's actions from the global torch stream, so each slot saw
different draws. Here the draws are keyed instead: :class:`AddExplorationKeys`,
the first piece of the env-to-module pipeline, gives every acting agent a key
derived from the policy seed, the env runner, the step and the agent
(:func:`exploration_key`), and the exploration forward pass of
:class:`CommonRandomPPOTorchRLModule` and :class:`CommonRandomAPPOTorchRLModule`
draws each row from a generator seeded with its key. ``GetActions`` then finds
the actions in place and leaves them. Two slots of one seed therefore receive
the same draws at the same step, while two seeds, two env runners or two
agents receive different ones.

Equal keys give equal draws only if the slots of one seed step in lockstep,
which the vectorised layout of the env runner guarantees: one connector call
per vector step, so the step counter is shared by every slot of the runner.
The examples evaluate with ``explore=False``, which adds no key and leaves
the greedy actions to RLlib; an evaluation that explores keys its draws in
the same way, each evaluation runner hosting every slot of its policy seed.
"""

from __future__ import annotations

import dataclasses
import zlib
from collections.abc import Callable
from typing import Any, Optional

import numpy as np
import torch
import tree
from ray.rllib.algorithms.appo.torch.default_appo_torch_rl_module import (
    DefaultAPPOTorchRLModule,
)
from ray.rllib.algorithms.ppo.ppo_catalog import PPOCatalog
from ray.rllib.algorithms.ppo.torch.default_ppo_torch_rl_module import (
    DefaultPPOTorchRLModule,
)
from ray.rllib.connectors.connector_v2 import ConnectorV2
from ray.rllib.core.columns import Columns
from ray.rllib.core.distribution.torch.torch_distribution import (
    TorchDiagGaussian,
    TorchMultiDistribution,
)
from ray.rllib.core.models.base import Model

from core.adaptors.ray.utils import parse_learner_id

#: Batch column holding the exploration key of each row (``int64``).
EXPLORATION_KEY = "exploration_key"

#: Seed stream of the pi and vf heads, distinct from the encoder's.
HEAD_STREAM = "heads"

_UINT64 = (1 << 64) - 1


def _derive_seed(*entropy: int) -> int:
    """Hash integers into a seed in ``[0, 2**63)``, valid for torch and int64."""
    sequence = np.random.SeedSequence([int(value) & _UINT64 for value in entropy])
    return int(sequence.generate_state(1, dtype=np.uint64)[0] >> np.uint64(1))


def _stream_code(name: str) -> int:
    """Return the CRC-32 of a name, which unlike ``hash`` ignores PYTHONHASHSEED."""
    return zlib.crc32(str(name).encode("utf-8"))


class SeededXavierUniform:
    """Xavier-uniform initializer that seeds every layer from a base seed.

    The ``i``-th call since the last :meth:`reset` seeds torch's CPU generator
    with ``seed + i`` (no stream) or with a hash of ``(seed, stream, i)``,
    applies ``torch.nn.init.xavier_uniform_`` (Glorot & Bengio, 2010,
    https://proceedings.mlr.press/v9/glorot10a.html) and restores the
    generator, so the global stream is left untouched. Two modules built from
    the same seed and the same layer order receive identical weights.

    Parameters
    ----------
    seed : int
        Base seed, the policy seed of the module.
    stream : str or None, optional
        ``None`` for the encoder layers, whose seeds stay ``seed + i``;
        :data:`HEAD_STREAM` for the heads, so that a head layer never replays
        the uniform draws of an encoder layer (Xavier-uniform weights are a
        scaled copy of those draws).

    When to use: built by ``RayOptimizerConfig._seeded_xavier_uniform`` for
    the ``fcnet_*`` and ``head_fcnet_*`` initializers of each module.
    :class:`SeededHeadsPPOCatalog` calls :meth:`reset` before every build,
    which makes a rebuild (learner, env runners, pickled copies) reproduce
    the first build.

    Examples
    --------
    >>> init = SeededXavierUniform(123)
    >>> a, b = torch.empty(4, 3), torch.empty(4, 3)
    >>> init(a)
    >>> init.reset()
    >>> init(b)
    >>> torch.equal(a, b)
    True
    """

    def __init__(self, seed: int, *, stream: Optional[str] = None) -> None:
        self.seed = int(seed)
        self.stream = stream
        self.calls = 0

    def reset(self) -> None:
        """Restart the layer count, so the next call reseeds layer 0."""
        self.calls = 0

    def layer_seed(self, index: int) -> int:
        """Return the torch seed of the ``index``-th layer."""
        if self.stream is None:
            return self.seed + index

        return _derive_seed(self.seed, _stream_code(self.stream), index)

    def __call__(self, tensor: torch.Tensor, **kwargs: Any) -> None:
        """Initialise ``tensor`` in place and advance the layer count."""
        layer_seed = self.layer_seed(self.calls)
        self.calls += 1

        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(layer_seed)
            torch.nn.init.xavier_uniform_(tensor, **kwargs)

    def __repr__(self) -> str:
        return f"SeededXavierUniform(seed={self.seed}, stream={self.stream!r})"


def exploration_key(
    *, policy_seed: int, worker_index: int, step: int, agent_id: str
) -> int:
    """Derive the exploration key of one agent at one step.

    Parameters
    ----------
    policy_seed : int
        Training seed of the module the agent maps to.
    worker_index : int
        Index of the env runner (0 for the local runner).
    step : int
        Number of exploring env-to-module calls this runner made before this
        one, i.e. the vector step.
    agent_id : str
        Agent identifier, e.g. ``"fisherman:0"``.

    Returns
    -------
    int
        A key in ``[0, 2**63)``, usable as a torch seed and as an ``int64``.

    When to use: by :class:`AddExplorationKeys`; the mechanism slot is not an
    argument, which is what makes the draws common to the slots of one seed.

    Notes
    -----
    The four values go through ``numpy.random.SeedSequence``, whose hashing
    spreads nearby inputs over the whole output range (negative values enter
    modulo ``2**64``). The agent identifier enters through its CRC-32, which,
    unlike ``hash``, does not depend on ``PYTHONHASHSEED``.

    Examples
    --------
    >>> key = exploration_key(policy_seed=11, worker_index=0, step=4, agent_id="f:0")
    >>> key == exploration_key(policy_seed=11, worker_index=0, step=4, agent_id="f:0")
    True
    >>> key == exploration_key(policy_seed=11, worker_index=0, step=5, agent_id="f:0")
    False
    """
    return _derive_seed(policy_seed, worker_index, step, _stream_code(agent_id))


class AddExplorationKeys(ConnectorV2):
    """Env-to-module piece that adds an exploration key to every acting agent.

    The key of an agent is :func:`exploration_key` of the policy seed of its
    module (read from the module id ``<policy>_m<slot>_s<seed>``; for an
    unseeded module, ``_sNone``, a seed drawn once from OS entropy when the
    piece is built, so the draws stay common to the slots of the run without
    making it reproducible), the index of
    the env runner (read from the sub-environments, see
    ``RLlibMultiAgentEnvAdapter.worker_index``), the number of exploring calls
    this piece has seen and the agent id. It is added as the batch column
    :data:`EXPLORATION_KEY`, one item per agent, so RLlib's default pieces map
    it to the agent's module and batch it alongside the observations.

    Parameters
    ----------
    env : gymnasium.vector.VectorEnv or None, optional
        The env runner's vectorised environment; its first sub-environment
        gives the runner index. ``None`` for a runner that hosts no
        environment: the piece then passes non-exploring batches through and
        refuses exploring ones.
    **kwargs
        Passed to ``ConnectorV2`` (observation and action spaces).

    Raises
    ------
    ValueError
        If the sub-environment of ``env`` has no ``worker_index``.
    RuntimeError
        On an exploring call, if the piece was built without environment.

    When to use: installed by ``RayOptimizerConfig._apply_agents_to_rllib`` as
    the first env-to-module piece; it is not meant to be added by hand.

    Examples
    --------
    >>> from types import SimpleNamespace
    >>> env = SimpleNamespace(
    ...     envs=[SimpleNamespace(unwrapped=SimpleNamespace(worker_index=0))]
    ... )
    >>> piece = AddExplorationKeys(env=env)
    >>> piece(rl_module=None, batch={}, episodes=[], explore=False)
    {}
    """

    def __init__(self, env: Any = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        # A local runner that hosts no environment (remote runners sample
        # instead) builds its pipeline with env=None; it never explores, and
        # RLlib does not rebuild the pipeline if an environment comes later.
        self.worker_index: Optional[int] = None

        if env is not None:
            worker_index = getattr(env.envs[0].unwrapped, "worker_index", None)

            if worker_index is None:
                raise ValueError(
                    "The sub-environments carry no `worker_index`; build them "
                    + "with RayOptimizerConfig, whose env creator records it."
                )

            self.worker_index = int(worker_index)

        self.step = 0
        self.unseeded_seed = int(np.random.SeedSequence().generate_state(1)[0])

    def __call__(
        self,
        *,
        rl_module: Any,
        batch: dict[str, Any],
        episodes: list[Any],
        explore: Optional[bool] = None,
        shared_data: Optional[dict] = None,
        **kwargs: Any,
    ) -> Any:
        """Add one key per acting agent when exploring, then advance the step.

        Parameters
        ----------
        rl_module : RLModule
            Unused.
        batch : dict
            Batch being built by the env-to-module pipeline.
        episodes : list of MultiAgentEpisode
            Ongoing episodes, one per sub-environment.
        explore : bool or None, optional
            Keys are added only when ``True``; otherwise the batch is returned
            unchanged and the step is not advanced.
        shared_data : dict or None, optional
            Unused.

        Returns
        -------
        dict
            The batch, with the column :data:`EXPLORATION_KEY` when exploring.
        """
        if not explore:
            return batch

        if self.worker_index is None:
            raise RuntimeError(
                "AddExplorationKeys was built without an environment, so it "
                + "cannot key the exploration draws of this env runner."
            )

        # The slot is not part of the key, so the M slots of a seed share each
        # key: derive it once per (seed, agent) rather than once per episode.
        keys: dict[tuple[str, Any], np.int64] = {}

        for sa_episode in self.single_agent_episode_iterator(
            episodes, agents_that_stepped_only=True
        ):
            _, _, policy_seed = parse_learner_id(sa_episode.module_id)
            identity = (policy_seed, sa_episode.agent_id)
            key = keys.get(identity)

            if key is None:
                key = keys[identity] = np.int64(
                    exploration_key(
                        policy_seed=self.unseeded_seed
                        if policy_seed == "None"
                        else int(policy_seed),
                        worker_index=self.worker_index,
                        step=self.step,
                        agent_id=sa_episode.agent_id,
                    )
                )

            self.add_batch_item(
                batch, EXPLORATION_KEY, item_to_add=key, single_agent_episode=sa_episode
            )

        self.step += 1

        return batch


def with_exploration_keys(user_connector: Optional[Callable] = None) -> Callable:
    """Build an ``env_to_module_connector`` that puts the keys first.

    Parameters
    ----------
    user_connector : callable or None, optional
        The ``env_to_module_connector`` already configured, if any. It is
        called as RLlib would call it, ``(env, spaces, device)`` with a
        fallback to the older ``(env)`` signature, and its pieces follow
        :class:`AddExplorationKeys`.

    Returns
    -------
    callable
        ``(env, spaces=None, device=None) -> list of ConnectorV2``.

    When to use: by ``RayOptimizerConfig._apply_agents_to_rllib``.

    Examples
    --------
    >>> from types import SimpleNamespace
    >>> env = SimpleNamespace(
    ...     envs=[SimpleNamespace(unwrapped=SimpleNamespace(worker_index=0))]
    ... )
    >>> [type(p).__name__ for p in with_exploration_keys()(env)]
    ['AddExplorationKeys']
    """

    def env_to_module_connector(
        env: Any, spaces: Any = None, device: Any = None
    ) -> list[Any]:
        """Return the key piece followed by the user's pieces."""
        pieces: list[Any] = [AddExplorationKeys(env=env)]

        if user_connector is None:
            return pieces

        try:
            extra = user_connector(env, spaces, device)
        except TypeError as error:
            if "positional argument" not in str(error):
                raise
            extra = user_connector(env)

        if isinstance(extra, (list, tuple)):
            pieces.extend(extra)
        else:
            pieces.append(extra)

        return pieces

    return env_to_module_connector


class SeededHeadsPPOCatalog(PPOCatalog):
    """``PPOCatalog`` whose pi and vf heads use the configured initializers.

    RLlib 2.53's ``PPOCatalog`` reads ``head_fcnet_hiddens`` and
    ``head_fcnet_activation`` but not ``head_fcnet_kernel_initializer`` nor
    ``head_fcnet_bias_initializer``, so the heads were initialised from
    torch's global stream. This catalog gives those two initializers to the
    hidden and the output layers of both heads. The pi head config is only
    known once the action distribution is, so it is first built by the parent
    inside a forked RNG (leaving the global stream as it was), then rebuilt
    from the completed config. The vf head config is complete from the start
    and is completed with the initializers when the head is built.

    When to use: set as ``catalog_class`` of every module spec by
    ``RayOptimizerConfig._apply_agents_to_rllib``; works for PPO and APPO.
    """

    def __init__(
        self,
        observation_space: Any,
        action_space: Any,
        model_config_dict: dict,
        **kwargs: Any,
    ) -> None:
        # One catalog is built per module build, before any layer exists:
        # restarting the seeded initializers here makes every build replay
        # the same layer seeds.
        for initializer in model_config_dict.values():
            if isinstance(initializer, SeededXavierUniform):
                initializer.reset()

        super().__init__(observation_space, action_space, model_config_dict, **kwargs)

    def _head_initializers(self) -> dict[str, Any]:
        """Return the initializer fields of ``MLPHeadConfig`` from the model config."""
        kernel = self._model_config_dict.get("head_fcnet_kernel_initializer")
        bias = self._model_config_dict.get("head_fcnet_bias_initializer")
        fields: dict[str, Any] = {}

        if kernel is not None:
            fields["hidden_layer_weights_initializer"] = kernel
            fields["output_layer_weights_initializer"] = kernel

        if bias is not None:
            fields["hidden_layer_bias_initializer"] = bias
            fields["output_layer_bias_initializer"] = bias

        return fields

    def build_vf_head(self, framework: str) -> Model:
        """Build the vf head with the configured initializers.

        Parameters
        ----------
        framework : str
            ``"torch"``.

        Returns
        -------
        Model
            The value head.
        """
        self.vf_head_config = dataclasses.replace(
            self.vf_head_config, **self._head_initializers()
        )

        return self.vf_head_config.build(framework=framework)

    def build_pi_head(self, framework: str) -> Model:
        """Build the pi head with the configured initializers.

        Parameters
        ----------
        framework : str
            ``"torch"``.

        Returns
        -------
        Model
            The policy head.
        """
        with torch.random.fork_rng(devices=[]):
            super().build_pi_head(framework=framework)

        self.pi_head_config = dataclasses.replace(
            self.pi_head_config, **self._head_initializers()
        )

        return self.pi_head_config.build(framework=framework)


def _gaussian_leaves(dist: Any) -> Optional[list[TorchDiagGaussian]]:
    """Return the diagonal-Gaussian leaves of ``dist``, or ``None`` if any is not."""
    if isinstance(dist, TorchDiagGaussian):
        return [dist]

    if isinstance(dist, TorchMultiDistribution):
        leaves = dist._flat_child_distributions
        if all(isinstance(leaf, TorchDiagGaussian) for leaf in leaves):
            return list(leaves)

    return None


def _keyed_gaussian_actions(
    dist: Any, leaves: list[TorchDiagGaussian], key_list: list[int]
) -> Any:
    """Draw ``loc + scale * noise`` with one keyed standard-normal row per key.

    The noise of row ``i`` is ``torch.randn`` over all the action dimensions,
    in the flattened order of the leaves, from a private generator seeded with
    ``key_list[i]``; the global generators are not touched.
    """
    first = leaves[0]._dist.loc
    generator = torch.Generator(device=first.device)
    widths = [leaf._dist.loc[0].numel() for leaf in leaves]
    noise_rows = []

    for key in key_list:
        generator.manual_seed(key)
        noise_rows.append(
            torch.randn(
                sum(widths), generator=generator, dtype=first.dtype, device=first.device
            )
        )

    noise = torch.stack(noise_rows)
    actions = []
    offset = 0

    for leaf, width in zip(leaves, widths):
        loc, scale = torch.broadcast_tensors(leaf._dist.loc, leaf._dist.scale)
        leaf_noise = noise[:, offset : offset + width].reshape(loc.shape)
        actions.append(loc + scale * leaf_noise)
        offset += width

    if isinstance(dist, TorchMultiDistribution):
        return tree.unflatten_as(dist._original_struct, actions)

    return actions[0]


class CommonRandomExploration:
    """Mixin drawing exploration actions from keyed generators.

    ``_forward_exploration`` runs the parent's forward pass, then, when the
    batch carries :data:`EXPLORATION_KEY`, draws the action of each row from
    the exploration distribution with a generator seeded by that row's key,
    leaving the global generators untouched. The actions and their
    log-probabilities are written to the output, so RLlib's ``GetActions``
    connector keeps them. Without the column the output is the parent's, and
    ``GetActions`` samples as usual.

    Two paths draw the rows. When every leaf of the distribution is a
    diagonal Gaussian (continuous actions, as in the examples), the
    distribution is built once for the batch, each row receives one keyed
    standard-normal vector from a private generator, and the actions are
    ``loc + scale * noise`` for the whole batch. Any other distribution is
    rebuilt row by row and sampled with torch's generator seeded by the key
    and restored afterwards. In the env runner every small torch call goes
    through a device-mode override, so the number of calls per row sets the
    cost: the row-by-row path made the full fishery's sampling 34 % slower
    than RLlib's own draws (12.8 s against 9.5 s for 4000 env steps).

    When to use: through :class:`CommonRandomPPOTorchRLModule` and
    :class:`CommonRandomAPPOTorchRLModule`.
    """

    def _forward_exploration(self, batch: dict[str, Any], **kwargs: Any) -> dict:
        """Run the forward pass and draw keyed actions when keys are present."""
        output = super()._forward_exploration(batch, **kwargs)
        keys = batch.get(EXPLORATION_KEY)

        if keys is None or Columns.ACTIONS in output:
            return output

        logits = output[Columns.ACTION_DIST_INPUTS]
        dist_class = self.get_exploration_action_dist_cls()
        dist = dist_class.from_logits(logits)
        key_list = [int(key) for key in torch.as_tensor(keys).reshape(-1).tolist()]
        leaves = _gaussian_leaves(dist)

        if leaves is not None:
            actions = _keyed_gaussian_actions(dist, leaves, key_list)
        else:
            actions = self._keyed_rows(dist_class, logits, key_list)

        output[Columns.ACTIONS] = actions
        output[Columns.ACTION_LOGP] = dist.logp(actions)

        return output

    @staticmethod
    def _keyed_rows(dist_class: Any, logits: torch.Tensor, key_list: list[int]) -> Any:
        """Sample each row from its own distribution under its seeded generator."""

        def draw_row(index: int) -> Any:
            return dist_class.from_logits(logits[index : index + 1]).sample()

        rows = []

        if logits.device.type == "cpu":
            # Seed only the CPU generator, which makes these draws, and save
            # its state once per call: torch.manual_seed would also seed the
            # CUDA and XPU generators, whose lazy seeding records a stack
            # trace on every call (1.11 ms against 0.52 ms for 10 rows).
            generator = torch.random.default_generator
            saved = generator.get_state()
            try:
                for index, key in enumerate(key_list):
                    generator.manual_seed(key)
                    rows.append(draw_row(index))
            finally:
                generator.set_state(saved)
        else:
            for index, key in enumerate(key_list):
                # fork_rng saves the CUDA generators unless told the device
                # type, and fails on any other accelerator.
                with torch.random.fork_rng(
                    devices=[logits.device], device_type=logits.device.type
                ):
                    torch.manual_seed(key)
                    rows.append(draw_row(index))

        return tree.map_structure(lambda *parts: torch.cat(parts, dim=0), *rows)


class CommonRandomPPOTorchRLModule(CommonRandomExploration, DefaultPPOTorchRLModule):
    """RLlib's default PPO torch module with keyed exploration draws."""


class CommonRandomAPPOTorchRLModule(CommonRandomExploration, DefaultAPPOTorchRLModule):
    """RLlib's default APPO torch module with keyed exploration draws."""


#: Default module class of each supported algorithm, mapped to its keyed variant.
_COMMON_RANDOM_CLASSES: dict[type, type] = {
    DefaultPPOTorchRLModule: CommonRandomPPOTorchRLModule,
    DefaultAPPOTorchRLModule: CommonRandomAPPOTorchRLModule,
}


def common_random_module_class(default_class: type) -> type:
    """Return the keyed-exploration variant of an algorithm's default module.

    Parameters
    ----------
    default_class : type
        ``module_class`` of the algorithm's default RLModule spec.

    Returns
    -------
    type
        :class:`CommonRandomPPOTorchRLModule` or
        :class:`CommonRandomAPPOTorchRLModule`.

    Raises
    ------
    ValueError
        For any other class: its slots could not share exploration draws, and
        the outer comparison would silently measure their differences.

    When to use: by ``RayOptimizerConfig._apply_agents_to_rllib``.

    Examples
    --------
    >>> common_random_module_class(DefaultPPOTorchRLModule).__name__
    'CommonRandomPPOTorchRLModule'
    """
    try:
        return _COMMON_RANDOM_CLASSES[default_class]
    except KeyError:
        supported = ", ".join(cls.__name__ for cls in _COMMON_RANDOM_CLASSES)
        raise ValueError(
            "No common random exploration variant of "
            + f"{getattr(default_class, '__name__', default_class)!r}; "
            + f"supported default modules: {supported}."
        ) from None
