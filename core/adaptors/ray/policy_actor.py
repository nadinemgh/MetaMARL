"""Ray actor that owns the RLlib ``Algorithm`` of the inner optimizer.

``RayOptimizer`` never holds an ``Algorithm`` itself; it drives this actor
through ``.remote()`` calls. Building the algorithm inside the actor keeps the
learner, its env runners and their evaluation counterparts in one process tree
and lets the outer loop rebuild the policy between ES generations without
touching the driver.
"""

from __future__ import annotations

import logging

import numpy as np
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.utils.typing import ResultDict

from core.adaptors.ray.learner_drain import stop_learner_thread, wait_for_learner_thread
from core.adaptors.ray.utils import hash_weights

logger = logging.getLogger(__name__)


@ray.remote(num_cpus=1)
class PolicyActor:
    """Own the RLlib ``Algorithm`` of one inner optimizer.

    The ``Algorithm`` is built from ``algo_config`` inside the actor and never
    leaves it. The weights right after construction are kept in
    ``_init_weights`` so that ``reset`` can restore the same starting point
    at each outer iteration.

    The class is decorated with ``@ray.remote(num_cpus=1)``: callers create it
    with ``PolicyActor.remote(algo_config)`` and invoke its methods with
    ``actor.<method>.remote(...)``, collecting results with ``ray.get``.

    Parameters
    ----------
    algo_config : AlgorithmConfig
        Fully resolved RLlib configuration (environment registered, RLModule
        specs and policy mapping applied). Kept so that ``reset`` can rebuild.

    Attributes
    ----------
    algo_config : AlgorithmConfig
        The configuration the actor was created with.
    algo : Algorithm
        The live RLlib algorithm; replaced by ``reset``.

    When to use: you do not create it directly. ``RayOptimizer`` spawns one per
    inner optimizer and forwards ``train``, ``evaluate``, ``reset`` and ``stop``
    to it, so the algorithm and its workers live outside the driver process.

    Examples
    --------
    The plain Python class behind the decorator can be exercised without a Ray
    runtime when the configuration is a stand-in whose ``build_algo`` returns a
    fake algorithm:

    >>> from types import SimpleNamespace
    >>> algo = SimpleNamespace(
    ...     get_weights=lambda: {"w": 1},
    ...     train=lambda: {"training_iteration": 1},
    ... )
    >>> config = SimpleNamespace(build_algo=lambda: algo)
    >>> actor = PolicyActor.__ray_metadata__.modified_class(config)
    >>> actor.train()
    {'training_iteration': 1}
    >>> actor._init_weights
    {'w': 1}
    """

    def __init__(self, algo_config: AlgorithmConfig):
        # Store config for reset capability
        self.algo_config = algo_config

        # Build Algorithm INSIDE actor (critical)
        self.algo: Algorithm = algo_config.build_algo()

        # Store initial weights
        self._init_weights = self.algo.get_weights()

    def train(self) -> ResultDict:
        """Run one ``Algorithm.train()`` iteration.

        Returns
        -------
        ResultDict
            The raw RLlib result dictionary of the iteration.
        """

        return self.algo.train()

    def evaluate(self) -> ResultDict:
        """Run one ``Algorithm.evaluate()`` pass on the evaluation env runners.

        With the configuration produced by ``RayOptimizerConfig.evaluation``,
        this executes ``_evaluate_with_fixed_duration_once``: exactly one
        episode per ``(mechanism, policy seed, eval seed)`` environment.

        Returns
        -------
        ResultDict
            RLlib evaluation results.

        Raises
        ------
        Exception
            Whatever ``wait_for_learner_thread`` raised on a learner (for
            example ``TimeoutError``), re-raised before evaluating.

        Notes
        -----
        APPO applies its updates on a background learner thread, so
        ``train`` can return before the last one is done, and ``evaluate``
        reads the learner's weights straight away. Waiting for that thread
        first makes the evaluated policy the fully trained one in every run.
        """

        for result in self.algo.learner_group.foreach_learner(wait_for_learner_thread):
            if not result.ok:
                raise result.get()

        return self.algo.evaluate()

    def compute_actions(self, policy_id: str, obs_batch: np.ndarray) -> np.ndarray:
        """Sample actions for a batch of observations from one RLModule.

        Parameters
        ----------
        policy_id : str
            RLModule ID (``<policy>_m<mechanism_idx>_s<seed>``).
        obs_batch : numpy.ndarray
            Observations, shape ``(B, obs_dim)``.

        Returns
        -------
        numpy.ndarray
            Actions, shape ``(B, act_dim)``. The new API stack path samples
            from the inference distribution; if it fails, the old-stack
            ``Policy.compute_single_action`` fallback is used without
            exploration.
        """

        try:
            module = self.algo.get_module(policy_id)
            out = module.forward_inference({"obs": obs_batch})
            dist_cls = module.get_inference_action_dist_cls()
            dist = dist_cls.from_logits(out["action_dist_inputs"])

            return dist.sample().cpu().numpy()
        except Exception:
            policy = self.algo.get_policy(policy_id)
            actions = []

            for obs in obs_batch:
                a, _, _ = policy.compute_single_action(obs, explore=False)

                actions.append(a)

            return np.asarray(actions)

    def reset(self) -> None:
        """Rebuild the ``Algorithm`` and restore the initial weights.

        The previous ``Algorithm`` is stopped first (its learner thread, env
        runners and evaluation env runners are released), then a brand-new one
        is built from the stored config and the weights captured at actor
        construction are loaded into it, so every outer iteration starts the
        inner policy from the same parameters. The hash of the restored
        weights is logged for cross-run comparison.

        Raises
        ------
        Exception
            Whatever ``stop_learner_thread`` raised on a learner of the
            previous algorithm (for example ``TimeoutError``); the algorithm
            is stopped all the same and no new one is built.
        """

        self.stop()

        self.algo = self.algo_config.build_algo()

        self.algo.set_weights(self._init_weights)

        weights = self.algo.get_weights()

        logger.info("[PPO] Initial policy weight hash: %s", hash_weights(weights))

    def stop(self) -> None:
        """Stop the owned ``Algorithm`` and release its workers.

        The learner threads are ended first: ``Algorithm.stop`` leaves the
        thread of a learner running in this process polling its empty buffer
        (see ``stop_learner_thread``). The algorithm is stopped even when
        ending a thread fails, and that failure is then re-raised.

        Raises
        ------
        Exception
            Whatever ``stop_learner_thread`` raised on a learner (for example
            ``TimeoutError``), re-raised after the algorithm has been stopped.
        """

        try:
            for result in self.algo.learner_group.foreach_learner(stop_learner_thread):
                if not result.ok:
                    raise result.get()
        finally:
            self.algo.stop()
