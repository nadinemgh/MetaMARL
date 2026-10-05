"""Regulator environment of the cart-pole example: scores mechanism candidates.

``CartpoleRegulatorEnv`` is the environment the ES outer optimizer steps. One
step publishes the population of candidates to the World, lets the inner
optimizer train and evaluate the balancing agent against them (all handled by
the ``RegulatorEnv`` base class) and then calls
:meth:`CartpoleRegulatorEnv.reward`, which turns the inner metrics into one
fitness per candidate: the mean per-step reward of the agent. The candidate is
the inert ``dial`` mechanism, so the fitness measures how well the agent
balances, not what the mechanism does.
"""

import logging
from collections import defaultdict
from typing import Any

import numpy as np
import ray
from gymnasium.core import ObsType

from core.annotations import override
from core.envs.regulator import RegulatorEnv
from core.metrics.schemas import MetricSchema
from core.world.context import Context, MechanismContext, MechanismStatus
from examples.cartpole.contexts import FitnessContext

logger = logging.getLogger(__name__)


class CartpoleRegulatorEnv(RegulatorEnv):
    """Outer-loop environment that scores cart-pole mechanism candidates.

    The fitness of a candidate is the mean of the per-episode ``reward_mean``
    (the mean per-step reward, which is ``1.0`` for an agent that never drops
    the pole before the episode ends and smaller otherwise) over every episode
    and seed of the candidate, read from the ``train`` or ``eval`` split of the
    inner metrics selected by ``aggregation_status``.

    Parameters
    ----------
    aggregation_status : str, optional
        Split of the inner metrics the fitness is computed from, ``"train"`` or
        ``"eval"`` (default ``"eval"``).
    **kwargs : Any
        Forwarded to :class:`core.envs.regulator.RegulatorEnv`: ``world``,
        ``optimizer``, ``horizon``, ``agents_cfgs``, ``seeds``, ``schema``,
        ``queries`` and the other options of the base class.

    Attributes
    ----------
    aggregation_status : MechanismStatus
        Split of the inner metrics the fitness is computed from.
    last_metrics : list[dict[str, float]]
        One summary dictionary per scored candidate from the latest call of
        :meth:`reward`.

    Raises
    ------
    ValueError
        If ``aggregation_status`` is neither ``"train"`` nor ``"eval"``.

    When to use: as the ``env`` of the outer ``ESConfig`` of the cart-pole
    experiment, with the inner environment logging ``CartpoleMetricSchema``.

    Examples
    --------
    The environment is built with a stand-in World and no inner optimizer, and
    scores one candidate whose two episodes have a mean step reward of 1.0 and
    0.5:

    >>> from types import SimpleNamespace
    >>> from unittest import mock
    >>> import ray
    >>> world = SimpleNamespace(
    ...     append_context=SimpleNamespace(remote=lambda context: None)
    ... )
    >>> env = CartpoleRegulatorEnv(
    ...     world=world, optimizer=None, horizon=1, agents_cfgs={}, seeds=[0]
    ... )
    >>> episodes = {
    ...     "0": SimpleNamespace(reward_mean=[1.0]),
    ...     "1": SimpleNamespace(reward_mean=[0.5]),
    ... }
    >>> seed = SimpleNamespace(by_episode=episodes)
    >>> rollout = SimpleNamespace(
    ...     by_mechanism={"0": SimpleNamespace(by_seed={"0": seed})}
    ... )
    >>> metrics = SimpleNamespace(eval=SimpleNamespace(rollout=rollout))
    >>> with mock.patch.object(ray, "get", lambda ref: ref):
    ...     env.reward(metrics)
    [0.75]
    """

    def __init__(self, *, aggregation_status: str = "eval", **kwargs: Any):
        super().__init__(**kwargs)

        self.aggregation_status = MechanismStatus(aggregation_status)
        if self.aggregation_status not in (MechanismStatus.train, MechanismStatus.eval):
            raise ValueError(
                "aggregation_status must be 'train' or 'eval', "
                + f"got {aggregation_status!r}"
            )
        self.last_metrics: list[dict[str, float]] = []

    @override(RegulatorEnv)
    def observation(self, obs: ObsType) -> ObsType:
        """Return a constant ``0.0``; the ES outer loop ignores observations.

        Parameters
        ----------
        obs : ObsType
            Ignored.

        Returns
        -------
        ObsType
            The float ``0.0``.
        """

        return 0.0

    @override(RegulatorEnv)
    def reward(self, metrics: MetricSchema) -> list[float]:
        """Compute one fitness per candidate from the inner optimizer's metrics.

        For every candidate the method averages the ``reward_mean`` of each
        episode of each seed, then averages over the seeds, appends a ``done``
        ``MechanismContext`` that carries the ``FitnessContext`` to the World
        (blocking on the Ray call) and logs one summary line.

        Parameters
        ----------
        metrics : MetricSchema
            Inner metrics with a ``train`` or ``eval`` branch (the one named by
            ``aggregation_status``), each holding
            ``rollout.by_mechanism[id].by_seed[id].by_episode[id].reward_mean``.
            Mechanism identifiers must be convertible to ``int``.

        Returns
        -------
        list[float]
            Fitness indexed by mechanism id, of length ``max(id) + 1``. An
            index with no rollouts holds ``-inf``.

        Raises
        ------
        ValueError
            If the selected branch holds no mechanism.
        """

        branch = getattr(metrics, self.aggregation_status.value)
        if not branch.rollout.by_mechanism:
            raise ValueError("The inner metrics hold no mechanism to score.")

        per_mechanism: dict[int, list[float]] = defaultdict(list)
        for mechanism_id, mechanism_metrics in branch.rollout.by_mechanism.items():
            for seed_metrics in mechanism_metrics.by_seed.values():
                seed_means = [
                    float(np.mean(np.atleast_1d(np.asarray(e.reward_mean, np.float32))))
                    for e in seed_metrics.by_episode.values()
                ]
                per_mechanism[int(mechanism_id)].append(float(np.mean(seed_means)))

        fitness = np.full(max(per_mechanism) + 1, -np.inf, dtype=np.float32)
        summaries: list[dict[str, float]] = []

        for idx, seed_values in per_mechanism.items():
            mean_reward = float(np.mean(seed_values))
            fitness[idx] = mean_reward
            context = FitnessContext.from_mean_reward(mean_reward)
            ray.get(
                self.world.append_context.remote(
                    Context(
                        id=None,
                        opt_id=self._opt_id,
                        step=self._t,
                        env=self.__class__.__name__,
                        payload=MechanismContext(
                            index=idx,
                            seed=None,
                            status=MechanismStatus.done,
                            env_id=None,
                            mechanism=None,
                            metrics=context,
                        ),
                    )
                )
            )
            summaries.append(
                {
                    "idx": float(idx),
                    "objective": mean_reward,
                    "num_seeds": float(len(seed_values)),
                }
            )

        logger.info(
            "[Regulator][summary] mean_obj=%.4f | best_obj=%.4f | worst_obj=%.4f",
            float(np.mean([s["objective"] for s in summaries])),
            max(s["objective"] for s in summaries),
            min(s["objective"] for s in summaries),
        )
        self.last_metrics = summaries
        return fitness.tolist()
