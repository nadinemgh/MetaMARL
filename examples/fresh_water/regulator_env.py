"""Regulator environment of the fresh-water example: scores water policies.

``WaterRegulatorEnv`` is the environment the ES outer optimizer steps. One step
publishes the population of candidate policies to the World, lets the inner
optimizer train the farms against them (handled by the ``RegulatorEnv`` base
class) and then calls :meth:`WaterRegulatorEnv.reward`, which turns the inner
metrics into one fitness per candidate:

    fitness = economic_weight * mean reward
              + sustainability_weight / (1 + level deviation),

where the deviation is the relative change of the lake's filled level caused by
the withdrawals, ``sum|L - L0| / sum|L0|``, against the world with no
withdrawals that the lake model runs in lockstep (see
:mod:`examples.fresh_water.hydrology`). The level is the measure because it is
the one quantity the irrigation moves in both lake models: the inflow is
upstream of the reservoir and the release only changes when the reservoir spills
or runs dry.
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
from examples.fresh_water.contexts import FitnessContext

logger = logging.getLogger(__name__)

EPS = 1e-8

LEVEL_FIELD = "reservoir_level_norm_series"
"""Metric field holding the filled fraction of the lake at every step."""

BASELINE_LEVEL_FIELD = "baseline_reservoir_level_norm_series"
"""Metric field holding the filled fraction of the lake with no withdrawal."""


def level_deviation(level: Any, baseline: Any) -> float:
    """Return the relative change of the lake level against its baseline.

    Parameters
    ----------
    level : array_like
        Filled fraction of the lake with the withdrawals, shape ``(T,)``,
        dimensionless (``0`` empty, ``1`` full).
    baseline : array_like
        Filled fraction of the lake without withdrawals on the same days, shape
        ``(T,)``.

    Returns
    -------
    float
        ``sum|level - baseline| / max(EPS, sum|baseline|)``; 0 when the two
        series are equal.

    Raises
    ------
    ValueError
        If the two series have different lengths.

    When to use: to score how far the farms lower the lake, once per episode.

    Examples
    --------
    >>> level_deviation([0.5, 0.75], [1.0, 1.0])
    0.375
    >>> level_deviation([0.5], [0.5])
    0.0
    """
    level_array = np.asarray(level, dtype=np.float64)
    baseline_array = np.asarray(baseline, dtype=np.float64)
    if level_array.shape != baseline_array.shape:
        raise ValueError(
            f"level has shape {level_array.shape} but its baseline has shape "
            + f"{baseline_array.shape}."
        )
    numerator = float(np.abs(level_array - baseline_array).sum())
    return numerator / max(EPS, float(np.abs(baseline_array).sum()))


class WaterRegulatorEnv(RegulatorEnv):
    """Outer-loop environment that scores water-policy candidates.

    Parameters
    ----------
    ecology_cfg : dict, optional
        Options of the score, read with these keys: ``economic_weight`` (1.0),
        ``sustainability_weight`` (1.0) and ``aggregation_status`` (``"train"``
        or ``"eval"``, default ``"train"``, the split of the inner metrics the
        fitness is computed from). The deviation is always read on the lake
        level: the inflow is upstream of the reservoir, so the withdrawals
        cannot change it, and the release moves only when the reservoir spills
        or runs dry, whereas the level moves with every withdrawal.
    **kwargs : Any
        Forwarded to :class:`core.envs.regulator.RegulatorEnv`: ``world``,
        ``optimizer``, ``horizon``, ``agents_cfgs``, ``seeds``, ``schema``,
        ``queries`` and the other options of the base class.

    Attributes
    ----------
    economic_weight, sustainability_weight : float
        Weights of the two terms of the fitness.
    aggregation_status : MechanismStatus
        Split of the inner metrics the fitness is computed from.
    last_metrics : list[dict[str, float]]
        One summary dictionary per scored candidate from the latest call of
        :meth:`reward`.

    Raises
    ------
    ValueError
        If ``aggregation_status`` is neither ``"train"`` nor ``"eval"``.

    When to use: as the ``env`` of the outer ``ESConfig`` of the fresh-water
    experiment, with the inner environment logging ``WaterMetricSchema``.

    Examples
    --------
    One candidate whose single episode has a mean reward of 0.5 and a lake level
    that is 10 % below its baseline on every day (so ``sum|L - L0| / sum|L0|``
    is 0.1, the sustainability score ``1 / 1.1`` and the fitness
    ``0.5 + 1 / 1.1``):

    >>> from types import SimpleNamespace
    >>> from unittest import mock
    >>> import ray
    >>> world = SimpleNamespace(
    ...     append_context=SimpleNamespace(remote=lambda context: None)
    ... )
    >>> env = WaterRegulatorEnv(
    ...     world=world,
    ...     optimizer=None,
    ...     horizon=1,
    ...     agents_cfgs={},
    ...     seeds=[0],
    ... )
    >>> episode = SimpleNamespace(
    ...     reward_mean=[0.5],
    ...     reservoir_level_norm_series=[[0.45, 0.9]],
    ...     baseline_reservoir_level_norm_series=[[0.5, 1.0]],
    ... )
    >>> seed = SimpleNamespace(by_episode={"0": episode})
    >>> rollout = SimpleNamespace(
    ...     by_mechanism={"0": SimpleNamespace(by_seed={"0": seed})}
    ... )
    >>> metrics = SimpleNamespace(train=SimpleNamespace(rollout=rollout))
    >>> with mock.patch.object(ray, "get", lambda ref: ref):
    ...     fitness = env.reward(metrics)
    >>> [round(value, 4) for value in fitness]
    [1.4091]
    """

    def __init__(self, *, ecology_cfg: dict[str, Any] | None = None, **kwargs: Any):
        super().__init__(**kwargs)

        options = ecology_cfg or {}
        self.economic_weight = float(options.get("economic_weight", 1.0))
        self.sustainability_weight = float(options.get("sustainability_weight", 1.0))
        self.aggregation_status = MechanismStatus(
            options.get("aggregation_status", "train")
        )
        if self.aggregation_status not in (MechanismStatus.train, MechanismStatus.eval):
            raise ValueError(
                "aggregation_status must be 'train' or 'eval', "
                + f"got {self.aggregation_status.value!r}."
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

        For every logged episode the method takes the mean per-step reward and
        the level deviation against the baseline; both are averaged over the
        episodes and seeds of the candidate and combined by
        :meth:`FitnessContext.from_scores`. A ``done`` ``MechanismContext`` that
        carries the ``FitnessContext`` is appended to the World (blocking on the
        Ray call) and one summary line is logged.

        Parameters
        ----------
        metrics : MetricSchema
            Inner metrics with a ``train`` or ``eval`` branch (the one named by
            ``aggregation_status``), each holding
            ``rollout.by_mechanism[id].by_seed[id].by_episode[id]`` records
            with ``reward_mean`` (one value per logged episode) and the pair of
            series ``reservoir_level_norm_series`` and
            ``baseline_reservoir_level_norm_series`` (each a list holding one
            list of days per logged episode); an entry is ``None`` where the
            metric logger recorded a gap, and gaps are skipped. Mechanism
            identifiers must be convertible to ``int``.

        Returns
        -------
        list[float]
            Fitness indexed by mechanism id, of length ``max(id) + 1``. An
            index with no rollouts holds ``-inf``.

        Raises
        ------
        ValueError
            If the selected split holds no rollout, so there is no candidate to
            score; the message names the split.
        """
        branch = getattr(metrics, self.aggregation_status.value)

        rewards: dict[int, list[float]] = defaultdict(list)
        deviations: dict[int, list[float]] = defaultdict(list)
        for mechanism_id, mechanism_metrics in branch.rollout.by_mechanism.items():
            idx = int(mechanism_id)
            for seed_metrics in mechanism_metrics.by_seed.values():
                for episode in seed_metrics.by_episode.values():
                    # Each leaf holds one entry per logged episode: its mean
                    # reward, or the list of its days.
                    logged_episodes = zip(
                        np.atleast_1d(episode.reward_mean),
                        getattr(episode, LEVEL_FIELD),
                        getattr(episode, BASELINE_LEVEL_FIELD),
                        strict=True,
                    )
                    for mean_reward, level, baseline in logged_episodes:
                        # A gap of the metric logger: no episode under this key
                        # in that inner iteration.
                        if mean_reward is None:
                            continue

                        rewards[idx].append(float(mean_reward))
                        deviations[idx].append(level_deviation(level, baseline))

        if not rewards:
            raise ValueError(
                f"No rollout found for the {self.aggregation_status.value!r} "
                + "split: the inner optimizer logged no mechanism, seed and "
                + "episode to score."
            )

        fitness = np.full(max(rewards) + 1, -np.inf, dtype=np.float32)
        summaries: list[dict[str, float]] = []
        for idx in rewards:
            context = FitnessContext.from_scores(
                economic_score=float(np.mean(rewards[idx])),
                level_deviation=float(np.mean(deviations[idx])),
                economic_weight=self.economic_weight,
                sustainability_weight=self.sustainability_weight,
            )
            fitness[idx] = context.objective_score
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
                    "objective": context.objective_score,
                    "economic_score": context.economic_score,
                    "level_deviation": context.level_deviation,
                    "sustainability_score": context.sustainability_score,
                }
            )

        logger.info(
            "[Regulator][summary] mean_obj=%.4f | best_obj=%.4f | worst_obj=%.4f | "
            + "mean_dev=%.4f",
            float(np.mean([s["objective"] for s in summaries])),
            max(s["objective"] for s in summaries),
            min(s["objective"] for s in summaries),
            float(np.mean([s["level_deviation"] for s in summaries])),
        )
        self.last_metrics = summaries
        return fitness.tolist()
