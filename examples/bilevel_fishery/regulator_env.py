"""Regulator environment of the fishery example: scores mechanism candidates.

``FisheryRegulatorEnv`` is the environment the ES outer optimizer steps. One
step publishes the population of candidate mechanisms to the World, lets the
inner optimizer train and evaluate the fishers against them (all of this is
handled by the ``RegulatorEnv`` base class) and then calls
:meth:`FisheryRegulatorEnv.reward`, which turns the inner metrics into one
fitness per candidate through a ``FitnessContext``. The objective rewards the
harvest relative to the maximum sustainable yield and the mean biomass of the
stock; see the class docstring for what the aggregated series contain.
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
from examples.bilevel_fishery.contexts import FitnessContext

logger = logging.getLogger(__name__)


class FisheryRegulatorEnv(RegulatorEnv):
    """Outer-loop environment that scores fishery mechanism candidates.

    The fitness of a candidate is computed from the inner rollouts of the
    ``train`` or ``eval`` split selected by ``aggregation_status``. For each
    mechanism, seed and episode the environment reads the step-by-step series
    that the inner environment logged for that episode, ``reward_series``,
    ``fish_norm_next_series`` and ``H_realized_series``, and the episode's
    ``MSY``, keeps the last ``fitness_tail_steps`` steps of each series, and
    summarizes them. The statistics of every episode and seed of a candidate
    are then averaged and folded into a ``FitnessContext``, whose objective is
    ``harvest_score + sustainability_weight * mean_fish``.

    The tail window measures the steady state of the last steps of an episode:
    ``mean_fish`` is the mean normalized biomass over those steps, ``min_fish``
    their minimum, ``collapse_rate`` the fraction of them below the
    sustainability threshold, and ``harvest_score`` the mean realized harvest
    over them divided by the maximum sustainable yield. An episode shorter than
    ``fitness_tail_steps`` contributes all its steps. With the ``train`` split
    an episode that was logged at several iterations contributes once per
    iteration.

    Parameters
    ----------
    ecology_cfg : dict[str, Any]
        Objective settings, read with these keys: ``sustainability_weight``
        (weight of the mean normalized biomass in the objective, dimensionless,
        default 5.0), ``sustainability_threshold`` (normalized biomass in
        ``[0, 1]`` below which an entry counts as collapsed, default 0.1),
        ``K`` (carrying capacity in biomass units, used to denormalize the
        threshold for plots; there is no default, and omitting it raises a
        ``ValueError``), ``aggregation_status`` (``"train"`` or ``"eval"``,
        default ``"eval"``) and ``fitness_tail_steps`` (number of trailing
        steps of each episode the fitness is computed on, default 50; an
        episode with fewer steps contributes all of them).
    **kwargs : Any
        Forwarded to :class:`core.envs.regulator.RegulatorEnv`: ``world``,
        ``optimizer``, ``horizon``, ``agents_cfgs``, ``seeds``, ``schema``,
        ``queries`` and the other options of the base class.

    Attributes
    ----------
    sustainability_weight : float
        Weight of ``mean_fish`` in the objective.
    sustainability_threshold : float
        Normalized biomass below which an entry counts as collapsed.
    K : float
        Carrying capacity (biomass units).
    raw_sustainability_threshold : float
        ``sustainability_threshold * K`` (biomass units), for plots.
    aggregation_status : MechanismStatus
        Split of the inner metrics the fitness is computed from.
    fitness_tail_steps : int
        Number of trailing steps of each episode the fitness is computed on.
    trajectories : dict[int, list[dict[str, Any]]]
        Reset to ``{}`` at every call of :meth:`reward` and never filled.
    last_metrics : list[dict[str, float]]
        One summary dictionary per scored candidate from the latest call of
        :meth:`reward`, in the order the candidates were visited.

    Raises
    ------
    ValueError
        If ``aggregation_status`` is neither ``"train"`` nor ``"eval"``, or if
        ``ecology_cfg`` has no ``K``.

    When to use: as the ``env`` of the outer ``ESConfig`` of a fishery
    experiment, with an ``ecology_cfg`` whose ``K`` matches the inner
    environment's carrying capacity. The objective depends on the series the
    inner environment logs, so use it with ``FisheryMetricSchema``.

    Examples
    --------
    The environment is built with a stand-in World and no inner optimizer, and
    scores one candidate whose single episode of two steps has a biomass of half
    the carrying capacity and a harvest of half the maximum sustainable yield
    at every step (``0.5 + 2.0 * 0.5``):

    >>> from types import SimpleNamespace
    >>> from unittest import mock
    >>> import ray
    >>> world = SimpleNamespace(
    ...     append_context=SimpleNamespace(remote=lambda context: None)
    ... )
    >>> env = FisheryRegulatorEnv(
    ...     world=world,
    ...     optimizer=None,
    ...     horizon=1,
    ...     agents_cfgs={},
    ...     seeds=[0],
    ...     ecology_cfg={"K": 5000.0, "sustainability_weight": 2.0},
    ... )
    >>> episode = SimpleNamespace(
    ...     reward_series=[[0.5, 0.5]],
    ...     fish_norm_next_series=[[0.5, 0.5]],
    ...     H_realized_series=[[37.5, 37.5]],
    ...     MSY=[75.0],
    ... )
    >>> seed = SimpleNamespace(by_episode={"0": episode})
    >>> rollout = SimpleNamespace(
    ...     by_mechanism={"0": SimpleNamespace(by_seed={"0": seed})}
    ... )
    >>> metrics = SimpleNamespace(eval=SimpleNamespace(rollout=rollout))
    >>> with mock.patch.object(ray, "get", lambda ref: ref):
    ...     env.reward(metrics)
    [1.5]
    """

    def __init__(self, *, ecology_cfg: dict[str, Any], **kwargs: Any):
        super().__init__(**kwargs)

        self.sustainability_weight = ecology_cfg.get("sustainability_weight", 5.0)
        self.sustainability_threshold = ecology_cfg.get("sustainability_threshold", 0.1)
        if "K" not in ecology_cfg:
            raise ValueError(
                "ecology_cfg must define 'K', the carrying capacity in biomass "
                + "units: the regulator needs it to denormalize the "
                + "sustainability threshold."
            )
        self.K = ecology_cfg["K"]

        # Denormalized threshold for visualization
        self.raw_sustainability_threshold = self.sustainability_threshold * self.K
        self.trajectories: dict[int, list[dict[str, Any]]] = {}
        self.last_metrics: list[dict[str, float]] = []
        target_status = ecology_cfg.get("aggregation_status", "eval")
        self.aggregation_status = MechanismStatus(target_status)

        # tail averaging
        self.fitness_tail_steps = int(ecology_cfg.get("fitness_tail_steps", 50))

    @override(RegulatorEnv)
    def observation(self, obs: ObsType) -> ObsType:
        """Return a constant ``0.0``.

        The ES outer loop is stateless and ignores observations. The step of
        the base class does not call this method, so it is never reached
        during a run.

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

        ``metrics`` is the inner ``RaySchema`` peeked after training; the
        ``aggregation_status`` split (``train`` or ``eval``) is read, then the
        rollouts are walked by mechanism, seed and episode. Each episode
        contributes the statistics of the last ``fitness_tail_steps`` steps of
        its reward, biomass and harvest series (all its steps when it is
        shorter than the window); the episodes of all seeds of a mechanism are
        averaged and folded into a ``FitnessContext``, whose
        ``objective_score`` is the fitness.

        For each candidate the method also appends a ``done``
        ``MechanismContext`` carrying the ``FitnessContext`` to the World
        (blocking on the Ray call), stores a summary dictionary in
        ``last_metrics`` and logs one summary line with the mean, best and
        worst objective and the collapse rates.

        Parameters
        ----------
        metrics : MetricSchema
            Inner metrics with a ``train`` or ``eval`` branch (the one named by
            ``aggregation_status``), each holding
            ``rollout.by_mechanism[id].by_seed[id].by_episode[id]`` records
            with ``reward_series``, ``fish_norm_next_series``,
            ``H_realized_series`` (each a list holding one list of steps per
            logged episode) and ``MSY`` (one value per logged episode).
            Mechanism and seed identifiers must be convertible to
            ``int``.

        Returns
        -------
        list[float]
            Fitness indexed by mechanism id, of length ``max(id) + 1``. An
            index with no rollouts holds ``-inf``.

        Raises
        ------
        ValueError
            If the selected branch holds no mechanism (``max`` of an empty
            sequence).
        """

        metrics = getattr(metrics, self.aggregation_status.value)

        per_mech_metrics: list[dict[str, float]] = []

        metrics_by_mechanism: dict[int, list[dict[str, Any]]] = defaultdict(list)

        self.trajectories = {}

        for mechanism_id, mechanism_metrics in metrics.rollout.by_mechanism.items():
            idx = int(mechanism_id)

            for seed_id, seed_metrics in mechanism_metrics.by_seed.items():
                seed = int(seed_id)

                for episode_metrics in seed_metrics.by_episode.values():
                    # Each leaf holds one entry per logged episode: the list of
                    # its steps, or its MSY.
                    logged_episodes = zip(
                        episode_metrics.reward_series,
                        episode_metrics.fish_norm_next_series,
                        episode_metrics.H_realized_series,
                        np.atleast_1d(episode_metrics.MSY),
                        strict=True,
                    )
                    for reward_steps, fish_steps, harvest_steps, msy in logged_episodes:
                        rewards = np.asarray(reward_steps, dtype=np.float64)
                        fish = np.asarray(fish_steps, dtype=np.float64)
                        realized_harvest = np.asarray(harvest_steps, dtype=np.float64)
                        harvest_scores = realized_harvest / max(float(msy), 1e-6)
                        num_steps = len(fish)
                        tail_steps = min(self.fitness_tail_steps, num_steps)
                        tail_start = num_steps - tail_steps
                        tail_rewards = rewards[tail_start:]
                        tail_fish = fish[tail_start:]
                        tail_realized_harvest = realized_harvest[tail_start:]
                        tail_harvest_scores = harvest_scores[tail_start:]
                        sustainability_penalties = np.maximum(
                            0.0,
                            (self.sustainability_threshold - tail_fish)
                            / max(1e-6, self.sustainability_threshold),
                        )

                        metrics_by_mechanism[idx].append(
                            {
                                "seed": seed,
                                "mean_reward": float(tail_rewards.mean()),
                                "reward_std": float(tail_rewards.std()),
                                "mean_realized_harvest": float(
                                    tail_realized_harvest.mean()
                                ),
                                "harvest_score": float(tail_harvest_scores.mean()),
                                "collapse_rate": float(
                                    (tail_fish < self.sustainability_threshold).mean()
                                ),
                                "sustainability_penalty": float(
                                    sustainability_penalties.mean()
                                ),
                                "min_fish": float(tail_fish.min()),
                                "mean_fish": float(tail_fish.mean()),
                            }
                        )

        max_idx = max(metrics_by_mechanism)
        fitness = np.full(max_idx + 1, -np.inf, dtype=np.float32)

        for idx, seed_metrics in metrics_by_mechanism.items():
            mean_reward = float(np.mean([m["mean_reward"] for m in seed_metrics]))
            reward_std = float(np.mean([m["reward_std"] for m in seed_metrics]))
            mean_realized_harvest = float(
                np.mean([m["mean_realized_harvest"] for m in seed_metrics])
            )
            harvest_score = float(np.mean([m["harvest_score"] for m in seed_metrics]))
            collapse_rate = float(np.mean([m["collapse_rate"] for m in seed_metrics]))
            sustainability_penalty = float(
                np.mean([m["sustainability_penalty"] for m in seed_metrics])
            )
            min_fish = float(np.mean([m["min_fish"] for m in seed_metrics]))
            mean_fish = float(np.mean([m["mean_fish"] for m in seed_metrics]))
            fitness_ctx = FitnessContext.from_metrics(
                mean_reward=mean_reward,
                collapse_rate=collapse_rate,
                sustainability_penalty=sustainability_penalty,
                sustainability_weight=self.sustainability_weight,
                mean_fish=mean_fish,
                min_fish=min_fish,
                mean_realized_harvest=mean_realized_harvest,
                harvest_score=harvest_score,
            )
            objective = float(fitness_ctx.objective_score)
            fitness[idx] = objective

            ctx = Context(
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
                    metrics=fitness_ctx,
                ),
            )
            ray.get(self.world.append_context.remote(ctx))

            per_mech_metrics.append(
                {
                    "idx": idx,
                    "objective": objective,
                    "mean_reward": mean_reward,
                    "reward_std": reward_std,
                    "mean_realized_harvest": mean_realized_harvest,
                    "harvest_score": harvest_score,
                    "collapse_rate": collapse_rate,
                    "min_fish": min_fish,
                    "mean_fish": mean_fish,
                    "num_seeds": float(len(seed_metrics)),
                }
            )

        objectives = np.asarray(
            [m["objective"] for m in per_mech_metrics], dtype=np.float32
        )
        collapse_rates = np.asarray(
            [m["collapse_rate"] for m in per_mech_metrics], dtype=np.float32
        )
        best_position = int(np.argmax(objectives))
        worst_position = int(np.argmin(objectives))
        best = per_mech_metrics[best_position]
        worst = per_mech_metrics[worst_position]

        logger.info(
            "[Regulator][summary] "
            + "mean_obj=%.4f | best_obj=%.4f (θ=%d) | "
            + "worst_obj=%.4f (θ=%d) | "
            + "collapse(mean=%.3f max=%.3f)",
            float(objectives.mean()),
            best["objective"],
            int(best["idx"]),
            worst["objective"],
            int(worst["idx"]),
            float(collapse_rates.mean()),
            float(collapse_rates.max()),
        )

        self.last_metrics = per_mech_metrics

        return fitness.tolist()
